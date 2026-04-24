"""
Dataset adapter that plugs the `asymmetric-llm-duos` (ib_edl) dataset classes,
splits and prompt templates into bayesian-peft's BLoB training loop.

Why this exists
---------------
The asymmetric-duos project needs BLoB predictions on six datasets
(arc_c, arc_e, obqa, csqa, race, sciq) under the exact same splits, prompts
and tokenisation used by the rest of that project, so that downstream duo /
deep-ensemble / calibration code can consume the resulting logits without
modification.

Rather than re-implement those dataset classes here, we import them directly
from the asymmetric-llm-duos repo (its path is passed via the IB_EDL_ROOT env
var) and wrap them in a bayesian-peft-compatible `DatasetBase` subclass that
yields the `(tokenized_prompts_dict, classes_long, targets_long)` batch tuple
expected by `modelwrappers/blob.py::forward_logits`.

Registered as `NAME = "ib_edl_mcdataset"`. The sbatch scripts pass
`--dataset-type ib_edl_mcdataset`. Downstream wrapper code only special-cases
`dataset_type == "mcdataset"`, so we mutate `args.dataset_type = "mcdataset"`
in-place at the end of `__init__` (the registry lookup has already happened by
then via `run.get_dataset`).
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.utils.datasetbase import DatasetBase


# ---------------------------------------------------------------------------
# Mapping from our --dataset value to (ib_edl class name, extra kwargs,
# per-split dataset_cfg). Mirrors configs/_base_/<dataset>.yaml in the
# asymmetric-llm-duos repo. Keep in sync with those files.
# ---------------------------------------------------------------------------
_DATASET_SPEC: Dict[str, Dict[str, Any]] = {
    "arc_c": {
        "cls": "ARCDataset",
        "kwargs": {"name_suffix": "C", "add_space": True},
        "splits": {
            "train": {"split": "train"},
            "val":   {"split": "validation"},
            "test":  {"split": "test"},
        },
        "num_labels": 5,
    },
    "arc_e": {
        "cls": "ARCDataset",
        "kwargs": {"name_suffix": "E", "add_space": True},
        "splits": {
            "train": {"split": "train"},
            "val":   {"split": "validation"},
            "test":  {"split": "test"},
        },
        "num_labels": 5,
    },
    "obqa": {
        "cls": "OBQADataset",
        "kwargs": {"add_space": True},
        "splits": {
            "train": {"split": "train"},
            "val":   {"split": "validation"},
            "test":  {"split": "test"},
        },
        "num_labels": 4,
    },
    "csqa": {
        "cls": "CSQADataset",
        "kwargs": {"add_space": True},
        "splits": {
            # CSQA test set has no labels; we use 10%/90% of train as val/train
            # and the official validation split as test. Matches configs/_base_/csqa.yaml.
            "train": {"split": "train[10%:]"},
            "val":   {"split": "train[:10%]"},
            "test":  {"split": "validation"},
        },
        "num_labels": 5,
    },
    "race": {
        "cls": "RaceDataset",
        "kwargs": {"add_space": True},
        "splits": {
            "train": {"split": "train"},
            "val":   {"split": "validation"},
            "test":  {"split": "test"},
        },
        "num_labels": 4,
    },
    "sciq": {
        "cls": "SciQDataset",
        "kwargs": {"add_space": True},
        "splits": {
            "train": {"split": "train"},
            "val":   {"split": "validation"},
            "test":  {"split": "test"},
        },
        "num_labels": 4,
    },
}


def _import_ib_edl_datasets():
    """Add IB_EDL_ROOT to sys.path and return the ib_edl.datasets module."""
    root = os.environ.get("IB_EDL_ROOT")
    if root is None:
        raise RuntimeError(
            "IB_EDL_ROOT environment variable is not set. The ib_edl_mcdataset "
            "adapter imports dataset classes and prompt templates from the "
            "asymmetric-llm-duos repo; point IB_EDL_ROOT at its clone."
        )
    if not os.path.isdir(root):
        raise RuntimeError(f"IB_EDL_ROOT={root!r} is not a directory.")
    if root not in sys.path:
        sys.path.insert(0, root)

    import ib_edl.datasets as ib_edl_datasets  # noqa: E402
    return ib_edl_datasets


class IbEdlMcDataset(DatasetBase):
    """Multi-choice dataset adapter backed by ib_edl's dataset classes + prompts."""

    NAME = "ib_edl_mcdataset"

    def __init__(self, accelerator, args):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        if args.dataset not in _DATASET_SPEC:
            raise NotImplementedError(
                f"ib_edl_mcdataset does not know dataset {args.dataset!r}. "
                f"Known: {sorted(_DATASET_SPEC.keys())}"
            )
        spec = _DATASET_SPEC[args.dataset]
        self.num_labels = spec["num_labels"]

        ib_mod = _import_ib_edl_datasets()
        cls = getattr(ib_mod, spec["cls"])

        # Tokenizer — matches configs/_base_/qwen2_7b.yaml / qwen2_15b.yaml:
        # Qwen2 has no dedicated pad token; reuse <|endoftext|>. Left padding
        # for causal LM (matters for per-batch last-token logit extraction).
        accelerator.wait_for_everyone()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, use_fast=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            # Qwen2 family: endoftext is always present
            self.tokenizer.pad_token = "<|endoftext|>"

        # Instantiate one ib_edl dataset per split.
        common_kwargs = dict(spec["kwargs"])
        self._splits: Dict[str, Any] = {}
        for split_name, dcfg in spec["splits"].items():
            self._splits[split_name] = cls(
                dataset_cfg=dcfg,
                tokenizer=self.tokenizer,
                **common_kwargs,
            )

        # Target token ids come from the ib_edl dataset (they are computed
        # from the tokenizer's encoding of " A" .. " E", taking the last
        # sub-token). Shape: (n_labels,). Wrapper code does `.squeeze(-1)` on
        # the attribute, so we re-add a trailing dim to match S2SDataset_Classification.
        tids = self._splits["train"].target_ids  # already squeezed, shape (n_labels,)
        if tids.dim() == 1:
            tids = tids.unsqueeze(-1)
        self.target_ids = tids  # (n_labels, 1)
        self._label2target = OrderedDict(
            [(i, self.target_ids[i]) for i in range(self.num_labels)]
        )

        if accelerator.is_local_main_process:
            print("=====================================")
            print(f"[ib_edl_mcdataset] Loaded {args.dataset}")
            print(f"  sizes: train={len(self._splits['train'])} "
                  f"val={len(self._splits['val'])} test={len(self._splits['test'])}")
            print(f"  num_labels: {self.num_labels}")
            print(f"  target_ids: {self.target_ids.squeeze(-1).tolist()}")
            print("=====================================")

        # Rewrite dataset_type so downstream wrapper checks
        # (`args.dataset_type == "mcdataset"`) continue to work unchanged.
        args.dataset_type = "mcdataset"

    # ------------------------------------------------------------------
    # Collation: produces the (inputs_dict, classes_long, targets_long)
    # 3-tuple that blob.forward_logits / wrapperbase.fit / evaluate expect.
    # ------------------------------------------------------------------
    def _make_collate_fn(self, ib_split) -> Callable:
        tokenizer = self.tokenizer
        max_seq_len = self.args.max_seq_len
        label2target = self._label2target

        def _collate(batch: List[Dict[str, Any]]) -> Tuple[Any, torch.Tensor, torch.Tensor]:
            prompts = [s["prompt"] for s in batch]
            classes = torch.tensor([int(s["label"]) for s in batch], dtype=torch.long)
            tok = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_seq_len,
            )
            targets = torch.cat([label2target[c.item()] for c in classes])
            return tok, classes, targets

        return _collate

    def _build_loader(self, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        ib_split = self._splits[split]
        return DataLoader(
            ib_split,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self._make_collate_fn(ib_split),
        )

    # ------------------------------------------------------------------
    # DatasetBase API
    # ------------------------------------------------------------------
    def get_loaders(self):
        # drop_last=True on train to match the paper reference script
        # (S2SDataset_Classification uses the same default through `ClassificationDataset.loader`).
        self.train_dataloader = self._build_loader("train", shuffle=True,  drop_last=True)
        self.val_dataloader   = self._build_loader("val",   shuffle=False, drop_last=False)
        self.test_dataloader  = self._build_loader("test",  shuffle=False, drop_last=False)

        # num_samples = true train count (counted from the loader to exclude dropped tail).
        total = 0
        for batch in self.train_dataloader:
            total += batch[1].size(0)
        self.num_samples = total

    # ------------------------------------------------------------------
    # Helpers used by the prediction-dump hook in wrapperbase.py to write
    # npz files compatible with ib_edl.utils.misc.save_predictions.
    # ------------------------------------------------------------------
    def get_data_indices(self, split: str):
        return self._splits[split].get_data_indices()

    def get_input_text(self, split: str):
        return self._splits[split].get_input_text()


# Required for run.get_all_datasets() to discover the class — it looks for
# subclasses of DatasetBase in any module under dataset/.
ib_edl_mcdataset = IbEdlMcDataset
