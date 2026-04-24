#!/bin/bash
# Shared logic for asymmetric-duos BLoB runs.
#
# Each per-dataset sbatch sets:
#   DATASET         — arc_c / arc_e / obqa / csqa / race / sciq
#   MAX_SEQ_LEN     — 300 for most, 512 for race
# Then sources this file to run 5 seeds × {base, sidekick}.
#
# Produces npz files at:
#   $HOME/results/blob/{base,sidekick}/{val_preds,test_preds}/<dataset>.npz
# matching the schema used by the asymmetric-llm-duos `average_metrics.py`
# and `optimize_weights_blob` pipelines.

set -euo pipefail

: "${DATASET:?must set DATASET before sourcing _common.sh}"
: "${MAX_SEQ_LEN:?must set MAX_SEQ_LEN before sourcing _common.sh}"

cd "$HOME/bayesian-peft"

source "$HOME/miniforge3/bin/activate"
eval "$(mamba shell hook --shell bash)"
mamba activate adl-asymmetric-lm-duos

# Make sure our fork is on PYTHONPATH (sys.path append in run/main.py relies
# on __file__, so `cd bayesian-peft` is enough; nothing extra needed here).

# Point the adapter at our asymmetric-llm-duos clone so it can import
# ib_edl.datasets (prompt templates, splits, get_data_indices / get_input_text).
export IB_EDL_ROOT="$HOME/asymmetric-llm-duos"

BASE_MODEL="Qwen/Qwen2-7B"
SIDE_MODEL="Qwen/Qwen2-1.5B"
RESULTS_ROOT="$HOME/results"

for ROLE in base sidekick; do
    if [ "$ROLE" = "base" ]; then
        MODEL="$BASE_MODEL"
    else
        MODEL="$SIDE_MODEL"
    fi
    for SEED in 1 2 3 4 5; do
        NAME="blob-${DATASET}-${ROLE}-seed${SEED}"
        echo "=============================================================="
        echo "[BLoB $DATASET] role=$ROLE seed=$SEED model=$MODEL max_seq_len=$MAX_SEQ_LEN"
        echo "=============================================================="
        CUDA_VISIBLE_DEVICES=0 python run/main.py \
            --dataset-type ib_edl_mcdataset --dataset "$DATASET" \
            --model-type causallm --model "$MODEL" --modelwrapper blob \
            --lora-target-modules q_proj k_proj v_proj o_proj \
            --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --lr 1e-4 --batch-size 4 --opt adamw --warmup-ratio 0.06 \
            --max-seq-len "$MAX_SEQ_LEN" \
            --max-train-steps 5000 --eval-per-steps 6000 \
            --seed "$SEED" --evaluate \
            --bayes-eps 0.05 --bayes-beta 0.2 --bayes-gamma 8 --bayes-kllr 0.01 \
            --bayes-klreweighting --bayes-datasetrescaling \
            --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final 10 \
            --model-role "$ROLE" \
            --results-root "$RESULTS_ROOT" \
            --dump-val-predictions \
            --wandb-name "$NAME" --wandb-project "BLoB-qwen2-asymmetric-duos" \
            --log-path "$NAME"
        echo "[BLoB $DATASET] role=$ROLE seed=$SEED — done."
    done
done

echo "All $DATASET runs complete. Predictions under $RESULTS_ROOT/blob/{base,sidekick}/{val_preds,test_preds}/$DATASET.npz"
