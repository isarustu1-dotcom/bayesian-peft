#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH -J blob_race_asymduos

# RACE articles are substantially longer than other datasets; use 512 tokens
# (matches configs/race_qwen2_*/... in the asymmetric-llm-duos repo).
export DATASET=race
export MAX_SEQ_LEN=512

source "$HOME/bayesian-peft/scripts/asymmetric_duos/_common.sh"
