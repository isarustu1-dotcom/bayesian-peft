#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH -J blob_csqa_asymduos

export DATASET=csqa
export MAX_SEQ_LEN=300

source "$HOME/bayesian-peft/scripts/asymmetric_duos/_common.sh"
