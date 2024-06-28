#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="train_RNN"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_RNN.out
#SBATCH --error=train_RNN.err
#SBATCH --partition=caps-gpu

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

python train_RNN.py