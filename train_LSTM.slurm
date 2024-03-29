#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="train_LSTM"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_LSTM.out
#SBATCH --error=train_LSTM.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

python train_LSTM_model.p