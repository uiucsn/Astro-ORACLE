#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --job-name="train_RNN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=train_RNN.out
#SBATCH --error=train_RNN.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=45G
#SBATCH --gres=gpu

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

nvidia-smi
python train_RNN.py --model_dir models/lsst_alpha_0.5 --num_epochs 1000 --alpha 0.5 --lr 0.00007