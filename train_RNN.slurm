#!/bin/bash
#SBATCH --time=1-15:30:00
#SBATCH --job-name="train_RNN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=train_RNN.out
#SBATCH --error=train_RNN.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=45G
#SBATCH --gres=gpu

module load anaconda3/2024.06
source activate /projects/ncsa/caps/uiucsn/ved/HC_env

nvidia-smi
python train_RNN.py --model_dir models/lsst_alpha_0.5_no_md --num_epochs 2000 --alpha 0.5 --lr 0.0002 --no_md