#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="train_RNN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=train_RNN.out
#SBATCH --error=train_RNN.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=45G
#SBATCH --gres=gpu:TeslaM2090

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

nvidia-smi
python train_RNN.py --model_dir test_models