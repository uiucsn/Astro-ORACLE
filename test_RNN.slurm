#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --job-name="test_RNN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=test_RNN.out
#SBATCH --error=test_RNN.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=25G
#SBATCH --gres=gpu

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

nvidia-smi
python test_RNN.py --model_dir models/lsst_alpha_0.5/ --max_class_count 10000