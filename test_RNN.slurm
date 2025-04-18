#!/bin/bash
#SBATCH --time=15:30:00
#SBATCH --job-name="test_RNN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=test_RNN.out
#SBATCH --error=test_RNN.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=65G
#SBATCH --gres=gpu

module load anaconda3/2024.06
source activate /projects/ncsa/caps/uiucsn/ved/HC_env
pip install seaborn

nvidia-smi
python test_RNN.py --model_dir models/lsst_alpha_0.5_no_md/ --max_class_count 20000