#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --job-name="time_benchmark"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --output=time_benchmark.out
#SBATCH --error=time_benchmark.err
#SBATCH --partition=caps-gpu
#SBATCH --mem-per-cpu=45G
#SBATCH --gres=gpu

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

nvidia-smi
python time_benchmark.py --model_dir models/lsst_alpha_0.5/