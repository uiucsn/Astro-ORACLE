#!/bin/bash
#SBATCH --time=6-15:30:00
#SBATCH --job-name="HC_tables"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vedgs2@illinois.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=HC_tables.out
#SBATCH --error=HC_tables.err
#SBATCH --partition=caps

module load anaconda/2023-Mar/3
source activate /projects/caps/uiucsn/ved/HC_env

cd data/data/elasticc2_train

#ls raw | sed 's/ELASTICC2_TRAIN_02_//' | xargs -IXXX -P32 python3 ../../../fits_to_parquet.py raw/ELASTICC2_TRAIN_02_XXX parquet/XXX.parquet
ls parquet | xargs -IXXX -P32 python ../../../parquet_to_LSST_Source.py parquet/XXX event_tables/