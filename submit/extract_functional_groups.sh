#!/bin/bash

#SBATCH --job-name=Eddie
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --account=MST111483
#SBATCH --output=../logs/job_output_%j.txt
#SBATCH --error=../logs/job_error_%j.txt   

python ../src/extract_functional_groups.py --input ../data/processed_pka_data.csv --output ../data/functional_groups_analysis_v5.csv