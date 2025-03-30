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

python integrated_nist_converter.py --input ../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv --output ../data/nist_pka_data.csv