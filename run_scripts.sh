#!/bin/bash

#SBATCH --job-name=optimise_cscape     # Job name
#SBATCH --output=logs/optimise_cscape_%j.out    # Output and error log
#SBATCH --error=logs/optimise_cscape_%j.err     # Error log
#SBATCH --partition=compute,gpu                 # Partition with GPU resources
##SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=8G                       # Total memory limit
#SBATCH --time=8:00:00                 # Time limit: 48 hours
#SBATCH --account=sscm013903
#SBATCH --chdir=/user/home/uw20204/CanDrivR_data/cscape-xf

source /user/home/uw20204/.bashrc

conda activate DrivR-Base

#python -u 4_prepare_data.py
python -u 6_hyperparam_tuning.py
#python -u active_learning_sampling.py
#python -u optimise_sample_size.py
#python -u optimise_features.py
#python -u model_comparison.py

#python -u cscape-xf.py 
