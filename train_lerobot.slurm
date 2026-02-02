#!/bin/bash

#SBATCH --job-name=lerobot_test
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda/12.1.0
conda init bash
source ~/.bashrc
conda activate lerobot
cd /public/home/202420116260/Project/lerobot
lerobot-train --config_path=configs/train_config.json