#!/bin/bash
#SBATCH --job-name=yolo_detect
#SBATCH -o out-err/%j.out
#SBATCH -e out-err/%j.err
#SBATCH --time=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-106

# conda 활성화
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate yolo11

# 실행
python run_yolo11.py predict