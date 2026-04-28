#!/bin/bash
#SBATCH --job-name=face_timestamp
#SBATCH -o out-err/%j.out
#SBATCH -e out-err/%j.err
#SBATCH --time=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-106

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov8n

# --frame-skip 1  : 모든 프레임 추론 (정밀도 최대)
# --frame-skip 3  : 3프레임마다 1장 추론 (속도 3배↑)
python run_face_timestamp.py predict
