#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=llava_caption
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/caption_%j.out
#SBATCH --error=logs/caption_%j.err

cd ~/projects/amp
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
mkdir -p "$DATA" /data/anantaraha/huggingface

nvidia-smi

srun python caption_laion_art_llava.py \
    --clean-dir "$DATA/clean" \
    --output-path "$DATA/llava_captions.csv" \
    --batch-size 16