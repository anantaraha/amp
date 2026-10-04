#!/bin/bash -l
#SBATCH -p general
#SBATCH --job-name=amp_qwen25_vl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/amp_qwen25_vl_%j.out
#SBATCH --error=logs/amp_qwen25_vl_%j.err

cd ~/projects/amp
source adversarial_mislabeling_attack/qwen25_vl/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art

nvidia-smi

srun python -u generate_amp_perturbations.py \
    --vlm qwen25_vl \
    --manifest "$DATA/attack_set/manifest.csv" \
    --output-root "$DATA/attack_set" \
    --device cuda \
    --verbose