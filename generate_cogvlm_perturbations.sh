#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=amp_cogvlm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/amp_cogvlm_%j.out
#SBATCH --error=logs/amp_cogvlm_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/cogvlm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art

nvidia-smi

srun python generate_amp_perturbations.py \
    --vlm cogvlm \
    --manifest "$DATA/attack_set/manifest.csv" \
    --output-root "$DATA/attack_set" \
    --device cuda

echo "=== CogVLM AMP generation completed ==="