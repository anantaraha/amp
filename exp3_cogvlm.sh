#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=cogvlm_exp3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cogvlm_exp3_%j.out
#SBATCH --error=logs/cogvlm_exp3_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/cogvlm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
COGVLM=adversarial_mislabeling_attack/cogvlm

EXP3_OUT="$DATA/output/cogvlm/exp3"
CACHE="$DATA/attack_set/representations/cogvlm_chat_hf"

rm -rf "$EXP3_OUT"

echo "=== Exp3 ==="
srun python "$COGVLM/exp3.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/cogvlm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP3_OUT" \
    --cache-only \
    --no-per-image-plots

echo "=== Exp3 completed successfully ==="