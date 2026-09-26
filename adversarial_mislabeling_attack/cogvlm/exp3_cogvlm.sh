#!/bin/bash -l
# Submit from the AMP repository root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=cogvlm_exp3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cogvlm_exp3_%j.out
#SBATCH --error=logs/cogvlm_exp3_%j.err

set -euo pipefail

cd "${AMP_ROOT:-$HOME/projects/amp}"
MODEL_DIR=adversarial_mislabeling_attack/cogvlm
source "$MODEL_DIR/.venv/bin/activate"

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA="${AMP_DATA_ROOT:-/data/anantaraha/amp/dataset/laion_art}"
OUTPUT="$DATA/output/cogvlm"
CACHE="$DATA/attack_set/representations/cogvlm_chat_hf"

# Each script replaces its result tables and reuses valid representations.
# Preserve existing files/directories; do not recursively delete prior outputs.
mkdir -p "$OUTPUT"

echo "=== Exp3 ==="
srun python "$MODEL_DIR/exp3.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/cogvlm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$OUTPUT/exp3" \
    --cache-only \
    --no-per-image-plots

echo "=== Exp3 completed successfully ==="
