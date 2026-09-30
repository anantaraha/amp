#!/bin/bash -l
# Submit from the AMP repository root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=llava_layer_selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/llava_layer_selection_%j.out
#SBATCH --error=logs/llava_layer_selection_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
SCRIPT=adversarial_mislabeling_attack/xgen_mm/test_exp3_layer_selection.py
CACHE="$DATA/attack_set/representations/llava_1_5_7b"
OUTPUT="$DATA/output/llava/extra"

mkdir -p "$OUTPUT"

echo "=== LLaVA clean-only layer selection and frozen-range Exp3 comparison ==="
srun python -u "$SCRIPT" \
    --vlm llava \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/llava/attack_results.csv" \
    --cache-dir "$CACHE" \
    --existing-exp3-dir "$DATA/output/exp3" \
    --output-dir "$OUTPUT" \
    --selection-kernel rbf \
    --min-encoder-layers 3 \
    --device cuda

echo "=== LLaVA layer-selection comparison completed successfully ==="