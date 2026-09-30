#!/bin/bash -l
# Submit from the AMP repository root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_layer_selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/xgen_mm_layer_selection_%j.out
#SBATCH --error=logs/xgen_mm_layer_selection_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/xgen_mm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
XGEN=adversarial_mislabeling_attack/xgen_mm
CACHE="$DATA/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1"
OUTPUT="$DATA/output/xgen_mm/extra"

# Reuse full-token caches; do not delete existing experiment directories.
mkdir -p "$OUTPUT"

echo "=== Clean-only layer selection and frozen-range Exp3 comparison ==="
srun python -u "$XGEN/test_exp3_layer_selection.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --existing-exp3-dir "$DATA/output/xgen_mm/exp3" \
    --output-dir "$OUTPUT" \
    --selection-kernel rbf \
    --min-encoder-layers 3 \
    --device cuda

echo "=== Layer-selection comparison completed successfully ==="
