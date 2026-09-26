#!/bin/bash -l
# Submit from the AMP repository root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/xgen_mm_exp1_2_%j.out
#SBATCH --error=logs/xgen_mm_exp1_2_%j.err

set -euo pipefail

cd "${AMP_ROOT:-$HOME/projects/amp}"
MODEL_DIR=adversarial_mislabeling_attack/xgen_mm
source "$MODEL_DIR/.venv/bin/activate"

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA="${AMP_DATA_ROOT:-/data/anantaraha/amp/dataset/laion_art}"
OUTPUT="$DATA/output/xgen_mm"
CACHE="$DATA/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1"

# Each script replaces its result tables and reuses valid representations.
# Preserve existing files/directories; do not recursively delete prior outputs.
mkdir -p "$OUTPUT"

echo "=== Exp1 ==="
srun python "$MODEL_DIR/exp1.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$OUTPUT/exp1" \
    --device cuda

echo "=== Exp2 ==="
srun python "$MODEL_DIR/exp2.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$OUTPUT/exp2" \
    --device cuda

echo "=== Analysis ==="
srun python "$MODEL_DIR/analyze_results.py" \
    --exp1-results "$OUTPUT/exp1/exp1_layerwise_cosine.csv" \
    --exp2-dir "$OUTPUT/exp2" \
    --output-dir "$OUTPUT/analysis"

echo "=== Pipeline completed successfully ==="
