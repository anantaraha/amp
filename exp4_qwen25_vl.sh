#!/bin/bash -l
# Submit from the AMP root after: mkdir -p logs
# Requires Qwen-specific attack_results.csv, including a vlm=qwen25_vl column.
# Optional pilot: MAX_SAMPLES=1 QWEN_MAX_PIXELS=200704 sbatch exp4_qwen25_vl.sh
# Pixel limits change representations: keep them consistent with future Qwen attacks/caches.
# Official maximum resolution can be costly: token CKA requires quadratic memory.
#SBATCH -p nopreempt
#SBATCH --job-name=qwen25_vl_exp4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen25_vl_exp4_%j.out
#SBATCH --error=logs/qwen25_vl_exp4_%j.err

set -euo pipefail

cd ~/projects/amp
# This existing environment has Transformers 4.52.3 with native Qwen2.5-VL support.
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
ATTACK_RESULTS="${ATTACK_RESULTS:-$DATA/attack_set/qwen25_vl/attack_results.csv}"
CACHE="$DATA/attack_set/representations/qwen25_vl_3b_instruct"
OUTPUT="${OUTPUT:-$DATA/output/qwen25_vl/exp4}"

if [[ ! -f "$ATTACK_RESULTS" ]]; then
    echo "Qwen attack results are required: $ATTACK_RESULTS" >&2
    echo "Generate Qwen-specific adversarial examples first; this job does not generate attacks." >&2
    exit 1
fi
options=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    options+=(--max-samples "$MAX_SAMPLES")
fi
mkdir -p "$OUTPUT"

srun python -u exp4.py \
    --vlm qwen25_vl \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$ATTACK_RESULTS" \
    --cache-dir "$CACHE" \
    --output-dir "$OUTPUT" \
    --diag-width "${DIAG_WIDTH:-0}" \
    --qwen-min-pixels "${QWEN_MIN_PIXELS:-3136}" \
    --qwen-max-pixels "${QWEN_MAX_PIXELS:-12845056}" \
    --device cuda \
    --cka both \
    --no-per-image-plots \
    "${options[@]}"

echo "=== Qwen Exp4 completed: $OUTPUT ==="
