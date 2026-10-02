#!/bin/bash -l
# Submit from the AMP root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_exp5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/xgen_mm_exp5_%j.out
#SBATCH --error=logs/xgen_mm_exp5_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/xgen_mm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
OUTPUT="$DATA/output/xgen_mm/exp5"
mkdir -p "$OUTPUT"
nvidia-smi

# Shared sweep/budget settings; retain each model's native AMP steps/LR defaults.
# Optional: MAX_SAMPLES=10 ALPHA_STEP=0.2 sbatch exp5_xgen_mm.sh
options=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    options+=(--max-samples "$MAX_SAMPLES")
fi

srun python -u exp5.py \
    --vlm xgen_mm \
    --manifest "$DATA/attack_set/manifest.csv" \
    --output-dir "$OUTPUT" \
    --cache-dir "$OUTPUT/perturbations" \
    --alpha-step "${ALPHA_STEP:-0.1}" \
    --cka both \
    --diag-width "${DIAG_WIDTH:-0}" \
    --budget 16/255 \
    --seed 0 \
    --device cuda \
    "${options[@]}"

echo "=== Exp5 completed: $OUTPUT ==="

