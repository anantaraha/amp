#!/bin/bash -l
# Submit from the AMP root after: mkdir -p logs
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_exp4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xgen_mm_exp4_%j.out
#SBATCH --error=logs/xgen_mm_exp4_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/xgen_mm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
MODEL=adversarial_mislabeling_attack/xgen_mm
CACHE="$DATA/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1"
OUTPUT="$DATA/output/xgen_mm/exp4"
DIAG_WIDTH="${DIAG_WIDTH:-0}"

# CPU CKA from existing caches; preserve representations and other experiment outputs.
mkdir -p "$OUTPUT"

echo "=== xGen-MM Exp4: linear + RBF, k=$DIAG_WIDTH ==="
srun python -u exp4.py --vlm xgen_mm \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$OUTPUT" \
    --diag-width "$DIAG_WIDTH" \
    --cache-only \
    --cka both \
    --no-per-image-plots

echo "=== Exp4 completed successfully: $OUTPUT ==="

