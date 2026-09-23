#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=llava_exp3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
LLAVA=adversarial_mislabeling_attack/llava

EXP3_OUT="$DATA/output/exp3"
CACHE="$DATA/attack_set/representations/llava_1_5_7b"

# Fresh Exp3 result directory; representation cache is preserved.
rm -rf "$EXP3_OUT"

echo "=== Exp3 ==="
srun python "$LLAVA/exp3.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/llava/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP3_OUT" \
    --cache-only \
    --no-per-image-plots

echo "=== Exp3 completed successfully ==="