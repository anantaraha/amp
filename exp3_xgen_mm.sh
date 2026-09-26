#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_exp3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xgen_mm_exp3_%j.out
#SBATCH --error=logs/xgen_mm_exp3_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/xgen_mm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
XGEN=adversarial_mislabeling_attack/xgen_mm

EXP3_OUT="$DATA/output/xgen_mm/exp3"
CACHE="$DATA/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1"

# Fresh Exp3 outputs; preserve representation cache.
rm -rf "$EXP3_OUT"

echo "=== Exp3 ==="
srun python "$XGEN/exp3.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP3_OUT" \
    --cache-only \
    --cka both \
    --no-per-image-plots

echo "=== Exp3 completed successfully ==="