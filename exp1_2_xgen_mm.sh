#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=xgen_mm_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/xgen_mm_analysis_%j.out
#SBATCH --error=logs/xgen_mm_analysis_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/xgen_mm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
XGEN=adversarial_mislabeling_attack/xgen_mm

EXP1_OUT="$DATA/output/xgen_mm/exp1"
EXP2_OUT="$DATA/output/xgen_mm/exp2"
ANALYSIS_OUT="$DATA/output/xgen_mm/analysis"
CACHE="$DATA/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1"

# Fresh result directories; representation cache is preserved.
rm -rf "$EXP1_OUT" "$EXP2_OUT" "$ANALYSIS_OUT"

echo "=== Exp1 ==="
srun python "$XGEN/exp1.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP1_OUT" \
    --device cuda

echo "=== Exp2 ==="
srun python "$XGEN/exp2.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/xgen_mm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP2_OUT" \
    --device cuda

echo "=== Analysis ==="
srun python "$XGEN/analyze_results.py" \
    --exp1-results "$EXP1_OUT/exp1_layerwise_cosine.csv" \
    --exp2-dir "$EXP2_OUT" \
    --output-dir "$ANALYSIS_OUT"

echo "=== Pipeline completed successfully ==="