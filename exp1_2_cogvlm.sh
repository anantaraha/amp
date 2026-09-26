#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=cogvlm_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/cogvlm_analysis_%j.out
#SBATCH --error=logs/cogvlm_analysis_%j.err

set -euo pipefail

cd ~/projects/amp
source adversarial_mislabeling_attack/cogvlm/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/general/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art
COGVLM=adversarial_mislabeling_attack/cogvlm

EXP1_OUT="$DATA/output/cogvlm/exp1"
EXP2_OUT="$DATA/output/cogvlm/exp2"
ANALYSIS_OUT="$DATA/output/cogvlm/analysis"
CACHE="$DATA/attack_set/representations/cogvlm_chat_hf"

# Fresh result directories; representation cache is preserved.
rm -rf "$EXP1_OUT" "$EXP2_OUT" "$ANALYSIS_OUT"

echo "=== Exp1 ==="
srun python "$COGVLM/exp1.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/cogvlm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP1_OUT" \
    --device cuda

echo "=== Exp2 ==="
srun python "$COGVLM/exp2.py" \
    --manifest "$DATA/attack_set/manifest.csv" \
    --attack-results "$DATA/attack_set/cogvlm/attack_results.csv" \
    --cache-dir "$CACHE" \
    --output-dir "$EXP2_OUT" \
    --device cuda

echo "=== Analysis ==="
srun python "$COGVLM/analyze_results.py" \
    --exp1-results "$EXP1_OUT/exp1_layerwise_cosine.csv" \
    --exp2-dir "$EXP2_OUT" \
    --output-dir "$ANALYSIS_OUT"

echo "=== Pipeline completed successfully ==="