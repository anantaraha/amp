#!/bin/bash -l
#SBATCH -p general
#SBATCH --job-name=laion100k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/laion_%j.out
#SBATCH --error=logs/laion_%j.err

cd ~/projects/amp
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

DATA=/data/anantaraha/amp/dataset/laion_art
mkdir -p "$DATA"

srun python prepare_laion_art.py \
    --num-samples 100000 \
    --metadata-cache "$DATA/cache" \
    --download-root "$DATA/tmp" \
    --clean-dir "$DATA/clean" \
    --metadata-path "$DATA/metadata.csv" \
    --workers 15 \
    --download-processes 8 \
    --download-threads 8