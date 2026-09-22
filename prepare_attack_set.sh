#!/bin/bash -l
#SBATCH -p nopreempt
#SBATCH --job-name=amp_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/prepare_attack_%j.out
#SBATCH --error=logs/prepare_attack_%j.err

cd ~/projects/amp
source adversarial_mislabeling_attack/llava/.venv/bin/activate

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/data/anantaraha/huggingface

DATA=/data/anantaraha/amp/dataset/laion_art

nvidia-smi

srun python prepare_amp_attack_set.py \
    --clean-dir "$DATA/clean" \
    --captions-path "$DATA/llava_captions.csv" \
    --assignments-path "$DATA/concept_assignments.csv" \
    --pairs-path "$DATA/concept_pairs.csv" \
    --attack-dir "$DATA/attack_set" \
    --num-concept-pairs 25 \
    --images-per-pair 2