# Generating Adversarial Images 
In this directory, you will find code that we used to generate ***white-box targeted*** adversarial perturbations for the three VLMS: CogVLM, BLIP-3 (xgen-MM), and LLaVA-1.5 (7b variant). This attack was used for the main result in our paper (section 5).

> [!WARNING]
> Although each VLM is available on HuggingFace, there are cross-dependencies (transformer version) that are not compatible for each VLM to be used in a single Python environment. Thus, we include separate `requirements.txt` for each VLM, under their own separate directory.

## Setup & Installation
As mentioned above, each VLM requires a different Python environment. Below, we detail the installation procedure for one such environment and VLM. We recommend repeating the process for each VLM as necessary.

> [!NOTE]
> We tested our code on a Linux machine running Ubuntu 22.04.5 LTS, using Python 3.10.13, with pyenv virtualenv. We recommend using pyenv and Python 3.10.13, as different versions may cause things to break. The steps below will assume you have pyenv and Python 3.10.13 already installed.

1. cd into model directory

        cd cogvlm

2. Create and activate virtualenv

        uv venv --python 3.10.13
        source .venv/bin/activate

3. Install required packages

        uv pip install -r requirements.txt --torch-backend=cu128

## Usage
Each script generates adversarial perturbation against a single model, using both an input image, and a target image (will add perturbation to the input image to cause it to mislabel as the target image). The arguments for all three attack scripts (`attack_cogvlm.py`, `attack_xgenmm.py`, `attack_llava.py`) are the same. Below are three examples, one for each model. We provide 2 example images to use for quick testing.

Attacking CogVLM (~5.5 minutes on a single A100 GPU)
```
pyenv activate amp-cogvlm-3.10.13
CUDA_VISIBLE_DEVICES=0 python3 cogvlm/attack_cogvlm.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./cogvlm/amp-cogvlm.png --verbose
```

Attacking XGen-MM (~5 minutes on a single A100 GPU)
```
pyenv activate {xgenmm_environment}
CUDA_VISIBLE_DEVICES=1 python3 xgen_mm/attack_xgenmm.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./xgenmm/amp-xgenmm.png --verbose
```

Attacking LLaVA (~2 minutes on a single A100 GPU)
```
pyenv activate {llava_environment}
CUDA_VISIBLE_DEVICES=2 python3 llava/attack_llava.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./llava/amp-llava.png --verbose
```

### Testing
We also provide notebooks to test our adversarial attack's mislabeling effect. For the provided two source and target images, the image should look like the source, but get labeled as if it was the target.

### New Experiments
Run with defaults:
```
cd amp
source .venv/bin/activate
cd adversarial_mislabeling_attack/llava

python exp1.py
python exp2.py
```

**Every argument is optional**. The following arguments work in both scripts. Default paths below are relative to the repository root, regardless of your working directory.
| Argument | Meaning | Default | 
|----------|---------|---------| 
| -h, --help | Print usage and exit | Not enabled | 
| --manifest PATH | Sample manifest | dataset/laion_art/attack_set/manifest.csv | 
| --attack-results PATH | Attack statuses and adversarial-image paths | dataset/laion_art/attack_set/attack_results.csv | 
| --cache-dir PATH | Shared cache containing clean/ and adv/ | dataset/laion_art/attack_set/representations/llava_1_5_7b | 
| --output-dir PATH | Automatically created results directory | adversarial_mislabeling_attack/llava/output/exp1/ or output/exp2/, respectively | 
| --device DEVICE | CUDA device for representation extraction; accepts cuda or cuda:N | cuda | 

Exclusive to `exp1.py` only:
| Argument | Meaning | Default | 
|----------|---------|---------| 
| --force-recompute-representations | Recompute and replace caches instead of reusing them | Off; reuse valid caches | 
| --no-images | Disable source/adversarial/target triplet plots | Off; save triplets | 
| --no-per-sample-heatmaps | Disable individual heatmaps; aggregate still saved | Off; save individual heatmaps | 


Example:
```
python exp1.py --no-images --no-per-sample-heatmaps
python exp2.py --force-recompute-representations
```

Saved outputs:
```
// exp1.py
- exp1_layerwise_cosine.csv
- exp1_layerwise_cosine_mean.csv
- exp1_aggregate_mean_heatmap.png
- PNGs in triplets/ and sample_heatmaps/

// exp2.py
- the four exp2_linear_cka_*.csv matrices
- exp2_samples.csv
- exp2_linear_cka_four_panel.png
```
Run rules and dependencies:
- Both scripts require prepared manifests and generated adversarial images.
- Exp1 requires CUDA and loads model at startup.
- Running Exp1 first populates caches for Exp2. Exp2 can run independently: it requires CUDA/model weights only when representations need extraction. With complete caches, it processes them on CPU without loading the model.
- Use matching input files and --cache-dir when sharing caches. Exp2 requires at least three valid triplets.
- Explicit relative CLI paths resolve from your terminal’s current directory; image paths inside CSVs remain repository-root-relative.
- Repeated runs reuse compatible caches and overwrite corresponding result files.
- Use the existing .venv; dependencies include PyTorch, torchvision, Transformers, Accelerate, NumPy, pandas, Pillow, and Matplotlib.

### Analyze saved Exp1 and Exp2 results

`llava/analyze_results.py` reads the saved experiment CSVs, computes numerical measurements, and then generates supported findings. It does not load LLaVA, rerun either experiment, or inspect plots. It requires NumPy and pandas, available in the existing `.venv`.

From the repository root:

```bash
source .venv/bin/activate
cd adversarial_mislabeling_attack/llava
python analyze_results.py
```

All arguments are optional. Default paths in this table are relative to the script's directory and are located automatically; explicitly supplied relative paths resolve from the terminal's current directory.

| Argument | Meaning | Default |
| --- | --- | --- |
| `-h`, `--help` | Show usage and exit | Not enabled |
| `--exp1-results PATH` | Per-sample, per-state Exp1 cosine CSV | `output/exp1/exp1_layerwise_cosine.csv` |
| `--exp2-dir PATH` | Directory containing the saved Exp2 matrices and sample list | `output/exp2/` |
| `--output-dir PATH` | Automatically created analysis destination | `output/analysis/` |
| `--bootstrap-reps N` | Exp1 bootstrap replicate count; at least 2 | `5000` |
| `--seed N` | Nonnegative bootstrap random seed | `2025` |

Custom paths example, from the script directory:

```bash
python analyze_results.py \
  --exp1-results /path/to/run/exp1/exp1_layerwise_cosine.csv \
  --exp2-dir /path/to/run/exp2 \
  --output-dir /path/to/run/analysis \
  --bootstrap-reps 10000 --seed 42
```

Expected inputs are the Exp1 result CSV and these five files inside `--exp2-dir`:

- `exp2_linear_cka_source.csv`
- `exp2_linear_cka_adversarial.csv`
- `exp2_linear_cka_target.csv`
- `exp2_linear_cka_adv_minus_source.csv`
- `exp2_samples.csv`

Matrices are checked for compatible state labels, symmetry, finite values, and agreement of the saved difference with adversarial minus source. Early/middle/deep blocks are three near-equal groups of ordered encoder states; embedding state 0 is separate. CKA measurements count each off-diagonal state pair once. Exp1 retains its pair-cluster bootstrap, with sample-level bootstrap when `pair_id` is absent. Combined findings additionally require matching sample IDs, image provenance, states, and complete Exp1 coverage. The report states the numerical decision rules before listing findings.

Each successful run overwrites these outputs under `--output-dir`:

- `analysis_decisions.txt`, organized as **Experiment Summary**, **Numerical Measurements**, and **Supported Findings**.
- Preserved Exp1 measurements: `exp1_validation.csv`, `exp1_layer_summary.csv`, `exp1_sample_transitions.csv`, `exp1_transition_summary.csv`, `exp1_pair_summary.csv`, `exp1_objective_summary.csv`, and `exp1_sample_metrics.csv`.
- Block definitions and Exp1 block measurements: `exp1_layer_blocks.csv`, `exp2_layer_blocks.csv`, and `exp1_block_summary.csv`.
- Exp2 measurements: `exp2_cka_adv_minus_source.csv`, `exp2_cka_layer_pairs.csv`, `exp2_cka_region_summary.csv`, and `exp2_cka_extrema.csv`.
- Cross-experiment checks and numerical criteria: `cross_experiment_summary.csv`.

The updated `llava/exp1just.ipynb` contains the same analysis functions and defaults for interactive use. Input experiment CSVs and representation caches are not modified.
