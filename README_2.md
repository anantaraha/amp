Run these stages from the repository root. The three preparation scripts use working-directory-relative paths. All scripts support `-h` / `--help` without loading models.

## 1. `prepare_laion_art.py`

Selects 100,000 streamed Re-LAION-Art records with a seeded shuffle, then downloads and prepares 1024×1024 PNGs. Uses direct Pillow LANCZOS resizing without cropping; valid PNGs already at the target size retain their original bytes.

```bash
python prepare_laion_art.py
```

Requires `datasets`, `img2dataset` (on PATH), `pyarrow`, `pandas`, and Pillow. Authenticate separately with `hf auth login` if needed.

| Argument | Meaning | Default |
| --- | --- | --- |
| `--dataset-id ID` | Hub dataset, streamed train split | `laion/relaion-art` |
| `--num-samples N` | Selected records, including failures | `100000` |
| `--seed N` | Streaming shuffle seed | `2025` |
| `--shuffle-buffer N` | Streaming shuffle buffer size | `10000` |
| `--metadata-cache PATH` | Hub metadata cache | `dataset/laion_art/cache` |
| `--download-root PATH` | Selection/download checkpoint parent | `dataset/laion_art/tmp` |
| `--clean-dir PATH` | Final PNG directory | `dataset/laion_art/clean` |
| `--metadata-path PATH` | Metadata/status CSV | `dataset/laion_art/metadata.csv` |
| `--workers N` | Image-processing processes | `max(1, CPU count - 1)` |
| `--download-processes N` | Download processes | `min(16, max(1, CPU count - 1))` |
| `--download-threads N` | Download threads per process | `32` |
| `--samples-per-shard N` | Records per download shard | `1000` |
| `--image-size N` | Square PNG side length | `1024` |

Outputs: `clean/<12-digit amp_sample_id>.png`, `metadata.csv`, and `tmp/n100000_seed2025_buffer10000/{selected.parquet,selection.json,downloads/}` under `dataset/laion_art/`. Failed rows remain in the CSV as `download_failed` or `decode_failed`; successful rows are `complete`. Failures are not replaced, so the image count may be below 100,000 (the notebook's older 20,000-image heading is stale).

Reruns reuse matching selection metadata, completed download shards, and valid final PNGs; raw images are deleted only after successful processing. Final image reuse checks format/size, not dataset identity. Use separate clean, metadata, and download locations for a different selection or image size; use a fresh download root if changing shard layout.

## 2. `caption_laion_art_llava.py`

Captions clean images in sorted filename order using LLaVA-1.5-7B in FP16. Uses the notebook's `USER: <image>…ASSISTANT:` template and left padding.

```bash
python caption_laion_art_llava.py
```

Use the LLaVA-compatible environment in `adversarial_mislabeling_attack/llava/requirements.txt`; CUDA is required.

| Argument | Meaning | Default |
| --- | --- | --- |
| `--clean-dir PATH` | Input image directory | `dataset/laion_art/clean` |
| `--output-path PATH` | Caption CSV and checkpoint | `dataset/laion_art/llava_captions.csv` |
| `--model ID_OR_PATH` | LLaVA model and processor | `llava-hf/llava-1.5-7b-hf` |
| `--prompt TEXT` | Caption instruction | `Describe the image in twenty words or less.` |
| `--batch-size N` | Images per batch/checkpoint | `16` |
| `--max-new-tokens N` | Generated-token limit | `77` |
| `--device DEVICE` | CUDA device, e.g. `cuda:1` | `cuda` |

For less GPU memory use:

```bash
python caption_laion_art_llava.py --batch-size 4
```

Output: `llava_captions.csv` with `image_id`, `image_path`, and `llava_caption`. Each batch atomically checkpoints the CSV, sorted by image ID. Reruns skip nonempty captions for current image IDs and retry blanks; filenames must have unique stems. Resume does not detect changed images, prompts, or models: use a new `--output-path` for those changes. The model still loads on a completed rerun, matching the notebook.

## 3. `prepare_amp_attack_set.py`

Assigns each image its highest-scoring caption noun/proper-noun concept using WordNet lemmatization and FP16 OpenCLIP. Samples 25 disjoint concept pairs × 16 image pairs from the top 100 concepts, with scores strictly **greater than 0.99** (400 manifest rows).

```bash
python prepare_amp_attack_set.py
```

Requires CUDA, `open-clip-torch`, PyTorch, NumPy, pandas, Pillow, tqdm, NLTK, spaCy, and `en_core_web_sm`; it requests the NLTK WordNet data at startup.

| Argument | Meaning | Default |
| --- | --- | --- |
| `--clean-dir PATH` | Input PNG directory | `dataset/laion_art/clean` |
| `--captions-path PATH` | LLaVA caption CSV | `dataset/laion_art/llava_captions.csv` |
| `--assignments-path PATH` | Assignment CSV and scoring cache | `dataset/laion_art/concept_assignments.csv` |
| `--pairs-path PATH` | Concept-pairs CSV | `dataset/laion_art/concept_pairs.csv` |
| `--attack-dir PATH` | Common clean copies and manifest | `dataset/laion_art/attack_set` |
| `--num-concept-pairs N` | Disjoint source/target concept pairs | `25` |
| `--images-per-pair N` | Images per concept; image pairs per concept pair | `16` |
| `--seed N` | Torch, NumPy, Python, and sampling seed | `42` |
| `--image-batch-size N` | Image scoring batch size | `32` |
| `--text-batch-size N` | Concept encoding batch size | `256` |
| `--rebuild-assignments` | Force concept/score recomputation | Off |
| `--top-concepts N` | Most frequent concepts before confidence filtering | `100` |
| `--confidence-threshold VALUE` | Strict lower score bound, in [0, 1] | `0.99` |
| `--model-name NAME` | OpenCLIP architecture/tokenizer | `EVA02-E-14-plus` |
| `--pretrained TAG_OR_PATH` | OpenCLIP weights | `laion2b_s9b_b144k` |
| `--spacy-model NAME` | Noun extraction model | `en_core_web_sm` |
| `--device DEVICE` | CUDA device | `cuda` |

Outputs: `concept_assignments.csv`, `concept_pairs.csv`, and `attack_set/{manifest.csv,source/<sample_id>.png,target/<sample_id>.png}` under `dataset/laion_art/`. Copies preserve image bytes. Defaults require at least 50 top-100 concepts with 16 qualifying images each; otherwise the script stops.

Assignments are saved after all scoring and reused when image IDs and captions match in order. Use `--rebuild-assignments` after changing image contents, image locations, or scoring/extraction settings. Every run rewrites the pairs CSV and manifest and **replaces the common `source/` and `target/` directories**. Model-specific AMP outputs remain; after changing pairings, use a new generation `--output-root` to avoid reusing old adversarial images.

## 4. `generate_amp_perturbations.py`

Reads the common manifest and generates adversarial images for `llava`, `cogvlm`, or `xgen_mm`. Each VLM gets separate input copies, adversarial images, and a results CSV.

Use the selected model's `adversarial_mislabeling_attack/<vlm>/requirements.txt`, plus pandas, Pillow, and tqdm. Transformers versions differ, so use separate compatible environments. Generation requires CUDA; completed runs can skip model loading.

```bash
python generate_amp_perturbations.py
```

```bash
python generate_amp_perturbations.py --vlm llava
python generate_amp_perturbations.py --vlm cogvlm
python generate_amp_perturbations.py --vlm xgen_mm
```

No arguments selects LLaVA. Unless overridden, model defaults are:

| VLM | Optimization steps | Initial learning rate |
| --- | ---: | ---: |
| `llava` | 4000 | 0.005 |
| `cogvlm` | 2000 | 0.003 |
| `xgen_mm` | 2000 | 0.01 |

All arguments are optional:

| Argument | Meaning | Default |
| --- | --- | --- |
| `-h`, `--help` | Show usage and exit | Off |
| `--vlm {llava,cogvlm,xgen_mm}` | Selected VLM | `llava` |
| `--manifest PATH` | Common source/target manifest | `dataset/laion_art/attack_set/manifest.csv` |
| `--output-root PATH` | Parent of the VLM-specific output folders | `dataset/laion_art/attack_set/` |
| `--budget VALUE` | L-infinity budget, decimal or fraction in [0, 1] | `16/255` |
| `--optimization-steps N` | Positive iteration-count override | Model-specific table above |
| `--initial-lr VALUE` | Positive initial learning-rate override | Model-specific table above |
| `--max-attacks N` | First N manifest rows, including skipped rows | All rows |
| `--device DEVICE` | CUDA device, e.g. `cuda:1` | `cuda` |
| `--verbose` | Show per-image optimization progress | Off |

The manifest requires unique `sample_id` values and `source_path` / `target_path` columns. Results retain extra metadata. Manifest image paths are repository-root-relative or absolute. Default CLI paths are repository-anchored; explicit relative CLI paths use the working directory.

Outputs are created automatically:

```text
dataset/laion_art/attack_set/
├── manifest.csv                      # Common input, unchanged
├── llava/
│   ├── source/<sample_id>.png
│   ├── target/<sample_id>.png
│   ├── adv/<sample_id>.png
│   └── attack_results.csv
├── cogvlm/                            # Same structure
└── xgen_mm/                           # Same structure
```

`--output-root` changes the parent; `<vlm>/` is always appended. Inputs are copied without re-encoding. Common-manifest preparation and legacy outputs stay unchanged.

Repeat the command to resume: valid adversarial images are `skipped`; missing/corrupt images are regenerated. Sample errors are recorded as `failed`; processing continues. `attack_results.csv` checkpoints after every sample and contains only selected rows, including with `--max-attacks`. Input copies are refreshed from the manifest. Resume checks image validity, not changed settings or inputs: use a new `--output-root` for a different configuration.

Custom example:

```bash
python generate_amp_perturbations.py --vlm cogvlm \
  --output-root dataset/laion_art/amp_trial --budget 8/255 \
  --optimization-steps 1000 --initial-lr 0.002 \
  --max-attacks 2 --device cuda:0 --verbose
```

For downstream LLaVA scripts run from the repository root, use `--attack-results dataset/laion_art/attack_set/llava/attack_results.csv`. The common manifest stays unchanged.
