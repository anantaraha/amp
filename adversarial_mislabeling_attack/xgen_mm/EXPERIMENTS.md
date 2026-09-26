Use this directory's xGen-MM environment and `requirements.txt`, plus NumPy, pandas, Pillow and Matplotlib. CLI options follow the LLaVA experiments; each script supports `--help`. Model inference requires CUDA; complete caches can be analyzed on CPU.

From the AMP repository root:

```bash
python adversarial_mislabeling_attack/xgen_mm/exp1.py
python adversarial_mislabeling_attack/xgen_mm/exp2.py
python adversarial_mislabeling_attack/xgen_mm/analyze_results.py
python adversarial_mislabeling_attack/xgen_mm/exp3.py --cache-only --no-per-image-plots
```

Default manifest: `dataset/laion_art/attack_set/manifest.csv`. Default attack results: `dataset/laion_art/attack_set/xgen_mm/attack_results.csv`. Default shared cache: `dataset/laion_art/attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1/{clean,adv}/`. Explicit relative CLI paths use the working directory; CSV image paths use the repository root.

Results go to this directory's `output/{exp1,exp2,analysis,exp3}/`. Each run replaces its result tables and reuses compatible representation caches. `--force-recompute-representations` in Exp1/Exp3 replaces selected caches; Exp3 `--cache-only` skips missing/invalid caches. `--no-per-image-plots` retains every numerical Exp3 result and its aggregate plot; `--no-plots` disables all Exp3 PNGs. Exp1 retains `--no-images` and `--no-per-sample-heatmaps`.

The output names match LLaVA. Exp1 additionally writes `exp1_attack_output_cosine.csv`; analysis requires this measured objective-output table (by default beside the layer-wise CSV, or supplied with `--attack-output-results`). `representation_metadata.json` records the model, source revision, actual encoder layout and preprocessing. Exp1/Exp2 also save skipped-sample CSVs. Exp3 keeps its image scores, condition/paired summaries, mean matrices, four-panel heatmap and optional individual heatmaps.

Architecture and measurement conventions:

- Official [config](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1/blob/1d91d356d3b6fbc141140edf490b39890417af44/config.json), [configuration defaults](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1/blob/1d91d356d3b6fbc141140edf490b39890417af44/configuration_xgenmm.py), and [model implementation](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1/blob/1d91d356d3b6fbc141140edf490b39890417af44/modeling_xgenmm.py) were inspected. Extraction pins revision `1d91d356d3b6fbc141140edf490b39890417af44` of `Salesforce/xgen-mm-phi3-mini-instruct-r-v1`.
- The wrapper selects OpenCLIP `ViT-H-14-378-quickgelu`, enables `output_tokens`, and exposes its visual encoder. The project's pinned OpenCLIP 2.24 [vision config](https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/model_configs/ViT-H-14-378-quickgelu.json) and [transformer source](https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/transformer.py) establish the layer and token conventions.
- Preprocessing follows `attack_xgenmm.py`: RGB FP32 tensor, direct 378×378 bicubic tensor resize and the author's CLIP normalization. These experiments use the attack's single resized image, without the captioning pipeline's any-resolution tiling or Perceiver resampling.
- The published encoder has 32 blocks, width 1280 and a 27×27 patch grid. The loader derives dimensions from the native module and validates the layout. Full encoder states contain CLS followed by 729 row-major spatial patches.
- State 0 is the input to the first encoder block, after positional embeddings and `ln_pre`; states 1…N are block outputs before the final `ln_post`. OpenCLIP 2.24 uses token-first block tensors; hooks read the actual attention layout and store batch-first CPU FP32 tensors. Runtime compatibility checks reject unsupported pooling/token layouts.
- The attack output is recorded separately as exactly `vision_encoder(...)[1]`: patches after `ln_post`, without CLS or the pooled contrastive projection. Analysis applies the same gain statistics and pair-cluster bootstrap to that measured output, without inferring an objective from an encoder index.
- Exp1 flattens each full encoder state for cosine. Exp2 averages all encoder tokens, including CLS, and uses images as CKA observations. Exp3 removes CLS from the full encoder states and uses each image's patch tokens as observations. Middle/deep blocks are thirds of encoder depth; embedding state 0 and the separate attack output are excluded from those groups.
- Caches include native full tokens and the actual attack output. Model/revision, tensor layout, precision, preprocessing and image SHA-256 must match before reuse. Cache keys and sample provenance are checked. Matched source/adversarial/target triplets contribute together. No cross-model caches are accepted.
- Exp3 remains exploratory: `S_gap = A_DD - A_MD`; it adds no detection threshold or classification claim.

Slurm submission from the repository root:

```bash
mkdir -p logs
sbatch adversarial_mislabeling_attack/xgen_mm/exp1_2_xgen_mm.sh
# Submit after the pipeline has populated the representation cache:
sbatch adversarial_mislabeling_attack/xgen_mm/exp3_xgen_mm.sh
```

The jobs retain LLaVA's partition, CPU/memory/time settings. Exp1→Exp2→analysis requests one A100; Exp3 uses CPU with `--cache-only --no-per-image-plots`. Jobs activate `xgen_mm/.venv` and use `$DATA/output/xgen_mm/` and the xGen-MM cache. `DATA` defaults to `/data/anantaraha/amp/dataset/laion_art`; override with `AMP_DATA_ROOT`. `AMP_ROOT` defaults to `$HOME/projects/amp`. Existing result directories are not deleted recursively.
