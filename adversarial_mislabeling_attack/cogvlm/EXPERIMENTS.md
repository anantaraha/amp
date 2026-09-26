Use this directory's CogVLM environment and `requirements.txt`, plus NumPy, pandas, Pillow and Matplotlib. CLI options follow the LLaVA experiments; each script supports `--help`. Model inference requires CUDA; complete caches can be analyzed on CPU.

From the AMP repository root:

```bash
python adversarial_mislabeling_attack/cogvlm/exp1.py
python adversarial_mislabeling_attack/cogvlm/exp2.py
python adversarial_mislabeling_attack/cogvlm/analyze_results.py
python adversarial_mislabeling_attack/cogvlm/exp3.py --cache-only --no-per-image-plots
```

Default manifest: `dataset/laion_art/attack_set/manifest.csv`. Default attack results: `dataset/laion_art/attack_set/cogvlm/attack_results.csv`. Default shared cache: `dataset/laion_art/attack_set/representations/cogvlm_chat_hf/{clean,adv}/`. Explicit relative CLI paths use the working directory; CSV image paths use the repository root.

Results go to this directory's `output/{exp1,exp2,analysis,exp3}/`. Each run replaces its result tables and reuses compatible representation caches. `--force-recompute-representations` in Exp1/Exp3 replaces selected caches; Exp3 `--cache-only` skips missing/invalid caches. `--no-per-image-plots` retains every numerical Exp3 result and its aggregate plot; `--no-plots` disables all Exp3 PNGs. Exp1 retains `--no-images` and `--no-per-sample-heatmaps`.

The output names match LLaVA. Exp1 additionally writes `exp1_attack_output_cosine.csv`; analysis requires this measured objective-output table (by default beside the layer-wise CSV, or supplied with `--attack-output-results`). `representation_metadata.json` records the model, source revision, actual encoder layout and preprocessing. Exp1/Exp2 also save skipped-sample CSVs. Exp3 keeps its image scores, condition/paired summaries, mean matrices, four-panel heatmap and optional individual heatmaps.

Architecture and measurement conventions:

- Official [config](https://huggingface.co/THUDM/cogvlm-chat-hf/blob/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/config.json), [vision implementation](https://huggingface.co/THUDM/cogvlm-chat-hf/blob/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/visual.py), and [model implementation](https://huggingface.co/THUDM/cogvlm-chat-hf/blob/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/modeling_cogvlm.py) were inspected. Extraction pins revision `e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c` of `THUDM/cogvlm-chat-hf` (the official URL currently redirects to `zai-org`).
- Preprocessing follows `attack_cogvlm.py`: RGB tensor, FP16, direct 490×490 bicubic tensor resize and the author's CLIP normalization. It does not use the captioning processor's PIL-first resize.
- The published config has 63 EVA vision blocks, width 1792 and a 35×35 patch grid. The loader derives and validates these dimensions from the native model rather than using LLaVA's dimensions. Full encoder states contain one leading CLS and 1225 row-major spatial patches.
- State 0 is the input to the first encoder block (patch/CLS plus position embeddings); states 1…N are the block outputs. Hooks observe the unmodified model forward pass and store CPU FP16 tensors.
- The attack output is a separate tensor: CLS is removed, a GLU projects patches to the language width, and BOI/EOI are added. The separate objective CSV includes exactly this returned tensor, including BOI/EOI. Analysis applies the same gain statistics and pair-cluster bootstrap to it; it does not infer an objective from an encoder index.
- Exp1 flattens each full encoder state for cosine. Exp2 averages all encoder tokens, including CLS, and uses images as CKA observations. Exp3 excludes CLS and uses each image's patch tokens as observations. Its middle/deep groups are thirds of encoder depth; embedding state 0 and the projected attack output are excluded from those groups.
- Caches include native full tokens and the actual attack output. Model/revision, tensor layout, precision, preprocessing and image SHA-256 must match before reuse. Cache keys and sample provenance are checked. Matched source/adversarial/target triplets contribute together. No cross-model caches are accepted.
- Exp3 remains exploratory: `S_gap = A_DD - A_MD`; it adds no detection threshold or classification claim.

Slurm submission from the repository root:

```bash
mkdir -p logs
sbatch adversarial_mislabeling_attack/cogvlm/exp1_2_cogvlm.sh
# Submit after the pipeline has populated the representation cache:
sbatch adversarial_mislabeling_attack/cogvlm/exp3_cogvlm.sh
```

The jobs retain LLaVA's partition, CPU/memory/time settings. Exp1→Exp2→analysis requests one A100; Exp3 uses CPU with `--cache-only --no-per-image-plots`. Jobs activate `cogvlm/.venv` and use `$DATA/output/cogvlm/` and the CogVLM cache. `DATA` defaults to `/data/anantaraha/amp/dataset/laion_art`; override with `AMP_DATA_ROOT`. `AMP_ROOT` defaults to `$HOME/projects/amp`. Existing result directories are not deleted recursively.
