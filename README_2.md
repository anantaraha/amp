# Generate AMP perturbations

`generate_amp_perturbations.py` reads the common attack-set manifest and generates adversarial images for `llava`, `cogvlm`, or `xgen_mm`. Each model has separate input copies, adversarial outputs, and a results CSV.

Run from the repository root, in an environment compatible with the selected VLM. Use that model's `adversarial_mislabeling_attack/<vlm>/requirements.txt`; pandas, Pillow, and tqdm are also needed. The models require different Transformers versions, so use separate compatible environments. CUDA is required for generation; a fully completed run can be skipped without loading a model.

```bash
python generate_amp_perturbations.py --vlm llava
python generate_amp_perturbations.py --vlm cogvlm
python generate_amp_perturbations.py --vlm xgen_mm
```

With no arguments, the script uses LLaVA. Model-specific defaults apply unless explicitly overridden:

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

The manifest needs unique `sample_id`, `source_path`, and `target_path` columns. Additional manifest metadata is retained in the results. Image paths in the manifest are repository-root-relative or absolute. Default CLI paths are anchored to the repository; explicit relative CLI paths resolve from your current directory.

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

`--output-root` changes the parent; the script always appends `<vlm>/`. Original source/target files are copied without re-encoding. Existing common-manifest preparation and legacy outputs remain unchanged.

Resume by repeating the command: existing valid adversarial images are marked `skipped`; missing/corrupt outputs are regenerated. Sample errors are recorded as `failed`, and processing continues. `attack_results.csv` is checkpointed after every sample for the selected rows; `--max-attacks` limits that result table too. Input copies are refreshed from the manifest. Resume checks image validity, not changed attack settings or inputs: use a new `--output-root` for a different configuration.

Custom example:

```bash
python generate_amp_perturbations.py --vlm cogvlm \
  --output-root dataset/laion_art/amp_trial --budget 8/255 \
  --optimization-steps 1000 --initial-lr 0.002 \
  --max-attacks 2 --device cuda:0 --verbose
```

For downstream LLaVA experiments, select the new results with `--attack-results dataset/laion_art/attack_set/llava/attack_results.csv` when running their scripts from the repository root; the common manifest stays unchanged.
