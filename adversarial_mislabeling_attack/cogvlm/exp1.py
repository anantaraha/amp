#!/usr/bin/env python3
"""CogVLM full-token layer-wise analysis; reuse native vision representations across experiments.

Run python exp1.py --help for runtime options. All plots are saved headlessly.
Relative image paths in the input CSVs resolve against the repository root.
"""

import argparse
from pathlib import Path
import re
import warnings

from _vision import (
    MODEL_ID, MODEL_REVISION, MODEL_TITLE, VLM, PRECISION, PREFIX_TOKENS,
    CACHE_VERSION, SUCCESS_STATUSES, SCRIPT_DIR, REPO_ROOT, ATTACK_DIR, DEFAULT_CACHE_DIR,
    cache_name, read_samples, validate_sample,
    Representations,
)


def cuda_device(value):
    """Retain the notebooks' CUDA/native-precision extraction requirement."""
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N (for example cuda:1).")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Layer-wise full-token cosine analysis with the model-native vision encoder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CLI paths are relative to the current directory; image paths inside CSVs are repository-root-relative. "
               "Model revision and author preprocessing/precision are fixed; layer layout is validated against the native encoder.",
    )
    parser.add_argument("--manifest", type=Path, default=ATTACK_DIR / "manifest.csv",
                        help="AMP sample manifest CSV.")
    parser.add_argument("--attack-results", type=Path, default=ATTACK_DIR / VLM / "attack_results.csv",
                        help="Attack-status and adversarial-image-path CSV.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Shared representation cache root, containing clean/ and adv/.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output/exp1",
                        help="Directory for result CSVs and PNGs; created automatically.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for model extraction (cuda or cuda:N); cached tensors are processed on CPU.")

    parser.add_argument(
        "--force-recompute-representations", action="store_true",
        help="Re-extract and replace representations instead of reusing valid caches.",
    )
    parser.add_argument(
        "--no-images", dest="save_images", action="store_false",
        help="Disable source/adversarial/target triplet PNGs (saving enabled by default: %(default)s).",
    )
    parser.add_argument(
        "--no-per-sample-heatmaps", dest="save_per_sample_heatmaps", action="store_false",
        help="Disable individual cosine heatmap PNGs (saving enabled by default: %(default)s); always save the aggregate.",
    )
    return parser.parse_args()


def save_figure(fig, output_path):
    """Save notebook figure geometry to PNG and release it even if writing fails."""
    import matplotlib.pyplot as plt

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def flattened_cosine(left, right):
    """Flatten only for cosine and retain the original float32 calculation."""
    import torch.nn.functional as F
    return F.cosine_similarity(
        left.float().reshape(1, -1), right.float().reshape(1, -1)
    ).item()


def layer_labels(layers):
    return ["Emb." if int(layer) == 0 else str(int(layer)) for layer in layers]


def save_triplet(source_path, adversarial_path, target_path, sample, output_path):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        labels = [
            f"Source: {sample.source_concept}",
            f"Adversarial source\n{sample.sample_id}",
            f"Target: {sample.target_concept}",
        ]
        for axis, path, label in zip(axes, [source_path, adversarial_path, target_path], labels):
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(label)
            axis.axis("off")
        fig.suptitle(f"AMP sample {sample.sample_id}: {sample.source_concept} → {sample.target_concept}")
        fig.tight_layout()
        save_figure(fig, output_path)
    finally:
        plt.close(fig)


def save_heatmap(frame, title, output_path):
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = ["R", "T", "B", "G"]
    ordered = frame.sort_values("layer")
    values = ordered[metrics].to_numpy().T
    bound = max(1.0, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(figsize=(14, 3.5))
    try:
        image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        axis.set_xticks(range(len(ordered)), labels=layer_labels(ordered["layer"]), rotation=90)
        axis.set_yticks(range(len(metrics)), labels=metrics)
        axis.set_xlabel("Hidden-state index (Emb. is the embedding state)")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="Cosine similarity / target gain")
        fig.tight_layout()
        save_figure(fig, output_path)
    finally:
        plt.close(fig)


def run_experiment(args):
    import matplotlib
    matplotlib.use("Agg")  # Select a headless backend before importing pyplot.
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "exp1_layerwise_cosine.csv"
    manifest, samples = read_samples(args.manifest, args.attack_results)
    print(f"Manifest: {args.manifest.resolve()} ({len(manifest)} samples)")
    print(f"Shared cache: {args.cache_dir.resolve()}")
    print(f"Outputs: {args.output_dir.resolve()}")
    representations = Representations(args.device, args.cache_dir, args.force_recompute_representations)
    provenance_columns = [
        "sample_id", "pair_id", "source_image_id", "target_image_id",
        "source_concept", "target_concept",
    ]
    all_rows = []
    attack_rows = []
    skipped = []
    for sample_index, sample in enumerate(samples.itertuples(index=False), start=1):
        print(f"[{sample_index}/{len(samples)}] {sample.sample_id}", flush=True)
        status = str(sample.status).strip().lower()
        if status not in SUCCESS_STATUSES:
            skipped.append((sample.sample_id, f"attack status={sample.status!r}"))
            warnings.warn(f"Skipping {sample.sample_id}: attack status={sample.status!r}")
            continue

        source_path = REPO_ROOT / sample.source_path
        target_path = REPO_ROOT / sample.target_path
        adversarial_path = REPO_ROOT / sample.adv_path
        try:
            validate_sample(samples.iloc[sample_index - 1].to_dict(), representations.fingerprints)
            if args.save_images:
                save_triplet(source_path, adversarial_path, target_path, sample,
                             args.output_dir / "triplets" / f"{sample_index:05d}_{cache_name(sample.sample_id)[:-3]}_triplet.png")
            sample_rows, attack_row = representations.analyze_sample(
                source_path,
                target_path,
                adversarial_path,
                sample.sample_id,
                sample.source_image_id,
                sample.target_image_id,
            )
            provenance = {column: getattr(sample, column) for column in provenance_columns}
            attack_rows.append({**provenance, **attack_row})
            for row in sample_rows:
                all_rows.append({**provenance, **row})
            if args.save_per_sample_heatmaps:
                save_heatmap(
                    pd.DataFrame(sample_rows),
                    f"AMP Experiment 1 — {sample.sample_id}: {sample.source_concept} → {sample.target_concept}",
                    args.output_dir / "sample_heatmaps" / f"{sample_index:05d}_{cache_name(sample.sample_id)[:-3]}_heatmap.png",
                )
        except Exception as error:
            skipped.append((sample.sample_id, f"{type(error).__name__}: {error}"))
            warnings.warn(f"Skipping {sample.sample_id}: {type(error).__name__}: {error}")

    columns = provenance_columns + ["layer", "R", "T", "B", "G", "tensor_shape"]
    results = pd.DataFrame(all_rows, columns=columns)
    if not results.empty:
        results = results.sort_values(["sample_id", "layer"], kind="stable")
        # Restore deterministic manifest order rather than lexical sample-ID order.
        sample_order = {sample_id: index for index, sample_id in enumerate(manifest["sample_id"])}
        results["_sample_order"] = results["sample_id"].map(sample_order)
        results = results.sort_values(["_sample_order", "layer"], kind="stable").drop(columns="_sample_order")
    results.to_csv(output_csv, index=False)  # Replace, never append.
    attack_csv = args.output_dir / "exp1_attack_output_cosine.csv"
    pd.DataFrame(attack_rows, columns=provenance_columns + ["representation", "attack_output", "R", "T", "B", "G", "tensor_shape"]).to_csv(attack_csv, index=False)
    representations.save_metadata(args.output_dir)
    pd.DataFrame(skipped, columns=["sample_id", "reason"]).to_csv(args.output_dir / "exp1_skipped_samples.csv", index=False)
    print(f"Actual attacked-output cosine: {attack_csv.resolve()}")

    analyzed_count = results["sample_id"].nunique() if not results.empty else 0
    print(
        "Summary: "
        f"manifest={len(manifest)}, analyzed={analyzed_count}, skipped/failed={len(skipped)}, "
        f"cache_hits={representations.cache_counts['hits']}, newly_extracted={representations.cache_counts['extracted']}"
    )
    if skipped:
        print("Skipped samples:", "; ".join(f"{sample_id} ({reason})" for sample_id, reason in skipped))
    if analyzed_count == 0:
        raise RuntimeError("No valid AMP attack samples remain; aggregate heatmap was not produced.")

    # Always compute the requested aggregate from the dataframe saved above.
    saved_results = pd.read_csv(output_csv, dtype={"sample_id": str})
    aggregate = (
        saved_results.groupby("layer", sort=True, as_index=False)[["R", "T", "B", "G"]]
        .mean()
        .sort_values("layer")
    )
    save_heatmap(
        aggregate,
        f"AMP Experiment 1 aggregate — mean across {analyzed_count} analyzed AMP samples",
        args.output_dir / "exp1_aggregate_mean_heatmap.png",
    )
    aggregate_path = args.output_dir / "exp1_layerwise_cosine_mean.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print(f"Saved sample results: {output_csv.resolve()}")
    print(f"Saved layer means: {aggregate_path.resolve()}")
    print(f"Saved aggregate heatmap: {(args.output_dir / 'exp1_aggregate_mean_heatmap.png').resolve()}")
    if args.save_images:
        print(f"Triplet PNGs: {(args.output_dir / 'triplets').resolve()}")
    if args.save_per_sample_heatmaps:
        print(f"Per-sample heatmaps: {(args.output_dir / 'sample_heatmaps').resolve()}")


def main():
    # Parse first: --help does not import ML packages or initialize CUDA/models.
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
