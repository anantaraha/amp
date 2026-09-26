#!/usr/bin/env python3
"""CogVLM dataset-level CKA using all-token means of native encoder states.

Run python exp2.py --help for runtime options. All plots are saved headlessly.
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
        description='Dataset-level all-token mean-pooled linear CKA; reuse model-specific Exp1 caches.',
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
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output/exp2",
                        help="Directory for result CSVs and PNGs; created automatically.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for model extraction (cuda or cuda:N); cached tensors are processed on CPU.")

    return parser.parse_args()


def save_figure(fig, output_path):
    """Save notebook figure geometry to PNG and release it even if writing fails."""
    import matplotlib.pyplot as plt

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def validate_image(path):
    from PIL import Image
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()


def mean_pool(states):
    """Produce [layers, native_width] float32 features by averaging every token, including CLS."""
    import torch
    return torch.stack([state.float().mean(dim=1).squeeze(0) for state in states]).numpy()


def center_gram(gram, unbiased=False):
    import numpy as np
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric Gram matrix.")
    gram = gram.copy()
    if unbiased:
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, axis=0, dtype=np.float64) / (gram.shape[0] - 2)
        means -= np.sum(means) / (2 * (gram.shape[0] - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, axis=0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram


def cka(gram_x, gram_y, debiased=False):
    import numpy as np
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def linear_cka_matrix(layer_features, debiased=False):
    """Compute all layer pairs with images as Gram-matrix observations."""
    import numpy as np
    grams = [layer @ layer.T for layer in layer_features]
    count = len(grams)
    matrix = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(row, count):
            value = cka(grams[row], grams[column], debiased=debiased)
            matrix[row, column] = matrix[column, row] = value
    return matrix


def draw_cka_heatmap(axis, matrix, title, cmap, vmin, vmax, colorbar_label, labels):
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Layer")
    axis.set_title(title)
    axis.figure.colorbar(image, ax=axis, label=colorbar_label)


def run_experiment(args):
    import matplotlib
    matplotlib.use("Agg")  # Select a headless backend before importing pyplot.
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "source": args.output_dir / "exp2_linear_cka_source.csv",
        "adversarial": args.output_dir / "exp2_linear_cka_adversarial.csv",
        "target": args.output_dir / "exp2_linear_cka_target.csv",
        "delta": args.output_dir / "exp2_linear_cka_adv_minus_source.csv",
        "samples": args.output_dir / "exp2_samples.csv",
    }
    manifest, samples = read_samples(args.manifest, args.attack_results)
    print(f"Manifest: {args.manifest.resolve()} ({len(manifest)} samples)")
    print(f"Shared cache: {args.cache_dir.resolve()}")
    print(f"Outputs: {args.output_dir.resolve()}")
    representations = Representations(args.device, args.cache_dir)
    provenance_columns = [
        "sample_id", "pair_id", "source_image_id", "target_image_id",
        "source_concept", "target_concept",
    ]
    feature_rows = {"source": [], "adversarial": [], "target": []}
    provenance_rows = []
    skipped = []
    pooled_shape = None

    for sample_index, sample in enumerate(samples.itertuples(index=False), start=1):
        print(f"[{sample_index}/{len(samples)}] {sample.sample_id}", flush=True)
        status = str(sample.status).strip().lower()
        if status not in SUCCESS_STATUSES:
            reason = f"attack status={sample.status!r}"
            skipped.append((sample.sample_id, reason))
            warnings.warn(f"Skipping {sample.sample_id}: {reason}")
            continue

        paths = {
            "source": REPO_ROOT / sample.source_path,
            "adversarial": REPO_ROOT / sample.adv_path,
            "target": REPO_ROOT / sample.target_path,
        }
        cache_paths = {
            "source": args.cache_dir / "clean" / cache_name(sample.source_image_id),
            "adversarial": args.cache_dir / "adv" / cache_name(sample.sample_id),
            "target": args.cache_dir / "clean" / cache_name(sample.target_image_id),
        }
        try:
            validate_sample(samples.iloc[sample_index - 1].to_dict(), representations.fingerprints)
            # Reject missing/corrupt members before adding any part of the triplet.
            for image_path in paths.values():
                validate_image(image_path)
            pooled = {
                condition: mean_pool(representations.get_hidden_states(paths[condition], cache_paths[condition]))
                for condition in ("source", "adversarial", "target")
            }
            current_shape = pooled["source"].shape
            if any(value.shape != current_shape for value in pooled.values()) or (pooled_shape is not None and current_shape != pooled_shape):
                raise ValueError(f"Unexpected pooled representation shapes: { {k: v.shape for k, v in pooled.items()} }")
            pooled_shape = current_shape
            for condition in feature_rows:
                feature_rows[condition].append(pooled[condition])
            provenance_rows.append({column: getattr(sample, column) for column in provenance_columns})
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            skipped.append((sample.sample_id, reason))
            warnings.warn(f"Skipping {sample.sample_id}: {reason}")

    representations.save_metadata(args.output_dir)
    pd.DataFrame(skipped, columns=["sample_id", "reason"]).to_csv(args.output_dir / "exp2_skipped_samples.csv", index=False)
    valid_count = len(provenance_rows)
    if valid_count < 3:
        raise RuntimeError(
            f"Linear CKA requires at least 3 valid AMP samples; only {valid_count} passed "
            "status, image, and representation validation."
        )

    # Each stack begins [samples, layers, width]; transpose to [layers, samples, width].
    source_features = np.stack(feature_rows["source"], axis=0).transpose(1, 0, 2)
    adversarial_features = np.stack(feature_rows["adversarial"], axis=0).transpose(1, 0, 2)
    target_features = np.stack(feature_rows["target"], axis=0).transpose(1, 0, 2)
    print("feature arrays [layers, samples, dimensions]:", {
        "source": source_features.shape,
        "adversarial": adversarial_features.shape,
        "target": target_features.shape,
    })

    cka_source = linear_cka_matrix(source_features, debiased=False)
    cka_adversarial = linear_cka_matrix(adversarial_features, debiased=False)
    cka_target = linear_cka_matrix(target_features, debiased=False)
    cka_delta = cka_adversarial - cka_source

    labels = ["Emb.", *map(str, range(1, source_features.shape[0]))]
    frames = {
        "source": pd.DataFrame(cka_source, index=labels, columns=labels),
        "adversarial": pd.DataFrame(cka_adversarial, index=labels, columns=labels),
        "target": pd.DataFrame(cka_target, index=labels, columns=labels),
        "delta": pd.DataFrame(cka_delta, index=labels, columns=labels),
    }
    for condition, frame in frames.items():
        frame.to_csv(output_paths[condition], index_label="layer")  # Replace, never append.
    pd.DataFrame(provenance_rows, columns=provenance_columns).to_csv(
        output_paths["samples"], index=False
    )

    difference_bound = float(np.max(np.abs(cka_delta)))
    if difference_bound == 0:
        difference_bound = np.finfo(np.float64).eps
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    try:
        draw_cka_heatmap(
            axes[0, 0], cka_source,
            f"CogVLM source linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA", labels,
        )
        draw_cka_heatmap(
            axes[0, 1], cka_adversarial,
            f"CogVLM adversarial linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA", labels,
        )
        draw_cka_heatmap(
            axes[1, 0], cka_target,
            f"CogVLM target linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA", labels,
        )
        draw_cka_heatmap(
            axes[1, 1], cka_delta,
            f"CogVLM adversarial − source CKA (N={valid_count} AMP samples)",
            "coolwarm", -difference_bound, difference_bound, "Linear CKA difference", labels,
        )
        fig.tight_layout()
        save_figure(fig, args.output_dir / "exp2_linear_cka_four_panel.png")
    finally:
        plt.close(fig)

    print(
        "Summary: "
        f"manifest={len(manifest)}, valid/analyzed={valid_count}, skipped={len(skipped)}, "
        f"cache_hits={representations.cache_counts['hits']}, newly_extracted={representations.cache_counts['extracted']}"
    )
    if skipped:
        print("Skipped samples:", "; ".join(f"{sample_id} ({reason})" for sample_id, reason in skipped))
    for path in output_paths.values():
        print(f"Saved: {path.resolve()}")
    print(f"Saved four-panel CKA figure: {(args.output_dir / 'exp2_linear_cka_four_panel.png').resolve()}")


def main():
    # Parse first: --help does not import ML packages or initialize CUDA/models.
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
