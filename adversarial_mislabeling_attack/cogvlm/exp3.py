#!/usr/bin/env python3
"""Test a single-image, cross-depth patch-token CKA gap using Exp1 representations.

Each image is analyzed independently: observations are spatial patch tokens,
never pooled images. The embedding hidden state is shown in heatmaps but excluded
from encoder-depth groups. This experiment tests a hypothesis, not a classifier.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import warnings

from _vision import (
    MODEL_ID, MODEL_REVISION, MODEL_TITLE, VLM, PRECISION, PREFIX_TOKENS,
    CACHE_VERSION, SUCCESS_STATUSES, SCRIPT_DIR, REPO_ROOT, ATTACK_DIR, DEFAULT_CACHE_DIR,
    cache_name, read_samples, validate_sample,
    Representations as NativeRepresentations, image_path, sha256_file,
)


CONDITIONS = ("source", "adversarial", "target")
METRICS = ("A_MD", "A_DD", "S_gap")
PROVENANCE = ("sample_id", "pair_id", "source_image_id", "target_image_id",
              "source_concept", "target_concept")


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N (for example cuda:1).")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-image centered linear or RBF CKA across patch tokens: S_gap = A_DD - A_MD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CLI paths are working-directory-relative; image paths in CSVs are repository-root-relative. "
               "Reuse full-token Exp1 caches; CUDA/native-precision extraction is needed only for missing/invalid caches. "
               "CKA runs on CPU in float64. CLS is excluded; encoder states are split into thirds as in analyze_results.py. "
               "Caches validate model revision, preprocessing, layout, precision and image SHA-256.",
    )
    parser.add_argument("--manifest", type=Path, default=ATTACK_DIR / "manifest.csv",
                        help="AMP source/target sample manifest CSV.")
    parser.add_argument("--attack-results", type=Path, default=ATTACK_DIR / VLM / "attack_results.csv",
                        help="Attack status/adversarial-path CSV; use cogvlm/attack_results.csv for model-specific generation outputs.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Exp1 full-token representation cache root, containing clean/ and adv/.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output/exp3",
                        help="Directory for Exp3 CSVs, run metadata, and PNGs; created automatically.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for missing representation extraction; cache-only computation needs no GPU.")
    parser.add_argument("--max-samples", type=positive_integer, default=None,
                        help="Process the first N manifest rows, including skipped rows; omitted means all.")
    parser.add_argument("--cka", choices=("linear", "rbf", "both"), default="linear",
                        help="CKA kernel; RBF uses a per-image/per-layer median patch-distance bandwidth. "
                             "Both reuses the linear pass's caches and writes separate exp3_rbf_* outputs.")
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument("--cache-only", action="store_true",
                               help="Skip triplets with missing/invalid caches instead of loading a model.")
    cache_options.add_argument("--force-recompute-representations", action="store_true",
                               help="Re-extract and replace selected caches, adding image-content hashes.")
    parser.add_argument("--no-per-image-plots", dest="per_image_plots", action="store_false",
                        help="Disable individual image heatmaps; retain numbers and aggregate plots "
                             "(per-image saving enabled by default: %(default)s).")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Disable all PNGs; retain numerical results (plotting enabled by default: %(default)s).")
    return parser.parse_args(argv)


class Representations:
    """Use exactly the same native full-token cache loader as Exp1 and Exp2."""

    def __init__(self, args):
        self.backend = NativeRepresentations(args.device, args.cache_dir, args.force_recompute_representations, args.cache_only)
        self.cache_counts = self.backend.cache_counts

    def get_hidden_states(self, path, cache_path, image_digest):
        payload = self.backend.get_payload(path, cache_path, image_digest)
        return payload["hidden_states"], "sha256_verified"


def layer_blocks(state_count):
    """Match analyze_results.layer_blocks: sorted encoders in thirds, embedding separate."""
    import numpy as np

    blocks = {0: "embedding"}
    for name, states in zip(("early", "middle", "deep"), np.array_split(np.arange(1, state_count), 3)):
        blocks.update({int(state): name for state in states})
    middle = [layer for layer, block in blocks.items() if block == "middle"]
    deep = [layer for layer, block in blocks.items() if block == "deep"]
    if not middle or len(deep) < 2:
        raise ValueError(f"Need a middle group and at least two deep layers; got M={middle}, D={deep}")
    return blocks, middle, deep


def token_cka_matrix(states):
    """Biased centered linear CKA, with PATCH TOKENS of ONE image as observations.

    For each [P, d] layer X, center each feature over its P tokens. Its Gram matrix
    is K = X_centered @ X_centered.T = H @ (X @ X.T) @ H. Frobenius-normalized
    Gram inner products give ||X_centered.T @ Y_centered||_F^2 divided by the
    corresponding self-norms. Layers/images are never pooled before this step.
    """
    import numpy as np

    normalized_grams = []
    patch_count = None
    for layer, state in enumerate(states):
        # This model's raw encoder states have a verified leading CLS token; attack-output tokens are separate.
        features = state[0, PREFIX_TOKENS:, :].detach().cpu().numpy().astype(np.float64)
        if features.ndim != 2 or features.shape[0] < 2 or not np.isfinite(features).all():
            raise ValueError(f"Invalid patch-token features at layer {layer}")
        if patch_count is not None and features.shape[0] != patch_count:
            raise ValueError("Layers must use the same spatial token observations")
        patch_count = features.shape[0]
        features -= features.mean(axis=0, keepdims=True)
        gram = features @ features.T
        norm = np.linalg.norm(gram)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Undefined CKA at layer {layer}: zero/nonfinite centered token variance")
        normalized_grams.append((gram / norm).ravel())
    if not normalized_grams:
        raise ValueError("No layers to compare")
    grams = np.stack(normalized_grams)
    matrix = grams @ grams.T
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite token CKA matrix")
    return matrix


def token_rbf_cka_matrix(states):
    """Centered RBF CKA across patch-token observations of a single image.

    For EACH layer/image, sigma = median(||x_i - x_j||_2 for i < j), with
    the diagonal excluded and duplicate-token distances retained. Then
    K_ij = exp(-||x_i - x_j||_2**2 / (2 * sigma**2)). Compare HKH matrices
    using their Frobenius-normalized inner products, as for linear CKA.
    A zero/nonfinite median bandwidth is undefined and rejects the triplet;
    no arbitrary bandwidth floor or cross-image bandwidth is substituted.
    """
    import numpy as np

    normalized_grams = []
    patch_count = None
    for layer, state in enumerate(states):
        features = state[0, PREFIX_TOKENS:, :].detach().cpu().numpy().astype(np.float64)
        if features.ndim != 2 or features.shape[0] < 2 or not np.isfinite(features).all():
            raise ValueError(f"Invalid patch-token features at layer {layer}")
        if patch_count is not None and features.shape[0] != patch_count:
            raise ValueError("Layers must use the same spatial token observations")
        patch_count = features.shape[0]
        # Translation does not change distances; centering improves numerical stability.
        features -= features.mean(axis=0, keepdims=True)
        products = features @ features.T
        squared_norms = np.diag(products).copy()
        squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * products
        np.maximum(squared_distances, 0, out=squared_distances)
        np.fill_diagonal(squared_distances, 0)
        distances = np.sqrt(squared_distances[np.triu_indices(patch_count, k=1)])
        sigma = float(np.median(distances))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Undefined RBF CKA at layer {layer}: zero/nonfinite median pairwise distance")
        gram = np.exp(-squared_distances / (2 * sigma ** 2))
        # Explicit double centering, H K H, over token observations.
        row_means = gram.mean(axis=1, keepdims=True)
        column_means = gram.mean(axis=0, keepdims=True)
        grand_mean = gram.mean()
        gram -= row_means
        gram -= column_means
        gram += grand_mean
        norm = np.linalg.norm(gram)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Undefined RBF CKA at layer {layer}: zero/nonfinite centered kernel norm")
        normalized_grams.append((gram / norm).ravel())
    if not normalized_grams:
        raise ValueError("No layers to compare")
    grams = np.stack(normalized_grams)
    matrix = grams @ grams.T
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite RBF token CKA matrix")
    return matrix


def gap_scores(matrix, middle, deep):
    import numpy as np

    md = matrix[np.ix_(middle, deep)]
    dd = matrix[np.ix_(deep, deep)][np.triu_indices(len(deep), k=1)]
    if not md.size or not dd.size:
        raise ValueError("A_MD and A_DD require nonempty cross-depth and unique deep-layer pairs")
    a_md, a_dd = float(md.mean()), float(dd.mean())
    return {"A_MD": a_md, "A_DD": a_dd, "S_gap": a_dd - a_md}


def save_csv(frame, path, **kwargs):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def summarize_scores(frame, group_column):
    """Descriptive image-level statistics; no classification or significance tests."""
    import pandas as pd

    rows = []
    for group, subset in frame.groupby(group_column, sort=False):
        for metric in METRICS:
            values = subset[metric]
            row = {
                group_column: group, "metric": metric, "n_samples": len(subset),
                "n_pairs": subset.loc[subset.pair_id.ne(""), "pair_id"].nunique(),
                "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
                "min": values.min(), "max": values.max(),
            }
            if "image_id" in subset:
                row["n_unique_images"] = subset.image_id.nunique()
            rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(results):
    """Compare the same sample IDs, preserving manifest order and concept-pair IDs."""
    import pandas as pd

    reference = results.loc[results.condition.eq("source")].set_index("sample_id")
    conditions = {name: results.loc[results.condition.eq(name)].set_index("sample_id").reindex(reference.index)
                  for name in CONDITIONS}
    rows = []
    for left, right in [("adversarial", "source"), ("adversarial", "target"), ("target", "source")]:
        delta = conditions[left][list(METRICS)] - conditions[right][list(METRICS)]
        for sample_id, values in delta.iterrows():
            rows.append({"sample_id": sample_id, "pair_id": reference.loc[sample_id, "pair_id"],
                         "comparison": f"{left}_minus_{right}", **values.to_dict()})
    return pd.DataFrame(rows)


def save_figure(fig, output_path):
    import matplotlib.pyplot as plt

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def draw_cka_heatmap(axis, matrix, blocks, title, difference=False, bound=1):
    labels = ["Emb." if layer == 0 else str(layer) for layer in blocks]
    image = axis.imshow(matrix, vmin=-bound if difference else 0, vmax=bound if difference else 1,
                        cmap="coolwarm" if difference else "viridis", origin="upper")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Hidden-state index")
    axis.set_ylabel("Hidden-state index")
    axis.set_title(title)
    for layer in range(len(blocks) - 1):
        if blocks[layer] != blocks[layer + 1]:
            axis.axhline(layer + 0.5, color="white", linewidth=0.6)
            axis.axvline(layer + 0.5, color="white", linewidth=0.6)
    axis.figure.colorbar(image, ax=axis, label="Token CKA difference" if difference else "Token CKA")


def save_per_image_plot(matrix, blocks, sample_id, condition, output_path, cka="linear"):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 8))
    try:
        kernel_label = "RBF " if cka == "rbf" else ""
        draw_cka_heatmap(axis, matrix, blocks, f"{sample_id} — {condition}\nSingle-image patch-token {kernel_label}CKA")
        fig.tight_layout()
        save_figure(fig, output_path)
    finally:
        plt.close(fig)


def save_aggregate_plot(means, delta, blocks, count, output_path, cka="linear"):
    import matplotlib.pyplot as plt
    import numpy as np

    kernel_label = "RBF " if cka == "rbf" else ""
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    try:
        for axis, condition in zip(axes.flat, CONDITIONS):
            draw_cka_heatmap(axis, means[condition], blocks,
                             f"{condition.title()}: mean single-image token {kernel_label}CKA (N={count})")
        bound = max(float(np.max(np.abs(delta))), np.finfo(np.float64).eps)
        draw_cka_heatmap(axes[1, 1], delta, blocks,
                         f"Adversarial − source mean token {kernel_label}CKA (N={count})", difference=True, bound=bound)
        fig.tight_layout()
        save_figure(fig, output_path)
    finally:
        plt.close(fig)


def _run_experiment(args, cka="linear"):
    import numpy as np
    import pandas as pd

    prefix = "exp3" if cka == "linear" else "exp3_rbf"

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, samples = read_samples(args.manifest, args.attack_results)
    if args.max_samples is not None:
        samples = samples.head(args.max_samples)
    print(f"Manifest: {args.manifest.resolve()} ({len(samples)}/{len(manifest)} selected rows)")
    print(f"Shared full-token cache: {args.cache_dir.resolve()}")
    print(f"Outputs: {args.output_dir.resolve()}")
    print("CKA observations: patch tokens within one image; CLS excluded. S_gap = A_DD - A_MD.", flush=True)
    if cka == "rbf":
        print("RBF kernel: sigma = median off-diagonal Euclidean patch distance per image/layer.", flush=True)
    representations = Representations(args)
    fingerprints, sums = {}, {}
    rows, skipped, plot_errors = [], [], []
    signature, blocks = None, None
    analyzed = 0
    for sample_index, row in enumerate(samples.to_dict(orient="records"), start=1):
        sample_id = row["sample_id"]
        print(f"[{sample_index}/{len(samples)}] {sample_id}", flush=True)
        status = str(row.get("_attack_status", "")).strip().lower()
        if status not in SUCCESS_STATUSES:
            skipped.append({"sample_id": sample_id, "pair_id": row.get("pair_id", ""),
                            "reason": f"attack status={status!r} (unsuccessful or unmatched)"})
            continue
        try:
            paths = validate_sample(row, fingerprints)
            cache_paths = {
                "source": args.cache_dir / "clean" / cache_name(row["source_image_id"]),
                "target": args.cache_dir / "clean" / cache_name(row["target_image_id"]),
                "adversarial": args.cache_dir / "adv" / cache_name(sample_id),
            }
            sample_rows, matrices = [], {}
            current_signature = None
            for condition in CONDITIONS:
                states, validation = representations.get_hidden_states(
                    paths[condition], cache_paths[condition], fingerprints[paths[condition]],
                )
                state_signature = (len(states), tuple(states[0].shape))
                if current_signature is not None and state_signature != current_signature:
                    raise ValueError("Source/adversarial/target layer or token shapes differ")
                if signature is not None and state_signature != signature:
                    raise ValueError("Layer/token layout differs from previously analyzed samples")
                current_signature = state_signature
                current_blocks, middle, deep = layer_blocks(len(states))
                matrix = token_cka_matrix(states) if cka == "linear" else token_rbf_cka_matrix(states)
                del states
                matrices[condition] = matrix
                sample_rows.append({
                    **{column: row.get(column, "") for column in PROVENANCE},
                    "condition": condition,
                    "image_id": sample_id if condition == "adversarial" else row[f"{condition}_image_id"],
                    "image_path": str(paths[condition]), "attack_status": status,
                    "cache_path": str(cache_paths[condition].resolve()), "cache_validation": validation,
                    "n_layers": state_signature[0], "n_patch_tokens": state_signature[1][1] - 1,
                    **gap_scores(matrix, middle, deep),
                })
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            skipped.append({"sample_id": sample_id, "pair_id": row.get("pair_id", ""), "reason": reason})
            warnings.warn(f"Skipping entire triplet {sample_id}: {reason}")
            continue

        # Commit all three conditions together: every aggregate uses the same sample set.
        if signature is None:
            signature, blocks = current_signature, current_blocks
            sums = {condition: np.zeros_like(matrices[condition]) for condition in CONDITIONS}
            print(f"Inferred {signature[0]} states, {signature[1][1] - 1} patch tokens; M={middle}, D={deep}")
        rows.extend(sample_rows)
        analyzed += 1
        for condition in CONDITIONS:
            sums[condition] += matrices[condition]
            if args.plots and args.per_image_plots:
                filename = f"{'rbf_' if cka == 'rbf' else ''}{sample_index:06d}_{cache_name(sample_id)[:-3]}_{condition}.png"
                try:
                    save_per_image_plot(matrices[condition], blocks, sample_id, condition,
                                        args.output_dir / "per_image" / filename, cka=cka)
                except Exception as error:
                    plot_errors.append({"sample_id": sample_id, "condition": condition, "error": str(error)})
                    warnings.warn(f"Could not plot {sample_id}/{condition}: {error}")

    columns = [*PROVENANCE, "condition", "image_id", "image_path", "attack_status", "cache_path", "cache_validation",
               "n_layers", "n_patch_tokens", *METRICS]
    results = pd.DataFrame(rows, columns=columns)
    score_path = args.output_dir / f"{prefix}_image_scores.csv"
    save_csv(results, score_path, index=False)
    save_csv(pd.DataFrame(skipped, columns=["sample_id", "pair_id", "reason"]),
             args.output_dir / f"{prefix}_skipped_samples.csv", index=False)
    representations.backend.save_metadata(args.output_dir)
    metadata = {
        "experiment": "Single-image centered linear CKA across spatial patch-token observations",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "cache_version": CACHE_VERSION, "representation_precision": PRECISION,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": sha256_file(args.manifest), "attack_results_sha256": sha256_file(args.attack_results),
        "manifest_rows": len(manifest), "selected_rows": len(samples), "analyzed_triplets": analyzed,
        "skipped_triplets": len(skipped), "cache_counts": representations.cache_counts,
        "hidden_state_count_and_shape": signature, "layer_blocks": blocks,
        "cls_tokens_removed_per_layer": 1, "embedding_state_in_scores": False,
        "cka_precision": "float64", "cka_estimator": "biased centered linear CKA",
        "A_MD": "Mean over every middle x deep layer pair",
        "A_DD": "Mean over unique unordered deep-layer pairs, excluding the diagonal",
        "S_gap": "A_DD - A_MD",
        "aggregation": "Arithmetic mean of independently computed per-image matrices/scores over matched triplets. "
                       "Each manifest sample has equal weight; repeated clean image IDs are not independent images.",
        "cache_provenance": "Model revision, encoder layout, precision, preprocessing and image SHA-256 are validated for every cache.",
        "image_id": "Source image ID for source; target image ID for target; sample ID for the adversarial image. "
                    "The sample ID and condition identify each evaluated occurrence.",
        "interpretation": "Exploratory score, not an established detector. No thresholds or classification performance. "
                          "Exp2 uses images as observations and does not establish this per-image hypothesis.",
        "plot_errors": plot_errors,
    }
    if cka == "rbf":
        metadata.update(
            experiment="Single-image centered RBF CKA across spatial patch-token observations",
            cka_estimator="biased centered RBF CKA",
            rbf_kernel="exp(-squared_distance / (2 * sigma**2))",
            rbf_bandwidth="For each image/layer independently: sigma is the median Euclidean distance "
                          "over unique unordered off-diagonal patch-token pairs; zero distances are retained.",
            rbf_degenerate_policy="Skip the whole RBF triplet if any layer has zero/nonfinite median "
                                  "distance or zero/nonfinite centered kernel norm; linear results are independent.",
        )
    (args.output_dir / f"{prefix}_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Summary: analyzed={analyzed}, skipped={len(skipped)}, cache={representations.cache_counts}")
    print(f"Image scores: {score_path.resolve()}")
    print(f"Skipped samples: {(args.output_dir / f'{prefix}_skipped_samples.csv').resolve()}")
    if analyzed == 0:
        raise RuntimeError(f"No complete valid triplets remain; see {prefix}_skipped_samples.csv. Aggregate outputs unavailable.")

    counts = results.groupby("sample_id").condition.agg(list)
    if results.duplicated(["sample_id", "condition"]).any() or any(value != list(CONDITIONS) for value in counts):
        raise ValueError("Unmatched conditions in image score results")
    save_csv(summarize_scores(results, "condition"), args.output_dir / f"{prefix}_condition_summary.csv", index=False)
    differences = paired_differences(results)
    save_csv(differences, args.output_dir / f"{prefix}_paired_differences.csv", index=False)
    save_csv(summarize_scores(differences, "comparison"), args.output_dir / f"{prefix}_paired_summary.csv", index=False)
    save_csv(pd.DataFrame({"layer": list(blocks), "block": list(blocks.values())}),
             args.output_dir / f"{prefix}_layer_blocks.csv", index=False)
    labels = ["Emb." if layer == 0 else str(layer) for layer in blocks]
    means = {condition: sums[condition] / analyzed for condition in CONDITIONS}
    delta = means["adversarial"] - means["source"]
    for condition, matrix in {**means, "adv_minus_source": delta}.items():
        save_csv(pd.DataFrame(matrix, index=labels, columns=labels),
                 args.output_dir / f"{prefix}_mean_token_cka_{condition}.csv", index_label="layer")
    if args.plots:
        save_aggregate_plot(means, delta, blocks, analyzed, args.output_dir / f"{prefix}_token_cka_four_panel.png", cka=cka)
    print(f"Condition/paired summaries, mean CKA matrices, layer groups, and run metadata: {args.output_dir.resolve()}")
    if args.plots:
        print(f"Aggregate figure: {(args.output_dir / f'{prefix}_token_cka_four_panel.png').resolve()}")
        if args.per_image_plots:
            print(f"Per-image figures: {(args.output_dir / 'per_image').resolve()}")
    if plot_errors:
        print(f"Per-image plotting errors: {len(plot_errors)} (details in {prefix}_run.json)")
    return results


def run_experiment(args):
    """Keep the original linear pass; RBF uses its own results and skip accounting."""
    cka = getattr(args, "cka", "linear")
    if cka not in ("linear", "rbf", "both"):
        raise ValueError("cka must be linear, rbf, or both")
    if cka != "both":
        return _run_experiment(args, cka)

    results, failures = {}, []
    for kernel in ("linear", "rbf"):
        kernel_args = argparse.Namespace(**vars(args))
        if kernel == "rbf":
            # The linear pass already refreshed requested caches. Reuse them so
            # --cka both --force-recompute-representations does not extract twice.
            kernel_args.force_recompute_representations = False
        print(f"=== {kernel.upper()} single-image token CKA ===", flush=True)
        try:
            results[kernel] = _run_experiment(kernel_args, kernel)
        except RuntimeError as error:
            # Save both kernels' diagnostics even if one has no valid triplets.
            failures.append((kernel, error))
            warnings.warn(f"{kernel.upper()} CKA did not complete: {error}")
    if failures:
        raise RuntimeError("; ".join(f"{kernel}: {error}" for kernel, error in failures)) from failures[0][1]
    return results


def main():
    # --help works without importing ML/plotting packages or loading a model.
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
