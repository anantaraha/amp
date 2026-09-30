#!/usr/bin/env python3
"""xGen-MM Exp4: mean single-image token CKA outside an encoder-layer diagonal band.

Reuse this model's Exp3 cache validation and linear/RBF CKA functions. Patch
tokens are observations; CLS and embedding state 0 are excluded from the score.
Lower S_global is the prespecified adversarial direction; no threshold is fitted.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
import warnings

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/anantaraha/amp/dataset/laion_art")
VLM = "xgen_mm"
MODEL_TITLE = "xGen-MM"
CONDITIONS = ("source", "adversarial", "target")
KERNELS = ("linear", "rbf")


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer.")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=f"{MODEL_TITLE} Exp4: S_global = mean encoder CKA(i,j), i<j and j-i>k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CKA runs on CPU in float64 using the existing Exp3 functions. Embedding state 0 and CLS "
               "are excluded. No boundary padding or per-row averaging. AUROC uses -S_global, with "
               "adversarial positive. Reruns recompute scores and replace this run's outputs; full-token "
               "caches are reused unless force-recompute is requested. CLI paths are working-directory-relative; "
               "CSV image paths are repository-root-relative. --cache-only needs no GPU.",
    )
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "attack_set/manifest.csv",
                        help="Source/target manifest CSV.")
    parser.add_argument("--attack-results", type=Path, default=DATA_ROOT / "attack_set" / VLM / "attack_results.csv",
                        help="Model-specific attack statuses and adversarial image paths.")
    parser.add_argument("--cache-dir", type=Path, default=DATA_ROOT / "attack_set/representations/xgen_mm_phi3_mini_instruct_r_v1",
                        help="Existing Exp1/Exp3 full-token caches, with clean/ and adv/ subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "output" / VLM / "exp4",
                        help="Exp4 CSVs, metadata and PNGs.")
    parser.add_argument("--diag-width", type=nonnegative_integer, default=0,
                        help="Exclude encoder-layer pairs with distance <= k; 0 excludes self-similarity only.")
    parser.add_argument("--cka", choices=("linear", "rbf", "both"), default="linear",
                        help="Both kernels share one cache-loading pass and produce separate outputs.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="Device for missing-cache extraction only; CKA itself runs on CPU.")
    parser.add_argument("--max-samples", type=positive_integer, default=None,
                        help="First N manifest rows, including unsuccessful/unmatched rows; omitted means all.")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-only", action="store_true",
                       help="Skip whole triplets with missing/invalid caches; never load a model or write caches.")
    cache.add_argument("--force-recompute-representations", action="store_true",
                       help="Re-extract and replace selected representation caches using this model's Exp3 convention.")
    parser.add_argument("--no-per-image-plots", dest="per_image_plots", action="store_false",
                        help="Keep aggregate plots and all numbers; skip individual heatmaps (default enabled: %(default)s).")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Skip all PNGs, retaining all numerical outputs (default enabled: %(default)s).")
    return parser.parse_args(argv)


def load_module(name, filename):
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = old_bytecode


@contextmanager
def model_backend(args):
    """Import this xGen-MM Exp3 with its own native encoder/cache implementation."""
    previous = sys.modules.get("_vision")
    try:
        native = load_module("_xgen_exp4_vision", "_vision.py")
        sys.modules["_vision"] = native
        experiment = load_module("_xgen_exp4_exp3", "exp3.py")
    finally:
        if previous is None:
            sys.modules.pop("_vision", None)
        else:
            sys.modules["_vision"] = previous
    yield experiment


def model_metadata(exp3):
    return {
        "representation_precision": exp3.PRECISION, "model_revision": exp3.MODEL_REVISION,
        "vision_source_sha256": exp3.sha256_file(SCRIPT_DIR / "_vision.py"),
        "cache_provenance": "xGen-MM validates model revision, preprocessing, full-token layout, native "
                            "precision, attack-output representation and image SHA-256 for every cache.",
    }


class NoLayerPairs(ValueError):
    """The requested diagonal band excludes every encoder-layer pair."""


def encoder_layer_pairs(state_count, diag_width):
    """Return one-based encoder-state indices; embedding state 0 is never used."""
    import numpy as np

    if not isinstance(diag_width, (int, np.integer)) or diag_width < 0:
        raise ValueError("diag_width must be a nonnegative integer.")
    n_encoder_layers = state_count - 1
    if n_encoder_layers < 2 or diag_width >= n_encoder_layers - 1:
        raise NoLayerPairs(f"No encoder-layer pairs remain for diag_width={diag_width} and "
                           f"{n_encoder_layers} encoder layers; require 0 <= k <= {n_encoder_layers - 2}.")
    left, right = np.triu_indices(n_encoder_layers, k=diag_width + 1)
    return left + 1, right + 1


def global_cka_score(matrix, diag_width=0):
    """Arithmetic mean over every eligible unordered encoder pair, with no padding."""
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square full layer-to-layer CKA matrix, including embedding state 0.")
    left, right = encoder_layer_pairs(matrix.shape[0], diag_width)
    values = matrix[left, right]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite CKA for an eligible encoder pair.")
    return float(values.mean())


def lower_score_roc(negative_scores, adversarial_scores):
    """ROC/AUROC with fixed detection score -S_global; ties receive half credit."""
    import numpy as np
    import pandas as pd

    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(adversarial_scores, dtype=np.float64)
    if negative.ndim != 1 or positive.ndim != 1 or not len(negative) or not len(positive):
        raise ValueError("AUROC requires nonempty one-dimensional negative and adversarial scores.")
    values = -np.concatenate([negative, positive])
    if not np.isfinite(values).all():
        raise ValueError("AUROC scores must be finite.")
    n_negative, n_positive = len(negative), len(positive)
    ranks = pd.Series(values).rank(method="average").to_numpy()
    area = float((ranks[n_negative:].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))
    labels = np.r_[np.zeros(n_negative, dtype=int), np.ones(n_positive, dtype=int)]
    order = np.argsort(-values, kind="stable")
    ordered, labels = values[order], labels[order]
    # Advance both counts together at tied thresholds.
    endpoints = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    tpr = np.r_[0.0, np.cumsum(labels)[endpoints] / n_positive]
    fpr = np.r_[0.0, np.cumsum(1 - labels)[endpoints] / n_negative]
    return area, pd.DataFrame({"FPR": fpr, "TPR": tpr})


class Outputs:
    def __init__(self, args, exp3, kernel):
        self.directory, self.exp3 = args.output_dir, exp3
        self.prefix = "exp4" if kernel == "linear" else "exp4_rbf"
        self.files, self.plot_errors = [], []

    def csv(self, suffix, frame, **kwargs):
        path = self.directory / f"{self.prefix}_{suffix}.csv"
        self.exp3.save_csv(frame, path, **kwargs)
        self.files.append(str(path.relative_to(self.directory)))
        return path

    def figure(self, figure, suffix):
        path = self.directory / f"{self.prefix}_{suffix}.png"
        self.exp3.save_figure(figure, path)
        self.files.append(str(path.relative_to(self.directory)))

    def per_image(self, figure, filename):
        path = self.directory / "per_image" / filename
        self.exp3.save_figure(figure, path)
        self.files.append(str(path.relative_to(self.directory)))

    def metadata(self, value):
        path = self.directory / f"{self.prefix}_run.json"
        value["output_files"] = [*self.files, path.name]
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def plot_cka(axis, matrix, diag_width, title, difference=False, bound=1):
    """Show encoder CKA with the excluded band masked, preserving numerical inputs."""
    import numpy as np

    encoder = np.asarray(matrix)[1:, 1:]
    layers = np.arange(1, len(encoder) + 1)
    mask = np.abs(layers[:, None] - layers[None, :]) <= diag_width
    image = axis.imshow(np.ma.array(encoder, mask=mask), origin="upper",
                        cmap="coolwarm" if difference else "viridis",
                        vmin=-bound if difference else 0, vmax=bound if difference else 1,
                        extent=(0.5, len(encoder) + 0.5, len(encoder) + 0.5, 0.5))
    axis.set_facecolor("#eeeeee")
    ticks = np.unique(np.linspace(1, len(encoder), min(len(encoder), 10), dtype=int))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlabel("Encoder layer")
    axis.set_ylabel("Encoder layer")
    axis.set_title(title)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def condition_summary(frame):
    import pandas as pd

    rows = []
    for condition in CONDITIONS:
        subset = frame.loc[frame.condition.eq(condition)]
        scores = subset.S_global
        rows.append({"condition": condition, "n_samples": len(subset),
                     "n_pairs": subset.loc[subset.pair_id.ne(""), "pair_id"].nunique(),
                     "n_unique_images": subset.image_id.nunique(),
                     "n_unique_image_contents": subset.image_sha256.nunique(),
                     "mean": scores.mean(), "median": scores.median(), "std": scores.std(ddof=1),
                     "min": scores.min(), "max": scores.max()})
    return pd.DataFrame(rows)


def paired_statistics(frame, provenance):
    import pandas as pd

    source = frame.loc[frame.condition.eq("source")].set_index("sample_id")
    adversarial = frame.loc[frame.condition.eq("adversarial")].set_index("sample_id").reindex(source.index)
    paired = source[list(provenance[1:])].copy()
    paired["source_S_global"] = source.S_global
    paired["adversarial_S_global"] = adversarial.S_global
    paired["adv_minus_source"] = paired.adversarial_S_global - paired.source_S_global
    paired["adv_below_source"] = paired.adv_minus_source < 0
    paired["diag_width"] = source.diag_width
    delta = paired.adv_minus_source
    summary = pd.DataFrame([{
        "comparison": "adversarial_minus_source", "n_samples": len(paired),
        "n_pairs": paired.loc[paired.pair_id.ne(""), "pair_id"].nunique(),
        "mean": delta.mean(), "median": delta.median(), "std": delta.std(ddof=1),
        "min": delta.min(), "max": delta.max(), "adv_below_source": int((delta < 0).sum()),
        "adv_equal_source": int((delta == 0).sum()), "adv_above_source": int((delta > 0).sum()),
        "fraction_adv_below_source": float((delta < 0).mean()),
    }])
    return paired.reset_index(), summary


def auroc_statistics(frame):
    import pandas as pd

    rows, curves = [], []
    positive = frame.loc[frame.condition.eq("adversarial")]
    for name, conditions in [("source", ("source",)), ("target", ("target",)),
                              ("source+target", ("source", "target"))]:
        negative = frame.loc[frame.condition.isin(conditions)]
        area, curve = lower_score_roc(negative.S_global, positive.S_global)
        rows.append({"positive_condition": "adversarial", "negative_condition": name,
                     "n_positive_occurrences": len(positive), "n_negative_occurrences": len(negative),
                     "n_unique_negative_images": negative.image_sha256.nunique(),
                     "detection_score": "-S_global", "AUROC": area})
        curves.append(curve.assign(negative_condition=name))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def aggregate_plots(frame, paired, auroc, curves, means, args, kernel, outputs):
    import numpy as np
    import matplotlib.pyplot as plt

    def save(suffix, create):
        try:
            outputs.figure(create(), suffix)
        except Exception as error:
            plt.close("all")
            outputs.plot_errors.append({"plot": suffix, "error": f"{type(error).__name__}: {error}"})
            warnings.warn(f"Could not save {suffix}: {error}")

    def scores_plot():
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        low, high = float(frame.S_global.min()), float(frame.S_global.max())
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        bins = np.linspace(low, high, 21)
        for condition in CONDITIONS:
            values = frame.loc[frame.condition.eq(condition), "S_global"]
            axes[0].hist(values, bins=bins, alpha=0.5, label=f"{condition} (n={len(values)})")
        axes[0].set_xlabel(r"$S_{global}$ (lower indicates adversarial)")
        axes[0].set_ylabel("Image occurrences")
        axes[0].legend()
        axes[1].scatter(paired.source_S_global, paired.adversarial_S_global, alpha=0.7, s=30)
        axes[1].plot([low, high], [low, high], "--", color="black")
        axes[1].set_xlim(low, high)
        axes[1].set_ylim(low, high)
        axes[1].set_xlabel(r"Source $S_{global}$")
        axes[1].set_ylabel(r"Adversarial $S_{global}$")
        axes[1].set_title(f"{int(paired.adv_below_source.sum())}/{len(paired)} adversarial scores below source")
        figure.suptitle(f"{MODEL_TITLE} Exp4 — {kernel.upper()}, k={args.diag_width}")
        figure.tight_layout()
        return figure

    def roc_plot():
        figure, axis = plt.subplots(figsize=(7, 6))
        for row in auroc.to_dict("records"):
            curve = curves.loc[curves.negative_condition.eq(row["negative_condition"])]
            axis.plot(curve.FPR, curve.TPR, label=f"vs {row['negative_condition']}: AUROC={row['AUROC']:.3f}")
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
                 title=f"{MODEL_TITLE} {kernel.upper()} Exp4 ROC; k={args.diag_width}\n"
                       "Adversarial positive; detection score = −S_global")
        axis.legend()
        figure.tight_layout()
        return figure

    def matrices_plot():
        figure, axes = plt.subplots(2, 2, figsize=(15, 13))
        for axis, condition in zip(axes.flat, CONDITIONS):
            plot_cka(axis, means[condition], args.diag_width, f"{condition.title()}: mean token CKA")
        delta = means["adversarial"] - means["source"]
        left, right = encoder_layer_pairs(len(delta), args.diag_width)
        bound = max(float(np.max(np.abs(delta[left, right]))), np.finfo(np.float64).eps)
        plot_cka(axes[1, 1], delta, args.diag_width, "Adversarial − source", difference=True, bound=bound)
        figure.suptitle(f"{MODEL_TITLE} Exp4 — {kernel.upper()}, k={args.diag_width}; gray band excluded")
        figure.tight_layout()
        return figure

    save("score_distributions_and_pairs", scores_plot)
    save("roc_curves", roc_plot)
    save("token_cka_four_panel", matrices_plot)


def run_experiment(args, exp3):
    import numpy as np
    import pandas as pd

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, samples = exp3.read_samples(args.manifest, args.attack_results)
    if args.max_samples is not None:
        samples = samples.head(args.max_samples)
    kernels = KERNELS if args.cka == "both" else (args.cka,)
    functions = {"linear": exp3.token_cka_matrix, "rbf": exp3.token_rbf_cka_matrix}
    representations = exp3.Representations(args)
    prefix_tokens = exp3.PREFIX_TOKENS
    states_by_kernel = {
        kernel: {"rows": [], "skipped": [], "sums": None, "signature": None,
                 "outputs": Outputs(args, exp3, kernel)} for kernel in kernels
    }
    common_metadata = {
        "experiment": "Exp4 single-image global encoder token CKA",
        "model_id": exp3.MODEL_ID, "vlm": VLM, "cache_version": exp3.CACHE_VERSION,
        **model_metadata(exp3),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": exp3.sha256_file(args.manifest),
        "attack_results_sha256": exp3.sha256_file(args.attack_results),
        "exp3_source_sha256": exp3.sha256_file(SCRIPT_DIR / "exp3.py"),
        "exp4_source_sha256": exp3.sha256_file(Path(__file__).resolve()),
        "manifest_rows": len(manifest), "selected_rows": len(samples),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cka_precision": "float64", "cka_device": "cpu",
        "cls_tokens_removed_per_layer": prefix_tokens, "embedding_state_in_scores": False,
        "diag_width": args.diag_width,
        "S_global": "Arithmetic mean of CKA(i,j) over encoder layers i<j with j-i>diag_width",
        "pair_weighting": "One vote per eligible unordered layer pair; no padding or per-row averaging",
        "score_direction": "Lower S_global indicates adversarial; AUROC uses -S_global, adversarial positive",
        "aggregation": "Each kernel admits complete source/adversarial/target triplets only. Kernels can have "
                       "different valid cohorts. Manifest image occurrences are weighted equally.",
        "interpretation": "Exploratory separation, not held-out detector performance. Repeated clean images "
                          "and paired occurrences are correlated; no threshold or score direction is fitted.",
        "cache_counts_scope": "One shared cache-loading/extraction pass for all requested kernels",
    }
    for state in states_by_kernel.values():
        state["outputs"].metadata({**common_metadata, "status": "running"})

    print(f"{MODEL_TITLE} Exp4: {len(samples)}/{len(manifest)} manifest rows; kernels={','.join(kernels)}; k={args.diag_width}")
    print(f"Caches: {args.cache_dir.resolve()}")
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)
    fingerprints, signature, fatal = {}, None, None
    for sample_index, record in enumerate(samples.to_dict("records"), 1):
        sample_id = record["sample_id"]
        print(f"[{sample_index}/{len(samples)}] {sample_id}", flush=True)
        status = str(record.get("_attack_status", "")).strip().lower()
        matrices = {kernel: {} for kernel in kernels}
        sample_rows = {kernel: [] for kernel in kernels}
        errors = {}
        current_signature = None
        try:
            if status not in exp3.SUCCESS_STATUSES:
                raise ValueError(f"attack status={status!r} (unsuccessful or unmatched)")
            paths = exp3.validate_sample(record, fingerprints)
            for condition in CONDITIONS:
                identifier = sample_id if condition == "adversarial" else record[f"{condition}_image_id"]
                cache_path = args.cache_dir / ("adv" if condition == "adversarial" else "clean") / exp3.cache_name(identifier)
                states, validation = representations.get_hidden_states(paths[condition], cache_path,
                                                                       fingerprints[paths[condition]])
                state_signature = (len(states), tuple(states[0].shape))
                if current_signature is not None and state_signature != current_signature:
                    raise ValueError("Source/adversarial/target layer or token shapes differ")
                if signature is not None and state_signature != signature:
                    raise ValueError("Layer/token layout differs from previously analyzed samples")
                current_signature = state_signature
                left, right = encoder_layer_pairs(len(states), args.diag_width)
                for kernel in kernels:
                    if kernel in errors:
                        continue
                    try:
                        matrix = functions[kernel](states)
                        score = global_cka_score(matrix, args.diag_width)
                        matrices[kernel][condition] = matrix
                        sample_rows[kernel].append({
                            **{name: record.get(name, "") for name in exp3.PROVENANCE},
                            "condition": condition, "image_id": identifier, "image_path": str(paths[condition]),
                            "image_sha256": fingerprints[paths[condition]], "attack_status": status,
                            "cache_path": str(cache_path.resolve()), "cache_validation": validation,
                            "n_encoder_layers": len(states) - 1, "n_patch_tokens": states[0].shape[1] - prefix_tokens,
                            "diag_width": args.diag_width, "n_layer_pairs": len(left), "S_global": score,
                        })
                    except Exception as error:
                        errors[kernel] = f"{type(error).__name__}: {error}"
                del states
        except NoLayerPairs as error:
            fatal = str(error)
            errors = {kernel: f"NoLayerPairs: {error}" for kernel in kernels}
        except Exception as error:
            errors = {kernel: f"{type(error).__name__}: {error}" for kernel in kernels}

        for kernel, state in states_by_kernel.items():
            if kernel in errors:
                state["skipped"].append({"sample_id": sample_id, "pair_id": record.get("pair_id", ""),
                                         "reason": errors[kernel]})
                print(f"  {kernel}: skipped ({errors[kernel]})", flush=True)
                continue
            if len(sample_rows[kernel]) != 3:
                raise RuntimeError("Internal error: incomplete triplet would enter aggregate results")
            signature = current_signature
            state["signature"] = signature
            if state["sums"] is None:
                state["sums"] = {condition: np.zeros_like(matrices[kernel][condition]) for condition in CONDITIONS}
            state["rows"].extend(sample_rows[kernel])
            for row in sample_rows[kernel]:
                condition = row["condition"]
                state["sums"][condition] += matrices[kernel][condition]
                if args.plots and args.per_image_plots:
                    filename = f"{kernel}_{sample_index:06d}_{exp3.cache_name(sample_id)[:-3]}_{condition}.png"
                    try:
                        figure, axis = plt.subplots(figsize=(8, 7))
                        plot_cka(axis, matrices[kernel][condition], args.diag_width,
                                 f"{MODEL_TITLE} {kernel.upper()} — {sample_id}/{condition}\n"
                                 f"S_global={row['S_global']:.6g}; k={args.diag_width}; pairs={row['n_layer_pairs']}")
                        figure.tight_layout()
                        state["outputs"].per_image(figure, filename)
                    except Exception as error:
                        plt.close("all")
                        state["outputs"].plot_errors.append({"plot": filename, "error": f"{type(error).__name__}: {error}"})
                        warnings.warn(f"Could not plot {filename}: {error}")
        if fatal is not None:
            break

    results, failures = {}, []
    columns = [*exp3.PROVENANCE, "condition", "image_id", "image_path", "image_sha256", "attack_status",
               "cache_path", "cache_validation", "n_encoder_layers", "n_patch_tokens", "diag_width", "n_layer_pairs", "S_global"]
    for kernel, state in states_by_kernel.items():
        outputs = state["outputs"]
        frame = pd.DataFrame(state["rows"], columns=columns)
        skipped = pd.DataFrame(state["skipped"], columns=["sample_id", "pair_id", "reason"])
        outputs.csv("image_scores", frame, index=False)
        outputs.csv("skipped_samples", skipped, index=False)
        metadata = {
            **common_metadata, "cka": kernel, "cka_estimator": f"biased centered {kernel} CKA",
            "cache_counts": representations.cache_counts.copy(),
            "analyzed_triplets": len(frame) // 3, "skipped_triplets": len(skipped),
            "hidden_state_count_and_shape": state["signature"],
        }
        if kernel == "rbf":
            metadata["rbf_bandwidth"] = "Per image/layer: median Euclidean patch distance over i<j; duplicates retained"
            metadata["rbf_kernel"] = "exp(-squared_distance / (2 * sigma**2)); Gram matrices double centered"
            metadata["rbf_degenerate_policy"] = "Reject whole triplet for zero/nonfinite bandwidth or centered norm"
        if frame.empty:
            reason = fatal or "No complete valid triplets; inspect skipped_samples."
            metadata.update(status="failed", error=reason, finished_utc=datetime.now(timezone.utc).isoformat(),
                            plot_errors=outputs.plot_errors)
            outputs.metadata(metadata)
            failures.append(f"{kernel}: {reason}")
            continue
        counts = frame.groupby("sample_id", sort=False).condition.agg(list)
        if frame.duplicated(["sample_id", "condition"]).any() or any(v != list(CONDITIONS) for v in counts):
            raise ValueError("Unmatched or duplicate sample conditions")
        outputs.csv("condition_summary", condition_summary(frame), index=False)
        paired, paired_summary = paired_statistics(frame, exp3.PROVENANCE)
        outputs.csv("paired_differences", paired, index=False)
        outputs.csv("paired_summary", paired_summary, index=False)
        auroc, curves = auroc_statistics(frame)
        outputs.csv("auroc_summary", auroc, index=False)
        outputs.csv("roc_curves", curves, index=False)
        left, right = encoder_layer_pairs(state["signature"][0], args.diag_width)
        outputs.csv("layer_pairs", pd.DataFrame({"layer_i": left, "layer_j": right, "distance": right - left}), index=False)
        metadata["n_layer_pairs"] = len(left)
        metadata["n_encoder_layers"] = state["signature"][0] - 1
        means = {condition: state["sums"][condition] / len(counts) for condition in CONDITIONS}
        labels = ["Emb.", *range(1, state["signature"][0])]
        for condition, matrix in {**means, "adv_minus_source": means["adversarial"] - means["source"]}.items():
            outputs.csv(f"mean_token_cka_{condition}", pd.DataFrame(matrix, index=labels, columns=labels), index_label="layer")
        if args.plots:
            aggregate_plots(frame, paired, auroc, curves, means, args, kernel, outputs)
        metadata.update(status="complete" if not outputs.plot_errors else "complete_with_plot_errors",
                        finished_utc=datetime.now(timezone.utc).isoformat(), plot_errors=outputs.plot_errors)
        outputs.metadata(metadata)
        results[kernel] = frame
        print(f"{kernel.upper()}: {len(counts)} matched triplets, {len(left)} layer pairs, {len(skipped)} skipped.")
        print(f"Scores/summaries/AUROC: {args.output_dir.resolve()}/{outputs.prefix}_*.csv", flush=True)
        if outputs.plot_errors:
            print(f"Plot errors: {len(outputs.plot_errors)}; see {outputs.prefix}_run.json.")
    if failures:
        raise RuntimeError("; ".join(failures))
    return results


def main():
    args = parse_args()
    with model_backend(args) as exp3:
        run_experiment(args, exp3)


if __name__ == "__main__":
    main()

