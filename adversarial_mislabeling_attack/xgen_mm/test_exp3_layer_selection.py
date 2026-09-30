#!/usr/bin/env python3
"""Standalone clean-only xGen-MM/LLaVA layer selection and Exp3 comparison.

Select contiguous depth blocks by normalized cut on the mean clean layer-to-layer CKA matrix.
CPU CKA calls the existing Exp3 functions unchanged. Optional CUDA CKA uses the
same float64 kernel definitions.
Only existing full-token caches are read; models are never loaded or caches written.
Run --help for paths, selection settings, device and plotting controls.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import functools
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/anantaraha/amp/dataset/laion_art")
CONDITIONS = ("source", "adversarial", "target")
KERNELS = ("linear", "rbf")
METRICS = ("A_MD", "A_DD", "S_gap")
MODEL_LABELS = {"xgen_mm": "xGen-MM", "llava": "LLaVA"}
CACHE_FOLDERS = {"xgen_mm": "xgen_mm_phi3_mini_instruct_r_v1", "llava": "llava_1_5_7b"}


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def minimum_layers(value):
    number = positive_integer(value)
    if number < 3:
        raise argparse.ArgumentTypeError("Require at least three encoder layers per segment.")
    return number


def compute_device(value):
    if value != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cpu, cuda or cuda:N.")
    return value


def parse_args(argv=None):
    # Resolve model defaults before formatting --help; explicit paths still take precedence.
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument("--vlm", choices=tuple(MODEL_LABELS), default="xgen_mm",
                              help="Model whose existing Exp3 code and full-token caches to use.")
    selected, _ = model_parser.parse_known_args(argv)
    model_dir = SCRIPT_DIR.parent / selected.vlm
    parser = argparse.ArgumentParser(
        parents=[model_parser],
        description="Select three contiguous depth blocks minimizing normalized cut of mean clean CKA, then compare with equal-thirds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Always cache-only; missing/invalid selection caches stop the run. CPU uses the existing Exp3 NumPy CKA functions. "
               "CUDA uses float64 CKA with the same definitions; floating-point roundoff can differ. "
               "Reruns replace this script's reports/plots, preserving representation caches and original Exp3 outputs. "
               "Paths default to the selected model. LLaVA legacy caches without image hashes retain legacy_id_only provenance.",
    )
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "attack_set/manifest.csv",
                        help="Source/target manifest; CSV image paths resolve against the AMP root.")
    parser.add_argument("--attack-results", type=Path, default=DATA_ROOT / "attack_set" / selected.vlm / "attack_results.csv",
                        help="Read only after clean boundaries have been selected.")
    parser.add_argument("--cache-dir", type=Path,
                        default=DATA_ROOT / "attack_set/representations" / CACHE_FOLDERS[selected.vlm],
                        help="Existing model-specific full-token caches, with clean/ and adv/ subdirectories.")
    parser.add_argument("--existing-exp3-dir", type=Path, default=DATA_ROOT / "output" / selected.vlm / "exp3",
                        help="Existing score CSVs for an optional equal-thirds reproducibility check.")
    parser.add_argument("--output-dir", type=Path, default=model_dir / "output/extra",
                        help="Save all result CSVs, JSON metadata, text reports/logs and PNGs here.")
    parser.add_argument("--selection-kernel", choices=KERNELS, default="linear",
                        help="Prespecified clean selection kernel; both kernels use the resulting frozen boundaries.")
    parser.add_argument("--min-encoder-layers", type=minimum_layers, default=3,
                        help="Minimum encoder-layer count in each of the three segments.")
    parser.add_argument("--max-samples", type=positive_integer, default=None,
                        help="First N manifest rows for both selection and evaluation; omitted means all.")
    parser.add_argument("--device", type=compute_device, default="cpu",
                        help="CKA compute device. Cached representations remain native precision on CPU.")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Skip PNGs while retaining every numerical result (plots enabled: %(default)s).")
    return parser.parse_args(argv)


class Tee:
    """Mirror stdout/stderr into a persistent run log."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


class Artifacts:
    def __init__(self, output_dir, arguments):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.report = self.directory / "report.txt"
        notes = EXPERIMENT_NOTES.replace("xGen-MM", MODEL_LABELS[arguments.vlm])
        if arguments.vlm == "llava":
            notes += ("\nLLaVA uses its existing Exp3 FP16 cache validation and removes one leading CLS token. "
                      "Legacy caches without image hashes retain legacy_id_only provenance; current image "
                      "hashes do not verify the original contents used to extract those caches.\n")
        self.report.write_text(notes + "\n\n", encoding="utf-8")
        self.files = []
        self.metadata = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()},
            "cka_precision": "float64", "cache_only": True,
            "cpu_backend": "Unmodified existing Exp3 NumPy functions",
            "cuda_backend": "Same float64 kernel definitions; numerical roundoff can differ from NumPy",
        }
        self.json("run", self.metadata)

    def text(self, title, text):
        with self.report.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{title}\n{'=' * len(title)}\n{text}\n")

    def table(self, name, frame, index=False):
        frame.to_csv(self.directory / f"{name}.csv", index=index)
        self.files.append(f"{name}.csv")
        self.text(name, frame.round(4).to_string(index=index, max_rows=None, max_cols=None))
        print(f"Saved {name}.csv ({len(frame)} rows)", flush=True)

    def json(self, name, value):
        text = json.dumps(value, indent=2) + "\n"
        (self.directory / f"{name}.json").write_text(text, encoding="utf-8")

    def figure(self, figure, name):
        import matplotlib.pyplot as plt
        try:
            figure.savefig(self.directory / f"{name}.png")
            self.files.append(f"{name}.png")
        finally:
            plt.close(figure)
        print(f"Saved {name}.png", flush=True)


def token_cka_torch(states, *, kernel, device, prefix_tokens):
    """Float64 Torch equivalent of the existing linear/RBF patch-token CKA.

    RBF uses the median of unique unordered off-diagonal Euclidean distances.
    The explicit midpoint for even counts matches np.median (torch.median alone
    would choose the lower middle value). No autocast, sampling or bandwidth floor.
    """
    import torch

    if kernel not in KERNELS:
        raise ValueError("Unknown CKA kernel.")
    normalized_grams, patch_count = [], None
    with torch.inference_mode():
        for layer, state in enumerate(states):
            features = state[0, prefix_tokens:, :].detach().to(device=device, dtype=torch.float64)
            if features.ndim != 2 or features.shape[0] < 2 or not torch.isfinite(features).all().item():
                raise ValueError(f"Invalid patch-token features at layer {layer}")
            if patch_count is not None and features.shape[0] != patch_count:
                raise ValueError("Layers must use the same spatial token observations")
            patch_count = features.shape[0]
            features = features - features.mean(dim=0, keepdim=True)
            gram = features @ features.T
            if kernel == "rbf":
                squared_norms = torch.diagonal(gram).clone()
                squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * gram
                squared_distances.clamp_min_(0)
                squared_distances.fill_diagonal_(0)
                pairs = torch.triu_indices(patch_count, patch_count, offset=1, device=features.device)
                distances = squared_distances[pairs[0], pairs[1]].sqrt().sort().values
                count = distances.numel()
                sigma = (distances[(count - 1) // 2] + distances[count // 2]) * 0.5
                if not torch.isfinite(sigma).item() or sigma.item() <= 0:
                    raise ValueError(f"Undefined RBF CKA at layer {layer}: zero/nonfinite median pairwise distance")
                gram = torch.exp(-squared_distances / (2 * sigma ** 2))
                row_means = gram.mean(dim=1, keepdim=True)
                column_means = gram.mean(dim=0, keepdim=True)
                grand_mean = gram.mean()
                gram = gram - row_means
                gram = gram - column_means
                gram = gram + grand_mean
            norm = torch.linalg.vector_norm(gram)
            if not torch.isfinite(norm).item() or norm.item() == 0:
                raise ValueError(f"Undefined {kernel} CKA at layer {layer}: zero/nonfinite centered norm")
            normalized_grams.append((gram / norm).reshape(-1))
        if not normalized_grams:
            raise ValueError("No layers to compare")
        grams = torch.stack(normalized_grams)
        matrix = grams @ grams.T
        if not torch.isfinite(matrix).all().item():
            raise ValueError("Nonfinite token CKA matrix")
        return matrix.cpu().numpy()


def cka_functions(exp3, native, device):
    if device == "cpu":
        return {"linear": exp3.token_cka_matrix, "rbf": exp3.token_rbf_cka_matrix}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or a GPU allocation.")
    selected_device = torch.device(device)
    print(f"CUDA CKA device: {selected_device} ({torch.cuda.get_device_name(selected_device)}), float64")
    return {kernel: functools.partial(token_cka_torch, kernel=kernel, device=selected_device,
                                      prefix_tokens=native.PREFIX_TOKENS)
            for kernel in KERNELS}


EXPERIMENT_NOTES = """xGen-MM clean-only layer selection and frozen-range Exp3 comparison

Distinct source/target image contents receive equal selection weight. Clean selection does not read
attack results or adversarial data. Full-token caches are validated and read only; no model
inference.

## Replaceable selection rule

Compute each clean image's full layer-to-layer patch-token CKA matrix using the prespecified
selection kernel, then take the arithmetic mean across distinct clean image contents. CLS tokens
are excluded by the model's CKA function. Save the full mean matrix, including embedding state 0
for reference, but exclude that state from selection and all encoder-depth groups.

For `L` encoder layers, search every contiguous split after `b1` and `b2` with at least the
configured minimum number of layers per block (default 3). The encoder groups are `1..b1`,
`b1+1..b2`, and `b2+1..L`; report `middle_start=b1+1` and `deep_start=b2+1`.

Minimize the contiguous three-way normalized cut on the mean clean encoder CKA matrix `W`:
`Ncut = cut(E,~E)/vol(E) + cut(M,~M)/vol(M) + cut(D,~D)/vol(D)`.
For each block A, `cut(A,~A) = sum(W[i,j] for i in A, j outside A)` and
`vol(A) = sum(W[i,j] for i in A, j in all encoder layers)`. Zero the diagonal on a selection copy:
self-similarity contributes to neither cut nor volume. Saved full CKA matrices and downstream
Exp3 evaluation retain their original values. Each connection between different
blocks contributes to both endpoint blocks' cuts, as specified by the three-term objective.
The matrix must be finite, symmetric and nonnegative; a zero/nonfinite block volume is undefined
and stops selection. No weight clipping, bandwidth change or volume floor is applied.
Exact Ncut ties choose the smallest `b1`, then `b2`.
The modular selection function accepts only the clean mean encoder matrix and minimum block size;
no adversarial data or evaluation scores can influence it.

## Frozen-range Exp3 evaluation

Only now read attack results and adversarial caches. Reuse the existing Exp3 linear/RBF CKA
functions and `gap_scores`, including CLS removal, RBF bandwidths, unique off-diagonal deep-layer
pairs, and `S_gap = A_DD - A_MD`. Both kernels use the **same frozen clean-selected ranges** and the
same equal-thirds reference.

Every admitted sample contributes source/adversarial/target together. Cache/provenance failures or
undefined kernels are reported. Comparisons below use the common complete sample set across both
kernels and both range definitions. Clean geometry is reused only for byte-identical images verified
by SHA-256.

## Exploratory AUROC comparison

Standard empirical AUROC uses **adversarial = positive** and **larger `S_gap` = more positive**,
fixed in advance. Do not reverse the score direction, maximize AUROC over partitions, or choose a
detection threshold.

Report adversarial versus source, target, and pooled source+target. Like the Exp3 summaries, these
metrics weight manifest image occurrences; repeated clean images are correlated. This is an
in-sample exploratory comparison, not held-out detection performance: the clean selection images
overlap the clean evaluation images.

The selected starts and candidate Ncut values describe a clean-representation depth partition. The paired,
distribution and AUROC tables describe what happens **after freezing it**; they do not feed back
into selection. Keep the signed AUROC difference, including negative differences, and the
prespecified score direction.

A larger `S_gap` means deep layers align more strongly with one another than with middle layers.
Improvements on this sample set would motivate independent validation, not establish a detector. To
try a different selection rule later, replace `select_three_blocks` while retaining the clean-only
inputs, explicit layer indexing and fixed downstream evaluation.
"""

def load_xgen_modules(model_dir):
    """Import the existing xGen-MM code without reusing another VLM's _vision module."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    old_bytecode = sys.dont_write_bytecode
    previous_vision = sys.modules.get("_vision")
    sys.dont_write_bytecode = True
    try:
        native = load("_xgen_selection_vision", model_dir / "_vision.py")
        sys.modules["_vision"] = native
        experiment = load("_xgen_selection_exp3", model_dir / "exp3.py")
    finally:
        sys.dont_write_bytecode = old_bytecode
        if previous_vision is None:
            sys.modules.pop("_vision", None)
        else:
            sys.modules["_vision"] = previous_vision
    return native, experiment


class LlavaRepresentations:
    """Expose LLaVA Exp3's cache-only loader through the shared payload interface."""

    def __init__(self, experiment, *, device, cache_dir, cache_only=True):
        if not cache_only:
            raise ValueError("Layer selection only supports existing full-token caches.")
        args = argparse.Namespace(device=device, cache_dir=cache_dir, cache_only=True,
                                  force_recompute_representations=False)
        self.backend = experiment.Representations(args)
        self.cache_counts = self.backend.cache_counts
        self.validations = {}
        self.signature = None

    def get_payload(self, image_path, cache_path, image_digest):
        states, validation = self.backend.get_hidden_states(image_path, cache_path, image_digest)
        # LLaVA Exp3 enforces a common depth/token/width signature across images.
        signature = (len(states), tuple(states[0].shape))
        if self.signature is not None and signature != self.signature:
            raise ValueError("Layer/token layout differs from previously analyzed images")
        self.signature = signature
        self.validations[cache_path.resolve()] = validation
        return {"hidden_states": states}


def load_model_modules(vlm):
    """Reuse the selected model's own cache, provenance and CKA implementations."""
    model_dir = SCRIPT_DIR.parent / vlm
    if vlm == "xgen_mm":
        return load_xgen_modules(model_dir)
    if vlm != "llava":
        raise ValueError(f"Unsupported VLM: {vlm}")
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_llava_selection_exp3", model_dir / "exp3.py")
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
    finally:
        sys.dont_write_bytecode = old_bytecode
    native = SimpleNamespace(**{
        name: getattr(experiment, name)
        for name in ("MODEL_ID", "PROVENANCE", "SUCCESS_STATUSES", "cache_name", "image_path",
                     "sha256_file", "read_samples", "validate_sample")
    })
    # LLaVA exp3.validate_states requires FP16; both of its CKA functions use state[0, 1:, :].
    # That implementation does not pin/record a model revision.
    native.PREFIX_TOKENS, native.PRECISION, native.MODEL_REVISION = 1, "float16", None
    native.Representations = functools.partial(LlavaRepresentations, experiment)
    return native, experiment


def cache_validation(cache_loader, cache_path):
    """Preserve LLaVA legacy provenance; xGen-MM's loader always verifies image hashes."""
    if isinstance(cache_loader, LlavaRepresentations):
        return cache_loader.validations[cache_path.resolve()]
    return "sha256_verified"


def select_three_blocks(mean_encoder_cka, min_layers=3):
    """Minimize contiguous three-way normalized cut over all valid encoder splits.

    Input rows/columns correspond to encoder layers 1..L (embedding excluded).
    Cut sums rows in a block and columns outside it. Volume sums those rows over
    every encoder column after zeroing the diagonal on a copy. No volume floor.
    """
    import numpy as np
    import pandas as pd

    matrix = np.array(mean_encoder_cka, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square mean encoder CKA matrix, excluding embedding state 0.")
    n_encoder_layers = matrix.shape[0]
    if not isinstance(min_layers, (int, np.integer)) or min_layers < 3:
        raise ValueError("Require at least three encoder layers per segment.")
    if n_encoder_layers < 3 * min_layers:
        raise ValueError("The encoder is too shallow for three valid segments.")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The mean encoder CKA matrix must be finite and symmetric.")
    if (matrix < 0).any():
        raise ValueError("Normalized-cut similarities must be nonnegative.")
    # Exclude self-similarity only from selection; never mutate the shared CKA matrix.
    np.fill_diagonal(matrix, 0.0)

    candidates = []
    for b1 in range(min_layers, n_encoder_layers - 2 * min_layers + 1):
        for b2 in range(b1 + min_layers, n_encoder_layers - min_layers + 1):
            terms = {}
            for name, start, stop in (("early", 0, b1), ("middle", b1, b2),
                                      ("deep", b2, n_encoder_layers)):
                rows = matrix[start:stop, :]
                volume = float(rows.sum())
                if not np.isfinite(volume) or volume <= 0:
                    raise ValueError(f"Undefined normalized cut: {name} block at ({b1}, {b2}) "
                                     "has zero/nonfinite volume.")
                cut = float(rows[:, :start].sum() + rows[:, stop:].sum())
                terms.update({f"{name}_cut": cut, f"{name}_volume": volume,
                              f"{name}_ncut": cut / volume})
            candidates.append({
                "b1": b1, "b2": b2,
                "middle_start": b1 + 1, "deep_start": b2 + 1,
                "n_early_layers": b1, "n_middle_layers": b2 - b1,
                "n_deep_layers": n_encoder_layers - b2,
                **terms,
                "Ncut": sum(terms[f"{name}_ncut"] for name in ("early", "middle", "deep")),
            })
    candidates = pd.DataFrame(candidates).sort_values(
        ["Ncut", "b1", "b2"], kind="stable"
    ).reset_index(drop=True)
    best = candidates.iloc[0].to_dict()
    for name in ["b1", "b2", "middle_start", "deep_start",
                 "n_early_layers", "n_middle_layers", "n_deep_layers"]:
        best[name] = int(best[name])
    return best, candidates


def encoder_ranges(n_encoder_layers, b1, b2):
    if not 1 <= b1 < b2 < n_encoder_layers:
        raise ValueError("Invalid encoder boundaries.")
    return {
        "early": tuple(range(1, b1 + 1)),
        "middle": tuple(range(b1 + 1, b2 + 1)),
        "deep": tuple(range(b2 + 1, n_encoder_layers + 1)),
    }


def build_clean_inventory(manifest, native, cache_dir):
    """Collect only source/target images, checking cache keys, IDs and content identity."""
    import pandas as pd

    required = set(native.PROVENANCE) | {"source_path", "target_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest fields: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in the manifest.")
    if manifest[list(required)].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Blank manifest identifiers, concepts or paths.")

    rows, fingerprints, cache_owners, id_digests = [], {}, {}, {}
    for record in manifest.to_dict(orient="records"):
        for condition in ("source", "target"):
            image_id = record[f"{condition}_image_id"]
            filename = native.cache_name(image_id)
            if filename in cache_owners and cache_owners[filename] != image_id:
                raise ValueError(f"Clean cache key collision: {filename}")
            cache_owners[filename] = image_id
            path = native.image_path(record[f"{condition}_path"])
            if path not in fingerprints:
                fingerprints[path] = native.sha256_file(path)
            digest = fingerprints[path]
            if image_id in id_digests and id_digests[image_id] != digest:
                raise ValueError(f"Clean image ID refers to different contents: {image_id}")
            id_digests[image_id] = digest
            rows.append({
                "sample_id": record["sample_id"], "pair_id": record["pair_id"],
                "condition": condition, "image_id": image_id, "image_path": str(path),
                "image_sha256": digest, "cache_path": str(cache_dir / "clean" / filename),
            })
    occurrences = pd.DataFrame(rows)
    if occurrences.empty:
        raise ValueError("No clean images in the selected manifest rows.")
    # Exact duplicate image contents receive one vote, regardless of ID or source/target role.
    unique = occurrences.drop_duplicates("image_sha256", keep="first").reset_index(drop=True)
    return occurrences, unique


def compute_clean_geometry(unique_images, cache_loader, selection_kernel, cka_functions):
    """Compute clean matrices once and reuse them later; return no adversarial information."""
    import numpy as np
    import pandas as pd

    matrices, other_kernel_errors = {}, []
    selection_sum = None
    n_encoder_layers = None
    for index, record in enumerate(unique_images.to_dict(orient="records"), 1):
        print(f"Clean image {index}/{len(unique_images)}: {record['image_id']}", flush=True)
        try:
            payload = cache_loader.get_payload(
                Path(record["image_path"]), Path(record["cache_path"]), record["image_sha256"]
            )
            states = payload["hidden_states"]
            depth = len(states) - 1
            if n_encoder_layers is not None and depth != n_encoder_layers:
                raise ValueError("Clean caches have different encoder depths.")
            n_encoder_layers = depth
            # Compute the prespecified selection kernel first. Missing/undefined selection
            # data stops the run rather than silently changing the clean cohort.
            selected_matrix = cka_functions[selection_kernel](states)
        except Exception as error:
            raise RuntimeError(
                f"Cannot form the clean selection matrix for {record['image_id']}: {error}. "
                "No cache will be generated or replaced."
            ) from error
        image_matrices = {selection_kernel: selected_matrix}
        for kernel in KERNELS:
            if kernel == selection_kernel:
                continue
            try:
                image_matrices[kernel] = cka_functions[kernel](states)
            except Exception as error:
                image_matrices[kernel] = None
                other_kernel_errors.append({
                    "image_id": record["image_id"], "image_sha256": record["image_sha256"],
                    "cka": kernel, "reason": f"{type(error).__name__}: {error}",
                })
        matrices[record["image_sha256"]] = image_matrices
        # Average per-image full matrices; never pool token observations across images.
        if selection_sum is None:
            selection_sum = np.array(selected_matrix, dtype=np.float64, copy=True)
        else:
            selection_sum += selected_matrix
        del payload, states
    if selection_sum is None:
        raise ValueError("No clean images available for selection.")
    mean_clean_cka = selection_sum / len(unique_images)
    return n_encoder_layers, mean_clean_cka, matrices, pd.DataFrame(other_kernel_errors)


def evaluate_frozen_ranges(selected_manifest, occurrences, unique_images, matrices, cache_loader, groupings,
                           *, native, gap_scores, manifest_path, attack_results_path, cache_dir,
                           manifest_digest, n_encoder_layers, cka_functions):
    import pandas as pd

    if native.sha256_file(manifest_path) != manifest_digest:
        raise ValueError("Manifest changed after clean-only selection; restart the run.")
    _, samples = native.read_samples(manifest_path, attack_results_path)
    samples = samples.loc[samples["sample_id"].isin(selected_manifest["sample_id"])]
    clean_lookup = occurrences.set_index(["sample_id", "condition"])
    canonical = unique_images.set_index("image_sha256")
    fingerprints, rows, errors = {}, [], []
    for index, record in enumerate(samples.to_dict(orient="records"), 1):
        sample_id = record["sample_id"]
        print(f"Evaluation {index}/{len(samples)}: {sample_id}", flush=True)
        status = str(record.get("_attack_status", "")).strip().lower()
        if status not in native.SUCCESS_STATUSES:
            errors.append({"sample_id": sample_id, "cka": "both", "reason": f"attack status={status!r}"})
            continue
        try:
            paths = native.validate_sample(record, fingerprints)
            clean_records = {condition: clean_lookup.loc[(sample_id, condition)]
                             for condition in ("source", "target")}
            for condition, clean_record in clean_records.items():
                if fingerprints[paths[condition]] != clean_record["image_sha256"]:
                    raise ValueError(f"{condition} image changed after clean selection")
            adv_cache = cache_dir / "adv" / native.cache_name(sample_id)
            payload = cache_loader.get_payload(paths["adversarial"], adv_cache,
                                               fingerprints[paths["adversarial"]])
            states = payload["hidden_states"]
            if len(states) - 1 != n_encoder_layers:
                raise ValueError("Adversarial encoder depth differs from the frozen clean depth.")
        except Exception as error:
            errors.append({"sample_id": sample_id, "cka": "both",
                           "reason": f"{type(error).__name__}: {error}"})
            continue

        for kernel in KERNELS:
            try:
                sample_matrices = {
                    condition: matrices[clean_record["image_sha256"]][kernel]
                    for condition, clean_record in clean_records.items()
                }
                if any(matrix is None for matrix in sample_matrices.values()):
                    raise ValueError("Undefined clean CKA for this kernel; see clean_kernel_errors.")
                sample_matrices["adversarial"] = cka_functions[kernel](states)
                sample_rows = []
                for grouping_name, groups in groupings.items():
                    for condition in CONDITIONS:
                        digest = fingerprints[paths[condition]]
                        used_cache = (adv_cache if condition == "adversarial"
                                      else Path(canonical.loc[digest, "cache_path"]))
                        sample_rows.append({
                            **{column: record[column] for column in native.PROVENANCE},
                            "condition": condition,
                            "image_id": sample_id if condition == "adversarial" else record[f"{condition}_image_id"],
                            "image_path": str(paths[condition]), "image_sha256": digest,
                            "cache_path": str(used_cache), "cache_validation": cache_validation(cache_loader, used_cache),
                            "cka": kernel, "layer_ranges": grouping_name,
                            "middle_start": groups["middle"][0], "deep_start": groups["deep"][0],
                            "n_layers": len(states), "n_patch_tokens": states[0].shape[1] - native.PREFIX_TOKENS,
                            **gap_scores(sample_matrices[condition], groups["middle"], groups["deep"]),
                        })
                rows.extend(sample_rows)
            except Exception as error:
                errors.append({"sample_id": sample_id, "cka": kernel,
                               "reason": f"{type(error).__name__}: {error}"})
        del payload, states
    return pd.DataFrame(rows), pd.DataFrame(errors, columns=["sample_id", "cka", "reason"])


def empirical_auroc(negative_scores, positive_scores):
    """P(positive > negative) + 0.5 P(tie), computed using average ranks."""
    import numpy as np
    import pandas as pd

    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(positive_scores, dtype=np.float64)
    if negative.ndim != 1 or positive.ndim != 1 or not len(negative) or not len(positive):
        raise ValueError("AUROC requires nonempty one-dimensional positive and negative scores.")
    combined = np.concatenate([negative, positive])
    if not np.isfinite(combined).all():
        raise ValueError("AUROC scores must be finite.")
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    n_positive, n_negative = len(positive), len(negative)
    return float((ranks[n_negative:].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def run_experiment(args, writer):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MANIFEST_PATH, ATTACK_RESULTS_PATH = args.manifest, args.attack_results
    CACHE_DIR, EXISTING_EXP3_DIR = args.cache_dir, args.existing_exp3_dir
    SELECTION_KERNEL, MIN_ENCODER_LAYERS, MAX_SAMPLES = args.selection_kernel, args.min_encoder_layers, args.max_samples
    model_dir = SCRIPT_DIR.parent / args.vlm
    model_label = MODEL_LABELS[args.vlm]
    native, exp3 = load_model_modules(args.vlm)
    CKA_FUNCTIONS = cka_functions(exp3, native, args.device)
    loader = native.Representations(device="cpu", cache_dir=CACHE_DIR, cache_only=True)
    writer.metadata.update(model_id=native.MODEL_ID, model_revision=native.MODEL_REVISION,
                           representation_precision=native.PRECISION,
                           exp3_source_sha256=native.sha256_file(model_dir / "exp3.py"))
    if args.vlm == "xgen_mm":
        writer.metadata["vision_source_sha256"] = native.sha256_file(model_dir / "_vision.py")
    else:
        writer.metadata["exp1_source_sha256"] = native.sha256_file(model_dir / "exp1.py")
        writer.metadata["cache_provenance"] = (
            "LLaVA Exp3 validates CPU FP16 full-token caches. Caches without image hashes are "
            "legacy_id_only; their original extraction image content cannot be verified."
        )
    writer.json("run", writer.metadata)
    print(f"Model: {native.MODEL_ID}")
    print(f"Cache: {CACHE_DIR}")
    print(f"CKA device: {args.device}; selection kernel: {SELECTION_KERNEL}; evaluation kernels: linear, rbf")
    print(f"Outputs: {writer.directory.resolve()}", flush=True)

    manifest_digest = native.sha256_file(MANIFEST_PATH)
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    if MAX_SAMPLES is not None:
        manifest = manifest.head(MAX_SAMPLES).copy()

    clean_occurrences, unique_clean_images = build_clean_inventory(manifest, native, CACHE_DIR)
    print(f"Manifest samples: {len(manifest)}")
    print(f"Clean occurrences: {len(clean_occurrences)}; distinct clean image contents: {len(unique_clean_images)}")
    writer.table("clean_occurrences", clean_occurrences)
    writer.table("unique_clean_images", unique_clean_images)
    n_encoder_layers, mean_clean_cka, clean_matrices, clean_kernel_errors = compute_clean_geometry(
        unique_clean_images, loader, SELECTION_KERNEL, CKA_FUNCTIONS
    )
    labels = pd.Index(["Emb.", *range(1, n_encoder_layers + 1)], name="layer")
    writer.table("clean_mean_cka", pd.DataFrame(mean_clean_cka, index=labels, columns=labels), index=True)
    writer.table("clean_kernel_errors", clean_kernel_errors.reindex(columns=["image_id", "image_sha256", "cka", "reason"]))
    if not clean_kernel_errors.empty:
        print("The non-selection kernel has undefined clean matrices; affected evaluation triplets will be reported.")
        print(f"Non-selection clean-kernel errors: {len(clean_kernel_errors)}")

    selection, selection_candidates = select_three_blocks(
        mean_clean_cka[1:, 1:], MIN_ENCODER_LAYERS
    )
    selected_boundaries = (selection["b1"], selection["b2"])
    selected_ranges = encoder_ranges(n_encoder_layers, *selected_boundaries)

    # Exact equal-thirds convention from the selected model's existing Exp3 implementation.
    equal_blocks, equal_middle, equal_deep = exp3.layer_blocks(n_encoder_layers + 1)
    equal_ranges = {
        name: tuple(layer for layer, block in equal_blocks.items() if block == name)
        for name in ("early", "middle", "deep")
    }
    GROUPINGS = {"equal_thirds": equal_ranges, "clean_selected": selected_ranges}
    selection_record = {
        "selection_kernel": SELECTION_KERNEL,
        "manifest_sha256": manifest_digest,
        "n_unique_clean_images": len(unique_clean_images),
        "encoder_layers": n_encoder_layers,
        "minimum_encoder_layers_per_segment": MIN_ENCODER_LAYERS,
        "b1": selection["b1"], "b2": selection["b2"],
        "middle_start": selection["middle_start"], "deep_start": selection["deep_start"],
        "selection_method": "minimize_contiguous_three_way_normalized_cut",
        "Ncut": selection["Ncut"],
        "objective": "cut(E,~E)/vol(E) + cut(M,~M)/vol(M) + cut(D,~D)/vol(D)",
        "volume_definition": "Sum of CKA from block layers to all encoder layers, excluding the zeroed diagonal",
        "volume_includes_diagonal": False,
        "embedding_state_in_selection": False,
        "valid_candidates": len(selection_candidates),
        "tie_break": "smallest b1, then smallest b2",
    }
    print(json.dumps(selection_record, indent=2))
    range_table = pd.DataFrame([
        {"layer_ranges": name, "early": f"{groups['early'][0]}..{groups['early'][-1]}",
         "middle": f"{groups['middle'][0]}..{groups['middle'][-1]}",
         "deep": f"{groups['deep'][0]}..{groups['deep'][-1]}",
         "middle_start": groups["middle"][0], "deep_start": groups["deep"][0]}
        for name, groups in GROUPINGS.items()
    ])
    writer.table("layer_ranges", range_table)
    writer.table("selection_candidates", selection_candidates)
    writer.json("selection", selection_record)
    writer.text("Frozen clean-only selection", json.dumps(selection_record, indent=2))
    writer.metadata["selection"] = selection_record
    writer.metadata["layer_groups"] = GROUPINGS
    writer.json("run", writer.metadata)

    if args.plots:
        fig, ax = plt.subplots(figsize=(9, 8))
        heatmap = ax.imshow(mean_clean_cka[1:, 1:], origin="upper", cmap="viridis", vmin=0, vmax=1,
                            extent=(0.5, n_encoder_layers + 0.5, n_encoder_layers + 0.5, 0.5))
        fig.colorbar(heatmap, ax=ax, label=f"Mean clean {SELECTION_KERNEL.upper()} patch-token CKA")
        b1, b2 = selected_boundaries
        for boundary in (b1, b2):
            ax.axvline(boundary + 0.5, color="white", linestyle="--", linewidth=1)
            ax.axhline(boundary + 0.5, color="white", linestyle="--", linewidth=1)
        ticks = np.unique(np.r_[np.linspace(1, n_encoder_layers, min(n_encoder_layers, 12), dtype=int),
                                selection["middle_start"], selection["deep_start"]])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Encoder layer (embedding excluded)")
        ax.set_ylabel("Encoder layer (embedding excluded)")
        ax.set_title(f"{model_label}: mean clean {SELECTION_KERNEL.upper()} CKA ({len(unique_clean_images)} images)\n"
                     f"middle_start={selection['middle_start']}, deep_start={selection['deep_start']}; "
                     f"Ncut={selection['Ncut']:.6g}")
        plt.tight_layout()
        writer.figure(fig, "clean_mean_cka")

    image_scores, skipped_evaluations = evaluate_frozen_ranges(
        manifest, clean_occurrences, unique_clean_images, clean_matrices, loader, GROUPINGS,
        native=native, gap_scores=exp3.gap_scores, manifest_path=MANIFEST_PATH,
        attack_results_path=ATTACK_RESULTS_PATH, cache_dir=CACHE_DIR, manifest_digest=manifest_digest,
        n_encoder_layers=n_encoder_layers, cka_functions=CKA_FUNCTIONS
    )
    writer.table("image_scores", image_scores)
    writer.table("skipped_evaluations", skipped_evaluations)
    writer.metadata["cache_counts"] = loader.cache_counts.copy()
    writer.metadata["attack_results_sha256"] = native.sha256_file(ATTACK_RESULTS_PATH)
    writer.json("run", writer.metadata)
    if image_scores.empty:
        raise RuntimeError("No valid frozen-range evaluation triplets; inspect skipped_evaluations.")
    if image_scores.duplicated(["cka", "layer_ranges", "sample_id", "condition"]).any():
        raise ValueError("Duplicate evaluation rows.")
    print(f"Cache operations: {loader.cache_counts}; extraction must remain zero.")
    assert loader.cache_counts["extracted"] == 0
    writer.table("evaluation_counts", image_scores.groupby(["cka", "layer_ranges", "condition"]).size().rename("n_rows").to_frame(), index=True)

    # Optional reproducibility check. Existing score CSVs never participate in layer selection.
    existing_checks = []
    for kernel in KERNELS:
        filename = "exp3_image_scores.csv" if kernel == "linear" else "exp3_rbf_image_scores.csv"
        path = EXISTING_EXP3_DIR / filename
        if not path.is_file():
            existing_checks.append({"cka": kernel, "existing_csv": str(path), "status": "not present"})
            continue
        existing = pd.read_csv(path, dtype={column: str for column in native.PROVENANCE})
        recomputed = image_scores.loc[
            (image_scores["cka"] == kernel) & (image_scores["layer_ranges"] == "equal_thirds")
        ]
        matched = recomputed.merge(existing, on=["sample_id", "condition"],
                                   suffixes=("_new", "_existing"), validate="one_to_one")
        for column in native.PROVENANCE[1:]:
            if not matched[f"{column}_new"].fillna("").equals(matched[f"{column}_existing"].fillna("")):
                raise ValueError(f"Existing {kernel} results have mismatched {column}.")
        record = {"cka": kernel, "existing_csv": str(path), "n_matching_image_rows": len(matched)}
        for metric in METRICS:
            differences = (matched[f"{metric}_new"] - matched[f"{metric}_existing"]).abs()
            record[f"max_abs_{metric}_difference"] = differences.max()
        record["status"] = ("no common rows" if matched.empty else
                            "matches" if all(np.allclose(matched[f"{metric}_new"], matched[f"{metric}_existing"],
                                                         rtol=1e-10, atol=1e-12) for metric in METRICS)
                            else "differs: check run/cache provenance")
        existing_checks.append(record)
    writer.table("existing_output_checks", pd.DataFrame(existing_checks))

    variants = [(kernel, grouping) for kernel in KERNELS for grouping in GROUPINGS]
    complete_ids = {}
    for kernel, grouping in variants:
        frame = image_scores.loc[(image_scores["cka"] == kernel) & (image_scores["layer_ranges"] == grouping)]
        complete = frame.groupby("sample_id")["condition"].agg(
            lambda values: set(values) == set(CONDITIONS) and len(values) == len(CONDITIONS)
        )
        complete_ids[(kernel, grouping)] = set(complete.index[complete])
    common_ids = set.intersection(*complete_ids.values())
    if not common_ids:
        raise RuntimeError("No common complete triplets across both kernels and both range definitions.")
    comparison_scores = image_scores.loc[image_scores["sample_id"].isin(common_ids)].copy()
    if not np.isfinite(comparison_scores[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite comparison scores.")
    print(f"Matched comparison: {len(common_ids)} samples out of {len(manifest)} selected manifest samples.")
    writer.table("comparison_cohort", pd.DataFrame([
        {"cka": kernel, "layer_ranges": grouping, "available_complete_samples": len(ids),
         "comparison_samples": len(common_ids), "excluded_from_comparison": len(ids - common_ids)}
        for (kernel, grouping), ids in complete_ids.items()
    ]))

    writer.table("comparison_scores", comparison_scores)
    writer.metadata["comparison_samples"] = len(common_ids)

    condition_summary = (
        comparison_scores.groupby(["cka", "layer_ranges", "condition"])[list(METRICS)]
        .agg(["mean", "median", "std"])
        .reindex(pd.MultiIndex.from_product(
            [KERNELS, tuple(GROUPINGS), CONDITIONS], names=["cka", "layer_ranges", "condition"]
        ))
    )
    condition_summary.columns = [f"{metric}_{stat}" for metric, stat in condition_summary.columns]
    condition_counts = comparison_scores.groupby(["cka", "layer_ranges", "condition"]).agg(
        n_samples=("sample_id", "nunique"), n_unique_image_contents=("image_sha256", "nunique")
    )
    writer.table("condition_summary", condition_summary.join(condition_counts), index=True)

    paired_scores, paired_rows = {}, []
    for kernel, grouping in variants:
        frame = comparison_scores.loc[
            (comparison_scores["cka"] == kernel) & (comparison_scores["layer_ranges"] == grouping)
        ]
        pairs = frame.pivot(index="sample_id", columns="condition", values="S_gap")
        paired_scores[(kernel, grouping)] = pairs
        delta = pairs["adversarial"] - pairs["source"]
        above = int((delta > 0).sum())
        paired_rows.append({
            "cka": kernel, "layer_ranges": grouping, "n_samples": len(pairs),
            "n_concept_pairs": frame["pair_id"].nunique(),
            "adv_above_source": above, "fraction_adv_above_source": above / len(pairs),
            "mean_adv_minus_source": delta.mean(), "median_adv_minus_source": delta.median(),
            "std_adv_minus_source": delta.std(ddof=1),
        })
    paired_summary = pd.DataFrame(paired_rows).set_index(["cka", "layer_ranges"])
    writer.table("paired_summary", paired_summary, index=True)
    paired_export = pd.concat([
        values.assign(cka=kernel, layer_ranges=grouping,
                      adv_minus_source=values["adversarial"] - values["source"]).reset_index()
        for (kernel, grouping), values in paired_scores.items()
    ], ignore_index=True)
    writer.table("paired_sample_scores", paired_export)

    if args.plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for row_index, kernel in enumerate(KERNELS):
            values = comparison_scores.loc[comparison_scores["cka"] == kernel, "S_gap"]
            low, high = values.min(), values.max()
            if low == high:
                low, high = low - 1e-6, high + 1e-6
            for column_index, grouping in enumerate(GROUPINGS):
                ax = axes[row_index, column_index]
                pairs = paired_scores[(kernel, grouping)]
                ax.scatter(pairs["source"], pairs["adversarial"], s=45, alpha=0.8)
                ax.plot([low, high], [low, high], "--", linewidth=1.5)
                ax.set_xlim(low, high); ax.set_ylim(low, high)
                ax.set_xlabel(r"Source $S_{gap}$"); ax.set_ylabel(r"Adversarial $S_{gap}$")
                ax.set_title(f"{kernel.upper()} — {grouping}")
                above = int((pairs["adversarial"] > pairs["source"]).sum())
                ax.text(0.05, 0.95, f"{above}/{len(pairs)} adversarial samples above source",
                        transform=ax.transAxes, va="top")
        plt.tight_layout()
        writer.figure(fig, "paired_scores")

    if args.plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        for row_index, kernel in enumerate(KERNELS):
            values = comparison_scores.loc[comparison_scores["cka"] == kernel, "S_gap"]
            low, high = values.min(), values.max()
            if low == high:
                low, high = low - 1e-6, high + 1e-6
            bins = np.linspace(low, high, 20)  # Shared across range definitions for this kernel.
            for column_index, grouping in enumerate(GROUPINGS):
                ax = axes[row_index, column_index]
                frame = comparison_scores.loc[
                    (comparison_scores["cka"] == kernel) & (comparison_scores["layer_ranges"] == grouping)
                ]
                for condition in CONDITIONS:
                    scores = frame.loc[frame["condition"] == condition, "S_gap"]
                    ax.hist(scores, bins=bins, alpha=0.5, label=f"{condition.capitalize()} (n={len(scores)})")
                ax.set_xlabel(r"$S_{gap} = A_{DD} - A_{MD}$")
                ax.set_ylabel("Number of image occurrences")
                ax.set_title(f"{kernel.upper()} — {grouping}")
                ax.legend()
        plt.tight_layout()
        writer.figure(fig, "score_distributions")

    auroc_rows = []
    negative_sets = {"source": ("source",), "target": ("target",), "source+target": ("source", "target")}
    for kernel, grouping in variants:
        frame = comparison_scores.loc[
            (comparison_scores["cka"] == kernel) & (comparison_scores["layer_ranges"] == grouping)
        ]
        positive = frame.loc[frame["condition"] == "adversarial"]
        for name, conditions in negative_sets.items():
            negative = frame.loc[frame["condition"].isin(conditions)]
            auroc_rows.append({
                "cka": kernel, "layer_ranges": grouping, "negative_condition": name,
                "n_positive_occurrences": len(positive), "n_negative_occurrences": len(negative),
                "n_unique_negative_images": negative["image_sha256"].nunique(),
                "AUROC": empirical_auroc(negative["S_gap"], positive["S_gap"]),
            })
    auroc_summary = pd.DataFrame(auroc_rows)
    writer.table("auroc_summary", auroc_summary)

    auroc_comparison = auroc_summary.pivot(
        index=["cka", "negative_condition"], columns="layer_ranges", values="AUROC"
    )
    auroc_comparison["selected_minus_thirds"] = (
        auroc_comparison["clean_selected"] - auroc_comparison["equal_thirds"]
    )
    writer.table("auroc_comparison", auroc_comparison, index=True)

    if args.plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, kernel in zip(axes, KERNELS):
            table = auroc_comparison.loc[kernel].reindex(list(negative_sets))
            x = np.arange(len(table))
            for index, grouping in enumerate(GROUPINGS):
                ax.bar(x + (index - 0.5) * 0.35, table[grouping], width=0.35, label=grouping)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
            ax.set_xticks(x, labels=table.index)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Negative condition (adversarial is positive)")
            ax.set_ylabel(r"Empirical AUROC using $S_{gap}$")
            ax.set_title(f"{model_label} {kernel.upper()}")
            ax.legend()
        plt.tight_layout()
        writer.figure(fig, "auroc_comparison")

    writer.metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                           output_files=writer.files.copy())
    writer.json("run", writer.metadata)
    writer.text("Run metadata", json.dumps(writer.metadata, indent=2))
    print(f"Selected middle_start={selection['middle_start']}, deep_start={selection['deep_start']}, "
          f"Ncut={selection['Ncut']:.6g}")
    print(f"Completed: {len(common_ids)} matched evaluation samples; results and report: {writer.directory.resolve()}")
    return {"selection": selection_record, "image_scores": image_scores,
            "comparison_scores": comparison_scores, "paired_summary": paired_summary,
            "auroc_summary": auroc_summary}

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            writer = Artifacts(args.output_dir, args)
            try:
                run_experiment(args, writer)
            except Exception as error:
                writer.metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                                       finished_utc=datetime.now(timezone.utc).isoformat())
                writer.json("run", writer.metadata)
                writer.text("Failure", traceback.format_exc())
                traceback.print_exc()
                raise SystemExit(1) from error


if __name__ == "__main__":
    main()
