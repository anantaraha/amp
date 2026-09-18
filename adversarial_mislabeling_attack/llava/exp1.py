#!/usr/bin/env python3
"""Standalone conversion of exp1.ipynb; scientific computations and cache format preserved.

Run python exp1.py --help for runtime options. All plots are saved headlessly.
Relative image paths in the input CSVs resolve against the repository root.
"""

import argparse
from pathlib import Path
import re
import warnings

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CACHE_VERSION = 1
SUCCESS_STATUSES = {"completed", "skipped", "success", "successful"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ATTACK_DIR = REPO_ROOT / "dataset/laion_art/attack_set"
DEFAULT_CACHE_DIR = ATTACK_DIR / "representations/llava_1_5_7b"


def cuda_device(value):
    """Retain the notebooks' CUDA/FP16 extraction requirement."""
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N (for example cuda:1).")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Layer-wise full-token cosine analysis, matching exp1.ipynb.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CLI paths are relative to the current directory; image paths inside CSVs are repository-root-relative. "
               "Model, FP16 precision, preprocessing, layer definitions and scientific computations are fixed to the notebook.",
    )
    parser.add_argument("--manifest", type=Path, default=ATTACK_DIR / "manifest.csv",
                        help="AMP sample manifest CSV.")
    parser.add_argument("--attack-results", type=Path, default=ATTACK_DIR / "attack_results.csv",
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


def cache_name(identifier):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(identifier)).strip("._")
    if not value:
        raise ValueError(f"Invalid empty cache identifier: {identifier!r}")
    return f"{value}.pt"


def read_samples(manifest_path, attack_results_path):
    """Apply the notebook's one-to-one join, preserving manifest order and ID strings."""
    import pandas as pd

    manifest = pd.read_csv(manifest_path, dtype=str)
    attack_results = pd.read_csv(attack_results_path, dtype=str)
    required_manifest = {
        "sample_id", "pair_id", "source_path", "target_path", "source_image_id",
        "target_image_id", "source_concept", "target_concept",
    }
    required_results = {"sample_id", "adv_path", "status"}
    for path, frame, required in [
        (manifest_path, manifest, required_manifest),
        (attack_results_path, attack_results, required_results),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {path}")
    samples = manifest.merge(
        attack_results[["sample_id", "adv_path", "status"]],
        on="sample_id", how="left", sort=False, validate="one_to_one",
    )
    return manifest, samples


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


class Representations:
    """Exp1 FP16 extraction and the notebook-compatible full-token cache."""

    def __init__(self, device, cache_dir, force_recompute=False):
        import torch
        from torchvision import transforms
        from transformers import LlavaForConditionalGeneration

        self.dtype = torch.float16
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute
        self.cache_counts = {"hits": 0, "extracted": 0}
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required, matching exp1.ipynb and attack_llava.py.")
        print(f"Loading {MODEL_ID} vision tower on {self.device} in FP16...", flush=True)
        # Load the vision tower once and match AMP's preprocessing conventions exactly.
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.vision_tower = llava_model.vision_tower.to(self.device).eval()
        del llava_model

        self.to_tensor = transforms.ToTensor()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        vision_config = self.vision_tower.config
        self.expected_state_count = vision_config.num_hidden_layers + 1
        self.expected_state_shape = (
            1,
            (vision_config.image_size // vision_config.patch_size) ** 2 + 1,
            vision_config.hidden_size,
        )
        print(
            {
                "model_type": vision_config.model_type,
                "encoder_layers": vision_config.num_hidden_layers,
                "returned_hidden_states": self.expected_state_count,
                "embedding_state_included": True,
                "hidden_state_shape": self.expected_state_shape,
            }
        )

    def validate_states(self, states):
        import torch
        if not isinstance(states, (list, tuple)) or len(states) != self.expected_state_count:
            return False
        return all(
            isinstance(state, torch.Tensor)
            and tuple(state.shape) == self.expected_state_shape
            and state.dtype == self.dtype
            and state.device.type == "cpu"
            for state in states
        )

    def load_cached_states(self, cache_path):
        import torch
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            states = payload["hidden_states"]
            metadata = payload["metadata"]
            shapes = [tuple(state.shape) for state in states]
            compatible = (
                metadata.get("cache_version") == CACHE_VERSION
                and metadata.get("model_id") == MODEL_ID
                and metadata.get("hidden_state_count") == len(states)
                and [tuple(shape) for shape in metadata.get("tensor_shapes", [])] == shapes
                and self.validate_states(states)
            )
            if not compatible:
                raise ValueError("incompatible cache metadata or tensors")
            self.cache_counts["hits"] += 1
            return tuple(states)
        except Exception as error:
            warnings.warn(f"Ignoring invalid representation cache {cache_path}: {error}")
            return None

    def extract_hidden_states(self, image_path):
        """Return every full FP16 vision hidden-state token tensor on CPU."""
        import torch
        from PIL import Image
        with Image.open(image_path) as image:
            image_tensor = self.to_tensor(image.convert("RGB")).to(self.device, self.dtype)
        pixel_values = self.preprocess(image_tensor).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        states = tuple(state.detach().to(device="cpu", dtype=self.dtype) for state in outputs.hidden_states)
        if not self.validate_states(states):
            raise ValueError(
                f"Unexpected hidden states for {image_path}: "
                f"count={len(states)}, shapes={[tuple(state.shape) for state in states]}"
            )
        return states

    def get_hidden_states(self, image_path, cache_path):
        import torch
        if cache_path.is_file() and not self.force_recompute:
            cached = self.load_cached_states(cache_path)
            if cached is not None:
                return cached
        states = self.extract_hidden_states(image_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "cache_version": CACHE_VERSION,
                "model_id": MODEL_ID,
                "hidden_state_count": len(states),
                "tensor_shapes": [tuple(state.shape) for state in states],
                "dtype": str(self.dtype),
            },
            "hidden_states": states,
        }
        torch.save(payload, cache_path)
        self.cache_counts["extracted"] += 1
        return states

    def analyze_sample(self, source_path, target_path, adversarial_path, sample_id, source_image_id, target_image_id):
        """Extract/cache one triplet and return one result row per hidden state."""
        states = {
            "source": self.get_hidden_states(
                source_path, self.cache_dir / "clean" / cache_name(source_image_id)
            ),
            "target": self.get_hidden_states(
                target_path, self.cache_dir / "clean" / cache_name(target_image_id)
            ),
            "adversarial": self.get_hidden_states(
                adversarial_path, self.cache_dir / "adv" / cache_name(sample_id)
            ),
        }
        counts = {name: len(value) for name, value in states.items()}
        shapes = {name: [tuple(tensor.shape) for tensor in value] for name, value in states.items()}
        if len(set(counts.values())) != 1 or not (
            shapes["source"] == shapes["target"] == shapes["adversarial"]
        ):
            raise ValueError(f"Representation mismatch: counts={counts}, shapes={shapes}")

        rows = []
        for layer, (source, target, adversarial) in enumerate(
            zip(states["source"], states["target"], states["adversarial"])
        ):
            R = flattened_cosine(adversarial, source)
            T = flattened_cosine(adversarial, target)
            B = flattened_cosine(source, target)
            rows.append(
                {
                    "layer": layer,
                    "R": R,
                    "T": T,
                    "B": B,
                    "G": T - B,
                    "tensor_shape": str(tuple(source.shape)),
                }
            )
        del states
        return rows


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
            if args.save_images:
                save_triplet(source_path, adversarial_path, target_path, sample,
                             args.output_dir / "triplets" / f"{sample_index:05d}_{cache_name(sample.sample_id)[:-3]}_triplet.png")
            sample_rows = representations.analyze_sample(
                source_path,
                target_path,
                adversarial_path,
                sample.sample_id,
                sample.source_image_id,
                sample.target_image_id,
            )
            provenance = {column: getattr(sample, column) for column in provenance_columns}
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
