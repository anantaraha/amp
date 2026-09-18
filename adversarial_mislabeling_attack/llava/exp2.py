#!/usr/bin/env python3
"""Standalone conversion of exp2.ipynb; scientific computations and cache format preserved.

Run python exp2.py --help for runtime options. All plots are saved headlessly.
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

# Fixed to the LLaVA-1.5-7B notebook's 24 encoder blocks plus embedding state.
EXPECTED_STATE_COUNT = 25
EXPECTED_STATE_SHAPE = (1, 577, 1024)
LAYER_LABELS = ["Emb.", *map(str, range(1, EXPECTED_STATE_COUNT))]


def parse_args():
    parser = argparse.ArgumentParser(
        description='All-token mean-pooled linear CKA, matching exp2.ipynb; reuse Exp1 representation caches.',
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
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output/exp2",
                        help="Directory for result CSVs and PNGs; created automatically.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for model extraction (cuda or cuda:N); cached tensors are processed on CPU.")

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


class Representations:
    """Reuse Exp1 caches; initialize LLaVA only for missing/invalid states."""

    def __init__(self, device, cache_dir):
        import torch
        from torchvision import transforms

        self.dtype = torch.float16
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.cache_counts = {"hits": 0, "extracted": 0}
        self.vision_tower = None
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

    def validate_states(self, states):
        import torch
        return (
            isinstance(states, (list, tuple))
            and len(states) == EXPECTED_STATE_COUNT
            and all(
                isinstance(state, torch.Tensor)
                and tuple(state.shape) == EXPECTED_STATE_SHAPE
                and state.dtype == self.dtype
                and state.device.type == "cpu"
                for state in states
            )
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

    def get_vision_tower(self):
        """Load LLaVA lazily, so a cache-complete run never initializes the model."""
        import torch
        from transformers import LlavaForConditionalGeneration
        if self.vision_tower is None:
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required to recompute missing LLaVA representations.")
            print(f"Loading {MODEL_ID} vision tower on {self.device} in FP16...", flush=True)
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=self.dtype, low_cpu_mem_usage=True
            )
            self.vision_tower = llava_model.vision_tower.to(self.device).eval()
            config = self.vision_tower.config
            actual_shape = (
                1,
                (config.image_size // config.patch_size) ** 2 + 1,
                config.hidden_size,
            )
            if config.num_hidden_layers + 1 != EXPECTED_STATE_COUNT or actual_shape != EXPECTED_STATE_SHAPE:
                raise ValueError(
                    f"Incompatible LLaVA vision tower: states={config.num_hidden_layers + 1}, "
                    f"shape={actual_shape}"
                )
            del llava_model
        return self.vision_tower

    def extract_hidden_states(self, image_path):
        """Return every full FP16 vision hidden-state token tensor on CPU."""
        import torch
        from PIL import Image
        model = self.get_vision_tower()
        with Image.open(image_path) as image:
            image_tensor = self.to_tensor(image.convert("RGB")).to(self.device, self.dtype)
        pixel_values = self.preprocess(image_tensor).unsqueeze(0)
        with torch.inference_mode():
            outputs = model(pixel_values, output_hidden_states=True)
        states = tuple(state.detach().to(device="cpu", dtype=self.dtype) for state in outputs.hidden_states)
        if not self.validate_states(states):
            raise ValueError(
                f"Unexpected hidden states for {image_path}: "
                f"count={len(states)}, shapes={[tuple(state.shape) for state in states]}"
            )
        return states

    def get_hidden_states(self, image_path, cache_path):
        import torch
        if cache_path.is_file():
            cached = self.load_cached_states(cache_path)
            if cached is not None:
                return cached
        states = self.extract_hidden_states(image_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    "cache_version": CACHE_VERSION,
                    "model_id": MODEL_ID,
                    "hidden_state_count": len(states),
                    "tensor_shapes": [tuple(state.shape) for state in states],
                    "dtype": str(self.dtype),
                },
                "hidden_states": states,
            },
            cache_path,
        )
        self.cache_counts["extracted"] += 1
        return states


def validate_image(path):
    from PIL import Image
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()


def mean_pool(states):
    """Produce [layers, 1024] float32 features by averaging every token, including CLS."""
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


def draw_cka_heatmap(axis, matrix, title, cmap, vmin, vmax, colorbar_label):
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    axis.set_xticks(range(len(LAYER_LABELS)), labels=LAYER_LABELS, rotation=90)
    axis.set_yticks(range(len(LAYER_LABELS)), labels=LAYER_LABELS)
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
            # Reject missing/corrupt members before adding any part of the triplet.
            for image_path in paths.values():
                validate_image(image_path)
            pooled = {
                condition: mean_pool(representations.get_hidden_states(paths[condition], cache_paths[condition]))
                for condition in ("source", "adversarial", "target")
            }
            if any(value.shape != (EXPECTED_STATE_COUNT, EXPECTED_STATE_SHAPE[-1]) for value in pooled.values()):
                raise ValueError(f"Unexpected pooled representation shapes: { {k: v.shape for k, v in pooled.items()} }")
            for condition in feature_rows:
                feature_rows[condition].append(pooled[condition])
            provenance_rows.append({column: getattr(sample, column) for column in provenance_columns})
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            skipped.append((sample.sample_id, reason))
            warnings.warn(f"Skipping {sample.sample_id}: {reason}")

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

    frames = {
        "source": pd.DataFrame(cka_source, index=LAYER_LABELS, columns=LAYER_LABELS),
        "adversarial": pd.DataFrame(cka_adversarial, index=LAYER_LABELS, columns=LAYER_LABELS),
        "target": pd.DataFrame(cka_target, index=LAYER_LABELS, columns=LAYER_LABELS),
        "delta": pd.DataFrame(cka_delta, index=LAYER_LABELS, columns=LAYER_LABELS),
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
            f"LLaVA-1.5-7B source linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA",
        )
        draw_cka_heatmap(
            axes[0, 1], cka_adversarial,
            f"LLaVA-1.5-7B adversarial linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA",
        )
        draw_cka_heatmap(
            axes[1, 0], cka_target,
            f"LLaVA-1.5-7B target linear CKA (N={valid_count} AMP samples)",
            "viridis", 0, 1, "Linear CKA",
        )
        draw_cka_heatmap(
            axes[1, 1], cka_delta,
            f"LLaVA-1.5-7B adversarial − source CKA (N={valid_count} AMP samples)",
            "coolwarm", -difference_bound, difference_bound, "Linear CKA difference",
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
