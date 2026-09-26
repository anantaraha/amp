"""Model-specific full-token extraction shared by Exp1, Exp2 and Exp3.

Official architecture/source references and token conventions: EXPERIMENTS.md.
No model or third-party package is imported until representations are requested.
"""

import hashlib
import json
from pathlib import Path
import re
import warnings

VLM = "cogvlm"
MODEL_ID = "THUDM/cogvlm-chat-hf"
MODEL_REVISION = "e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c"
MODEL_TITLE = "CogVLM"
PRECISION = "float16"
IMAGE_SIZE = 490
ATTACK_OUTPUT = "BOI + GLU-projected patch tokens + EOI"
PREFIX_TOKENS = 1  # Verified in this model's encoder source, not inherited from LLaVA.
CACHE_VERSION = 1
CACHE_SCHEMA = "amp.encoder_full_tokens_and_attack_output.v1"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ATTACK_DIR = REPO_ROOT / "dataset/laion_art/attack_set"
DEFAULT_CACHE_DIR = ATTACK_DIR / "representations/cogvlm_chat_hf"
SUCCESS_STATUSES = {"completed", "skipped", "success", "successful"}
PROVENANCE = ("sample_id", "pair_id", "source_image_id", "target_image_id", "source_concept", "target_concept")


def cache_name(identifier):
    """Exactly the cache-key sanitization used by Exp1."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(identifier)).strip("._")
    if not value:
        raise ValueError(f"Invalid empty cache identifier: {identifier!r}")
    return f"{value}.pt"


def image_path(value):
    return (REPO_ROOT / value).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(manifest_path, attack_results_path):
    """Join once by unique sample ID; retain result provenance for cross-checks."""
    import pandas as pd

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    results = pd.read_csv(attack_results_path, dtype=str, keep_default_na=False)
    required_manifest = set(PROVENANCE) | {"source_path", "target_path"}
    for path, frame, required in [
        (manifest_path, manifest, required_manifest),
        (attack_results_path, results, {"sample_id", "adv_path", "status"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if frame["sample_id"].duplicated().any() or frame["sample_id"].str.strip().eq("").any():
            raise ValueError(f"Missing or duplicate sample IDs in {path}")
    if manifest[list(required_manifest)].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError(f"Empty image IDs or paths in {manifest_path}")
    if any(column.startswith("_attack_") for column in manifest.columns):
        raise ValueError("Manifest column prefix '_attack_' is reserved for the result join.")

    # Different raw identifiers must never silently share Exp1's sanitized cache key.
    owners = {}
    for row in manifest.to_dict(orient="records"):
        for namespace, identifier in [("adv", row["sample_id"]),
                                      ("clean", row["source_image_id"]), ("clean", row["target_image_id"])]:
            key = (namespace, cache_name(identifier))
            if key in owners and owners[key] != identifier:
                raise ValueError(f"Cache-key collision: {owners[key]!r} and {identifier!r} map to {key}")
            owners[key] = identifier
    samples = manifest.merge(results.add_prefix("_attack_"), left_on="sample_id", right_on="_attack_sample_id",
                             how="left", sort=False, validate="one_to_one")
    samples["status"] = samples["_attack_status"]
    samples["adv_path"] = samples["_attack_adv_path"]
    return manifest, samples


def validate_sample(row, fingerprints):
    """Check matching IDs, original paths, copied inputs, and all three image files."""
    from PIL import Image

    for column in PROVENANCE[1:]:
        reported = row.get("_attack_" + column)
        if column in row and isinstance(reported, str) and reported and reported != row[column]:
            raise ValueError(f"Manifest/result mismatch for {column}: {row[column]!r} != {reported!r}")
    vlm = row.get("_attack_vlm")
    if isinstance(vlm, str) and vlm and vlm.strip().lower() != VLM:
        raise ValueError(f"Expected a {MODEL_TITLE} attack, found vlm={vlm!r}")
    adv = row.get("_attack_adv_path")
    if not isinstance(adv, str) or not adv.strip():
        raise ValueError("Successful attack row has no adversarial image path")
    paths = {"source": image_path(row["source_path"]), "target": image_path(row["target_path"]),
             "adversarial": image_path(adv)}
    for path in paths.values():
        with Image.open(path) as image:
            image.verify()
        if path not in fingerprints:
            fingerprints[path] = sha256_file(path)
    for condition in ("source", "target"):
        original = row.get(f"_attack_manifest_{condition}_path")
        if isinstance(original, str) and original and image_path(original) != paths[condition]:
            raise ValueError(f"Manifest/result mismatch for original {condition} path")
        copied = row.get(f"_attack_{condition}_path")
        if isinstance(copied, str) and copied:
            copied = image_path(copied)
            if copied not in fingerprints:
                fingerprints[copied] = sha256_file(copied)
            if fingerprints[copied] != fingerprints[paths[condition]]:
                raise ValueError(f"Attack result's {condition} image differs from the manifest image")
    return paths


def preprocessing_metadata():
    return {"image_size": IMAGE_SIZE, "resize": "tensor_bicubic_square", "crop": False,
            "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
            "input_dtype": PRECISION, "image_mode": "RGB"}


def validate_payload(payload):
    """Reject other models, precisions, token layouts, revisions and incomplete caches."""
    import torch

    metadata, states, attacked = payload["metadata"], payload["hidden_states"], payload["attack_features"]
    if not (metadata.get("cache_version") == CACHE_VERSION and metadata.get("schema") == CACHE_SCHEMA
            and metadata.get("model_id") == MODEL_ID and metadata.get("model_revision") == MODEL_REVISION
            and metadata.get("dtype") == f"torch.{PRECISION}"
            and metadata.get("preprocessing") == preprocessing_metadata()
            and metadata.get("attack_output") == ATTACK_OUTPUT):
        raise ValueError("Cache model/revision/precision/preprocessing/schema mismatch")
    layout = metadata["layout"]
    count = int(layout["encoder_layers"]) + 1
    height, width = layout["patch_grid"]
    patches = int(height) * int(width)
    shape = (1, patches + PREFIX_TOKENS, int(layout["hidden_width"]))
    if (layout.get("prefix_tokens") != PREFIX_TOKENS or layout.get("state_zero") != "input_to_first_encoder_block"
            or count < 2 or patches < 2 or shape[-1] <= 0):
        raise ValueError("Invalid encoder/token layout")
    if not isinstance(states, (tuple, list)) or len(states) != count or metadata.get("hidden_state_count") != count:
        raise ValueError("Cache is missing encoder states")
    dtype = getattr(torch, PRECISION)
    for tensor in states:
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != dtype or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
            raise ValueError("Invalid full-token encoder state")
    attack_shape = tuple(layout["attack_shape"])
    if (not isinstance(attacked, torch.Tensor) or tuple(attacked.shape) != attack_shape
            or attacked.dtype != dtype or attacked.device.type != "cpu" or not torch.isfinite(attacked).all()):
        raise ValueError("Invalid attack-output representation")
    if VLM == "cogvlm":
        if len(attack_shape) != 3 or attack_shape[:2] != (1, patches + 2) or attack_shape[-1] <= 0:
            raise ValueError("CogVLM attack output must be BOI + projected patches + EOI")
    elif attack_shape != (1, patches, shape[-1]):
        raise ValueError("xGen-MM attack output must be the post-LN patch tokens without CLS")
    if [tuple(s) for s in metadata.get("tensor_shapes", [])] != [tuple(s.shape) for s in states]:
        raise ValueError("Cache shape metadata mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", metadata.get("image_sha256", "")):
        raise ValueError("Cache lacks an image-content fingerprint")
    return layout


def cosine_metrics(source, target, adversarial):
    import torch.nn.functional as F

    def cosine(a, b):
        return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    r, t, b = cosine(adversarial, source), cosine(adversarial, target), cosine(source, target)
    return {"R": r, "T": t, "B": b, "G": t - b, "tensor_shape": str(tuple(source.shape))}


class Representations:
    """Lazy model loading, strict model-specific caches, and exact author attack inputs."""

    def __init__(self, device, cache_dir, force_recompute=False, cache_only=False):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        self.cache_only = cache_only
        self.cache_counts = {"hits": 0, "extracted": 0}
        self.fingerprints, self.bindings = {}, {}
        self.vision_model = None
        self.layout = None
        self.startup_error = None

    def initialize_model(self):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=self.dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        config = model.config.vision_config
        vision = model.model.vision
        self.blocks = vision.transformer.layers
        patch = vision.patch_embedding
        if len(self.blocks) != config["num_hidden_layers"] or not self.blocks:
            raise ValueError("CogVLM transformer depth disagrees with the vision config")
        grid = config["image_size"] // config["patch_size"]
        self.patch_grid = (grid, grid)
        self.hidden_width = config["hidden_size"]
        if (config["image_size"] != IMAGE_SIZE or patch.cls_embedding.shape != (1, self.hidden_width)
                or patch.position_embedding.num_embeddings != grid * grid + 1
                or patch.proj.out_channels != self.hidden_width
                or tuple(patch.proj.kernel_size) != (config["patch_size"],) * 2
                or tuple(patch.proj.stride) != (config["patch_size"],) * 2):
            raise ValueError("Unsupported CogVLM patch/CLS layout")
        self.batch_first = True  # visual.py uses [batch, CLS + row-major patches, width].
        self.vision_model = vision.to(self.device).eval()
        del model


    def initialize(self):
        import torch
        from torchvision import transforms

        if self.vision_model is not None:
            return
        if self.startup_error is not None:
            raise RuntimeError(f"Model initialization already failed: {self.startup_error}")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required only for missing/invalid representation extraction")
            self.dtype = getattr(torch, PRECISION)
            print(f"Loading {MODEL_ID} at {MODEL_REVISION} on {self.device} ({PRECISION})...", flush=True)
            self.initialize_model()
            if any(p.dtype != self.dtype for p in self.vision_model.parameters() if p.is_floating_point()):
                raise ValueError("Vision parameter precision differs from the author attack")
            self.to_tensor = transforms.ToTensor()
            self.preprocess = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        except Exception as error:
            self.vision_model = None
            self.startup_error = error
            raise

    def capture(self, pixel_values):
        """Observe the original forward pass; do not reimplement normalization or attention."""
        import torch

        states = [None] * (len(self.blocks) + 1)

        def record(index, tensor):
            if states[index] is not None:
                raise ValueError(f"Encoder block {index} ran more than once")
            if not self.batch_first:
                tensor = tensor.transpose(0, 1)
            states[index] = tensor.detach().to("cpu").clone()

        handles = [self.blocks[0].register_forward_pre_hook(lambda module, inputs: record(0, inputs[0]))]
        for index, block in enumerate(self.blocks, 1):
            handles.append(block.register_forward_hook(lambda module, inputs, output, i=index: record(i, output)))
        try:
            with torch.inference_mode():
                output = self.vision_model(pixel_values)
            attacked = output if VLM == "cogvlm" else output[1]
            attacked = attacked.detach().to("cpu").clone()
        finally:
            for handle in handles:
                handle.remove()
        if any(state is None for state in states):
            raise ValueError("Not all encoder blocks were observed during the model forward pass")
        return tuple(states), attacked

    def extract_payload(self, path, digest):
        from PIL import Image

        self.initialize()
        with Image.open(path) as image:
            pixels = self.to_tensor(image.convert("RGB")).to(device=self.device, dtype=self.dtype)
        states, attacked = self.capture(self.preprocess(pixels).unsqueeze(0))
        layout = {"encoder_layers": len(self.blocks), "prefix_tokens": PREFIX_TOKENS,
                  "patch_grid": list(self.patch_grid), "hidden_width": self.hidden_width,
                  "state_zero": "input_to_first_encoder_block", "attack_shape": list(attacked.shape)}
        payload = {"metadata": {
            "cache_version": CACHE_VERSION, "schema": CACHE_SCHEMA, "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "dtype": f"torch.{PRECISION}", "preprocessing": preprocessing_metadata(),
            "layout": layout, "attack_output": ATTACK_OUTPUT, "hidden_state_count": len(states),
            "tensor_shapes": [list(state.shape) for state in states], "image_sha256": digest,
        }, "hidden_states": states, "attack_features": attacked}
        validate_payload(payload)
        return payload

    def get_payload(self, image_path, cache_path, image_digest=None):
        import torch
        from PIL import Image

        path, cache_path = Path(image_path).resolve(), Path(cache_path)
        if image_digest is None:
            if path not in self.fingerprints:
                with Image.open(path) as image:
                    image.verify()
                self.fingerprints[path] = sha256_file(path)
            image_digest = self.fingerprints[path]
        key = cache_path.resolve()
        if key in self.bindings and self.bindings[key] != image_digest:
            raise ValueError(f"The same cache key identifies different images: {key}")
        self.bindings[key] = image_digest
        payload = None
        if cache_path.is_file() and not self.force_recompute:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                validate_payload(payload)
                if payload["metadata"]["image_sha256"] != image_digest:
                    raise ValueError("Image content changed since cache extraction")
            except Exception as error:
                payload = None
                if self.cache_only:
                    raise ValueError(f"Invalid cache {cache_path}: {error}") from error
                warnings.warn(f"Ignoring cache {cache_path}: {error}")
        if payload is None:
            if self.cache_only:
                raise FileNotFoundError(f"Missing compatible full-token cache: {cache_path}")
            payload = self.extract_payload(path, image_digest)
            if self.layout is not None and payload["metadata"]["layout"] != self.layout:
                raise ValueError("Extracted encoder layout differs from the other representations in this run")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(cache_path)
            self.cache_counts["extracted"] += 1
        else:
            self.cache_counts["hits"] += 1
        layout = payload["metadata"]["layout"]
        if self.layout is not None and layout != self.layout:
            raise ValueError("Inconsistent encoder/token layout within this run")
        self.layout = layout
        return payload

    def get_hidden_states(self, path, cache_path):
        return self.get_payload(path, cache_path)["hidden_states"]

    def analyze_sample(self, source_path, target_path, adversarial_path, sample_id, source_image_id, target_image_id):
        payloads = [self.get_payload(path, self.cache_dir / directory / cache_name(identifier))
                    for path, directory, identifier in [(source_path, "clean", source_image_id),
                                                       (target_path, "clean", target_image_id),
                                                       (adversarial_path, "adv", sample_id)]]
        rows = [{"layer": layer, **cosine_metrics(*states)} for layer, states in enumerate(
            zip(*(payload["hidden_states"] for payload in payloads)))]
        attacked = {"representation": "attack_output", "attack_output": ATTACK_OUTPUT,
                    **cosine_metrics(*(payload["attack_features"] for payload in payloads))}
        return rows, attacked

    def save_metadata(self, output_dir):
        metadata = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "cache_schema": CACHE_SCHEMA,
                    "layout": self.layout, "attack_output": ATTACK_OUTPUT, "preprocessing": preprocessing_metadata()}
        (Path(output_dir) / "representation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
