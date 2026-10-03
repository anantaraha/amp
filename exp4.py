#!/usr/bin/env python3
"""Standalone Exp4: single-image patch-token CKA for LLaVA, xGen-MM and Qwen2.5-VL.

No Exp1/2/3 imports or runtime dependencies. LLaVA/xGen cache schemas, numerical
operations, triplet selection and result filenames match the original Exp4.
Qwen uses official preprocessing and pre-merger vision-block tokens, without CLS.
Only saved, explicitly Qwen-tagged adversarial examples are accepted for Qwen.

Qwen source inspected: Transformers 4.52.3 modeling_qwen2_5_vl.py and
image_processing_qwen2_vl.py; checkpoint config/preprocessor at:
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import warnings

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/anantaraha/amp/dataset/laion_art")
CONDITIONS = ("source", "adversarial", "target")
KERNELS = ("linear", "rbf")
PROVENANCE = ("sample_id", "pair_id", "source_image_id", "target_image_id", "source_concept", "target_concept")
SUCCESS_STATUSES = {"completed", "skipped", "success", "successful"}
MODEL_SPECS = {
    "llava": {"title": "LLaVA", "cache": "llava_1_5_7b"},
    "xgen_mm": {"title": "xGen-MM", "cache": "xgen_mm_phi3_mini_instruct_r_v1"},
    "qwen25_vl": {"title": "Qwen2.5-VL-3B-Instruct", "cache": "qwen25_vl_3b_instruct"},
}


def model_title(args):
    return MODEL_SPECS[args.vlm]["title"]


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
        description="Standalone Exp4: S_global = mean encoder CKA(i,j), i<j and j-i>k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Patch tokens are observations; encoder input state 0 is excluded from scores. "
               "CKA runs on CPU/float64; CUDA is needed only for missing-cache extraction. "
               "AUROC uses -S_global, adversarial positive. Reruns replace result files; use separate "
               "output directories to compare k. CSV image paths are repository-root-relative. "
               "Qwen requires its own attack-results CSV with vlm=qwen25_vl; no attacks are generated.",
    )
    parser.add_argument("--vlm", choices=tuple(MODEL_SPECS), default="llava", help="Model-specific representation backend.")
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "attack_set/manifest.csv", help="Source/target manifest CSV.")
    parser.add_argument("--attack-results", type=Path, default=None, help="Default: <DATA>/attack_set/<vlm>/attack_results.csv.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default: <DATA>/attack_set/representations/<model>; clean/ and adv/ subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <DATA>/output/<vlm>/exp4.")
    parser.add_argument("--diag-width", type=nonnegative_integer, default=0, help="Exclude encoder pairs with distance <= k; 0 excludes self-similarity only.")
    parser.add_argument("--cka", choices=("linear", "rbf", "both"), default="linear", help="Both kernels share one cache-loading pass and write separate outputs.")
    parser.add_argument("--device", type=cuda_device, default="cuda", help="Device for missing-cache extraction only.")
    parser.add_argument("--max-samples", type=positive_integer, default=None, help="First N manifest rows, including skipped rows; all if omitted.")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-only", action="store_true", help="Skip missing/invalid caches; never load a model or write representations.")
    cache.add_argument("--force-recompute-representations", action="store_true", help="Re-extract and replace selected representation caches.")
    parser.add_argument("--local-files-only", action="store_true", help="Extraction uses existing HF model/config files only; no downloads.")
    parser.add_argument("--no-per-image-plots", dest="per_image_plots", action="store_false", help="Skip individual heatmaps (enabled by default: %(default)s).")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Skip all PNGs; retain numerical outputs (plotting enabled: %(default)s).")
    parser.add_argument("--qwen-min-pixels", type=positive_integer, default=3136, help="Qwen official processor minimum image area; ignored for other models.")
    parser.add_argument("--qwen-max-pixels", type=positive_integer, default=12845056, help="Qwen official processor maximum image area; lower explicitly for smaller grids.")
    args = parser.parse_args(argv)
    if args.qwen_min_pixels > args.qwen_max_pixels:
        parser.error("--qwen-min-pixels must not exceed --qwen-max-pixels")
    args.attack_results = args.attack_results or DATA_ROOT / "attack_set" / args.vlm / "attack_results.csv"
    args.cache_dir = args.cache_dir or DATA_ROOT / "attack_set/representations" / MODEL_SPECS[args.vlm]["cache"]
    args.output_dir = args.output_dir or DATA_ROOT / "output" / args.vlm / "exp4"
    return args


def save_csv(frame, path, **kwargs):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def save_figure(fig, output_path):
    import matplotlib.pyplot as plt
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def tensor_features(state, prefix_tokens):
    """Losslessly promote BF16 via FP32; preserve legacy FP16/FP32 -> NumPy FP64."""
    import numpy as np
    import torch
    patches = state[0, prefix_tokens:, :].detach().cpu()
    if patches.dtype == torch.bfloat16:
        patches = patches.float()
    return patches.numpy().astype(np.float64)


def validate_full_states(states, dtype, prefix_tokens):
    import torch
    if not isinstance(states, (tuple, list)) or not states:
        raise ValueError("Expected a nonempty list/tuple of hidden states")
    shape = None
    for layer, state in enumerate(states):
        if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] != 1:
            raise ValueError(f"Layer {layer}: expected a single-image [1, tokens, width] tensor")
        if state.shape[1] < prefix_tokens + 2 or state.shape[2] < 1:
            raise ValueError(f"Layer {layer}: need at least two patch tokens and a nonempty width")
        if state.dtype != dtype or state.device.type != "cpu" or not torch.isfinite(state).all():
            raise ValueError(f"Layer {layer}: caches must contain finite CPU {dtype} tensors")
        if shape is not None and tuple(state.shape) != shape:
            raise ValueError("Layer token/width shapes differ within an image")
        shape = tuple(state.shape)
    return len(states), shape


class Representations:
    """Shared image binding/cache lifecycle; subclasses own extraction and schemas."""
    cache_version = 1
    prefix_tokens = 1
    require_full_provenance = False
    require_vlm_tag = False

    def __init__(self, args):
        self.args = args
        self.model = None
        self.startup_error = None
        self.bindings = {}
        self.cache_counts = {"hits": 0, "extracted": 0}
        self.layout = None

    def initialize(self):
        import torch
        if self.model is not None:
            return
        if self.startup_error is not None:
            raise RuntimeError(f"Model initialization previously failed: {self.startup_error}")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for missing representation extraction")
            device = torch.device(self.args.device)
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"Unavailable CUDA device: {device}")
            self.dtype = getattr(torch, self.precision)
            print(f"Loading {self.model_id} on {device} ({self.precision})...", flush=True)
            self.load_model()
            print("Vision model ready.", flush=True)
        except Exception as error:
            self.model = None
            self.startup_error = error
            raise

    def matching_signature(self, left, right):
        return left == right

    def compare_layout(self, layout):
        if self.layout is not None and layout != self.layout:
            raise ValueError("Inconsistent encoder/token layout within this run")
        self.layout = layout

    def get_hidden_states(self, path, cache_path, image_digest):
        import torch
        key = cache_path.resolve()
        if key in self.bindings and self.bindings[key] != image_digest:
            raise ValueError(f"One cache identifier refers to different image contents: {cache_path}")
        self.bindings[key] = image_digest
        payload = None
        if cache_path.is_file() and not self.args.force_recompute_representations:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                self.validate_payload(payload)
                saved_digest = payload["metadata"].get("image_sha256")
                if saved_digest is not None and saved_digest != image_digest:
                    raise ValueError("Cache image-content hash does not match the input image")
            except Exception as error:
                payload = None
                if self.args.cache_only:
                    raise ValueError(f"Invalid cache {cache_path}: {error}") from error
                warnings.warn(f"Re-extracting invalid cache {cache_path}: {error}")
        if payload is None:
            if self.args.cache_only:
                raise FileNotFoundError(f"Missing full-token cache: {cache_path}")
            self.initialize()
            payload = self.extract_payload(path, image_digest)
            self.validate_payload(payload)
            self.compare_layout(payload["metadata"].get("layout"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(cache_path)
            self.cache_counts["extracted"] += 1
            validation = self.extracted_validation
        else:
            self.cache_counts["hits"] += 1
            self.compare_layout(payload["metadata"].get("layout"))
            validation = self.cached_validation(payload)
        return tuple(payload["hidden_states"]), validation

    def cached_validation(self, payload):
        return "sha256_verified"

    def metadata(self):
        return {"representation_precision": self.precision, "model_revision": self.model_revision,
                "attack_output": self.attack_output, "cache_provenance": self.cache_provenance}

    def layout_metadata(self, frame):
        return {}


class LlavaRepresentations(Representations):
    model_id = "llava-hf/llava-1.5-7b-hf"
    model_revision = None
    precision = "float16"
    extracted_validation = "extracted_sha256"
    attack_output = "hidden_states[-2], including CLS; not used for patch-token CKA"
    cache_provenance = ("CPU FP16 full-token caches. Legacy caches without image hashes remain "
                        "legacy_id_only; hashes, when present, must match current inputs.")

    def __init__(self, args):
        super().__init__(args)
        self.cache_counts = {"hits": 0, "legacy_hits": 0, "extracted": 0}

    def cached_validation(self, payload):
        if payload["metadata"].get("image_sha256") is not None:
            return "sha256_verified"
        if not self.cache_counts["legacy_hits"]:
            warnings.warn("Reusing legacy LLaVA caches by ID/tensor metadata: image-content provenance cannot be verified.")
        self.cache_counts["legacy_hits"] += 1
        return "legacy_id_only"

    def validate_payload(self, payload):
        import torch
        states, metadata = payload["hidden_states"], payload["metadata"]
        count, _ = validate_full_states(states, torch.float16, self.prefix_tokens)
        if not (metadata.get("cache_version") == self.cache_version
                and metadata.get("model_id") == self.model_id
                and metadata.get("hidden_state_count") == count and metadata.get("dtype") == str(torch.float16)
                and [tuple(s) for s in metadata.get("tensor_shapes", [])] == [tuple(s.shape) for s in states]):
            raise ValueError("Incompatible LLaVA cache metadata")

    def compare_layout(self, layout):
        # Legacy LLaVA caches do not have layout dictionaries; shared scoring
        # retains the original cross-image state-shape checks.
        pass

    def load_model(self):
        import torch
        from transformers import LlavaForConditionalGeneration
        from torchvision import transforms
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=self.args.local_files_only,
        )
        self.model = model.vision_tower.to(self.args.device).eval()
        self.to_tensor = transforms.ToTensor()
        self.preprocess = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        config = self.model.config
        self.expected_count = config.num_hidden_layers + 1
        self.expected_shape = (1, (config.image_size // config.patch_size) ** 2 + 1, config.hidden_size)

    def extract_payload(self, path, digest):
        import torch
        from PIL import Image
        with Image.open(path) as image:
            pixels = self.to_tensor(image.convert("RGB")).to(self.args.device, self.dtype)
        pixel_values = self.preprocess(pixels).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(pixel_values, output_hidden_states=True)
        states = tuple(state.detach().to(device="cpu", dtype=self.dtype) for state in output.hidden_states)
        if len(states) != self.expected_count or any(tuple(s.shape) != self.expected_shape for s in states):
            raise ValueError("Unexpected LLaVA encoder layer/token layout")
        return {"hidden_states": states, "metadata": {
            "cache_version": self.cache_version, "model_id": self.model_id, "hidden_state_count": len(states),
            "tensor_shapes": [tuple(s.shape) for s in states], "dtype": str(self.dtype), "image_sha256": digest,
        }}


class XgenRepresentations(Representations):
    model_id = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
    model_revision = "1d91d356d3b6fbc141140edf490b39890417af44"
    precision = "float32"
    extracted_validation = "sha256_verified"
    require_full_provenance = True
    schema = "amp.encoder_full_tokens_and_attack_output.v1"
    attack_output = "post-ln_post patch tokens (vision_encoder output[1], without CLS)"
    cache_provenance = "Validate model revision, preprocessing, full-token layout, native precision, attack output and image SHA-256."

    def preprocessing_metadata(self):
        return {"image_size": 378, "resize": "tensor_bicubic_square", "crop": False,
                "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
                "input_dtype": self.precision, "image_mode": "RGB"}

    def validate_payload(self, payload):
        import torch
        metadata, states, attacked = payload["metadata"], payload["hidden_states"], payload["attack_features"]
        if not (metadata.get("cache_version") == self.cache_version and metadata.get("schema") == self.schema
                and metadata.get("model_id") == self.model_id and metadata.get("model_revision") == self.model_revision
                and metadata.get("dtype") == f"torch.{self.precision}"
                and metadata.get("preprocessing") == self.preprocessing_metadata()
                and metadata.get("attack_output") == self.attack_output):
            raise ValueError("Cache model/revision/precision/preprocessing/schema mismatch")
        layout = metadata["layout"]
        count = int(layout["encoder_layers"]) + 1
        height, width = layout["patch_grid"]
        patches = int(height) * int(width)
        shape = (1, patches + self.prefix_tokens, int(layout["hidden_width"]))
        if (layout.get("prefix_tokens") != self.prefix_tokens or layout.get("state_zero") != "input_to_first_encoder_block"
                or count < 2 or patches < 2 or shape[-1] <= 0):
            raise ValueError("Invalid encoder/token layout")
        if not isinstance(states, (tuple, list)) or len(states) != count or metadata.get("hidden_state_count") != count:
            raise ValueError("Cache is missing encoder states")
        dtype = getattr(torch, self.precision)
        for tensor in states:
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid full-token encoder state")
        attack_shape = tuple(layout["attack_shape"])
        if (not isinstance(attacked, torch.Tensor) or tuple(attacked.shape) != attack_shape
                or attacked.dtype != dtype or attacked.device.type != "cpu" or not torch.isfinite(attacked).all()):
            raise ValueError("Invalid attack-output representation")
        if attack_shape != (1, patches, shape[-1]):
            raise ValueError("xGen-MM attack output must be the post-LN patch tokens without CLS")
        if [tuple(s) for s in metadata.get("tensor_shapes", [])] != [tuple(s.shape) for s in states]:
            raise ValueError("Cache shape metadata mismatch")
        if not re.fullmatch(r"[a-f0-9]{64}", metadata.get("image_sha256", "")):
            raise ValueError("Cache lacks an image-content fingerprint")

    def load_model(self):
        from transformers import AutoModelForVision2Seq
        from torchvision import transforms
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, revision=self.model_revision, trust_remote_code=True,
            local_files_only=self.args.local_files_only,
        )
        config = model.config.vision_encoder_config
        vision = model.vlm.vision_encoder
        self.blocks = vision.transformer.resblocks
        self.patch_grid = tuple(vision.grid_size)
        self.hidden_width = vision.conv1.out_channels
        if (config.model_name != "ViT-H-14-378-quickgelu" or not self.blocks
                or tuple(vision.image_size) != (378, 378)
                or vision.class_embedding.shape != (self.hidden_width,)
                or tuple(vision.positional_embedding.shape) != (self.patch_grid[0] * self.patch_grid[1] + 1, self.hidden_width)
                or vision.attn_pool is not None or vision.final_ln_after_pool
                or vision.pool_type != "tok" or not vision.output_tokens):
            raise ValueError("Unsupported xGen-MM OpenCLIP token/pooling layout")
        layouts = {block.attn.batch_first for block in self.blocks}
        if len(layouts) != 1:
            raise ValueError("Inconsistent OpenCLIP block layouts")
        self.batch_first = layouts.pop()
        self.model = vision.to(self.args.device).eval()
        if any(p.dtype != self.dtype for p in self.model.parameters() if p.is_floating_point()):
            raise ValueError("Vision parameter precision differs from the author attack")
        self.to_tensor = transforms.ToTensor()
        self.preprocess = transforms.Compose([
            transforms.Resize((378, 378), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def capture(self, pixels):
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
                output = self.model(pixels)
            attacked = output[1].detach().to("cpu").clone()
        finally:
            for handle in handles:
                handle.remove()
        if any(state is None for state in states):
            raise ValueError("Not all encoder blocks were observed during the model forward pass")
        return tuple(states), attacked

    def extract_payload(self, path, digest):
        from PIL import Image
        with Image.open(path) as image:
            pixels = self.to_tensor(image.convert("RGB")).to(device=self.args.device, dtype=self.dtype)
        states, attacked = self.capture(self.preprocess(pixels).unsqueeze(0))
        layout = {"encoder_layers": len(self.blocks), "prefix_tokens": self.prefix_tokens,
                  "patch_grid": list(self.patch_grid), "hidden_width": self.hidden_width,
                  "state_zero": "input_to_first_encoder_block", "attack_shape": list(attacked.shape)}
        return {"metadata": {
            "cache_version": self.cache_version, "schema": self.schema, "model_id": self.model_id,
            "model_revision": self.model_revision, "dtype": f"torch.{self.precision}",
            "preprocessing": self.preprocessing_metadata(), "layout": layout, "attack_output": self.attack_output,
            "hidden_state_count": len(states), "tensor_shapes": [list(s.shape) for s in states], "image_sha256": digest,
        }, "hidden_states": states, "attack_features": attacked}


class QwenRepresentations(Representations):
    """Official Qwen visual tower; no language states, CLS, or patch-merger states in CKA.

    preprocess_image and forward_visual are separate for later attack-backend reuse.
    forward_visual remains differentiable. The official PIL image processor itself
    is not a differentiable pixel-space attack transform.
    """
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"
    precision = "bfloat16"
    prefix_tokens = 0
    require_vlm_tag = True
    require_full_provenance = True
    schema = "amp.qwen25_vl.premerger_raster_tokens.v1"
    extracted_validation = "sha256_verified"
    attack_output = "No AMP objective defined; final merged vision output is cached separately, excluded from CKA"
    cache_provenance = "Strict model/revision/processor/layout/dtype/image SHA-256; Qwen-specific attacks only."

    def preprocessing_metadata(self):
        return {"processor": "Qwen2VLImageProcessor", "min_pixels": self.args.qwen_min_pixels,
                "max_pixels": self.args.qwen_max_pixels, "patch_size": 14, "temporal_patch_size": 2,
                "merge_size": 2, "resize": "official_smart_resize_PIL_bicubic", "image_mode": "RGB",
                "rescale_factor": 1 / 255, "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711]}

    def matching_signature(self, left, right):
        # Dynamic resolution changes P, but never depth, batch size or hidden width.
        return left[0] == right[0] and left[1][0] == right[1][0] and left[1][2] == right[1][2]

    def compare_layout(self, layout):
        stable = {key: value for key, value in layout.items() if key not in ("grid_thw", "vision_output_shape")}
        super().compare_layout(stable)

    def layout_metadata(self, frame):
        return {"hidden_state_count_and_shape": None,
                "token_layout": "No CLS; pre-merger blocks, window permutation undone, raster [t,h,w] order",
                "state_zero": "patch embeddings before encoder block 1; excluded from S_global",
                "patch_counts": sorted(int(p) for p in frame.n_patch_tokens.unique()),
                "dynamic_resolution": "Patch counts can differ between images; CKA compares tokens within each image",
                "preprocessing": self.preprocessing_metadata(), "attention_implementation": "sdpa",
                "vision_output": "Final spatial patch-merger output, excluded from all CKA scores",
                "amp_attack": "Not generated here; attack CSV must explicitly tag vlm=qwen25_vl"}

    def validate_payload(self, payload):
        import torch
        metadata, states, output = payload["metadata"], payload["hidden_states"], payload["vision_output"]
        if not (metadata.get("schema") == self.schema and metadata.get("cache_version") == self.cache_version
                and metadata.get("model_id") == self.model_id and metadata.get("model_revision") == self.model_revision
                and metadata.get("dtype") == f"torch.{self.precision}"
                and metadata.get("preprocessing") == self.preprocessing_metadata()
                and metadata.get("attention_implementation") == "sdpa"):
            raise ValueError("Qwen cache model/revision/processor/precision/schema mismatch")
        count, shape = validate_full_states(states, torch.bfloat16, self.prefix_tokens)
        layout = metadata["layout"]
        t, h, w = layout["grid_thw"]
        merge = layout["spatial_merge_size"]
        if (count != layout["encoder_layers"] + 1 or count < 3 or metadata.get("hidden_state_count") != count
                or t != 1 or h < 1 or w < 1 or merge < 1 or h % merge or w % merge
                or shape != (1, t * h * w, layout["hidden_width"])
                or layout.get("prefix_tokens") != 0 or layout.get("token_order") != "raster_thw"
                or layout.get("state_zero") != "input_to_first_encoder_block"):
            raise ValueError("Invalid Qwen pre-merger encoder/token layout")
        output_shape = (1, t * h * w // (merge * merge), layout["merged_width"])
        if (tuple(layout["vision_output_shape"]) != output_shape or not isinstance(output, torch.Tensor)
                or tuple(output.shape) != output_shape or output.dtype != torch.bfloat16
                or output.device.type != "cpu" or not torch.isfinite(output).all()):
            raise ValueError("Invalid Qwen final merged vision output")
        if [tuple(s) for s in metadata.get("tensor_shapes", [])] != [tuple(s.shape) for s in states]:
            raise ValueError("Qwen cache shape metadata mismatch")
        if not re.fullmatch(r"[a-f0-9]{64}", metadata.get("image_sha256", "")):
            raise ValueError("Qwen cache lacks an image-content fingerprint")

    def load_model(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor
        self.processor = Qwen2VLImageProcessor.from_pretrained(
            self.model_id, revision=self.model_revision, local_files_only=self.args.local_files_only,
            min_pixels=self.args.qwen_min_pixels, max_pixels=self.args.qwen_max_pixels,
        )
        if (self.processor.patch_size != 14 or self.processor.temporal_patch_size != 2
                or self.processor.merge_size != 2 or not self.processor.do_resize
                or not self.processor.do_rescale or not self.processor.do_normalize
                or not self.processor.do_convert_rgb or int(self.processor.resample) != 3
                or self.processor.rescale_factor != 1 / 255
                or list(self.processor.image_mean) != self.preprocessing_metadata()["mean"]
                or list(self.processor.image_std) != self.preprocessing_metadata()["std"]):
            raise ValueError("Unexpected official Qwen image processor configuration")
        complete_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.model_revision, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=self.args.local_files_only,
        )
        if complete_model.config.model_type != "qwen2_5_vl":
            raise ValueError("Expected the native Qwen2.5-VL architecture")
        # Transformers 4.52 uses model.visual; older native Qwen versions used visual.
        base = getattr(complete_model, "model", complete_model)
        vision = getattr(base, "visual", None)
        if vision is None:
            vision = getattr(complete_model, "visual", None)
        if vision is None or type(vision).__name__ != "Qwen2_5_VisionTransformerPretrainedModel":
            raise ValueError("Unsupported Qwen visual tower implementation")
        config = vision.config
        if (len(vision.blocks) != config.depth or not vision.blocks or config.patch_size != self.processor.patch_size
                or config.temporal_patch_size != self.processor.temporal_patch_size
                or config.spatial_merge_size != self.processor.merge_size
                or not callable(getattr(vision, "get_window_index", None))):
            raise ValueError("Unexpected Qwen vision depth/patch/window layout")
        self.model = vision.to(self.args.device).eval()
        self.blocks = self.model.blocks

    def preprocess_image(self, image_or_path):
        """Official still-image preprocessing, including dynamic resize and temporal packing."""
        from PIL import Image
        if isinstance(image_or_path, Image.Image):
            inputs = self.processor(images=image_or_path.convert("RGB"), return_tensors="pt")
        else:
            with Image.open(image_or_path) as image:
                inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        if tuple(inputs["image_grid_thw"].shape) != (1, 3) or int(inputs["image_grid_thw"][0, 0]) != 1:
            raise ValueError("Qwen Exp4 expects exactly one still image")
        return (inputs["pixel_values"].to(device=self.args.device, dtype=self.dtype),
                inputs["image_grid_thw"].to(device=self.args.device))

    def forward_visual(self, pixel_values, grid_thw):
        """Native visual forward; returns final merged tokens and preserves autograd."""
        return self.model(pixel_values, grid_thw=grid_thw)

    def capture(self, pixels, grid):
        import torch
        t, h, w = (int(v) for v in grid[0])
        merge = self.model.spatial_merge_size
        window_index, _ = self.model.get_window_index(grid)
        inverse = torch.argsort(window_index).cpu()
        if not torch.equal(torch.sort(window_index.cpu()).values, torch.arange(t * h * w // merge ** 2)):
            raise ValueError("Qwen window index is not a permutation of image patch groups")
        states = [None] * (len(self.blocks) + 1)

        def record(index, tensor):
            if states[index] is not None or tuple(tensor.shape) != (t * h * w, self.model.config.hidden_size):
                raise ValueError(f"Unexpected Qwen vision-block output at state {index}")
            tensor = tensor.detach().cpu().reshape(-1, merge ** 2, tensor.shape[-1])[inverse]
            # Window-major -> merged-group row-major -> individual patch raster order.
            states[index] = tensor.reshape(t, h // merge, w // merge, merge, merge, -1).permute(
                0, 1, 3, 2, 4, 5).reshape(1, t * h * w, -1).contiguous()

        handles = [self.blocks[0].register_forward_pre_hook(lambda module, inputs: record(0, inputs[0]))]
        for index, block in enumerate(self.blocks, 1):
            handles.append(block.register_forward_hook(lambda module, inputs, output, i=index: record(i, output)))
        try:
            with torch.inference_mode():
                output = self.forward_visual(pixels, grid).detach().cpu().unsqueeze(0).clone()
        finally:
            for handle in handles:
                handle.remove()
        if any(state is None for state in states):
            raise ValueError("Not all Qwen vision encoder states were captured")
        return tuple(states), output

    def extract_payload(self, path, digest):
        pixels, grid = self.preprocess_image(path)
        states, output = self.capture(pixels, grid)
        config = self.model.config
        layout = {"encoder_layers": len(self.blocks), "prefix_tokens": 0,
                  "grid_thw": grid[0].cpu().tolist(), "hidden_width": config.hidden_size,
                  "spatial_merge_size": config.spatial_merge_size, "merged_width": config.out_hidden_size,
                  "state_zero": "input_to_first_encoder_block", "token_order": "raster_thw",
                  "vision_output_shape": list(output.shape)}
        return {"hidden_states": states, "vision_output": output, "metadata": {
            "schema": self.schema, "cache_version": self.cache_version, "model_id": self.model_id,
            "model_revision": self.model_revision, "dtype": f"torch.{self.precision}",
            "preprocessing": self.preprocessing_metadata(), "attention_implementation": "sdpa",
            "layout": layout, "hidden_state_count": len(states), "tensor_shapes": [list(s.shape) for s in states],
            "image_sha256": digest,
        }}


def make_representations(args):
    return {"llava": LlavaRepresentations, "xgen_mm": XgenRepresentations,
            "qwen25_vl": QwenRepresentations}[args.vlm](args)


def cache_name(identifier):
    """Exactly the cache-key sanitization used by Exp1."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(identifier)).strip("._")
    if not value:
        raise ValueError(f"Invalid empty cache identifier: {identifier!r}")
    return f"{value}.pt"


def image_path(value):
    return (ROOT / value).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(manifest_path, attack_results_path, backend):
    """Join once by unique sample ID; retain result provenance for cross-checks."""
    import pandas as pd

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    results = pd.read_csv(attack_results_path, dtype=str, keep_default_na=False)
    required_manifest = {"sample_id", "source_path", "target_path", "source_image_id", "target_image_id"}
    if backend.require_full_provenance:
        required_manifest.update(PROVENANCE)
    required_results = {"sample_id", "adv_path", "status"}
    if backend.require_vlm_tag:
        required_results.add("vlm")
    for path, frame, required in [
        (manifest_path, manifest, required_manifest),
        (attack_results_path, results, required_results),
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
    return manifest, samples


def validate_sample(row, fingerprints, backend):
    """Check matching IDs, original paths, copied inputs, and all three image files."""
    from PIL import Image

    for column in PROVENANCE[1:]:
        reported = row.get("_attack_" + column)
        if column in row and isinstance(reported, str) and reported and reported != row[column]:
            raise ValueError(f"Manifest/result mismatch for {column}: {row[column]!r} != {reported!r}")
    vlm = row.get("_attack_vlm")
    expected = backend.args.vlm
    if backend.require_vlm_tag and (not isinstance(vlm, str) or vlm.strip().lower() != expected):
        raise ValueError(f"Qwen requires explicitly tagged Qwen-specific attack results: vlm={expected}; found {vlm!r}")
    if isinstance(vlm, str) and vlm and vlm.strip().lower() != expected:
        raise ValueError(f"Expected a {expected} attack, found vlm={vlm!r}")
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


def token_cka_matrix(states, prefix_tokens=1):
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
        features = tensor_features(state, prefix_tokens)
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


def token_rbf_cka_matrix(states, prefix_tokens=1):
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
        features = tensor_features(state, prefix_tokens)
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
    def __init__(self, args, kernel):
        self.directory = args.output_dir
        self.prefix = "exp4" if kernel == "linear" else "exp4_rbf"
        self.files, self.plot_errors = [], []

    def csv(self, suffix, frame, **kwargs):
        path = self.directory / f"{self.prefix}_{suffix}.csv"
        save_csv(frame, path, **kwargs)
        self.files.append(str(path.relative_to(self.directory)))
        return path

    def figure(self, figure, suffix):
        path = self.directory / f"{self.prefix}_{suffix}.png"
        save_figure(figure, path)
        self.files.append(str(path.relative_to(self.directory)))

    def per_image(self, figure, filename):
        path = self.directory / "per_image" / filename
        save_figure(figure, path)
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
        figure.suptitle(f"{model_title(args)} Exp4 — {kernel.upper()}, k={args.diag_width}")
        figure.tight_layout()
        return figure

    def roc_plot():
        figure, axis = plt.subplots(figsize=(7, 6))
        for row in auroc.to_dict("records"):
            curve = curves.loc[curves.negative_condition.eq(row["negative_condition"])]
            axis.plot(curve.FPR, curve.TPR, label=f"vs {row['negative_condition']}: AUROC={row['AUROC']:.3f}")
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
                 title=f"{model_title(args)} {kernel.upper()} Exp4 ROC; k={args.diag_width}\n"
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
        figure.suptitle(f"{model_title(args)} Exp4 — {kernel.upper()}, k={args.diag_width}; gray band excluded")
        figure.tight_layout()
        return figure

    save("score_distributions_and_pairs", scores_plot)
    save("roc_curves", roc_plot)
    save("token_cka_four_panel", matrices_plot)


def run_experiment(args):
    import numpy as np
    import pandas as pd

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representations = make_representations(args)
    manifest, samples = read_samples(args.manifest, args.attack_results, representations)
    if args.max_samples is not None:
        samples = samples.head(args.max_samples)
    kernels = KERNELS if args.cka == "both" else (args.cka,)
    functions = {"linear": token_cka_matrix, "rbf": token_rbf_cka_matrix}
    prefix_tokens = representations.prefix_tokens
    states_by_kernel = {
        kernel: {"rows": [], "skipped": [], "sums": None, "signature": None,
                 "outputs": Outputs(args, kernel)} for kernel in kernels
    }
    common_metadata = {
        "experiment": "Exp4 single-image global encoder token CKA",
        "model_id": representations.model_id, "vlm": args.vlm, "cache_version": representations.cache_version,
        **representations.metadata(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": sha256_file(args.manifest),
        "attack_results_sha256": sha256_file(args.attack_results),
        "exp4_source_sha256": sha256_file(Path(__file__).resolve()),
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

    print(f"{model_title(args)} Exp4: {len(samples)}/{len(manifest)} manifest rows; kernels={','.join(kernels)}; k={args.diag_width}")
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
            if status not in SUCCESS_STATUSES:
                raise ValueError(f"attack status={status!r} (unsuccessful or unmatched)")
            paths = validate_sample(record, fingerprints, representations)
            for condition in CONDITIONS:
                identifier = sample_id if condition == "adversarial" else record[f"{condition}_image_id"]
                cache_path = args.cache_dir / ("adv" if condition == "adversarial" else "clean") / cache_name(identifier)
                states, validation = representations.get_hidden_states(paths[condition], cache_path,
                                                                       fingerprints[paths[condition]])
                state_signature = (len(states), tuple(states[0].shape))
                if current_signature is not None and not representations.matching_signature(state_signature, current_signature):
                    raise ValueError("Source/adversarial/target layer or token shapes differ")
                if signature is not None and not representations.matching_signature(state_signature, signature):
                    raise ValueError("Layer/token layout differs from previously analyzed samples")
                current_signature = state_signature
                left, right = encoder_layer_pairs(len(states), args.diag_width)
                for kernel in kernels:
                    if kernel in errors:
                        continue
                    try:
                        matrix = functions[kernel](states, prefix_tokens=prefix_tokens)
                        score = global_cka_score(matrix, args.diag_width)
                        matrices[kernel][condition] = matrix
                        sample_rows[kernel].append({
                            **{name: record.get(name, "") for name in PROVENANCE},
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
                    filename = f"{kernel}_{sample_index:06d}_{cache_name(sample_id)[:-3]}_{condition}.png"
                    try:
                        figure, axis = plt.subplots(figsize=(8, 7))
                        plot_cka(axis, matrices[kernel][condition], args.diag_width,
                                 f"{model_title(args)} {kernel.upper()} — {sample_id}/{condition}\n"
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
    columns = [*PROVENANCE, "condition", "image_id", "image_path", "image_sha256", "attack_status",
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
            **representations.layout_metadata(frame),
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
        paired, paired_summary = paired_statistics(frame, PROVENANCE)
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
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
