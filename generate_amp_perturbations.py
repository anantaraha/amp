#!/usr/bin/env python3
"""Manifest-driven AMP generation using the three local author attack definitions."""

import argparse
from fractions import Fraction
import math
from pathlib import Path
import re
import shutil


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ATTACK_ROOT = REPO_ROOT / "dataset/laion_art/attack_set"
ATTACK_DEFAULTS = {
    "llava": {"optimization_steps": 4000, "initial_lr": 0.005},
    "cogvlm": {"optimization_steps": 2000, "initial_lr": 0.003},
    "xgen_mm": {"optimization_steps": 2000, "initial_lr": 0.01},
}


def budget_value(value):
    """Accept a decimal or fraction without evaluating Python expressions."""
    try:
        number = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Use a finite decimal or fraction, e.g. 16/255") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Budget must lie between 0 and 1")
    return number


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N, for example cuda:1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate AMP perturbations from a common manifest into separate VLM directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Author defaults: llava=4000 steps / LR 0.005; cogvlm=2000 / 0.003; "
               "xgen_mm=2000 / 0.01. CUDA is required for generation. "
               "CLI paths are relative to your working directory; paths inside the manifest are repository-root-relative.",
    )
    parser.add_argument("--vlm", choices=list(ATTACK_DEFAULTS), default="llava",
                        help="VLM whose author-defined AMP attack to use.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ATTACK_ROOT / "manifest.csv",
                        help="Common source/target sample manifest CSV.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ATTACK_ROOT,
                        help="Parent directory; outputs go under <output-root>/<vlm>/.")
    parser.add_argument("--budget", type=budget_value, default=16 / 255,
                        help="L-infinity perturbation budget in [0,1], as decimal or fraction; default 16/255.")
    parser.add_argument("--optimization-steps", type=positive_integer, default=None,
                        help="Override iteration count; omitted uses the selected VLM's author default.")
    parser.add_argument("--initial-lr", type=positive_float, default=None,
                        help="Override initial learning rate; omitted uses the selected VLM's author default.")
    parser.add_argument("--max-attacks", type=positive_integer, default=None,
                        help="Process only the first N manifest rows, including rows skipped on resume; omitted means all.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for the selected model and input tensors.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the per-image optimization progress bar.")
    args = parser.parse_args(argv)
    for setting, default in ATTACK_DEFAULTS[args.vlm].items():
        if getattr(args, setting) is None:
            setattr(args, setting, default)
    return args


def clip_preprocessing(image_size):
    """The author files share normalization but specify different resize sizes."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class LlavaAttack:
    """Local authority: adversarial_mislabeling_attack/llava/attack_llava.py."""

    def __init__(self, device):
        import torch
        from transformers import LlavaForConditionalGeneration

        self.device = device
        self.input_dtype = torch.float16
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        self.vision_tower = model.vision_tower.to(device)
        self.transform = clip_preprocessing(336)

    def features(self, image_tensor):
        import torch

        return self.vision_tower(
            torch.stack([self.transform(image_tensor)]), output_hidden_states=True,
        ).hidden_states[-2]


class CogvlmAttack:
    """Local authority: adversarial_mislabeling_attack/cogvlm/attack_cogvlm.py."""

    def __init__(self, device):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = device
        self.input_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf", torch_dtype=torch.float16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        self.vision_model = model.model.vision.to(device).eval()
        self.transform = clip_preprocessing(490)

    def features(self, image_tensor):
        import torch

        return self.vision_model(torch.stack([self.transform(image_tensor)]))


class XgenMMAttack:
    """Local authority: adversarial_mislabeling_attack/xgen_mm/attack_xgenmm.py."""

    def __init__(self, device):
        from transformers import AutoModelForVision2Seq

        self.device = device
        # The author does not cast this model or its ToTensor inputs to FP16.
        self.input_dtype = None
        model = AutoModelForVision2Seq.from_pretrained(
            "Salesforce/xgen-mm-phi3-mini-instruct-r-v1", trust_remote_code=True,
        )
        self.vision_encoder = model.vlm.vision_encoder.to(device)
        self.transform = clip_preprocessing(378)

    def features(self, image_tensor):
        return self.vision_encoder(self.transform(image_tensor).unsqueeze(0))[1]


def load_attack(vlm, device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AMP generation, matching the author implementations")
    device = torch.device(device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is unavailable")
    print(f"Loading {vlm} once on {device}...", flush=True)
    return {"llava": LlavaAttack, "cogvlm": CogvlmAttack, "xgen_mm": XgenMMAttack}[vlm](device)


def generate_perturbation(attack, source_path, target_path, output_path, args):
    """Shared author PGD loop; architecture-specific features remain separate."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm.auto import tqdm

    to_tensor = transforms.ToTensor()
    with Image.open(source_path) as source_image:
        source_tensor = to_tensor(source_image).to(device=attack.device, dtype=attack.input_dtype)
    modifier = torch.clone(source_tensor) * 0.1
    with Image.open(target_path) as target_image:
        target_tensor = to_tensor(target_image).to(device=attack.device, dtype=attack.input_dtype)
    target_feature = attack.features(target_tensor)

    for i in tqdm(range(args.optimization_steps), desc=f"[{args.vlm}]: generating perturbation",
                  leave=False, disable=not args.verbose):
        alpha = (args.initial_lr - (args.initial_lr - args.initial_lr / 100) / args.optimization_steps * i)
        modifier.requires_grad_(True)
        adv_tensor = torch.clamp(modifier + source_tensor, 0, 1)
        adv_feature = attack.features(adv_tensor)
        loss = (adv_feature - target_feature).norm()
        grad = torch.autograd.grad(loss, modifier)[0]
        modifier = modifier.detach()
        modifier = modifier - torch.sign(grad) * alpha
        modifier = torch.clamp(modifier, min=-args.budget, max=args.budget)

    adv_image = source_tensor + modifier
    adv_image = torch.clamp(adv_image, 0.0, 1.0)
    # All three author implementations save through an FP16 ToPILImage tensor.
    transforms.ToPILImage()(adv_image.to(torch.float16)).save(output_path)


def is_valid_image(path):
    from PIL import Image

    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError):
        return False


def read_manifest(path):
    import pandas as pd

    manifest = pd.read_csv(path, dtype=str)  # Preserve leading zeros in image IDs.
    required = {"sample_id", "source_path", "target_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError(f"Missing sample IDs or source/target paths in {path}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")
    for sample_id in manifest.sample_id:
        if not sample_id.strip() or sample_id in {".", ".."} or any(c in sample_id for c in ["/", "\\", "\0"]):
            raise ValueError(f"Sample ID must be a nonempty filename component: {sample_id!r}")
    return manifest


def result_path(path):
    """Write repository-relative provenance, or absolute paths for external roots."""
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def copy_input(source, destination):
    """Copy original bytes; never resize or re-encode the model-specific input copy."""
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def save_results(records, columns, path):
    """Checkpoint this run's selected manifest rows after every sample."""
    import pandas as pd

    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(records, columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def run(args):
    from tqdm.auto import tqdm

    manifest = read_manifest(args.manifest)
    rows = manifest if args.max_attacks is None else manifest.head(args.max_attacks)
    model_root = args.output_root.resolve() / args.vlm
    source_dir, target_dir, adv_dir = (model_root / name for name in ["source", "target", "adv"])
    for directory in [source_dir, target_dir, adv_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    results_path = model_root / "attack_results.csv"
    print(f"Manifest: {args.manifest.resolve()} ({len(rows)}/{len(manifest)} selected rows)")
    print(f"Attack: {args.vlm}; steps={args.optimization_steps}; initial_lr={args.initial_lr}; "
          f"budget={args.budget:.10g}; device={args.device}")
    print(f"Outputs: {model_root}")

    # Model startup errors remain fatal, as in the notebook. Never repeatedly load per sample.
    needs_generation = any(not is_valid_image(adv_dir / f"{sample_id}.png") for sample_id in rows.sample_id)
    attack = load_attack(args.vlm, args.device) if needs_generation else None
    records = []
    core = ["sample_id", "source_path", "target_path", "adv_path", "status", "error"]
    extra = ["manifest_source_path", "manifest_target_path", "vlm"]
    columns = core + [c for c in manifest.columns if c not in core + extra] + extra
    for row in tqdm(rows.to_dict(orient="records"), total=len(rows), desc=f"AMP {args.vlm} attacks"):
        sample_id = row["sample_id"]
        source_path = REPO_ROOT / row["source_path"]
        target_path = REPO_ROOT / row["target_path"]
        source_copy, target_copy = source_dir / f"{sample_id}.png", target_dir / f"{sample_id}.png"
        adv_path = adv_dir / f"{sample_id}.png"
        result = {
            **row, "source_path": result_path(source_copy), "target_path": result_path(target_copy),
            "adv_path": result_path(adv_path), "manifest_source_path": row["source_path"],
            "manifest_target_path": row["target_path"], "vlm": args.vlm, "error": "",
        }
        try:
            copy_input(source_path, source_copy)
            copy_input(target_path, target_copy)
            if is_valid_image(adv_path):
                result["status"] = "skipped"
            else:
                generate_perturbation(attack, source_copy, target_copy, adv_path, args)
                result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"Failed {sample_id}: {result['error']}", flush=True)
        records.append(result)
        save_results(records, columns, results_path)
    if not records:
        save_results(records, columns, results_path)
    counts = {status: sum(row["status"] == status for row in records) for status in ["completed", "skipped", "failed"]}
    print(f"Summary: completed={counts['completed']}, skipped={counts['skipped']}, failed={counts['failed']}")
    print(f"Results: {results_path}")
    return records


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
