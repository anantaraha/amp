#!/usr/bin/env python3
"""Exp5: optimize one AMP modifier, then sweep its amplitude without re-optimizing.

Model adapters reuse the local AMP and Exp1/3 implementations. Shared scoring uses
Exp4's encoder-pair mean. Interpolations stay in memory, before PNG quantization.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import warnings

ROOT = Path(__file__).resolve().parent
DATA = Path("/data/anantaraha/amp/dataset/laion_art")
MODELS = ("llava", "xgen_mm", "cogvlm")
PROVENANCE = ("sample_id", "pair_id", "source_image_id", "target_image_id",
              "source_concept", "target_concept")
ATTACK_FILES = {"llava": "attack_llava.py", "xgen_mm": "attack_xgenmm.py", "cogvlm": "attack_cogvlm.py"}


def load_local(path, name):
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = old


AMP = load_local(ROOT / "generate_amp_perturbations.py", "_exp5_amp")


def alpha_step(value):
    try:
        step = Fraction(value)
        if not 0 < step <= 1:
            raise ValueError()
        return step
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Alpha step must be in (0,1], e.g. 0.1 or 1/10.") from error


def alpha_list(value):
    try:
        values = [float(Fraction(item.strip())) for item in value.split(",")]
        if (len(values) < 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or any(a >= b for a, b in zip(values, values[1:]))):
            raise ValueError()
        return values
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Use at least two increasing, unique alphas in [0,1], separated by commas.") from error


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="AMP alpha sweep: RBF S_global (primary), linear S_global and attacked-output target cosine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Native AMP defaults: llava=4000 steps/LR .005, xgen_mm=2000/.01, cogvlm=2000/.003. "
               "CUDA is required for new trajectories; CKA runs on CPU/float64. CSV image paths are "
               "repository-root-relative. No swept images or hidden-state caches are saved. Resume reuses "
               "content/settings-validated modifier and trajectory checkpoints. Use a separate output "
               "directory to retain aggregate files from different settings.",
    )
    parser.add_argument("--vlm", choices=MODELS, default="llava", help="Model and native AMP objective.")
    parser.add_argument("--manifest", type=Path, default=DATA / "attack_set/manifest.csv", help="Source/target manifest.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: /data/anantaraha/amp/dataset/laion_art/output/<vlm>/exp5.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Modifier checkpoints; default: <output-dir>/perturbations.")
    parser.add_argument("--max-samples", type=AMP.positive_integer, default=None, help="First N manifest rows, including resumed/failed rows; all if omitted.")
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--alpha-step", type=alpha_step, default=Fraction(1, 10), help="Sweep 0..1; append 1 if the step does not divide 1.")
    sweep.add_argument("--alphas", type=alpha_list, default=None, help="Explicit increasing comma-separated alpha list; overrides the default step.")
    parser.add_argument("--cka", choices=("rbf", "linear", "both"), default="both", help="CKA metrics to calculate.")
    parser.add_argument("--diag-width", type=nonnegative_integer, default=0, help="Exclude encoder pairs with layer distance <= k, as in Exp4.")
    parser.add_argument("--budget", type=AMP.budget_value, default=16 / 255, help="Native pixel-space L-infinity budget; accepts 16/255.")
    parser.add_argument("--optimization-steps", type=AMP.positive_integer, default=None, help="Use the selected model's AMP default unless overridden.")
    parser.add_argument("--initial-lr", type=AMP.positive_float, default=None, help="Use the selected model's AMP default unless overridden.")
    parser.add_argument("--device", type=AMP.cuda_device, default="cuda", help="CUDA device for attack and hidden-state extraction.")
    parser.add_argument("--seed", type=nonnegative_integer, default=0, help="Base seed; derive a stable seed per source-target sample.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Re-evaluate trajectories; still reuse valid modifiers (resume enabled: %(default)s).")
    parser.add_argument("--individual-trajectories", action="store_true", help="Also save one trajectory PNG per sample.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Save numerical results only (plotting enabled: %(default)s).")
    parser.add_argument("--plot-only", action="store_true", help="Render existing Exp5 scores/aggregate CSVs; no model, attack, or metric recomputation.")
    parser.add_argument("--verbose", action="store_true", help="Show AMP optimization progress bars.")
    args = parser.parse_args(argv)
    if args.plot_only and not args.plots:
        parser.error("--plot-only cannot be combined with --no-plots")
    if args.alphas is None:
        count = int(Fraction(1) // args.alpha_step)
        args.alphas = [float(i * args.alpha_step) for i in range(count + 1)]
        if args.alphas[-1] < 1:
            args.alphas.append(1.0)
    args.alpha_step = str(args.alpha_step)
    for setting, default in AMP.ATTACK_DEFAULTS[args.vlm].items():
        if getattr(args, setting) is None:
            setattr(args, setting, default)
    args.output_dir = args.output_dir or DATA / "output" / args.vlm / "exp5"
    args.cache_dir = args.cache_dir or args.output_dir / "perturbations"
    return args


def analysis_modules(vlm):
    directory = ROOT / "adversarial_mislabeling_attack" / vlm
    native = None
    old = sys.modules.get("_vision")
    try:
        if vlm != "llava":
            native = load_local(directory / "_vision.py", f"_exp5_{vlm}_vision")
            sys.modules["_vision"] = native
        exp3 = load_local(directory / "exp3.py", f"_exp5_{vlm}_exp3")
    finally:
        if old is None:
            sys.modules.pop("_vision", None)
        else:
            sys.modules["_vision"] = old
    # The Exp4 matrix-only pair selector has no model/token assumptions.
    exp4 = load_local(ROOT / "adversarial_mislabeling_attack/llava/exp4.py", "_exp5_exp4_math")
    return exp3, exp4, native


# Model-specific code: preserve the attacked representation separately from CKA.
class LlavaBackend:
    def __init__(self, args, native):
        self.attack = AMP.LlavaAttack(args.device)
        self.model = self.attack.vision_tower.eval()
        self.dtype, self.device = self.attack.input_dtype, args.device

    def features(self, image):
        return self.attack.features(image)  # Penultimate encoder output, INCLUDING CLS.

    def capture(self, image):
        import torch
        with torch.inference_mode():
            output = self.model(self.attack.transform(image).unsqueeze(0), output_hidden_states=True)
        states = tuple(state.detach().cpu().clone() for state in output.hidden_states)
        return states, states[-2]


class NativeBackend:
    def __init__(self, args, native):
        self.representations = native.Representations(args.device, args.cache_dir)
        self.representations.initialize()
        self.model = self.representations.vision_model
        self.dtype, self.device = self.representations.dtype, args.device

    def forward(self, image):
        return self.model(self.representations.preprocess(image).unsqueeze(0))

    def capture(self, image):
        # Native hooks preserve encoder order, CLS, and LND/NLD layout conventions.
        return self.representations.capture(self.representations.preprocess(image).unsqueeze(0))


class XgenBackend(NativeBackend):
    def features(self, image):
        return self.forward(image)[1]  # Post-ln_post patch tokens, no CLS.


class CogvlmBackend(NativeBackend):
    def features(self, image):
        return self.forward(image)  # BOI + GLU-projected patch tokens + EOI.


def load_backend(args, native):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate or extract Exp5 trajectories.")
    device = torch.device(args.device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Unavailable device: {device}")
    backend = {"llava": LlavaBackend, "xgen_mm": XgenBackend, "cogvlm": CogvlmBackend}[args.vlm](args, native)
    # Gradients are needed only with respect to the image modifier, not weights.
    backend.model.eval().requires_grad_(False)
    return backend


def read_pixels(path, backend):
    from PIL import Image
    from torchvision.transforms import ToTensor
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"AMP inputs must be RGB (no implicit conversion): {path}")
        return ToTensor()(image).to(device=backend.device, dtype=backend.dtype)


def optimize_delta(backend, source, target_features, args):
    """Exact shared AMP initialization, raw L2 objective, sign step and LR schedule."""
    import torch
    from tqdm.auto import tqdm
    modifier = source.clone() * 0.1
    for step in tqdm(range(args.optimization_steps), desc=f"{args.vlm} AMP", disable=not args.verbose, leave=False):
        lr = args.initial_lr - (args.initial_lr - args.initial_lr / 100) / args.optimization_steps * step
        modifier.requires_grad_(True)
        features = backend.features(torch.clamp(source + modifier, 0, 1))
        loss = (features - target_features).norm()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite AMP objective at step {step}")
        gradient = torch.autograd.grad(loss, modifier)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError(f"Nonfinite AMP gradient at step {step}")
        modifier = modifier.detach() - torch.sign(gradient) * lr
        modifier = torch.clamp(modifier, min=-args.budget, max=args.budget)
    return modifier.detach()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_sample(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def modifier_for(backend, source, target_features, args, contract, path):
    import torch
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        delta = payload["delta"]
        bound = torch.tensor(args.budget, dtype=source.dtype).item()
        if (payload.get("contract") != contract or not isinstance(delta, torch.Tensor)
                or delta.shape != source.shape or delta.dtype != source.dtype
                or not torch.isfinite(delta).all() or delta.abs().max().item() > bound):
            raise ValueError(f"Invalid modifier checkpoint: {path}; refusing to silently re-optimize.")
        return delta.to(source.device), "reused"
    delta = optimize_delta(backend, source, target_features, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"contract": contract, "delta": delta.cpu()}, temporary)
    temporary.replace(path)
    return delta, "generated"


def cosine_to_target(features, target):
    import torch
    import torch.nn.functional as F
    if features.shape != target.shape:
        raise ValueError("Source/swept and target attack-output shapes differ.")
    left, right = features.detach().cpu().float().reshape(1, -1), target.detach().cpu().float().reshape(1, -1)
    if not torch.isfinite(left).all() or not torch.isfinite(right).all() or left.norm() == 0 or right.norm() == 0:
        raise ValueError("Undefined target cosine: zero or nonfinite attack output.")
    return F.cosine_similarity(left, right).item()


def score_image(backend, image, target, args, exp3, exp4):
    states, features = backend.capture(image)
    signature = (len(states), tuple(states[0].shape))
    if any(tuple(state.shape) != signature[1] for state in states) or signature[1][0] != 1:
        raise ValueError("Expected consistent single-image full-token states at every encoder layer.")
    left, _ = exp4.encoder_layer_pairs(len(states), args.diag_width)
    result = {"target_cosine_similarity": cosine_to_target(features, target),
              "n_encoder_layers": len(states) - 1, "n_patch_tokens": states[0].shape[1] - getattr(exp3, "PREFIX_TOKENS", 1),
              "n_layer_pairs": len(left)}
    for kernel in (("rbf", "linear") if args.cka == "both" else (args.cka,)):
        function = exp3.token_rbf_cka_matrix if kernel == "rbf" else exp3.token_cka_matrix
        result[f"S_global_{kernel}"] = exp4.global_cka_score(function(states), args.diag_width)
    return result


def read_progress(path, analysis_key, alphas, metrics):
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    if (payload.get("analysis_key") != analysis_key or not isinstance(rows, list)
            or len(rows) > len(alphas) or [row["alpha"] for row in rows] != alphas[:len(rows)]
            or any(not math.isfinite(row[metric]) for row in rows for metric in metrics)):
        raise ValueError(f"Invalid trajectory checkpoint: {path}")
    return rows


def spearman(x, y):
    """Average ranks for ties; constant trajectories have undefined rho, not zero."""
    import numpy as np
    import pandas as pd
    a, b = pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy()
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(frame, metrics):
    import numpy as np
    import pandas as pd
    aggregates, trends, summaries = [], [], []
    for metric in metrics:
        for alpha, group in frame.groupby("alpha", sort=True):
            scores = group[metric]
            sd = scores.std(ddof=1)
            aggregates.append({"metric": metric, "alpha": alpha, "n_samples": len(scores),
                               "mean": scores.mean(), "std": sd, "sem": sd / math.sqrt(len(scores))})
        for sample_id, group in frame.groupby("sample_id", sort=False):
            group = group.sort_values("alpha")
            rho = spearman(group.alpha, group[metric])
            trends.append({"sample_id": sample_id, "pair_id": group.pair_id.iloc[0], "metric": metric,
                           "n_alphas": len(group), "alpha_first": group.alpha.iloc[0], "alpha_last": group.alpha.iloc[-1],
                           "spearman_rho": rho, "constant_trajectory": group[metric].nunique() == 1,
                           "change_first_to_last": group[metric].iloc[-1] - group[metric].iloc[0]})
        rho = np.array([row["spearman_rho"] for row in trends if row["metric"] == metric])
        valid = rho[np.isfinite(rho)]
        curve = [row for row in aggregates if row["metric"] == metric]
        summaries.append({"metric": metric, "n_samples": len(rho), "n_defined_rho": len(valid),
                          "mean_rho": float(valid.mean()) if len(valid) else float("nan"),
                          "median_rho": float(np.median(valid)) if len(valid) else float("nan"),
                          "std_rho": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
                          "negative_rho": int((valid < 0).sum()), "zero_rho": int((valid == 0).sum()),
                          "positive_rho": int((valid > 0).sum()),
                          "fraction_negative_rho": float((valid < 0).mean()) if len(valid) else float("nan"),
                          "mean_curve_rho": spearman([r["alpha"] for r in curve], [r["mean"] for r in curve])})
    return pd.DataFrame(aggregates), pd.DataFrame(trends), pd.DataFrame(summaries)


def make_plots(frame, summary, metrics, args, exp3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    labels = {"S_global_rbf": "RBF (primary)", "S_global_linear": "Linear (ablation)",
              "target_cosine_similarity": "Target cosine"}
    for filename, selected, ylabel in [
        ("exp5_s_global_vs_alpha.png", [m for m in metrics if m.startswith("S_global")], "S_global"),
        ("exp5_target_similarity_vs_alpha.png", ["target_cosine_similarity"], "Target cosine similarity"),
    ]:
        figure, axis = plt.subplots(figsize=(8, 5))
        for metric in selected:
            group = summary.loc[summary.metric.eq(metric)].sort_values("alpha")
            x, mean, sem = (group[column].to_numpy(dtype=float) for column in ("alpha", "mean", "sem"))
            line, = axis.plot(x, mean, marker="o", label=labels[metric])
            axis.fill_between(x, mean - sem, mean + sem, color=line.get_color(), alpha=0.18)
        axis.set(xlabel="Alpha", ylabel=ylabel,
                 title=f"{args.vlm} Exp5: mean ± SEM; n={frame.sample_id.nunique()} samples")
        axis.legend()
        figure.tight_layout()
        exp3.save_figure(figure, args.output_dir / filename)
        files.append(filename)
    if args.individual_trajectories:
        for sample_id, group in frame.groupby("sample_id", sort=False):
            group = group.sort_values("alpha")
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            for metric in metrics:
                axis = axes[1] if metric == "target_cosine_similarity" else axes[0]
                axis.plot(group.alpha, group[metric], marker="o", label=labels[metric])
            for axis, ylabel in zip(axes, ("S_global", "Target cosine similarity")):
                axis.set(xlabel="Alpha", ylabel=ylabel)
                axis.legend()
            figure.suptitle(f"{args.vlm}: {sample_id}")
            figure.tight_layout()
            filename = f"trajectories/{exp3.cache_name(sample_id)[:-3]}_{key_for(sample_id)[:12]}.png"
            exp3.save_figure(figure, args.output_dir / filename)
            files.append(filename)
    return files


def input_record(record):
    from PIL import Image
    values = {field: record.get(field, "") for field in PROVENANCE}
    for condition in ("source", "target"):
        path = (ROOT / record[f"{condition}_path"]).resolve()
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected an RGB AMP input: {path}")
            image.verify()
        values[f"{condition}_path"] = str(path)
        values[f"{condition}_sha256"] = sha256(path)
    return values


def plot_saved(args):
    """Allow CPU rendering in a plotting environment without touching model results."""
    import pandas as pd
    run_path = args.output_dir / "exp5_run.json"
    metadata = json.loads(run_path.read_text())
    if metadata["model"]["vlm"] != args.vlm or not metadata["status"].startswith("complete"):
        raise ValueError("Plot-only requires completed numerical results for the requested VLM.")
    saved = metadata["arguments"]
    kernels = ("rbf", "linear") if saved["cka"] == "both" else (saved["cka"],)
    metrics = [*(f"S_global_{kernel}" for kernel in kernels), "target_cosine_similarity"]
    frame = pd.read_csv(args.output_dir / "exp5_alpha_scores.csv", dtype={key: str for key in PROVENANCE}, keep_default_na=False)
    summary = pd.read_csv(args.output_dir / "exp5_aggregate.csv")
    if (frame.empty or not frame.vlm.eq(args.vlm).all() or frame.duplicated(["sample_id", "alpha"]).any()
            or any(list(group.alpha) != saved["alphas"] for _, group in frame.groupby("sample_id", sort=False))):
        raise ValueError("Saved image scores do not match the recorded complete trajectories.")
    exp3, _, _ = analysis_modules(args.vlm)
    files = make_plots(frame, summary, metrics, args, exp3)
    metadata["outputs"] = list(dict.fromkeys([*metadata["outputs"], *files]))
    if metadata["status"] == "complete_with_plot_error":
        metadata["status"] = "complete_with_failed_samples" if metadata.get("failed_samples") else "complete"
    metadata.pop("plot_error", None)
    metadata["plotting"] = {"python": sys.executable, "finished_utc": datetime.now(timezone.utc).isoformat(),
                            "individual_trajectories": args.individual_trajectories}
    save_json(metadata, run_path)
    print(f"Rendered {len(files)} PNGs from existing results in {args.output_dir.resolve()}")


def run(args, backend_factory=load_backend):
    import pandas as pd
    import torch

    exp3, exp4, native = analysis_modules(args.vlm)
    manifest = AMP.read_manifest(args.manifest).fillna("")
    selected = manifest.head(args.max_samples) if args.max_samples is not None else manifest
    if selected.empty:
        raise ValueError("Manifest has no selected samples.")
    kernels = ("rbf", "linear") if args.cka == "both" else (args.cka,)
    metrics = [*(f"S_global_{kernel}" for kernel in kernels), "target_cosine_similarity"]
    directory = ROOT / "adversarial_mislabeling_attack" / args.vlm
    source_paths = [Path(__file__), ROOT / "generate_amp_perturbations.py", directory / ATTACK_FILES[args.vlm],
                    directory / "exp3.py", Path(exp4.__file__)]
    if native is not None:
        source_paths.append(directory / "_vision.py")
    code = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "pandas", "open-clip-torch", "xformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    model = {"vlm": args.vlm, "model_id": exp3.MODEL_ID,
             "revision": getattr(exp3, "MODEL_REVISION", None),
             "dtype": getattr(exp3, "PRECISION", "float16"),
             "attack_output": "hidden_states[-2], including CLS" if native is None else native.ATTACK_OUTPUT}
    attack_settings = {**model, "budget": args.budget, "optimization_steps": args.optimization_steps,
                       "initial_lr": args.initial_lr, "seed": args.seed, "code": code, "packages": packages}
    metadata = {
        "experiment": "Exp5 AMP alpha sweep", "status": "running", "model": model,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "manifest_sha256": sha256(args.manifest), "selected_rows": len(selected),
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": code, "packages": packages,
        "delta_definition": "Final bounded PGD modifier before image clipping; optimize once per source-target pair.",
        "sweep": "clip(source + alpha * delta, 0, 1) in native attack input precision, before resize/normalization; no PNG round trip.",
        "S_global": "Exp4 arithmetic mean over encoder pairs i<j and j-i>diag_width; embedding 0 and CLS excluded.",
        "rbf": "Exp3 per-image/per-layer median unordered patch distance; double-centered RBF Gram CKA, CPU float64.",
        "cosine": "Flatten the exact AMP-optimized output, retain its special tokens, compute cosine in float32.",
        "uncertainty": "Descriptive ±1 sample SEM; NaN for n=1. Repeated source/target images can be correlated.",
        "trends": "Within-sample Spearman rho vs alpha (average ties), plus mean-curve rho; no independence-based p-values.",
        "aggregation": "Complete trajectories only, same cohort at every alpha and for every requested metric.",
        "resume": "Atomic modifier and per-alpha metric checkpoints, keyed by image content/model/settings/code/software.",
        "outputs": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "exp5_run.json"
    save_json(metadata, run_path)
    backend, complete, failures = None, [], []
    counts = {"generated_modifiers": 0, "reused_modifiers": 0, "resumed_alphas": 0}
    core_columns = [*PROVENANCE, "source_path", "target_path", "source_sha256", "target_sha256",
                    "vlm", "perturbation_key", "diag_width", "delta_linf", "alpha",
                    "n_encoder_layers", "n_patch_tokens", "n_layer_pairs", *metrics]
    print(f"Exp5 {args.vlm}: {len(selected)} samples, alphas={args.alphas}, CKA={args.cka}, k={args.diag_width}")
    print(f"Output: {args.output_dir.resolve()}\nModifier cache: {args.cache_dir.resolve()}", flush=True)
    try:
        for index, record in enumerate(selected.to_dict("records"), 1):
            sample_id = record["sample_id"]
            print(f"[{index}/{len(selected)}] {sample_id}", flush=True)
            try:
                provenance = input_record(record)
                contract = {**attack_settings, "sample_id": sample_id,
                            "source_sha256": provenance["source_sha256"], "target_sha256": provenance["target_sha256"]}
                perturbation_key = key_for(contract)
                analysis_key = key_for({"perturbation_key": perturbation_key, "alphas": args.alphas,
                                        "cka": args.cka, "diag_width": args.diag_width})
                checkpoint = args.output_dir / "checkpoints" / f"{analysis_key}.json"
                rows = read_progress(checkpoint, analysis_key, args.alphas, metrics) if args.resume else []
                counts["resumed_alphas"] += len(rows)
                if len(rows) < len(args.alphas):
                    if backend is None:
                        # Startup failures are fatal, not retried for every manifest row.
                        try:
                            seed_sample(args.seed % (2**32))
                            backend = backend_factory(args, native)
                        except Exception as error:
                            raise SystemError(f"Model startup failed: {error}") from error
                    seed_sample(int(key_for({"seed": args.seed, "sample_id": sample_id})[:8], 16))
                    source = read_pixels(Path(provenance["source_path"]), backend)
                    target = read_pixels(Path(provenance["target_path"]), backend)
                    with torch.no_grad():
                        target_features = backend.features(target).detach()
                    del target
                    cache_path = args.cache_dir / f"{perturbation_key}.pt"
                    delta, cache_status = modifier_for(backend, source, target_features, args, contract, cache_path)
                    counts[f"{cache_status}_modifiers"] += 1
                    target_cpu = target_features.cpu()
                    del target_features
                    delta_linf = delta.abs().max().item()
                    for alpha in args.alphas[len(rows):]:
                        with torch.no_grad():
                            image = torch.clamp(source + alpha * delta, 0, 1)
                        values = score_image(backend, image, target_cpu, args, exp3, exp4)
                        del image
                        if rows and any(values[name] != rows[0][name] for name in ("n_encoder_layers", "n_patch_tokens", "n_layer_pairs")):
                            raise ValueError("Encoder/token layout changed across alpha.")
                        rows.append({"alpha": alpha, "delta_linf": delta_linf, **values})
                        save_json({"analysis_key": analysis_key, "rows": rows}, checkpoint)
                        print(f"  alpha={alpha:g}: " + ", ".join(f"{metric}={values[metric]:.6g}" for metric in metrics), flush=True)
                    del source, delta, target_cpu
                complete.extend({**provenance, "vlm": args.vlm, "perturbation_key": perturbation_key,
                                 "diag_width": args.diag_width, **row} for row in rows)
            except SystemError:
                raise
            except Exception as error:
                failures.append({"sample_id": sample_id, "pair_id": record.get("pair_id", ""),
                                 "error": f"{type(error).__name__}: {error}"})
                print(f"  Failed: {failures[-1]['error']}", flush=True)
            exp3.save_csv(pd.DataFrame(complete, columns=core_columns), args.output_dir / "exp5_alpha_scores.csv", index=False)
            exp3.save_csv(pd.DataFrame(failures, columns=["sample_id", "pair_id", "error"]),
                          args.output_dir / "exp5_failed_samples.csv", index=False)

        frame = pd.DataFrame(complete, columns=core_columns)
        metadata.update(counts=counts, completed_samples=frame.sample_id.nunique(), failed_samples=len(failures))
        metadata["outputs"] = ["exp5_alpha_scores.csv", "exp5_failed_samples.csv", "exp5_run.json"]
        if frame.empty:
            raise RuntimeError("No complete trajectories. Inspect exp5_failed_samples.csv and the job log.")
        if frame.duplicated(["sample_id", "alpha"]).any() or any(
                list(group.alpha) != args.alphas for _, group in frame.groupby("sample_id", sort=False)):
            raise ValueError("Incomplete or duplicate alpha trajectories would enter aggregation.")
        aggregates, trends, trend_summary = summarize(frame, metrics)
        for name, table in [("aggregate", aggregates), ("spearman_per_sample", trends), ("spearman_summary", trend_summary)]:
            filename = f"exp5_{name}.csv"
            exp3.save_csv(table, args.output_dir / filename, index=False)
            metadata["outputs"].append(filename)
        metadata["status"] = "complete_with_failed_samples" if failures else "complete"
        if args.plots:
            try:
                metadata["outputs"].extend(make_plots(frame, aggregates, metrics, args, exp3))
            except Exception as error:
                metadata.update(status="complete_with_plot_error", plot_error=f"{type(error).__name__}: {error}")
                warnings.warn(f"Numerical results saved, but plotting failed: {error}")
        print(f"Completed {frame.sample_id.nunique()} samples; failed {len(failures)}.")
        print(f"Scores, summaries, trends and metadata: {args.output_dir.resolve()}/exp5_*", flush=True)
        return frame
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", counts=counts)
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(metadata, run_path)


def main():
    args = parse_args()
    if args.plot_only:
        plot_saved(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
