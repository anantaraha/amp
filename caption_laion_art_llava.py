#!/usr/bin/env python3
"""Caption the clean image pool with the notebook's LLaVA setup and checkpoints."""

import argparse
from pathlib import Path
import re


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N, for example cuda:1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate LLaVA captions for clean LAION-Art images, checkpointing after each batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run from the repository root; paths are relative to your working directory. "
               "CUDA is required. Existing nonempty captions are reused by image ID.",
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("dataset/laion_art/clean"),
                        help="Input image directory, scanned in sorted filename order.")
    parser.add_argument("--output-path", type=Path, default=Path("dataset/laion_art/llava_captions.csv"),
                        help="Caption CSV and resume checkpoint.")
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model ID or local model directory, also used for the processor.")
    parser.add_argument("--prompt", default="Describe the image in twenty words or less.",
                        help="Instruction inside the notebook's USER/image/ASSISTANT prompt template.")
    parser.add_argument("--batch-size", type=positive_integer, default=16,
                        help="Images per captioning batch and CSV checkpoint.")
    parser.add_argument("--max-new-tokens", type=positive_integer, default=77,
                        help="Maximum generated tokens per image.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for FP16 model and floating-point inputs.")
    return parser.parse_args(argv)


def save_checkpoint(frame, output_path):
    frame = frame.drop_duplicates("image_id", keep="last").sort_values("image_id")
    temporary_path = output_path.with_suffix(".csv.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return frame


def run(args):
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for LLaVA captioning.")
    device = args.device
    dtype = torch.float16

    print(f"Loading {args.model} on {device}...", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"

    image_paths = sorted(
        path for path in args.clean_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.clean_dir}")
    if len({path.stem for path in image_paths}) != len(image_paths):
        raise ValueError("Clean image filenames must have unique stems.")

    columns = ["image_id", "image_path", "llava_caption"]
    if args.output_path.exists():
        captions = pd.read_csv(args.output_path, dtype={"image_id": str})
        missing_columns = set(columns) - set(captions.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in {args.output_path}: {sorted(missing_columns)}")
        if captions["image_id"].duplicated().any():
            raise ValueError(f"Duplicate image IDs in {args.output_path}")
        captions = captions[columns].copy()
    else:
        captions = pd.DataFrame(columns=columns)

    clean_ids = {path.stem for path in image_paths}
    captions = captions[captions["image_id"].isin(clean_ids)].copy()
    completed_ids = set(
        captions.loc[captions["llava_caption"].fillna("").str.strip().ne(""), "image_id"]
    )
    pending = [path for path in image_paths if path.stem not in completed_ids]
    print(f"Clean images: {len(image_paths):,}; remaining: {len(pending):,}")

    prompt = f"USER: <image>\n{args.prompt}\nASSISTANT:"
    for start in tqdm(range(0, len(pending), args.batch_size), desc="Captioning batches"):
        batch_paths = pending[start:start + args.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        inputs = processor(
            text=[prompt] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        batch_captions = [caption.strip() for caption in processor.batch_decode(generated, skip_special_tokens=True)]

        batch_frame = pd.DataFrame({
            "image_id": [path.stem for path in batch_paths],
            "image_path": [path.as_posix() for path in batch_paths],
            "llava_caption": batch_captions,
        })
        captions = save_checkpoint(pd.concat([captions, batch_frame], ignore_index=True), args.output_path)

    expected_ids = {path.stem for path in image_paths}
    captioned_ids = set(captions.loc[captions["llava_caption"].fillna("").str.strip().ne(""), "image_id"])
    assert expected_ids <= captioned_ids, f"Missing captions for {len(expected_ids - captioned_ids)} images"
    print(f"Saved {len(expected_ids):,} captions to {args.output_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
