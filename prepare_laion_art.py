#!/usr/bin/env python3
"""Prepare the deterministic Re-LAION-Art image pool from prepare_laion_art.ipynb."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess


AMP_ID_COLUMN = "amp_sample_id"
CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select Re-LAION-Art records, download images, and prepare the clean PNG pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run from the repository root; paths are relative to your working directory. "
               "Selection and download checkpoints are reused automatically. Failed downloads are not replaced.",
    )
    parser.add_argument("--dataset-id", default="laion/relaion-art",
                        help="Hugging Face dataset ID; stream the train split.")
    parser.add_argument("--num-samples", type=positive_integer, default=100_000,
                        help="Number of metadata records to select, including failed downloads.")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Streaming shuffle seed.")
    parser.add_argument("--shuffle-buffer", type=positive_integer, default=10_000,
                        help="Bounded streaming shuffle buffer size.")
    parser.add_argument("--metadata-cache", type=Path, default=Path("dataset/laion_art/cache"),
                        help="Hugging Face metadata cache directory.")
    parser.add_argument("--download-root", type=Path, default=Path("dataset/laion_art/tmp"),
                        help="Parent of the n<count>_seed<seed>_buffer<buffer> checkpoint directory.")
    parser.add_argument("--clean-dir", type=Path, default=Path("dataset/laion_art/clean"),
                        help="Final PNG output directory.")
    parser.add_argument("--metadata-path", type=Path, default=Path("dataset/laion_art/metadata.csv"),
                        help="Output CSV containing all selected metadata and processing statuses.")
    parser.add_argument("--workers", type=positive_integer, default=CPU_WORKERS,
                        help="Image-processing workers; defaults to max(1, CPU count - 1).")
    parser.add_argument("--download-processes", type=positive_integer, default=min(16, CPU_WORKERS),
                        help="img2dataset processes; defaults to min(16, max(1, CPU count - 1)).")
    parser.add_argument("--download-threads", type=positive_integer, default=32,
                        help="img2dataset threads per process.")
    parser.add_argument("--samples-per-shard", type=positive_integer, default=1000,
                        help="img2dataset records per incremental download shard.")
    parser.add_argument("--image-size", type=positive_integer, default=1024,
                        help="Final square PNG side length; direct LANCZOS resize with no crop.")
    return parser.parse_args(argv)


def selection_is_reusable(parquet_path, config_path, expected_config):
    import pyarrow.parquet as pq

    if not parquet_path.is_file() or not config_path.is_file():
        return False
    try:
        saved_config = json.loads(config_path.read_text())
        row_count = pq.ParquetFile(parquet_path).metadata.num_rows
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return saved_config == expected_config and row_count == expected_config["count"]


def select_streaming_records(dataset_id, count, seed, buffer_size, cache_dir):
    """Shuffle a bounded stream reproducibly and consume only the requested records."""
    from datasets import load_dataset

    stream = load_dataset(
        dataset_id,
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
    )
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)
    records = []
    for selection_index, record in enumerate(stream.take(count)):
        record = dict(record)  # Preserve every field supplied by the Hub dataset.
        if AMP_ID_COLUMN in record:
            raise KeyError(f"Reserved column already exists: {AMP_ID_COLUMN}")
        record[AMP_ID_COLUMN] = f"{selection_index:012d}"
        records.append(record)
    if len(records) != count:
        raise ValueError(f"Requested {count:,} records, but the stream yielded {len(records):,}.")
    return records


def valid_final_png(path, target_size):
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "PNG" and image.size == target_size
    except (OSError, ValueError):
        return False


def find_downloads(raw_directory):
    """Map amp_sample_id to downloaded file using img2dataset's JSON sidecars."""
    found = {}
    for sidecar in raw_directory.rglob("*.json"):
        try:
            payload = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        amp_sample_id = payload.get(AMP_ID_COLUMN)
        if amp_sample_id is None:
            continue
        candidates = [
            path for path in sidecar.parent.glob(sidecar.stem + ".*")
            if path.suffix.lower() not in {".json", ".txt"}
        ]
        if candidates:
            found[str(amp_sample_id)] = candidates[0]
    return found


def process_image(task):
    """Validate, normalize to the required PNG/size, and delete raw only on success."""
    from PIL import Image, ImageOps

    amp_sample_id, raw_path_string, final_path_string, target_size = task
    raw_path = Path(raw_path_string) if raw_path_string else None
    final_path = Path(final_path_string)

    if valid_final_png(final_path, target_size):
        if raw_path and raw_path.exists() and raw_path != final_path:
            raw_path.unlink()
        return amp_sample_id, "complete", "", str(final_path)
    if raw_path is None or not raw_path.is_file():
        return amp_sample_id, "download_failed", "No downloaded image was recorded", ""

    temporary = final_path.with_suffix(".png.part")
    try:
        with Image.open(raw_path) as image:
            image.verify()
        with Image.open(raw_path) as image:
            original_format = image.format
            original_size = image.size
            if original_format == "PNG" and original_size == target_size:
                shutil.copyfile(raw_path, temporary)
            else:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if image.size != target_size:
                    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
                image.save(temporary, format="PNG", optimize=False)
        if not valid_final_png(temporary, target_size):
            raise ValueError("Final PNG validation failed")
        temporary.replace(final_path)
        raw_path.unlink()
        return amp_sample_id, "complete", "", str(final_path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return amp_sample_id, "decode_failed", f"{type(error).__name__}: {error}", ""


def run(args):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    target_size = (args.image_size, args.image_size)
    run_dir = args.download_root / f"n{args.num_samples}_seed{args.seed}_buffer{args.shuffle_buffer}"
    selected_parquet = run_dir / "selected.parquet"
    selection_config_path = run_dir / "selection.json"
    raw_image_dir = run_dir / "downloads"
    for directory in (args.metadata_cache, run_dir, args.clean_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selection_config = {
        "dataset": args.dataset_id,
        "split": "train",
        "streaming": True,
        "count": args.num_samples,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
    }
    if not selection_is_reusable(selected_parquet, selection_config_path, selection_config):
        print(f"Selecting {args.num_samples:,} streamed records from {args.dataset_id}...", flush=True)
        selected_records = select_streaming_records(
            args.dataset_id, args.num_samples, args.seed, args.shuffle_buffer, args.metadata_cache,
        )
        temporary_parquet = selected_parquet.with_suffix(".parquet.part")
        pq.write_table(pa.Table.from_pylist(selected_records), temporary_parquet, compression="zstd")
        temporary_parquet.replace(selected_parquet)
        temporary_config = selection_config_path.with_suffix(".json.part")
        temporary_config.write_text(json.dumps(selection_config, indent=2) + "\n")
        temporary_config.replace(selection_config_path)

    selected_count = pq.ParquetFile(selected_parquet).metadata.num_rows
    assert selected_count == args.num_samples
    print(f"Saved deterministic selection of {selected_count:,} streamed records to {selected_parquet}")

    selected_columns = set(pq.read_schema(selected_parquet).names)
    url_column = next((name for name in ("URL", "url") if name in selected_columns), None)
    caption_column = next((name for name in ("TEXT", "text", "caption") if name in selected_columns), None)
    if url_column is None:
        raise KeyError(f"No URL column found in streamed metadata: {sorted(selected_columns)}")
    caption_args = ["--caption_col", caption_column] if caption_column else []
    command = [
        "img2dataset",
        "--url_list", str(selected_parquet),
        "--input_format", "parquet",
        "--url_col", url_column,
        *caption_args,
        "--output_format", "files",
        "--output_folder", str(raw_image_dir),
        "--resize_mode", "no",
        "--processes_count", str(args.download_processes),
        "--thread_count", str(args.download_threads),
        "--number_sample_per_shard", str(args.samples_per_shard),
        "--save_additional_columns", json.dumps([AMP_ID_COLUMN]),
        "--incremental_mode", "incremental",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    metadata = pq.read_table(selected_parquet).to_pandas()
    metadata[AMP_ID_COLUMN] = metadata[AMP_ID_COLUMN].astype(str)
    downloaded = find_downloads(raw_image_dir)
    # Pass the size explicitly so spawned workers honor --image-size as well.
    tasks = [
        (
            amp_sample_id,
            str(downloaded[amp_sample_id]) if amp_sample_id in downloaded else "",
            str(args.clean_dir / f"{amp_sample_id}.png"),
            target_size,
        )
        for amp_sample_id in metadata[AMP_ID_COLUMN]
    ]
    print(f"Processing {len(tasks):,} images with {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(executor.map(process_image, tasks, chunksize=16))

    status = pd.DataFrame(outcomes, columns=[AMP_ID_COLUMN, "status", "error", "final_path"])
    metadata = metadata.merge(status, on=AMP_ID_COLUMN, how="left", validate="one_to_one")
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.metadata_path, index=False)

    assert len(metadata) == args.num_samples, f"Expected {args.num_samples:,} metadata rows, found {len(metadata):,}"
    assert metadata[AMP_ID_COLUMN].is_unique, f"{AMP_ID_COLUMN} must be unique"
    expected_statuses = {"complete", "download_failed", "decode_failed"}
    unexpected_statuses = set(metadata["status"].dropna()) - expected_statuses
    assert not unexpected_statuses, f"Unexpected statuses: {sorted(unexpected_statuses)}"
    completed = metadata.loc[metadata["status"] == "complete", "final_path"]
    invalid = [path for path in completed if not valid_final_png(Path(path), target_size)]
    assert not invalid, f"Invalid final files: {invalid[:5]}"
    print(metadata["status"].value_counts(dropna=False))
    print(f"Validated {len(completed):,} final {args.image_size}×{args.image_size} PNG images")
    print("Images:", args.clean_dir)
    print("Metadata:", args.metadata_path)
    print("Selection checkpoints:", run_dir)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
