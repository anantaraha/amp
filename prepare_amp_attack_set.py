#!/usr/bin/env python3
"""Prepare clean AMP source/target pairs using the notebook's concept scoring."""

import argparse
import json
import math
from pathlib import Path
import random
import re
import shutil


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def confidence_value(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Confidence threshold must lie between 0 and 1")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use cuda or cuda:N, for example cuda:1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Assign caption concepts with OpenCLIP and sample reproducible clean AMP source/target pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run from the repository root; paths are relative to your working directory. "
               "CUDA is required. Each run replaces source/, target/, and manifest.csv under --attack-dir. "
               "Assignments are reused when image IDs and captions match, unless --rebuild-assignments is set.",
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("dataset/laion_art/clean"),
                        help="Input PNG image directory.")
    parser.add_argument("--captions-path", type=Path, default=Path("dataset/laion_art/llava_captions.csv"),
                        help="Input LLaVA captions CSV.")
    parser.add_argument("--assignments-path", type=Path, default=Path("dataset/laion_art/concept_assignments.csv"),
                        help="Concept assignments CSV, also used as the scoring cache.")
    parser.add_argument("--pairs-path", type=Path, default=Path("dataset/laion_art/concept_pairs.csv"),
                        help="Output concept-pairs CSV.")
    parser.add_argument("--attack-dir", type=Path, default=Path("dataset/laion_art/attack_set"),
                        help="Output directory for source/, target/, and manifest.csv.")
    parser.add_argument("--num-concept-pairs", type=positive_integer, default=25,
                        help="Number of disjoint source/target concept pairs.")
    parser.add_argument("--images-per-pair", type=positive_integer, default=16,
                        help="Images sampled per concept and image pairs written per concept pair.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Torch, NumPy, Python, and pair-sampling random seed.")
    parser.add_argument("--image-batch-size", type=positive_integer, default=32,
                        help="Image batch size for concept scoring.")
    parser.add_argument("--text-batch-size", type=positive_integer, default=256,
                        help="Concept batch size for text encoding.")
    parser.add_argument("--rebuild-assignments", action="store_true",
                        help="Recompute cached concepts and scores even if captions match.")
    parser.add_argument("--top-concepts", type=positive_integer, default=100,
                        help="Most frequent assigned concepts considered before confidence filtering.")
    parser.add_argument("--confidence-threshold", type=confidence_value, default=0.99,
                        help="Only images with selected-concept score strictly greater than this value qualify.")
    parser.add_argument("--model-name", default="EVA02-E-14-plus",
                        help="OpenCLIP model architecture and tokenizer name.")
    parser.add_argument("--pretrained", default="laion2b_s9b_b144k",
                        help="OpenCLIP pretrained weights tag or local checkpoint path.")
    parser.add_argument("--spacy-model", default="en_core_web_sm",
                        help="spaCy model used to extract NOUN/PROPN tokens.")
    parser.add_argument("--device", type=cuda_device, default="cuda",
                        help="CUDA device for FP16 concept scoring.")
    return parser.parse_args(argv)


def run(args):
    import nltk
    import numpy as np
    import open_clip
    import pandas as pd
    import spacy
    import torch
    from PIL import Image
    from nltk.stem import WordNetLemmatizer
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for efficient concept scoring.")
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Using {device}: {torch.cuda.get_device_name(device)}")

    captions = pd.read_csv(args.captions_path, dtype={"image_id": str})
    images = sorted(args.clean_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {args.clean_dir}")
    required_columns = {"image_id", "llava_caption"}
    missing_columns = required_columns - set(captions.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {args.captions_path}: {sorted(missing_columns)}")
    if captions["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image IDs in {args.captions_path}")
    caption_by_id = captions.set_index("image_id")["llava_caption"]
    records = pd.DataFrame({
        "image_path": [path.as_posix() for path in images],
        "image_id": [path.stem for path in images],
    })
    records["caption"] = records["image_id"].map(caption_by_id)
    missing_caption = records["caption"].isna() | records["caption"].fillna("").str.strip().eq("")
    if missing_caption.any():
        missing = records.loc[missing_caption, "image_id"].tolist()[:10]
        raise ValueError(f"Missing LLaVA captions for image IDs: {missing}")

    rebuild_assignments = args.rebuild_assignments or not args.assignments_path.exists()
    if not rebuild_assignments:
        cached = pd.read_csv(args.assignments_path, dtype={"image_id": str})
        cached_captions = cached.set_index("image_id")["caption"]
        current_captions = records.set_index("image_id")["caption"]
        rebuild_assignments = not cached_captions.equals(current_captions)
        if rebuild_assignments:
            print("Caption source changed; rebuilding concept assignments.")

    nltk.download("wordnet", quiet=True)
    nlp = spacy.load(args.spacy_model, disable=["ner", "parser"])
    lemmatizer = WordNetLemmatizer()

    def extract_concepts(caption):
        nouns = {token.text for token in nlp(str(caption)) if token.pos_ in {"NOUN", "PROPN"}}
        return sorted({lemmatizer.lemmatize(noun.lower()) for noun in nouns})

    if rebuild_assignments:
        records["candidate_concepts_list"] = [
            extract_concepts(caption) for caption in tqdm(records["caption"], desc="Extracting concepts")
        ]
        records["assignable"] = records["candidate_concepts_list"].map(bool)
    else:
        print(f"Using existing {args.assignments_path}; use --rebuild-assignments to recompute.")

    if rebuild_assignments:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.pretrained,
            precision="fp16",
            device=device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(args.model_name)

        assignable_records = records.loc[records["assignable"]]
        vocabulary = sorted({concept for concepts in assignable_records["candidate_concepts_list"] for concept in concepts})
        text_feature_batches = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(vocabulary), args.text_batch_size), desc="Encoding concepts"):
                tokens = tokenizer(vocabulary[start:start + args.text_batch_size]).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                text_feature_batches.append(features)
        text_features = torch.cat(text_feature_batches) if text_feature_batches else None
        concept_index = {concept: index for index, concept in enumerate(vocabulary)}

        selected, scores = [], []
        with torch.inference_mode():
            for start in tqdm(range(0, len(assignable_records), args.image_batch_size), desc="Scoring images"):
                batch = assignable_records.iloc[start:start + args.image_batch_size]
                pixels = torch.stack([
                    preprocess(Image.open(path).convert("RGB")) for path in batch["image_path"]
                ]).to(device=device, dtype=torch.float16)
                image_features = model.encode_image(pixels)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarities = 100.0 * image_features @ text_features.T

                for row_index, concepts in enumerate(batch["candidate_concepts_list"]):
                    indices = torch.tensor([concept_index[concept] for concept in concepts], device=device)
                    probabilities = similarities[row_index, indices].softmax(dim=-1)
                    best = int(probabilities.argmax())
                    selected.append(concepts[best])
                    scores.append(float(probabilities[best].cpu()))

        records["candidate_concepts"] = records["candidate_concepts_list"].map(json.dumps)
        records["selected_concept"] = pd.Series(selected, index=assignable_records.index)
        records["score"] = pd.Series(scores, index=assignable_records.index)
        args.assignments_path.parent.mkdir(parents=True, exist_ok=True)
        records[["image_path", "image_id", "caption", "candidate_concepts", "assignable", "selected_concept", "score"]].to_csv(
            args.assignments_path, index=False,
        )

    assignments = pd.read_csv(args.assignments_path, dtype={"image_id": str})
    assignable_count = assignments["selected_concept"].notna().sum()
    print(f"Assignable images: {assignable_count}; skipped: {len(assignments) - assignable_count}")

    # Rank all assignments first, then apply the strict confidence threshold.
    concept_counts = assignments["selected_concept"].value_counts()
    top_concepts = concept_counts.head(args.top_concepts)
    qualified_assignments = assignments[assignments["score"].gt(args.confidence_threshold)]
    qualified_counts = qualified_assignments["selected_concept"].value_counts()
    eligible = qualified_counts.reindex(top_concepts.index, fill_value=0)
    eligible = eligible[eligible >= args.images_per_pair]
    required_concepts = 2 * args.num_concept_pairs
    if len(eligible) < required_concepts:
        raise ValueError(
            f"Need {required_concepts} top-{args.top_concepts} concepts with at least {args.images_per_pair} images above "
            f"score {args.confidence_threshold}; found {len(eligible)}."
        )

    rng = np.random.default_rng(args.seed)
    chosen_concepts = rng.choice(eligible.index.to_numpy(), size=required_concepts, replace=False)
    concept_pairs = pd.DataFrame({
        "pair_id": [f"pair_{index:02d}" for index in range(args.num_concept_pairs)],
        "source_concept": chosen_concepts[0::2],
        "target_concept": chosen_concepts[1::2],
    })
    concept_pairs["source_frequency"] = concept_pairs["source_concept"].map(qualified_counts)
    concept_pairs["target_frequency"] = concept_pairs["target_concept"].map(qualified_counts)
    args.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    concept_pairs.to_csv(args.pairs_path, index=False)

    # Preserve the notebook's rebuild behavior for common clean inputs only.
    source_dir = args.attack_dir / "source"
    target_dir = args.attack_dir / "target"
    for directory in (source_dir, target_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    manifest_rows = []
    for pair in tqdm(concept_pairs.itertuples(index=False), total=len(concept_pairs), desc="Preparing pairs"):
        source_pool = qualified_assignments[qualified_assignments["selected_concept"].eq(pair.source_concept)]
        target_pool = qualified_assignments[qualified_assignments["selected_concept"].eq(pair.target_concept)]
        source_seed = int(rng.integers(0, 2**32 - 1))
        target_seed = int(rng.integers(0, 2**32 - 1))
        source_rows = source_pool.sample(args.images_per_pair, random_state=source_seed).reset_index(drop=True)
        target_rows = target_pool.sample(args.images_per_pair, random_state=target_seed).reset_index(drop=True)

        for image_index in range(args.images_per_pair):
            sample_id = f"{pair.pair_id}_{image_index:02d}"
            source = source_rows.iloc[image_index]
            target = target_rows.iloc[image_index]
            source_output = source_dir / f"{sample_id}.png"
            target_output = target_dir / f"{sample_id}.png"
            shutil.copy2(source["image_path"], source_output)
            shutil.copy2(target["image_path"], target_output)
            manifest_rows.append({
                "sample_id": sample_id,
                "pair_id": pair.pair_id,
                "source_path": source_output.as_posix(),
                "target_path": target_output.as_posix(),
                "source_image_id": source["image_id"],
                "target_image_id": target["image_id"],
                "source_score": source["score"],
                "target_score": target["score"],
                "source_concept": pair.source_concept,
                "target_concept": pair.target_concept,
                "source_caption": source["caption"],
                "target_caption": target["caption"],
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.attack_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Prepared {len(concept_pairs):,} concept pairs and {len(manifest):,} image pairs")
    print("Assignments:", args.assignments_path)
    print("Concept pairs:", args.pairs_path)
    print("Source images:", source_dir)
    print("Target images:", target_dir)
    print("Manifest:", manifest_path)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
