#!/usr/bin/env python3
"""Analyze saved Exp1/Exp2 results; no model inference or plot-based decisions.

Scientific analysis functions are converted directly from exp1just.ipynb.
Run python analyze_results.py --help for paths and deterministic bootstrap settings.
"""

import argparse

from pathlib import Path
import numpy as np
import pandas as pd

BOOTSTRAP_REPS = 5000
SEED = 2025
DISTANCE_EPS = 1e-8
CHANGE_EPS = 1e-6  # Numerical tolerance for CKA validation and change classification.
STRONG_CKA = 0.8  # Declared descriptive threshold, not a significance cutoff.
CROSS_DECREASE_FRACTION = 0.75
METRICS = ["R", "T", "B", "G", "M"]


def load_results(path):
    frame = pd.read_csv(path, dtype="string")  # Preserve leading zeros in all image IDs.
    required = ["sample_id", "layer", "R", "T", "B", "G"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("The Exp1 CSV has no result rows.")
    has_pairs = "pair_id" in frame.columns
    ids = ["sample_id"] + (["pair_id"] if has_pairs else [])
    for column in ids:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    null_counts = frame.isna().sum()
    checked = required + (["pair_id"] if has_pairs else [])
    if frame[checked].isna().any().any():
        raise ValueError(f"Missing analysis values: {null_counts[null_counts > 0].to_dict()}")
    numeric = ["layer", "R", "T", "B", "G"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Layers and metrics must be finite.")
    if ((frame.layer < 0) | (frame.layer % 1 != 0)).any():
        raise ValueError("Layer indices must be nonnegative integers.")
    frame["layer"] = frame.layer.astype(int)
    if frame.duplicated(["sample_id", "layer"]).any():
        raise ValueError("Duplicate (sample_id, layer) rows; resolve these in the input CSV.")
    if (frame[["R", "T", "B"]].abs() > 1).any().any():
        raise ValueError("Cosines must lie in [-1, 1].")
    if not np.allclose(frame.G, frame["T"] - frame.B, rtol=1e-6, atol=1e-7):
        raise ValueError("Stored G is inconsistent with T - B.")
    if has_pairs and (frame.groupby("sample_id").pair_id.nunique() != 1).any():
        raise ValueError("Each sample must belong to exactly one pair_id.")
    frame["G"] = frame["T"] - frame.B
    frame["M"] = frame["T"] - frame.R
    return frame.sort_values(["sample_id", "layer"]).reset_index(drop=True), null_counts


def summarize(frame, keys):
    groups = frame.groupby(keys, sort=True)
    table = groups[METRICS].agg(["mean", "median"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table["n_samples"] = groups.sample_id.nunique()
    for metric in ["G", "M"]:
        table[f"{metric}_positive_count"] = groups[metric].agg(lambda x: int((x > 0).sum()))
        table[f"{metric}_positive_pct"] = 100 * table[f"{metric}_positive_count"] / table.n_samples
    return table.reset_index()


def bootstrap_intervals(frame, reps=BOOTSTRAP_REPS, seed=SEED):
    if reps < 2:
        raise ValueError("Use at least two bootstrap replicates.")
    unit = "pair_id" if "pair_id" in frame.columns else "sample_id"
    units = sorted(frame[unit].unique())
    layers = sorted(frame.layer.unique())
    grouped = frame.groupby([unit, "layer"])
    counts = grouped.size().unstack("layer").reindex(index=units, columns=layers).fillna(0).to_numpy()
    sums = {m: grouped[m].sum().unstack("layer").reindex(index=units, columns=layers)
            .fillna(0).to_numpy() for m in ["G", "M"]}
    draws = {m: np.full((reps, len(layers)), np.nan) for m in sums}
    rng = np.random.default_rng(seed)
    # Batch cluster multiplicities to bound temporary memory as the dataset grows.
    for start in range(0, reps, 200):
        stop = min(start + 200, reps)
        weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=stop-start)
        denominator = weights @ counts
        for metric in sums:
            np.divide(weights @ sums[metric], denominator, out=draws[metric][start:stop],
                      where=denominator > 0)
    rows = []
    for j, layer in enumerate(layers):
        n_units = int((counts[:, j] > 0).sum())
        row = {"layer": layer, "bootstrap_unit": unit, "n_bootstrap_units": n_units,
               "bootstrap_reps": reps, "bootstrap_seed": seed}
        for metric in sums:
            valid = draws[metric][:, j]
            valid = valid[np.isfinite(valid)]
            low, high = np.quantile(valid, [0.025, 0.975]) if n_units >= 2 and len(valid) >= 2 else (np.nan, np.nan)
            row.update({f"{metric}_ci95_low": low, f"{metric}_ci95_high": high,
                        f"{metric}_valid_bootstrap_reps": len(valid)})
        rows.append(row)
    return pd.DataFrame(rows)


def sample_transitions(frame):
    maximum = int(frame.layer.max())
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        group = group.sort_values("layer")
        positive_g = group.loc[group.G > 0, "layer"]
        positive_m = group.loc[group.M > 0, "layer"]
        first_m = int(positive_m.min()) if len(positive_m) else None
        # Scan a complete positive suffix backward, stopping at a gap or M <= 0.
        expected, persistent = maximum, None
        for row in group.iloc[::-1].itertuples():
            if row.layer != expected or row.M <= 0:
                break
            persistent = int(row.layer)
            expected -= 1
        reversed_after_first = bool(first_m is not None and ((group.layer > first_m) & (group.M <= 0)).any())
        record = {"sample_id": sample_id, "n_layers": len(group),
                  "first_G_positive": int(positive_g.min()) if len(positive_g) else None,
                  "first_M_positive": first_m, "persistent_M_positive": persistent,
                  "M_reversal_after_first_positive": reversed_after_first,
                  "covers_all_indices_0_to_max": len(group) == maximum + 1}
        if "pair_id" in group.columns:
            record["pair_id"] = group.pair_id.iloc[0]
        rows.append(record)
    table = pd.DataFrame(rows)
    for column in ["first_G_positive", "first_M_positive", "persistent_M_positive"]:
        table[column] = table[column].astype("Int64")
    return table


def measure_exp1(exp1_csv, bootstrap_reps=BOOTSTRAP_REPS, seed=SEED):
    """Preserve the original Exp1 validation, measurements and clustered uncertainty."""
    results, missing_values = load_results(exp1_csv)
    HAS_PAIRS = "pair_id" in results.columns
    LAYERS = sorted(results.layer.unique().tolist())
    MAX_LAYER = max(LAYERS)
    N_SAMPLES = results.sample_id.nunique()
    N_PAIRS = results.pair_id.nunique() if HAS_PAIRS else None
    coverage = results.groupby("sample_id").layer.nunique()
    validation = pd.DataFrame([{
        "rows": len(results), "samples": N_SAMPLES, "pairs": N_PAIRS,
        "available_layers": str(LAYERS), "duplicate_sample_layers": 0,
        "missing_analysis_values": 0,
        "missing_sample_layer_combinations": N_SAMPLES * len(LAYERS) - len(results),
        "samples_covering_all_available_layers": int(coverage.eq(len(LAYERS)).sum()),
        "unrecorded_indices_between_0_and_max": str(sorted(set(range(MAX_LAYER + 1)) - set(LAYERS))),
    }])
    layer_summary = summarize(results, ["layer"])
    layer_summary["total_samples"] = N_SAMPLES
    layer_summary["coverage_pct"] = 100 * layer_summary.n_samples / N_SAMPLES
    layer_summary = layer_summary.merge(bootstrap_intervals(results, bootstrap_reps, seed), on="layer", validate="one_to_one")
    pair_summary = summarize(results, ["pair_id", "layer"]) if HAS_PAIRS else pd.DataFrame(
        columns=["pair_id", "layer", "n_samples", "G_mean", "M_mean"])
    if HAS_PAIRS:
        pair_counts = results.groupby("pair_id").sample_id.nunique()
        pair_summary["total_pair_samples"] = pair_summary.pair_id.map(pair_counts)
        pair_summary["coverage_pct"] = 100 * pair_summary.n_samples / pair_summary.total_pair_samples
    transitions = sample_transitions(results)
    persistent = transitions.persistent_M_positive.dropna()
    all_positive_layers = layer_summary.loc[
        layer_summary.n_samples.eq(N_SAMPLES) & layer_summary.M_positive_count.eq(N_SAMPLES), "layer"]
    earliest_all_M = int(all_positive_layers.min()) if len(all_positive_layers) else None
    common_persistent_M = int(persistent.max()) if len(persistent) == N_SAMPLES else None
    reversals_after_all = (results.loc[(results.layer > earliest_all_M) & (results.M <= 0), "sample_id"].nunique()
                           if earliest_all_M is not None else None)
    transition_summary = pd.DataFrame([{
        "total_samples": N_SAMPLES, "verified_persistent_samples": len(persistent),
        "persistent_state_median": persistent.median() if len(persistent) else np.nan,
        "persistent_state_min": persistent.min() if len(persistent) else np.nan,
        "persistent_state_max": persistent.max() if len(persistent) else np.nan,
        "earliest_all_samples_M_positive": earliest_all_M,
        "all_samples_persistent_from": common_persistent_M,
        "samples_reversing_after_first_M_positive": int(transitions.M_reversal_after_first_positive.sum()),
        "samples_reversing_after_earliest_all_positive": reversals_after_all,
    }])
    OBJECTIVE_LAYER = MAX_LAYER - 1
    denominator = 1 - results.B
    results["normalized_target_distance_reduction"] = results.G.div(denominator.where(denominator > DISTANCE_EPS))
    objective = results.loc[results.layer.eq(OBJECTIVE_LAYER)].copy()
    objective_record = {"objective_layer": OBJECTIVE_LAYER, "available": not objective.empty,
                        "n_samples": len(objective), "total_samples": N_SAMPLES}
    if not objective.empty:
        for metric in ["G", "M"]:
            objective_record.update({f"{metric}_mean": objective[metric].mean(),
                                     f"{metric}_median": objective[metric].median(),
                                     f"{metric}_positive_proportion": (objective[metric] > 0).mean()})
        objective_record.update(M_min=objective.M.min(), M_max=objective.M.max())
        ratios = objective.normalized_target_distance_reduction.dropna()
        objective_record.update(distance_reduction_valid=len(ratios),
                                distance_reduction_excluded=len(objective)-len(ratios))
        for stat in ["mean", "median", "min", "max"]:
            objective_record[f"distance_reduction_{stat}"] = getattr(ratios, stat)() if len(ratios) else np.nan
    objective_summary = pd.DataFrame([objective_record])
    return {
        "exp1_validation": validation,
        "exp1_layer_summary": layer_summary,
        "exp1_sample_transitions": transitions,
        "exp1_transition_summary": transition_summary,
        "exp1_pair_summary": pair_summary,
        "exp1_objective_summary": objective_summary,
        "exp1_sample_metrics": results,
    }


def layer_blocks(layers):
    """Split sorted encoder states into thirds; remainder goes to earlier blocks."""
    layers = sorted(int(layer) for layer in layers)
    encoders = [layer for layer in layers if layer != 0]
    assignment = {0: "embedding"} if 0 in layers else {}
    for name, states in zip(["early", "middle", "deep"], np.array_split(encoders, 3)):
        assignment.update({int(state): name for state in states})
    return pd.DataFrame({"layer": layers, "block": [assignment[layer] for layer in layers]})


def read_cka_matrix(path):
    """Validate labels before pandas can mangle duplicate column names."""
    import csv

    def state_index(label):
        label = str(label).strip()
        if label == "Emb.":
            return 0
        if not label.isdigit():
            raise ValueError(f"Invalid state label {label!r} in {path}")
        return int(label)

    with Path(path).open(newline="") as handle:
        header = next(csv.reader(handle), [])
    columns = [state_index(label) for label in header[1:]]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"Missing or duplicate CKA column states: {path}")
    frame = pd.read_csv(path, index_col=0)
    rows = [state_index(label) for label in frame.index]
    if len(set(rows)) != len(rows) or set(rows) != set(columns) or len(rows) != len(columns):
        raise ValueError(f"CKA must be square with matching unique row/column states: {path}")
    frame.index, frame.columns = rows, columns
    frame = frame.loc[sorted(rows), sorted(rows)].astype(float)
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite CKA entries (possibly degenerate features): {path}")
    if not np.allclose(values, values.T, rtol=0, atol=CHANGE_EPS):
        raise ValueError(f"Asymmetric CKA matrix: {path}")
    return frame


def measure_exp2(directory):
    """Measure saved matrices only; never reconstruct features or rerun CKA."""
    directory = Path(directory)
    matrices = {condition: read_cka_matrix(directory / f"exp2_linear_cka_{condition}.csv")
                for condition in ["source", "adversarial", "target"]}
    layers = matrices["source"].index.tolist()
    for name, frame in matrices.items():
        if frame.index.tolist() != layers:
            raise ValueError(f"Different CKA states in {name} matrix")
        values = frame.to_numpy()
        # Permit the small FP32 overshoot in the original CKA computation; never clip it.
        if (values < -CHANGE_EPS).any() or (values > 1 + CHANGE_EPS).any():
            raise ValueError(f"{name} biased linear CKA is outside [0, 1] beyond tolerance")
        if not np.allclose(np.diag(values), 1, rtol=0, atol=CHANGE_EPS):
            raise ValueError(f"{name} CKA diagonal is not approximately one")
    delta = matrices["adversarial"] - matrices["source"]
    supplied_delta = read_cka_matrix(directory / "exp2_linear_cka_adv_minus_source.csv")
    if supplied_delta.index.tolist() != layers or not np.allclose(
            delta.to_numpy(), supplied_delta.to_numpy(), rtol=0, atol=CHANGE_EPS):
        raise ValueError("Saved Exp2 difference matrix does not match adversarial - source")

    samples = pd.read_csv(directory / "exp2_samples.csv", dtype="string")
    if "sample_id" not in samples:
        raise ValueError("exp2_samples.csv requires sample_id")
    samples["sample_id"] = samples.sample_id.str.strip().replace("", pd.NA)
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any() or len(samples) < 3:
        raise ValueError("Exp2 requires at least three unique, nonmissing sample IDs")
    if "pair_id" in samples:
        samples["pair_id"] = samples.pair_id.str.strip().replace("", pd.NA)
        if samples.pair_id.isna().any():
            raise ValueError("Exp2 pair_id is present but contains missing values")

    blocks = layer_blocks(layers)
    block_of = blocks.set_index("layer").block.to_dict()
    i, j = np.triu_indices(len(layers), k=1)
    pairs = pd.DataFrame({"layer_i": np.array(layers)[i], "layer_j": np.array(layers)[j]})
    pairs["block_i"] = pairs.layer_i.map(block_of)
    pairs["block_j"] = pairs.layer_j.map(block_of)
    for name, frame in matrices.items():
        pairs[name] = frame.to_numpy()[i, j]
    pairs["delta_adv_minus_source"] = delta.to_numpy()[i, j]

    masks = {"all_off_diagonal": np.ones(len(pairs), dtype=bool)}
    names = ["embedding", "early", "middle", "deep"]
    for start, left in enumerate(names):
        for right in names[start:]:
            region = f"within_{left}" if left == right else f"{left}_to_{right}"
            masks[region] = ((pairs.block_i == left) & (pairs.block_j == right)) | (
                (pairs.block_i == right) & (pairs.block_j == left))
    encoders = (pairs.block_i != "embedding") & (pairs.block_j != "embedding")
    touches_deep = (pairs.block_i == "deep") | (pairs.block_j == "deep")
    masks["early_middle_to_deep"] = encoders & (pairs.block_i != pairs.block_j) & touches_deep
    masks["encoder_pairs_touching_deep"] = encoders & touches_deep
    masks["encoder_pairs_without_deep"] = encoders & ~touches_deep

    region_rows = []
    for region, mask in masks.items():
        subset = pairs.loc[mask]
        row = {"region": region, "n_layer_pairs": len(subset)}
        for condition in matrices:
            values = subset[condition]
            for stat in ["mean", "median", "min", "max"]:
                row[f"{condition}_{stat}"] = getattr(values, stat)()
            row[f"{condition}_strong_fraction"] = (values >= STRONG_CKA).mean() if len(values) else np.nan
        changes = subset.delta_adv_minus_source
        row.update(delta_mean=changes.mean(), delta_median=changes.median(),
                   delta_abs_mean=changes.abs().mean(), delta_abs_median=changes.abs().median(),
                   delta_min=changes.min(), delta_max=changes.max(),
                   decreased_count=int((changes < -CHANGE_EPS).sum()),
                   increased_count=int((changes > CHANGE_EPS).sum()),
                   unchanged_count=int((changes.abs() <= CHANGE_EPS).sum()),
                   decreased_fraction=(changes < -CHANGE_EPS).mean() if len(changes) else np.nan)
        region_rows.append(row)

    extrema = []
    for direction in ["positive", "negative"]:
        selected = pairs.loc[pairs.delta_adv_minus_source > CHANGE_EPS] if direction == "positive" else pairs.loc[
            pairs.delta_adv_minus_source < -CHANGE_EPS]
        record = {"direction": direction, "available": not selected.empty, "layer_i": np.nan,
                  "layer_j": np.nan, "delta": np.nan, "tied_pair_count": 0}
        if not selected.empty:
            extreme = selected.delta_adv_minus_source.max() if direction == "positive" else selected.delta_adv_minus_source.min()
            tied = selected.loc[selected.delta_adv_minus_source.eq(extreme)]
            # First pair in numerical layer order breaks exact ties deterministically.
            first = tied.iloc[0]
            record.update(layer_i=int(first.layer_i), layer_j=int(first.layer_j), delta=extreme,
                          tied_pair_count=len(tied))
        extrema.append(record)
    delta.index = ["Emb." if layer == 0 else str(layer) for layer in layers]
    delta.columns = delta.index
    delta.index.name = "layer"
    return {"exp2_layer_blocks": blocks, "exp2_cka_layer_pairs": pairs,
            "exp2_cka_region_summary": pd.DataFrame(region_rows),
            "exp2_cka_extrema": pd.DataFrame(extrema), "exp2_cka_adv_minus_source": delta}, samples


def measure_all(exp1_csv, exp2_dir, bootstrap_reps=BOOTSTRAP_REPS, seed=SEED):
    """Finish every measurement before any finding is constructed."""
    tables = measure_exp1(exp1_csv, bootstrap_reps, seed)
    exp2_tables, samples = measure_exp2(exp2_dir)
    tables.update(exp2_tables)
    results = tables["exp1_sample_metrics"]
    blocks = layer_blocks(results.layer.unique())
    tables["exp1_layer_blocks"] = blocks
    layer_means = tables["exp1_layer_summary"].merge(blocks, on="layer", validate="one_to_one")
    summary = layer_means.groupby("block", sort=False)[["G_mean", "M_mean", "R_mean", "T_mean", "B_mean"]].mean()
    summary["n_states"] = layer_means.groupby("block").size()
    summary["all_states_complete"] = layer_means.groupby("block").coverage_pct.agg(lambda x: bool(x.eq(100).all()))
    tables["exp1_block_summary"] = summary.reset_index()

    ids1, ids2 = set(results.sample_id), set(samples.sample_id)
    layers1, layers2 = set(results.layer), set(exp2_tables["exp2_layer_blocks"].layer)
    # Match recorded image/pair provenance too: identical sample IDs alone are insufficient.
    provenance = [c for c in ["pair_id", "source_image_id", "target_image_id", "source_concept", "target_concept"]
                  if c in results and c in samples]
    inconsistent = [c for c in provenance if (results.groupby("sample_id")[c].nunique(dropna=False) > 1).any()]
    if inconsistent:
        raise ValueError(f"Exp1 provenance changes across states: {inconsistent}")
    one = results.drop_duplicates("sample_id").set_index("sample_id")
    two = samples.set_index("sample_id")
    common = sorted(ids1 & ids2)
    mismatches = sum(int((one.loc[common, c].fillna("<missing>") != two.loc[common, c].fillna("<missing>")).sum())
                     for c in provenance)
    image_provenance_available = all(c in provenance for c in ["source_image_id", "target_image_id"])
    if image_provenance_available:
        image_provenance_available = all(
            not frame.loc[common, column].isna().any()
            and frame.loc[common, column].str.strip().ne("").all()
            for frame in [one, two] for column in ["source_image_id", "target_image_id"]
        )
    complete = bool(tables["exp1_layer_summary"].coverage_pct.eq(100).all())
    matched = ids1 == ids2 and layers1 == layers2 and complete and mismatches == 0 and image_provenance_available
    block_stats = tables["exp1_block_summary"].set_index("block")
    enough_blocks = all(name in block_stats.index for name in ["early", "middle", "deep"])
    deepest_gain = bool(enough_blocks and block_stats.loc["deep", "G_mean"] > 0 and
                        block_stats.loc["deep", "G_mean"] > block_stats.loc["early", "G_mean"] + CHANGE_EPS and
                        block_stats.loc["deep", "G_mean"] > block_stats.loc["middle", "G_mean"] + CHANGE_EPS)
    regions = tables["exp2_cka_region_summary"].set_index("region")
    deep_change = regions.loc["encoder_pairs_touching_deep", "delta_abs_mean"]
    other_change = regions.loc["encoder_pairs_without_deep", "delta_abs_mean"]
    deeper_cka_change = bool(np.isfinite(deep_change) and np.isfinite(other_change) and deep_change > other_change + CHANGE_EPS)
    tables["cross_experiment_summary"] = pd.DataFrame([{
        "exp1_samples": len(ids1), "exp2_samples": len(ids2), "common_samples": len(common),
        "exp1_only_samples": len(ids1-ids2), "exp2_only_samples": len(ids2-ids1),
        "same_sample_ids": ids1 == ids2, "same_states": layers1 == layers2,
        "exp1_complete_state_coverage": complete, "provenance_columns_checked": ",".join(provenance),
        "image_provenance_available": image_provenance_available, "provenance_mismatches": mismatches,
        "matched_for_combined_finding": matched, "deep_has_largest_positive_mean_G": deepest_gain,
        "deep_touching_mean_abs_delta": deep_change, "nondeep_mean_abs_delta": other_change,
        "deep_touching_change_larger": deeper_cka_change,
        "complementary_later_concentration": matched and deepest_gain and deeper_cka_change,
    }])
    return tables


def number(value, digits=4):
    return "unavailable" if pd.isna(value) else f"{value:.{digits}f}"


def supported_findings(tables):
    """Apply declared numerical rules to this run's measurements only."""
    layer = tables["exp1_layer_summary"]
    transition = tables["exp1_transition_summary"].iloc[0]
    objective = tables["exp1_objective_summary"].iloc[0]
    n = int(transition.total_samples)
    unanimous = layer.loc[layer.n_samples.eq(n) & layer.G_positive_count.eq(n), "layer"].tolist()
    findings = [f"Positive target-image gain is observed in every sample at {len(unanimous)} states: {unanimous}."
                if unanimous else "No state has positive target-image gain in every sample with complete coverage."]
    persistent = transition.all_samples_persistent_from
    if pd.notna(persistent):
        findings.append(f"All {n} samples remain target-closer (M > 0) from state {int(persistent)} "
                        f"through the last recorded state {int(layer.layer.max())}.")
    else:
        findings.append(f"A common persistent M > 0 state is not established; positive suffixes are verified for "
                        f"{int(transition.verified_persistent_samples)}/{n} samples.")
    if objective.available:
        state = int(objective.objective_layer)
        row = layer.set_index("layer").loc[state]
        consistent = bool(row.n_samples == n and row.G_positive_count == n and row.G_ci95_low > 0)
        findings.append(f"At inferred objective state {state}, the consistent-gain rule "
                        f"(all samples G > 0 and pointwise mean-G interval above zero) is {'met' if consistent else 'not met'}: "
                        f"G > 0 in {int(row.G_positive_count)}/{int(row.n_samples)}, "
                        f"M > 0 in {int(row.M_positive_count)}/{int(row.n_samples)}; "
                        f"mean-G interval [{number(row.G_ci95_low)}, {number(row.G_ci95_high)}].")
        pairs = tables["exp1_pair_summary"]
        if not pairs.empty:
            pairs = pairs.loc[pairs.layer.eq(state)]
            unanimous_pairs = ((pairs.M_positive_pct == 100) & (pairs.coverage_pct == 100)).sum()
            findings.append(f"At that state, {int((pairs.G_mean > 0).sum())}/{len(pairs)} observed concept pairs have positive mean G, "
                            f"{int((pairs.M_mean > 0).sum())}/{len(pairs)} have positive mean M, and "
                            f"{int(unanimous_pairs)} have M > 0 for every member with complete pair coverage.")
    else:
        findings.append(f"Inferred objective state {int(objective.objective_layer)} is absent; no objective-state finding is generated.")

    regions = tables["exp2_cka_region_summary"].set_index("region")
    cross = regions.loc["early_middle_to_deep"]
    if cross.n_layer_pairs:
        reduced = bool(cross.delta_mean < -CHANGE_EPS and cross.delta_median < -CHANGE_EPS and
                       cross.decreased_fraction >= CROSS_DECREASE_FRACTION)
        findings.append(f"Adversarial early/middle-to-deep CKA {'meets' if reduced else 'does not meet'} the consistent-reduction rule: "
                        f"mean change {cross.delta_mean:.4f}, median {cross.delta_median:.4f}, "
                        f"{int(cross.decreased_count)}/{int(cross.n_layer_pairs)} layer pairs decreased "
                        f"({cross.decreased_fraction:.1%}).")
    else:
        findings.append("Early/middle-to-deep consistency has no available layer pairs; the reduction rule is unevaluated.")
    deep = regions.loc["within_deep"]
    if deep.n_layer_pairs:
        strong = bool(deep.adversarial_min >= STRONG_CKA)
        findings.append(f"Deep adversarial layers {'meet' if strong else 'do not meet'} the strong mutual-alignment rule "
                        f"(every off-diagonal CKA >= {STRONG_CKA:g}): mean {deep.adversarial_mean:.4f}, "
                        f"minimum {deep.adversarial_min:.4f}, {deep.adversarial_strong_fraction:.1%} of pairs meet the threshold.")
    else:
        findings.append("Fewer than two deep states are available; deep mutual alignment is unevaluated.")
    combined = tables["cross_experiment_summary"].iloc[0]
    if combined.complementary_later_concentration:
        findings.append("The complementary later-concentration rule is met on matched samples/states: "
                        "deep states have the largest positive block-mean G, and mean absolute CKA change for "
                        f"encoder pairs touching deep states ({combined.deep_touching_mean_abs_delta:.4f}) exceeds "
                        f"that for other encoder pairs ({combined.nondeep_mean_abs_delta:.4f}).")
    else:
        findings.append("The complementary later-concentration rule is not established: "
                        f"matched complete samples/states/provenance={bool(combined.matched_for_combined_finding)}, "
                        f"deep positive mean G largest={bool(combined.deep_has_largest_positive_mean_G)}, "
                        f"deep-touching mean absolute CKA change larger={bool(combined.deep_touching_change_larger)}.")
    return findings


def build_report(tables, exp1_csv, exp2_dir):
    """Report measurements first, followed by mechanically supported findings."""
    validation = tables["exp1_validation"].iloc[0]
    layer = tables["exp1_layer_summary"]
    transition = tables["exp1_transition_summary"].iloc[0]
    objective = tables["exp1_objective_summary"].iloc[0]
    combined = tables["cross_experiment_summary"].iloc[0]
    bootstrap = layer.iloc[0]
    block_text = []
    for experiment in ["exp1", "exp2"]:
        blocks = tables[f"{experiment}_layer_blocks"]
        block_text.append(experiment + ": " + "; ".join(
            f"{name}={group.layer.tolist()}" for name, group in blocks.groupby("block", sort=False)))
    lines = [
        "Experiment Summary", "",
        f"Exp1 input: {Path(exp1_csv).resolve()}", f"Exp2 directory: {Path(exp2_dir).resolve()}",
        f"Exp1: {int(validation.samples)} samples; concept pairs={number(validation.pairs, 0)}; "
        f"states={validation.available_layers}; missing sample/state combinations={int(validation.missing_sample_layer_combinations)}.",
        f"Exp2: {int(combined.exp2_samples)} samples; shared sample IDs={int(combined.common_samples)}; "
        f"provenance mismatches={int(combined.provenance_mismatches)}.",
        "Scope: recorded samples. Exp1 measures paired target-image representation alignment; "
        "Exp2 measures within-condition cross-layer CKA across image observations.",
        "No class-prediction, caption, causal-effect or per-token outcomes are measured.",
        f"Exp1 uncertainty: pointwise 95% percentile {bootstrap.bootstrap_unit} bootstrap, "
        f"{int(bootstrap.bootstrap_reps)} replicates, seed {int(bootstrap.bootstrap_seed)}; "
        "whole units are resampled across states and sample-weighted means retained. "
        "Intervals assume approximately independent, representative units; they do not select transition states.",
        "CKA summaries are descriptive over unique unordered off-diagonal pairs; matrix entries are not independent samples. "
        "No sample-level CKA uncertainty is estimated from these aggregate CSVs.",
        "Blocks: split ordered available encoder states into three near-equal groups; extra states go to earlier groups. "
        "Embedding state 0 is separate. Blocks are set before inspecting matrix values.",
        *block_text,
        f"Rules: CKA change tolerance={CHANGE_EPS:g}; consistent cross-block reduction requires negative mean/median "
        f"beyond tolerance and >= {CROSS_DECREASE_FRACTION:.0%} decreasing pairs. "
        f"Strong deep alignment requires every unique within-deep pair >= {STRONG_CKA:g} (declared descriptive threshold).",
        "Combined later-concentration requires matching sample IDs/image provenance/states and complete Exp1 coverage, "
        "deep block mean G positive and larger than both earlier blocks, and larger mean absolute CKA change for "
        "encoder pairs touching deep states than for other encoder pairs (differences exceed the tolerance).",
        "", "Numerical Measurements", "",
        "Exp1: R=cos(adv,source); T=cos(adv,target); B=cos(source,target); G=T-B; M=T-R. "
        "For unit vectors, squared distance=2(1-cosine): 2G is target-distance improvement and 2M is the source-minus-target distance margin.",
        f"Positive suffixes through state {int(layer.layer.max())}: {int(transition.verified_persistent_samples)}/"
        f"{int(transition.total_samples)} samples; persistent crossing median={number(transition.persistent_state_median, 1)}, "
        f"range=[{number(transition.persistent_state_min, 0)}, {number(transition.persistent_state_max, 0)}].",
        f"Earliest all-sample M > 0 state={number(transition.earliest_all_samples_M_positive, 0)}; "
        f"samples reversing after their first M > 0={int(transition.samples_reversing_after_first_M_positive)}; "
        f"reversals after earliest all-positive state={number(transition.samples_reversing_after_earliest_all_positive, 0)}.",
        "Persistence requires every integer state in the suffix; missing states are not evidence of positivity. "
        "The inferred AMP objective is max recorded state - 1, assuming the final returned model state is recorded.",
    ]
    if objective.available:
        state = int(objective.objective_layer)
        row = layer.set_index("layer").loc[state]
        lines.append(f"Objective state {state}: {int(objective.n_samples)}/{int(objective.total_samples)} samples; "
                     f"G mean/median={objective.G_mean:.4f}/{objective.G_median:.4f}; "
                     f"M mean/median={objective.M_mean:.4f}/{objective.M_median:.4f}; "
                     f"M range=[{objective.M_min:.4f}, {objective.M_max:.4f}].")
        for metric in ["G", "M"]:
            lines.append(f"  {metric}>0: {int(row[f'{metric}_positive_count'])}/{int(row.n_samples)} "
                         f"({row[f'{metric}_positive_pct']:.1f}%); mean 95% interval="
                         f"[{number(row[f'{metric}_ci95_low'])}, {number(row[f'{metric}_ci95_high'])}].")
        lines.append(f"  Normalized squared target-distance reduction G/(1-B): "
                     f"valid={int(objective.distance_reduction_valid)}, excluded={int(objective.distance_reduction_excluded)} "
                     f"(1-B <= {DISTANCE_EPS:g}); mean={number(objective.distance_reduction_mean)}, "
                     f"median={number(objective.distance_reduction_median)}, "
                     f"range=[{number(objective.distance_reduction_min)}, {number(objective.distance_reduction_max)}]. "
                     "These are fractional distance reductions, not success probabilities.")
    else:
        lines.append(f"Objective state {int(objective.objective_layer)}: absent.")
    lines.extend(["", "Exp1 block means (equal weight per recorded state):",
                  tables["exp1_block_summary"].to_string(index=False, float_format=lambda x:f"{x:.4f}")])
    regions = tables["exp2_cka_region_summary"]
    keep = ["all_off_diagonal", "within_early", "within_middle", "within_deep", "early_to_middle",
            "early_to_deep", "middle_to_deep", "early_middle_to_deep", "encoder_pairs_touching_deep", "encoder_pairs_without_deep"]
    columns = ["region", "n_layer_pairs", "source_mean", "adversarial_mean", "target_mean",
               "delta_mean", "delta_median", "delta_abs_mean", "delta_abs_median", "decreased_fraction"]
    lines.extend(["", "Exp2 CKA: delta = adversarial - source (means, signed/absolute changes and decrease fractions):",
                  regions.loc[regions.region.isin(keep), columns].to_string(index=False, float_format=lambda x:f"{x:.4f}"),
                  "", "Strongest signed changes (exact ties: first pair in numerical order):",
                  tables["exp2_cka_extrema"].to_string(index=False, float_format=lambda x:f"{x:.6f}"),
                  "", "Supported Findings", ""])
    lines.extend(f"- {finding}" for finding in supported_findings(tables))
    return "\n".join(lines) + "\n"


def save_analysis(tables, report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=name == "exp2_cka_adv_minus_source")
    (output_dir / "analysis_decisions.txt").write_text(report, encoding="utf-8")
    print(f"Saved {len(tables)} numerical CSVs and analysis_decisions.txt to {output_dir.resolve()}")


def bootstrap_count(value):
    value = int(value)
    if value < 2:
        raise argparse.ArgumentTypeError("Bootstrap replicate count must be at least 2")
    return value


def nonnegative_seed(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Seed must be nonnegative")
    return value


def parse_args():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute numerical Exp1/Exp2 measurements, then generate supported findings; no model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp1-results", type=Path,
                        default=directory / "output/exp1/exp1_layerwise_cosine.csv",
                        help="Per-sample, per-state Exp1 cosine result CSV.")
    parser.add_argument("--exp2-dir", type=Path, default=directory / "output/exp2",
                        help="Directory containing the four Exp2 CKA CSVs and exp2_samples.csv.")
    parser.add_argument("--output-dir", type=Path, default=directory / "output/analysis",
                        help="Directory for derived CSVs and analysis_decisions.txt; created automatically.")
    parser.add_argument("--bootstrap-reps", type=bootstrap_count, default=BOOTSTRAP_REPS,
                        help="Number of Exp1 pair-cluster (or sample-level fallback) bootstrap replicates.")
    parser.add_argument("--seed", type=nonnegative_seed, default=SEED,
                        help="Random seed for deterministic Exp1 bootstrap intervals.")
    return parser.parse_args()


def main():
    args = parse_args()
    tables = measure_all(args.exp1_results, args.exp2_dir, args.bootstrap_reps, args.seed)
    report = build_report(tables, args.exp1_results, args.exp2_dir)
    save_analysis(tables, report, args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
