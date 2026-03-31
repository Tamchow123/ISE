"""
run_stat_tests.py - Run paired Wilcoxon tests for repeated experiment results.

Usage:
    python -m scripts.run_stat_tests
"""

from pathlib import Path

import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError as exc:
    raise ImportError(
        "SciPy is required for Wilcoxon tests. Install it with: pip install scipy"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.csv"
OUTPUT_DIR = RESULTS_DIR / "stat_tests"
OUTPUT_FILE = OUTPUT_DIR / "wilcoxon_results.csv"

BASELINE_ALGORITHM = "baseline_random_search"
BESTCONFIG_ALGORITHM = "improved_bestconfig"
ALGORITHM_ALIASES = {
    "baseline_random_search": BASELINE_ALGORITHM,
    "bestconfig": BESTCONFIG_ALGORITHM,
    "improved_bestconfig": BESTCONFIG_ALGORITHM,
}
REQUIRED_COLUMNS = {
    "algorithm",
    "dataset",
    "budget",
    "seed",
    "best_performance",
}
ROUND_DECIMALS = 12
ALPHA = 0.05


def ensure_required_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def ensure_required_columns(df, required_columns, file_path):
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def canonical_algorithm_name(name):
    normalized_name = str(name).strip()
    return ALGORITHM_ALIASES.get(normalized_name, normalized_name)


def load_raw_results():
    ensure_required_file(RAW_RESULTS_FILE)
    raw_results = pd.read_csv(RAW_RESULTS_FILE)
    ensure_required_columns(raw_results, REQUIRED_COLUMNS, RAW_RESULTS_FILE)

    prepared = raw_results.copy()
    prepared["algorithm"] = prepared["algorithm"].map(canonical_algorithm_name)
    prepared["seed"] = pd.to_numeric(prepared["seed"], errors="coerce")
    prepared["best_performance"] = pd.to_numeric(
        prepared["best_performance"], errors="coerce"
    )
    prepared = prepared.dropna(subset=["seed", "best_performance"])
    prepared["seed"] = prepared["seed"].astype(int)
    return prepared


def deduplicate_seed_runs(runs, algorithm_name, dataset_name, budget):
    duplicate_seed_mask = runs["seed"].duplicated(keep=False)
    if not duplicate_seed_mask.any():
        return runs, None

    duplicated_seeds = sorted(runs.loc[duplicate_seed_mask, "seed"].unique().tolist())
    deduplicated = runs.drop_duplicates(subset="seed", keep="first").copy()
    note = (
        f"duplicate {algorithm_name} seeds collapsed by keeping first occurrence: "
        f"{duplicated_seeds}"
    )
    print(
        f"Warning: {dataset_name} budget={budget} has duplicate {algorithm_name} seeds; "
        "keeping the first row for each duplicated seed."
    )
    return deduplicated, note


def extract_algorithm_runs(group_df, algorithm_name, dataset_name, budget):
    runs = group_df.loc[
        group_df["algorithm"] == algorithm_name,
        ["seed", "best_performance"],
    ].copy()
    runs = runs.sort_values("seed").reset_index(drop=True)
    return deduplicate_seed_runs(runs, algorithm_name, dataset_name, budget)


def align_paired_runs(group_df, dataset_name, budget):
    baseline_runs, baseline_note = extract_algorithm_runs(
        group_df,
        BASELINE_ALGORITHM,
        dataset_name,
        budget,
    )
    bestconfig_runs, bestconfig_note = extract_algorithm_runs(
        group_df,
        BESTCONFIG_ALGORITHM,
        dataset_name,
        budget,
    )

    baseline_runs = baseline_runs.rename(
        columns={"best_performance": "baseline_best_performance"}
    )
    bestconfig_runs = bestconfig_runs.rename(
        columns={"best_performance": "bestconfig_best_performance"}
    )

    paired = pd.merge(
        baseline_runs,
        bestconfig_runs,
        on="seed",
        how="inner",
        validate="one_to_one",
    ).sort_values("seed")

    missing_in_baseline = sorted(
        set(bestconfig_runs["seed"].tolist()) - set(baseline_runs["seed"].tolist())
    )
    missing_in_bestconfig = sorted(
        set(baseline_runs["seed"].tolist()) - set(bestconfig_runs["seed"].tolist())
    )

    notes = []
    if baseline_note:
        notes.append(baseline_note)
    if bestconfig_note:
        notes.append(bestconfig_note)
    if baseline_runs.empty:
        notes.append("missing baseline_random_search rows")
    if bestconfig_runs.empty:
        notes.append("missing improved_bestconfig rows")
    if missing_in_baseline:
        notes.append(f"missing baseline seeds: {missing_in_baseline}")
    if missing_in_bestconfig:
        notes.append(f"missing bestconfig seeds: {missing_in_bestconfig}")
    if len(baseline_runs) != len(bestconfig_runs):
        notes.append(
            "unequal run counts before alignment "
            f"(baseline={len(baseline_runs)}, bestconfig={len(bestconfig_runs)})"
        )

    return paired, "; ".join(notes)


def safe_wilcoxon_test(differences):
    if differences.empty:
        return float("nan"), float("nan"), "no paired seeds available"

    if (differences == 0).all():
        return 0.0, 1.0, "all paired differences are zero"

    result = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue), None


def winner_from_result(p_value, median_paired_difference):
    if pd.isna(p_value) or p_value >= ALPHA:
        return "no_significant_difference"
    if median_paired_difference > 0:
        return BESTCONFIG_ALGORITHM
    if median_paired_difference < 0:
        return BASELINE_ALGORITHM
    return "no_significant_difference"


def apply_holm_correction(results_df):
    corrected = results_df.copy()
    corrected["holm_p_value"] = float("nan")
    corrected["significant_holm_0_05"] = False
    corrected["winner_holm"] = "no_significant_difference"

    valid_mask = corrected["p_value"].notna()
    if not valid_mask.any():
        return corrected

    ordered = corrected.loc[valid_mask, ["p_value"]].sort_values("p_value")
    valid_count = len(ordered)
    adjusted_values = []
    running_max = 0.0

    for rank, p_value in enumerate(ordered["p_value"].tolist(), start=1):
        scaled_p_value = (valid_count - rank + 1) * float(p_value)
        running_max = max(running_max, scaled_p_value)
        adjusted_values.append(min(running_max, 1.0))

    corrected.loc[ordered.index, "holm_p_value"] = adjusted_values
    corrected["significant_holm_0_05"] = corrected["holm_p_value"] < ALPHA
    corrected["winner_holm"] = corrected.apply(
        lambda row: winner_from_result(
            row["holm_p_value"],
            row["median_paired_difference"],
        ),
        axis=1,
    )
    return corrected


def summarize_pair(dataset_name, budget, paired, paired_note):
    n_pairs = len(paired)
    baseline_values = paired["baseline_best_performance"]
    bestconfig_values = paired["bestconfig_best_performance"]
    differences = (baseline_values - bestconfig_values).round(ROUND_DECIMALS)

    baseline_median = float(baseline_values.median()) if n_pairs else float("nan")
    bestconfig_median = float(bestconfig_values.median()) if n_pairs else float("nan")
    baseline_mean = float(baseline_values.mean()) if n_pairs else float("nan")
    bestconfig_mean = float(bestconfig_values.mean()) if n_pairs else float("nan")
    median_paired_difference = float(differences.median()) if n_pairs else float("nan")
    mean_paired_difference = float(differences.mean()) if n_pairs else float("nan")
    statistic, p_value, test_note = safe_wilcoxon_test(differences)

    note_parts = [note for note in [paired_note, test_note] if note]
    winner = winner_from_result(p_value, median_paired_difference)
    significant = bool(not pd.isna(p_value) and p_value < ALPHA)

    result_row = {
        "dataset": dataset_name,
        "budget": budget,
        "n_pairs": n_pairs,
        "baseline_median": baseline_median,
        "bestconfig_median": bestconfig_median,
        "baseline_mean": baseline_mean,
        "bestconfig_mean": bestconfig_mean,
        "median_paired_difference": median_paired_difference,
        "mean_paired_difference": mean_paired_difference,
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_0_05": significant,
        "winner": winner,
        "note": "; ".join(note_parts),
    }
    return result_row


def print_console_summary(result_row):
    dataset_name = result_row["dataset"]
    budget = result_row["budget"]
    p_value = result_row["p_value"]
    holm_p_value = result_row.get("holm_p_value", float("nan"))
    statistic = result_row["statistic"]
    winner_holm = result_row.get("winner_holm", "no_significant_difference")
    p_value_text = "nan" if pd.isna(p_value) else f"{p_value:.6f}"
    holm_p_value_text = "nan" if pd.isna(holm_p_value) else f"{holm_p_value:.6f}"
    statistic_text = "nan" if pd.isna(statistic) else f"{statistic:.4f}"

    summary = (
        f"{dataset_name:>12s} | budget={int(budget):>3d} | pairs={result_row['n_pairs']:>2d} "
        f"| baseline_median={result_row['baseline_median']:.6g} "
        f"| bestconfig_median={result_row['bestconfig_median']:.6g} "
        f"| median_diff={result_row['median_paired_difference']:.6g} "
        f"| statistic={statistic_text} | p={p_value_text} "
        f"| holm_p={holm_p_value_text} "
        f"| winner={result_row['winner']} "
        f"| winner_holm={winner_holm}"
    )
    if result_row["note"]:
        summary += f" | note={result_row['note']}"
    print(summary)


def run_stat_tests(raw_results):
    result_rows = []
    grouped = raw_results.groupby(["dataset", "budget"], sort=True)

    for (dataset_name, budget), group_df in grouped:
        paired, paired_note = align_paired_runs(group_df, dataset_name, budget)
        result_row = summarize_pair(dataset_name, budget, paired, paired_note)
        result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows).sort_values(["dataset", "budget"])
    result_df = apply_holm_correction(result_df)

    for result_row in result_df.to_dict(orient="records"):
        print_console_summary(result_row)

    return result_df.reset_index(drop=True)


def main():
    raw_results = load_raw_results()
    results = run_stat_tests(raw_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_output_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}__tmp.csv")
    results.to_csv(temporary_output_file, index=False)
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    temporary_output_file.replace(OUTPUT_FILE)

    print(f"\nWilcoxon test results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()