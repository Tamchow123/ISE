"""
build_results_table.py - Build the report-facing table from experiment summaries
and Holm-adjusted Wilcoxon results.

Usage:
    python -m scripts.build_results_table
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
AGGREGATED_RESULTS_FILE = RESULTS_DIR / "aggregated_results.csv"
STAT_TESTS_FILE = RESULTS_DIR / "stat_tests" / "wilcoxon_results.csv"
OUTPUT_DIR = RESULTS_DIR / "report_tables"
OUTPUT_CSV_FILE = OUTPUT_DIR / "final_results_table.csv"
OUTPUT_MD_FILE = OUTPUT_DIR / "final_results_table.md"

BASELINE_ALGORITHM = "baseline_random_search"
BESTCONFIG_ALGORITHM = "improved_bestconfig"
ALGORITHM_ALIASES = {
    "baseline_random_search": BASELINE_ALGORITHM,
    "bestconfig": BESTCONFIG_ALGORITHM,
    "improved_bestconfig": BESTCONFIG_ALGORITHM,
}
AGGREGATED_REQUIRED_COLUMNS = {
    "algorithm",
    "dataset",
    "budget",
    "median_best_performance",
    "q1_best_performance",
    "q3_best_performance",
}
STAT_TEST_REQUIRED_COLUMNS = {
    "dataset",
    "budget",
    "holm_p_value",
    "significant_holm_0_05",
    "winner_holm",
}
OUTPUT_COLUMNS = [
    "dataset",
    "budget",
    "baseline_median",
    "bestconfig_median",
    "baseline_q1",
    "baseline_q3",
    "bestconfig_q1",
    "bestconfig_q3",
    "holm_p_value",
    "significant_holm_0_05",
    "winner_holm",
]
ROUND_DECIMALS = 4
SMALL_NUMBER_THRESHOLD = 10 ** (-ROUND_DECIMALS)


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


def load_aggregated_results():
    ensure_required_file(AGGREGATED_RESULTS_FILE)
    aggregated_results = pd.read_csv(AGGREGATED_RESULTS_FILE)
    ensure_required_columns(
        aggregated_results,
        AGGREGATED_REQUIRED_COLUMNS,
        AGGREGATED_RESULTS_FILE,
    )

    prepared = aggregated_results.copy()
    prepared["algorithm"] = prepared["algorithm"].map(canonical_algorithm_name)
    prepared["budget"] = pd.to_numeric(prepared["budget"], errors="coerce")
    for column_name in [
        "median_best_performance",
        "q1_best_performance",
        "q3_best_performance",
    ]:
        prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")

    prepared = prepared.dropna(subset=["dataset", "budget"])
    return prepared


def load_stat_tests():
    ensure_required_file(STAT_TESTS_FILE)
    stat_tests = pd.read_csv(STAT_TESTS_FILE)
    ensure_required_columns(stat_tests, STAT_TEST_REQUIRED_COLUMNS, STAT_TESTS_FILE)

    prepared = stat_tests.copy()
    prepared["budget"] = pd.to_numeric(prepared["budget"], errors="coerce")
    prepared["holm_p_value"] = pd.to_numeric(prepared["holm_p_value"], errors="coerce")
    prepared = prepared.dropna(subset=["dataset", "budget"])
    return prepared


def select_algorithm_summary(aggregated_results, algorithm_name, prefix):
    selected = aggregated_results.loc[
        aggregated_results["algorithm"] == algorithm_name,
        [
            "dataset",
            "budget",
            "median_best_performance",
            "q1_best_performance",
            "q3_best_performance",
        ],
    ].copy()

    duplicate_mask = selected.duplicated(subset=["dataset", "budget"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = selected.loc[duplicate_mask, ["dataset", "budget"]]
        raise ValueError(
            "Duplicate aggregated rows found for "
            f"{algorithm_name}: {duplicate_rows.to_dict(orient='records')}"
        )

    return selected.rename(
        columns={
            "median_best_performance": f"{prefix}_median",
            "q1_best_performance": f"{prefix}_q1",
            "q3_best_performance": f"{prefix}_q3",
        }
    )


def build_order_frame(aggregated_results):
    target_algorithms = {BASELINE_ALGORITHM, BESTCONFIG_ALGORITHM}
    ordered_pairs = aggregated_results.loc[
        aggregated_results["algorithm"].isin(target_algorithms),
        ["dataset", "budget"],
    ].drop_duplicates(ignore_index=True)

    if ordered_pairs.empty:
        raise ValueError("No baseline or improved_bestconfig rows found in aggregated_results.csv")

    return ordered_pairs


def round_output_values(results_table):
    normalized = results_table.copy()

    if pd.api.types.is_float_dtype(normalized["budget"]):
        integral_mask = normalized["budget"].dropna().map(float.is_integer)
        if integral_mask.all():
            normalized["budget"] = normalized["budget"].astype("Int64")

    return normalized


def build_results_table(aggregated_results, stat_tests):
    ordered_pairs = build_order_frame(aggregated_results)
    baseline_summary = select_algorithm_summary(
        aggregated_results,
        BASELINE_ALGORITHM,
        "baseline",
    )
    bestconfig_summary = select_algorithm_summary(
        aggregated_results,
        BESTCONFIG_ALGORITHM,
        "bestconfig",
    )
    stat_summary = stat_tests.loc[
        :,
        [
            "dataset",
            "budget",
            "holm_p_value",
            "significant_holm_0_05",
            "winner_holm",
        ],
    ].copy()

    duplicate_mask = stat_summary.duplicated(subset=["dataset", "budget"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = stat_summary.loc[duplicate_mask, ["dataset", "budget"]]
        raise ValueError(
            "Duplicate Wilcoxon rows found: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    results_table = ordered_pairs.merge(
        baseline_summary,
        on=["dataset", "budget"],
        how="left",
        validate="one_to_one",
    )
    results_table = results_table.merge(
        bestconfig_summary,
        on=["dataset", "budget"],
        how="left",
        validate="one_to_one",
    )
    results_table = results_table.merge(
        stat_summary,
        on=["dataset", "budget"],
        how="left",
        validate="one_to_one",
    )

    results_table = results_table.loc[:, OUTPUT_COLUMNS]
    return round_output_values(results_table)


def format_cell(value):
    if pd.isna(value):
        return ""
    if type(value) is bool:
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if value.is_integer():
            return str(int(value))
        if abs(value) < SMALL_NUMBER_THRESHOLD:
            return f"{value:.4g}"
        return f"{value:.{ROUND_DECIMALS}f}".rstrip("0").rstrip(".")
    return str(value)


def build_display_table(results_table):
    return results_table.map(format_cell)


def build_markdown_table(results_table):
    display_table = build_display_table(results_table)
    rows = [OUTPUT_COLUMNS]
    for row in display_table.itertuples(index=False, name=None):
        rows.append(list(row))

    column_widths = [
        max(len(str(row[column_index])) for row in rows)
        for column_index in range(len(OUTPUT_COLUMNS))
    ]

    def render_row(values):
        padded_values = [
            str(value).ljust(column_widths[index])
            for index, value in enumerate(values)
        ]
        return "| " + " | ".join(padded_values) + " |"

    header = render_row(OUTPUT_COLUMNS)
    separator = "| " + " | ".join("-" * width for width in column_widths) + " |"
    body = [render_row(row) for row in rows[1:]]
    return "\n".join([header, separator, *body]) + "\n"


def save_outputs(results_table):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_display_table(results_table).to_csv(OUTPUT_CSV_FILE, index=False)
    OUTPUT_MD_FILE.write_text(build_markdown_table(results_table), encoding="utf-8")


def main():
    aggregated_results = load_aggregated_results()
    stat_tests = load_stat_tests()
    results_table = build_results_table(aggregated_results, stat_tests)
    save_outputs(results_table)

    print(f"Final results table saved to {OUTPUT_CSV_FILE}")
    print(f"Markdown table saved to {OUTPUT_MD_FILE}")
    print(f"Rows written: {len(results_table)}")


if __name__ == "__main__":
    main()