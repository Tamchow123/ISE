"""
evaluation.py — Multi-dataset evaluation helpers.

Provides utilities to run search functions over every dataset, collect raw
results for repeated stochastic runs, and aggregate summary statistics.
"""

import os
from pathlib import Path

import pandas as pd

from src.data import is_maximization


def generate_seeds(base_seed, n_runs):
    """Return a deterministic list of seeds for repeated experiments."""
    if n_runs < 1:
        return []
    if base_seed is None:
        return list(range(n_runs))
    return [base_seed + i for i in range(n_runs)]


def _iter_dataset_files(datasets_folder):
    datasets_path = Path(datasets_folder)
    return sorted(path for path in datasets_path.iterdir() if path.suffix == ".csv")


def _resolve_dataset_seed(seed, dataset_index, offset_seed_by_dataset):
    if seed is None:
        return None
    if offset_seed_by_dataset:
        return seed + dataset_index
    return seed


def _validate_experiment_inputs(methods, budgets, seeds):
    if not methods:
        raise ValueError("methods must contain at least one algorithm")
    if not budgets:
        raise ValueError("budgets must contain at least one budget")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")


def _trace_folder_for_run(results_path, algorithm, budget, save_traces):
    if not save_traces:
        return None
    return results_path / "traces" / algorithm / str(budget)


def _raw_record_from_summary(algorithm, budget, seed, summary):
    return {
        "algorithm": algorithm,
        "dataset": summary["system"],
        "budget": budget,
        "seed": seed,
        "best_performance": summary["best_performance"],
        "best_solution": summary["best_solution"],
        "valid_count": summary["valid_count"],
        "invalid_count": summary["invalid_count"],
        "total_samples": summary["total_samples"],
    }


def _default_trace_file_name(system_name, dataset_seed):
    return f"{system_name}_search_results.csv"


def _seed_trace_file_name(system_name, dataset_seed):
    return f"{system_name}__seed_{dataset_seed}.csv"


def run_search_over_datasets(
    search_fn,
    datasets_folder,
    results_folder,
    budget,
    seed=None,
    save_traces=True,
    offset_seed_by_dataset=True,
    trace_file_name_fn=None,
):
    """Run *search_fn* on every CSV in *datasets_folder*.

    Parameters
    ----------
    search_fn : callable
        A search function with signature
        ``(file_path, budget, output_file, seed=None) -> (best_solution, best_perf, summary)``.
    datasets_folder : str or path-like
        Directory containing dataset CSVs.
    results_folder : str or path-like or None
        Directory where per-dataset search-trace CSVs are saved.
    budget : int
        Measurement budget per dataset.
    seed : int or None
        Base random seed.
    save_traces : bool
        Whether to write per-dataset search-trace CSVs.
    offset_seed_by_dataset : bool
        If true, dataset ``i`` receives ``seed + i``. This keeps the previous
        baseline behavior. For repeated experiments, set this to false so the
        recorded seed matches the actual run seed.
    trace_file_name_fn : callable or None
        Optional function ``(system_name, dataset_seed) -> file_name`` used
        when traces are written.

    Returns
    -------
    summaries : list[dict]
        One summary dict per dataset.
    """
    if save_traces and results_folder is None:
        raise ValueError("results_folder is required when save_traces=True")

    results_path = Path(results_folder) if results_folder is not None else None
    if save_traces:
        results_path.mkdir(parents=True, exist_ok=True)
        if trace_file_name_fn is None:
            trace_file_name_fn = _default_trace_file_name

    summaries = []

    for i, dataset_path in enumerate(_iter_dataset_files(datasets_folder)):
        system_name = dataset_path.stem
        dataset_seed = _resolve_dataset_seed(seed, i, offset_seed_by_dataset)

        output_file = None
        if save_traces:
            trace_file_name = trace_file_name_fn(system_name, dataset_seed)
            output_file = str(results_path / trace_file_name)

        best_solution, best_perf, summary = search_fn(
            str(dataset_path), budget, output_file, seed=dataset_seed
        )
        summary = dict(summary)
        summary["best_solution"] = best_solution
        summary["seed"] = dataset_seed
        summaries.append(summary)

        print(
            f"  {system_name:>12s}  |  best = {best_perf:.4f}  "
            f"|  valid = {summary['valid_count']}  "
            f"|  invalid = {summary['invalid_count']}"
        )

    return summaries


def run_repeated_experiments(
    methods, datasets_folder, results_base, budgets, seeds, save_traces=False
):
    """Run repeated experiments for all methods, budgets, and seeds.

    Parameters
    ----------
    methods : list[tuple[str, callable]]
        Each entry is ``(algorithm_name, search_fn)``.
    datasets_folder : str or path-like
        Directory containing dataset CSVs.
    results_base : str or path-like
        Base output directory used for optional trace storage.
    budgets : iterable[int]
        Budgets to evaluate.
    seeds : iterable[int]
        Seeds to evaluate.
    save_traces : bool
        Whether to save per-run search traces.

    Returns
    -------
    pandas.DataFrame
        One raw row per ``(algorithm, dataset, seed, budget)``.
    """
    methods = list(methods)
    budgets = list(budgets)
    seeds = list(seeds)

    _validate_experiment_inputs(methods, budgets, seeds)

    results_path = Path(results_base)
    results_path.mkdir(parents=True, exist_ok=True)

    raw_records = []

    for algorithm, search_fn in methods:
        print(f"\n=== {algorithm} ===")
        for budget in budgets:
            print(f"Budget: {budget}")
            for run_index, seed in enumerate(seeds, start=1):
                print(f"Run {run_index:>2d}/{len(seeds)}  |  seed = {seed}")

                trace_folder = _trace_folder_for_run(
                    results_path, algorithm, budget, save_traces
                )

                summaries = run_search_over_datasets(
                    search_fn=search_fn,
                    datasets_folder=datasets_folder,
                    results_folder=trace_folder,
                    budget=budget,
                    seed=seed,
                    save_traces=save_traces,
                    offset_seed_by_dataset=False,
                    trace_file_name_fn=_seed_trace_file_name,
                )
                for summary in summaries:
                    raw_records.append(
                        _raw_record_from_summary(algorithm, budget, seed, summary)
                    )

    raw_results_df = pd.DataFrame(raw_records)
    if raw_results_df.empty:
        return raw_results_df

    column_order = [
        "algorithm",
        "dataset",
        "budget",
        "seed",
        "best_performance",
        "best_solution",
        "valid_count",
        "invalid_count",
        "total_samples",
    ]
    return raw_results_df[column_order].sort_values(
        ["algorithm", "dataset", "budget", "seed"]
    ).reset_index(drop=True)


def aggregate_results(raw_results_df):
    """Aggregate repeated-run raw results by algorithm, dataset, and budget."""
    if raw_results_df.empty:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "dataset",
                "budget",
                "n_runs",
                "median_best_performance",
                "mean_best_performance",
                "std_best_performance",
                "min_best_performance",
                "max_best_performance",
                "q1_best_performance",
                "q3_best_performance",
                "median_valid_count",
                "median_invalid_count",
                "median_total_samples",
            ]
        )

    aggregated = (
        raw_results_df.groupby(["algorithm", "dataset", "budget"], sort=True)
        .agg(
            n_runs=("best_performance", "size"),
            median_best_performance=("best_performance", "median"),
            mean_best_performance=("best_performance", "mean"),
            std_best_performance=(
                "best_performance",
                lambda values: values.std(ddof=1),
            ),
            min_best_performance=("best_performance", "min"),
            max_best_performance=("best_performance", "max"),
            q1_best_performance=(
                "best_performance",
                lambda values: values.quantile(0.25),
            ),
            q3_best_performance=(
                "best_performance",
                lambda values: values.quantile(0.75),
            ),
            median_valid_count=("valid_count", "median"),
            median_invalid_count=("invalid_count", "median"),
            median_total_samples=("total_samples", "median"),
        )
        .reset_index()
    )

    return aggregated.sort_values(["algorithm", "dataset", "budget"]).reset_index(
        drop=True
    )


def compute_win_counts(raw_results_df):
    """Compute win counts from per-dataset median best performance.

    For each ``(dataset, budget)`` pair, the winning algorithm is the one with
    the best median performance across repeated runs. Ties are counted as wins
    for all tied algorithms.
    """
    if raw_results_df.empty:
        return pd.DataFrame(columns=["algorithm", "n_wins"])

    medians = (
        raw_results_df.groupby(["algorithm", "dataset", "budget"], sort=True)
        .agg(median_best_performance=("best_performance", "median"))
        .reset_index()
    )

    winner_records = []
    for (dataset, budget), group in medians.groupby(["dataset", "budget"], sort=True):
        maximization = is_maximization(dataset)
        if maximization:
            winning_value = group["median_best_performance"].max()
        else:
            winning_value = group["median_best_performance"].min()

        winners = group[group["median_best_performance"] == winning_value]
        for algorithm in winners["algorithm"]:
            winner_records.append({"algorithm": algorithm, "dataset": dataset, "budget": budget})

    winner_df = pd.DataFrame(winner_records)
    if winner_df.empty:
        return pd.DataFrame(columns=["algorithm", "n_wins"])

    return (
        winner_df.groupby("algorithm", sort=True)
        .size()
        .reset_index(name="n_wins")
        .sort_values(["n_wins", "algorithm"], ascending=[False, True])
        .reset_index(drop=True)
    )


def save_summary_csv(summaries, output_file):
    """Write a list of summary dicts to a CSV file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")
