"""
evaluation.py — Multi-dataset evaluation helpers.

Provides utilities to run a search function over every dataset in a folder
and persist a compact summary CSV.
"""

import os
import pandas as pd


def run_search_over_datasets(
    search_fn, datasets_folder, results_folder, budget, seed=None
):
    """Run *search_fn* on every CSV in *datasets_folder*.

    Parameters
    ----------
    search_fn : callable
        A search function with signature
        ``(file_path, budget, output_file, seed=None) -> (best_solution, best_perf, summary)``.
    datasets_folder : str
        Directory containing dataset CSVs.
    results_folder : str
        Directory where per-dataset search-trace CSVs are saved.
    budget : int
        Measurement budget per dataset.
    seed : int or None
        Base random seed.  Each dataset receives ``seed + i`` so runs are
        reproducible yet distinct.

    Returns
    -------
    summaries : list[dict]
        One summary dict per dataset.
    """
    os.makedirs(results_folder, exist_ok=True)
    summaries = []

    dataset_files = sorted(
        f for f in os.listdir(datasets_folder) if f.endswith(".csv")
    )

    for i, file_name in enumerate(dataset_files):
        file_path = os.path.join(datasets_folder, file_name)
        system_name = file_name.split(".")[0]
        output_file = os.path.join(results_folder, f"{system_name}_search_results.csv")

        dataset_seed = (seed + i) if seed is not None else None
        best_solution, best_perf, summary = search_fn(
            file_path, budget, output_file, seed=dataset_seed
        )
        summary["best_solution"] = best_solution
        summaries.append(summary)

        print(
            f"  {system_name:>12s}  |  best = {best_perf:.4f}  "
            f"|  valid = {summary['valid_count']}  "
            f"|  invalid = {summary['invalid_count']}"
        )

    return summaries


def save_summary_csv(summaries, output_file):
    """Write a list of summary dicts to a CSV file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
