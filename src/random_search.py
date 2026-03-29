"""
random_search.py — Baseline random-search algorithm.

Randomly samples configurations from the valid parameter domain and
evaluates them against the dataset.  Invalid configurations (those not
present in the dataset) are assigned a worst-case performance value but
do NOT consume the measurement budget.
"""

import os
import numpy as np
import pandas as pd

from src.data import (
    load_dataset,
    get_config_columns_and_performance_column,
    is_maximization,
    get_worst_value,
    sample_random_configuration,
    lookup_configuration_performance,
    system_name_from_path,
)


def _is_better(candidate_perf, best_perf, maximization):
    if maximization:
        return candidate_perf > best_perf
    return candidate_perf < best_perf


def _save_search_trace(output_file, config_columns, search_results):
    if output_file is None:
        return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)


def run_random_search(file_path, budget, output_file, seed=None):
    """Run random search on a single dataset.

    Parameters
    ----------
    file_path : str
        Path to the dataset CSV.
    budget : int
        Number of *valid* measurements to perform.
    output_file : str or None
        Path where the search-trace CSV will be saved. If None, the trace is
        not written.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_solution : list[int]
        Configuration parameter values of the best solution found.
    best_performance : float
        Performance value of the best solution.
    summary : dict
        Additional metadata (total_samples, valid_count, invalid_count).
    """
    rng = np.random.default_rng(seed)

    df = load_dataset(file_path)
    config_columns, performance_column = get_config_columns_and_performance_column(df)
    system_name = system_name_from_path(file_path)
    maximization = is_maximization(system_name)
    worst_value = get_worst_value(df, performance_column, maximization)

    best_performance = -np.inf if maximization else np.inf
    best_solution = []

    search_results = []
    valid_count = 0
    invalid_count = 0

    while valid_count < budget:
        config = sample_random_configuration(df, config_columns, rng)
        perf, is_valid = lookup_configuration_performance(
            df, config_columns, performance_column, config
        )

        if is_valid:
            valid_count += 1
        else:
            # Invalid configuration — assign worst-case, do NOT consume budget
            perf = worst_value
            invalid_count += 1

        # Update the best solution
        if _is_better(perf, best_performance, maximization):
            best_performance = perf
            best_solution = config

        search_results.append(config + [perf])

    # Save search trace only when requested by the caller.
    _save_search_trace(output_file, config_columns, search_results)

    summary = {
        "system": system_name,
        "budget": budget,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "total_samples": valid_count + invalid_count,
        "best_performance": best_performance,
    }

    return [int(x) for x in best_solution], best_performance, summary
