"""
run_experiments.py — Run repeated experiments for fair algorithm comparison.

Usage:
    python -m scripts.run_experiments
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import (
    aggregate_results,
    generate_seeds,
    run_repeated_experiments,
)
from src.random_search import run_random_search

# Uncomment later when the improved algorithm is implemented.
# from src.bestconfig_search import run_bestconfig_search

# Register algorithms here as (algorithm_name, search_fn).
METHODS = [
    ("baseline_random_search", run_random_search),
    # ("improved_bestconfig", run_bestconfig_search),
]

DATASETS_FOLDER = os.path.join("datasets")
RESULTS_BASE = os.path.join("results", "experiments")

BUDGETS = [50, 100]
N_RUNS = 30
BASE_SEED = 42
SAVE_TRACES = False


def main():
    print("=== Repeated Experiment Runner ===")
    print(f"Methods: {', '.join(name for name, _ in METHODS)}")
    print(f"Budgets: {BUDGETS}")
    print(f"Runs per setting: {N_RUNS}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Save traces: {SAVE_TRACES}\n")

    seeds = generate_seeds(BASE_SEED, N_RUNS)
    raw_results = run_repeated_experiments(
        methods=METHODS,
        datasets_folder=DATASETS_FOLDER,
        results_base=RESULTS_BASE,
        budgets=BUDGETS,
        seeds=seeds,
        save_traces=SAVE_TRACES,
    )
    aggregated_results = aggregate_results(raw_results)

    os.makedirs(RESULTS_BASE, exist_ok=True)
    raw_results_file = os.path.join(RESULTS_BASE, "raw_results.csv")
    aggregated_results_file = os.path.join(RESULTS_BASE, "aggregated_results.csv")

    raw_results.to_csv(raw_results_file, index=False)
    aggregated_results.to_csv(aggregated_results_file, index=False)

    print(f"\nRaw results saved to {raw_results_file}")
    print(f"Aggregated results saved to {aggregated_results_file}")
    if SAVE_TRACES:
        print(f"Trace CSVs saved under {os.path.join(RESULTS_BASE, 'traces')}")


if __name__ == "__main__":
    main()
