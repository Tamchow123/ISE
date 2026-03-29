"""
run_experiments.py — Compare baseline and improved search methods.

Usage:
    python -m scripts.run_experiments

Runs each registered search method over all datasets and saves summary
CSVs side-by-side for easy comparison.  Extend METHODS below as new
algorithms are implemented.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.random_search import run_random_search
from src.evaluation import run_search_over_datasets, save_summary_csv
from src.visualization import plot_all_search_results

# Register search methods here.
# Each entry: (label, search_fn, results_subfolder)
METHODS = [
    ("Baseline Random Search", run_random_search, "baseline_random_search"),
    # Uncomment when bestconfig_search is implemented:
    # ("Improved BestConfig", run_bestconfig_search, "improved_bestconfig"),
]

DATASETS_FOLDER = os.path.join("datasets")
BUDGET = 100
SEED = 42


def main():
    print("=== Experiment Runner ===\n")

    for label, search_fn, subfolder in METHODS:
        results_base = os.path.join("results", subfolder)
        search_results = os.path.join(results_base, "search_results")
        viz_results = os.path.join(results_base, "visualization_results")

        print(f"--- {label} ---")
        summaries = run_search_over_datasets(
            search_fn=search_fn,
            datasets_folder=DATASETS_FOLDER,
            results_folder=search_results,
            budget=BUDGET,
            seed=SEED,
        )

        summary_file = os.path.join(results_base, "summary.csv")
        save_summary_csv(summaries, summary_file)

        print("Generating visualizations...")
        plot_all_search_results(search_results, viz_results)
        print()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
