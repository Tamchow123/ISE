"""
run_improved.py — Run the improved BestConfig-style search over all datasets.

Usage:
    python -m scripts.run_improved

Note: This will raise NotImplementedError until bestconfig_search.py is
implemented.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bestconfig_search import run_bestconfig_search
from src.evaluation import run_search_over_datasets, save_summary_csv
from src.visualization import plot_all_search_results

DATASETS_FOLDER = os.path.join("datasets")
RESULTS_BASE = os.path.join("results", "improved_bestconfig")
SEARCH_RESULTS = os.path.join(RESULTS_BASE, "search_results")
VISUALIZATION_RESULTS = os.path.join(RESULTS_BASE, "visualization_results")

BUDGET = 100
SEED = 42


def main():
    print("=== Improved BestConfig Search ===")
    print(f"Budget: {BUDGET}  |  Seed: {SEED}\n")

    summaries = run_search_over_datasets(
        search_fn=run_bestconfig_search,
        datasets_folder=DATASETS_FOLDER,
        results_folder=SEARCH_RESULTS,
        budget=BUDGET,
        seed=SEED,
    )

    summary_file = os.path.join(RESULTS_BASE, "summary.csv")
    save_summary_csv(summaries, summary_file)

    print("\nGenerating visualizations...")
    plot_all_search_results(SEARCH_RESULTS, VISUALIZATION_RESULTS)

    print("\nDone.")


if __name__ == "__main__":
    main()
