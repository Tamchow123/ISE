"""
visualization.py — Plot search-trace results.

Generates per-dataset performance-over-iterations plots from the CSV
files produced by the search algorithms.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd

from src.data import is_maximization


def plot_search_results(csv_file, output_image, dataset_name=None):
    """Generate a performance-over-iterations plot for a single search trace.

    Parameters
    ----------
    csv_file : str
        Path to a search-results CSV (must have a ``Performance`` column).
    output_image : str
        Path where the PNG image will be saved.
    dataset_name : str or None
        Label used in the plot title; inferred from *csv_file* if None.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} does not exist — skipping.")
        return

    df = pd.read_csv(csv_file)

    if dataset_name is None:
        dataset_name = os.path.basename(csv_file).replace("_search_results.csv", "")

    maximization = is_maximization(dataset_name)

    if maximization:
        best_performance = df["Performance"].max()
    else:
        best_performance = df["Performance"].min()

    best_index = df[df["Performance"] == best_performance].index[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Performance"], marker="o", markersize=3, linestyle="-", label="Performance")
    plt.plot(best_index, best_performance, marker="*", color="red", markersize=12, label="Best Point")
    plt.xlabel("Sample (iteration)", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.title(f"Search Results — {dataset_name}", fontsize=16)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image)
    plt.close()
    print(f"  Saved plot: {output_image}")


def plot_all_search_results(results_folder, visualization_folder):
    """Generate plots for every ``*_search_results.csv`` in *results_folder*."""
    if not os.path.exists(results_folder):
        print(f"Error: {results_folder} does not exist.")
        return

    for file_name in sorted(os.listdir(results_folder)):
        if file_name.endswith("_search_results.csv"):
            dataset_name = file_name.replace("_search_results.csv", "")
            csv_file = os.path.join(results_folder, file_name)
            output_image = os.path.join(visualization_folder, f"{dataset_name}_visualization.png")
            plot_search_results(csv_file, output_image, dataset_name)
