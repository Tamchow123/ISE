"""
bestconfig_search.py — Improved BestConfig-style search (placeholder).

This module will implement a smarter configuration search strategy based on
the BestConfig algorithm (divide-and-diverge).  It shares the same interface
as ``random_search.run_random_search`` so it can be swapped in easily for
experiments.

TODO: Implement the improved search algorithm.
"""

import os


def run_bestconfig_search(file_path, budget, output_file, seed=None):
    """Run BestConfig-style search on a single dataset.

    Parameters
    ----------
    file_path : str
        Path to the dataset CSV.
    budget : int
        Number of valid measurements to perform.
    output_file : str
        Path where the search-trace CSV will be saved.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_solution : list[int]
        Configuration parameter values of the best solution found.
    best_performance : float
        Performance value of the best solution.
    summary : dict
        Additional metadata.

    Raises
    ------
    NotImplementedError
        Always — this is a placeholder.
    """
    raise NotImplementedError(
        "BestConfig search is not yet implemented. "
        "Replace this stub with the divide-and-diverge algorithm."
    )
