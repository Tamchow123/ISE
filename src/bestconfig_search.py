"""
bestconfig_search.py — Simple BestConfig-style search.

This module implements a lightweight, coursework-level approximation of the
BestConfig idea for discrete dataset-based tuning. It alternates between:

1. DDS-inspired global exploration using stratified sampling over each option's
   observed domain.
2. RBS-inspired local search inside a bounded neighborhood around the current
   best valid configuration.
"""

import os

import numpy as np
import pandas as pd

from src.data import (
    get_config_columns_and_performance_column,
    get_worst_value,
    is_maximization,
    load_dataset,
    system_name_from_path,
)

NUM_STRATA = 3
GLOBAL_SAMPLES_PER_ROUND = 12
LOCAL_SAMPLES_PER_ROUND = 12
MAX_LOCAL_PERTURB_DIMS = 3
MAX_CANDIDATE_GENERATION_MULTIPLIER = 10


def _is_better(candidate_perf, best_perf, maximization):
    if maximization:
        return candidate_perf > best_perf
    return candidate_perf < best_perf


def _build_ordered_domains(df, config_columns):
    return {
        column: sorted(int(value) for value in df[column].unique())
        for column in config_columns
    }


def _build_performance_lookup(df, config_columns, performance_column):
    lookup = {}
    for row in df[list(config_columns) + [performance_column]].itertuples(index=False, name=None):
        config = tuple(int(value) for value in row[:-1])
        lookup[config] = row[-1]
    return lookup


def _build_strata(values, n_strata):
    if not values:
        return []

    strata_count = min(n_strata, len(values))
    index_groups = np.array_split(np.arange(len(values)), strata_count)
    strata = []
    for index_group in index_groups:
        strata.append([values[int(index)] for index in index_group])
    return strata


def _sample_random_global_config(domains, rng):
    return [int(rng.choice(values)) for values in domains.values()]


def _sample_dds_candidate(domains, candidate_index, rng):
    config = []
    for dim_index, values in enumerate(domains.values()):
        strata = _build_strata(values, NUM_STRATA)
        if len(strata) == 1:
            config.append(int(strata[0][0]))
            continue

        base_bucket = (candidate_index + dim_index) % len(strata)
        if candidate_index >= len(strata):
            base_bucket = (base_bucket + int(rng.integers(len(strata)))) % len(strata)

        bucket = strata[base_bucket]
        config.append(int(rng.choice(bucket)))
    return config


def _candidate_batch_limit(batch_size):
    return max(batch_size * MAX_CANDIDATE_GENERATION_MULTIPLIER, batch_size + 20)


def _generate_global_candidates(domains, batch_size, attempted_configs, rng):
    candidates = []
    seen_in_batch = set()

    for try_index in range(_candidate_batch_limit(batch_size)):
        if len(candidates) >= batch_size:
            break

        candidate = _sample_dds_candidate(domains, try_index, rng)
        candidate_key = tuple(candidate)
        if candidate_key in attempted_configs or candidate_key in seen_in_batch:
            candidate = _sample_random_global_config(domains, rng)
            candidate_key = tuple(candidate)

        if candidate_key in attempted_configs or candidate_key in seen_in_batch:
            continue

        seen_in_batch.add(candidate_key)
        candidates.append(candidate)

    return candidates


def _derive_local_domains(domains, centre_config):
    local_domains = []
    for dim_index, values in enumerate(domains.values()):
        centre_value = centre_config[dim_index]
        centre_position = values.index(centre_value)
        lower_position = max(centre_position - 1, 0)
        upper_position = min(centre_position + 1, len(values) - 1)
        local_domains.append(values[lower_position : upper_position + 1])
    return local_domains


def _append_local_candidate(candidate, attempted_configs, seen_in_batch, candidates):
    candidate_key = tuple(candidate)
    if candidate_key in attempted_configs or candidate_key in seen_in_batch:
        return

    seen_in_batch.add(candidate_key)
    candidates.append(candidate)


def _systematic_local_candidates(
    centre_config,
    local_domains,
    mutable_dims,
    batch_size,
    attempted_configs,
    seen_in_batch,
    candidates,
):
    for dim_index in mutable_dims:
        centre_value = centre_config[dim_index]
        for neighbor_value in local_domains[dim_index]:
            if neighbor_value == centre_value:
                continue
            candidate = list(centre_config)
            candidate[dim_index] = int(neighbor_value)
            _append_local_candidate(candidate, attempted_configs, seen_in_batch, candidates)
            if len(candidates) >= batch_size:
                return


def _random_local_candidates(
    centre_config,
    local_domains,
    mutable_dims,
    batch_size,
    attempted_configs,
    seen_in_batch,
    candidates,
    rng,
):
    max_perturb_dims = min(MAX_LOCAL_PERTURB_DIMS, len(mutable_dims))
    for _ in range(_candidate_batch_limit(batch_size)):
        if len(candidates) >= batch_size:
            return

        candidate = list(centre_config)
        n_changes = int(rng.integers(1, max_perturb_dims + 1))
        chosen_dims = rng.choice(mutable_dims, size=n_changes, replace=False)

        for dim_index in np.atleast_1d(chosen_dims):
            dim_index = int(dim_index)
            options = [
                value
                for value in local_domains[dim_index]
                if value != centre_config[dim_index]
            ]
            if options:
                candidate[dim_index] = int(rng.choice(options))

        _append_local_candidate(candidate, attempted_configs, seen_in_batch, candidates)


def _generate_local_candidates(
    domains, centre_config, batch_size, attempted_configs, rng
):
    local_domains = _derive_local_domains(domains, centre_config)
    mutable_dims = [
        dim_index for dim_index, values in enumerate(local_domains) if len(values) > 1
    ]
    if not mutable_dims:
        return []

    candidates = []
    seen_in_batch = set()

    _systematic_local_candidates(
        centre_config,
        local_domains,
        mutable_dims,
        batch_size,
        attempted_configs,
        seen_in_batch,
        candidates,
    )
    _random_local_candidates(
        centre_config,
        local_domains,
        mutable_dims,
        batch_size,
        attempted_configs,
        seen_in_batch,
        candidates,
        rng,
    )

    return candidates


def _save_search_trace(output_file, config_columns, search_results):
    if output_file is None:
        return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)


def _evaluate_candidate(
    candidate,
    performance_lookup,
    worst_value,
    maximization,
    attempted_configs,
    search_results,
    state,
):
    candidate_key = tuple(int(value) for value in candidate)
    if candidate_key in attempted_configs:
        return False

    attempted_configs.add(candidate_key)

    perf = performance_lookup.get(candidate_key)
    is_valid = perf is not None
    if is_valid:
        state["valid_count"] += 1
    else:
        perf = worst_value
        state["invalid_count"] += 1

    if is_valid and _is_better(perf, state["best_performance"], maximization):
        state["best_performance"] = perf
        state["best_solution"] = list(candidate_key)
        improved = True
    else:
        improved = False

    search_results.append(list(candidate_key) + [perf])
    return improved


def _evaluate_batch(
    candidates,
    performance_lookup,
    worst_value,
    maximization,
    attempted_configs,
    search_results,
    state,
    budget,
):
    improved = False
    for candidate in candidates:
        if state["valid_count"] >= budget:
            break
        improved = _evaluate_candidate(
            candidate,
            performance_lookup,
            worst_value,
            maximization,
            attempted_configs,
            search_results,
            state,
        ) or improved
    return improved


def _remaining_valid_budget(valid_count, budget):
    return max(0, budget - valid_count)


def run_bestconfig_search(file_path, budget, output_file, seed=None):
    """Run a simple BestConfig-style search on a single dataset.

    Parameters
    ----------
    file_path : str
        Path to the dataset CSV.
    budget : int
        Number of valid measurements to perform.
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
        Additional metadata.
    """
    rng = np.random.default_rng(seed)

    df = load_dataset(file_path)
    config_columns, performance_column = get_config_columns_and_performance_column(df)
    system_name = system_name_from_path(file_path)
    maximization = is_maximization(system_name)
    worst_value = get_worst_value(df, performance_column, maximization)

    ordered_domains = _build_ordered_domains(df, config_columns)
    performance_lookup = _build_performance_lookup(df, config_columns, performance_column)
    total_possible_configs = int(
        np.prod([len(values) for values in ordered_domains.values()], dtype=np.int64)
    )

    attempted_configs = set()
    search_results = []
    state = {
        "best_solution": [],
        "best_performance": -np.inf if maximization else np.inf,
        "valid_count": 0,
        "invalid_count": 0,
    }
    global_rounds = 0
    local_rounds = 0

    while state["valid_count"] < budget and len(attempted_configs) < total_possible_configs:
        global_rounds += 1
        global_batch_size = min(
            GLOBAL_SAMPLES_PER_ROUND,
            _remaining_valid_budget(state["valid_count"], budget),
        )
        global_candidates = _generate_global_candidates(
            ordered_domains,
            global_batch_size,
            attempted_configs,
            rng,
        )
        if not global_candidates:
            break

        _evaluate_batch(
            global_candidates,
            performance_lookup,
            worst_value,
            maximization,
            attempted_configs,
            search_results,
            state,
            budget,
        )

        if state["valid_count"] >= budget or not state["best_solution"]:
            continue

        local_improved = True
        while (
            state["valid_count"] < budget
            and len(attempted_configs) < total_possible_configs
            and local_improved
            and state["best_solution"]
        ):
            local_rounds += 1
            previous_best = state["best_performance"]
            local_batch_size = min(
                LOCAL_SAMPLES_PER_ROUND,
                _remaining_valid_budget(state["valid_count"], budget),
            )
            local_candidates = _generate_local_candidates(
                ordered_domains,
                state["best_solution"],
                local_batch_size,
                attempted_configs,
                rng,
            )
            if not local_candidates:
                break

            _evaluate_batch(
                local_candidates,
                performance_lookup,
                worst_value,
                maximization,
                attempted_configs,
                search_results,
                state,
                budget,
            )
            local_improved = _is_better(
                state["best_performance"], previous_best, maximization
            )

    _save_search_trace(output_file, config_columns, search_results)

    summary = {
        "system": system_name,
        "budget": budget,
        "valid_count": state["valid_count"],
        "invalid_count": state["invalid_count"],
        "total_samples": state["valid_count"] + state["invalid_count"],
        "best_performance": state["best_performance"],
        "global_rounds": global_rounds,
        "local_rounds": local_rounds,
        "unique_configs_tried": len(attempted_configs),
    }

    return state["best_solution"], state["best_performance"], summary
