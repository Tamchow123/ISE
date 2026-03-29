"""
data.py — Dataset loading and configuration-space helpers.

Provides functions to:
  - load a dataset CSV
  - identify configuration columns vs. the performance column
  - sample a random configuration from the valid domain
  - look up the measured performance of a configuration
  - compute worst-case performance for invalid configurations
"""

import os
import numpy as np
import pandas as pd


def load_dataset(file_path):
    """Load a dataset CSV and return a pandas DataFrame."""
    return pd.read_csv(file_path)


def get_config_columns_and_performance_column(df):
    """Return (config_columns, performance_column).

    Convention: all columns except the last are configuration parameters;
    the last column is the performance metric.
    """
    config_columns = df.columns[:-1]
    performance_column = df.columns[-1]
    return config_columns, performance_column


def is_maximization(system_name):
    """Determine whether the objective for *system_name* should be maximised.

    Currently all benchmark systems in the lab use minimisation.
    Extend the list below if a maximisation system is added later.
    """
    # No systems currently require maximisation.
    maximisation_systems = set()
    return system_name.lower() in maximisation_systems


def get_worst_value(df, performance_column, maximization):
    """Return a worst-case performance value for invalid configurations.

    For minimisation problems the worst value is twice the observed maximum.
    For maximisation problems it is half the observed minimum.
    """
    if maximization:
        return df[performance_column].min() / 2
    else:
        return df[performance_column].max() * 2


def sample_random_configuration(df, config_columns, rng):
    """Sample one configuration uniformly at random from the valid domain.

    Each parameter is drawn independently from its unique values in *df*.
    *rng* must be a ``numpy.random.Generator`` instance.
    """
    config = [int(rng.choice(df[col].unique())) for col in config_columns]
    return config


def lookup_configuration_performance(df, config_columns, performance_column, config):
    """Look up the performance of *config* in *df*.

    Returns (performance, is_valid):
      - If the exact configuration exists, returns (measured_value, True).
      - Otherwise returns (None, False).
    """
    match = df.loc[
        (df[config_columns] == pd.Series(config, index=config_columns)).all(axis=1)
    ]
    if not match.empty:
        return match[performance_column].iloc[0], True
    return None, False


def system_name_from_path(file_path):
    """Extract the system name (e.g. '7z') from a dataset file path."""
    return os.path.basename(file_path).split(".")[0]
