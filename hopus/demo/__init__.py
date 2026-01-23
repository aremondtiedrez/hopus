"""
This module contains routines used in the demo notebook.
"""

from importlib import resources

import pandas as pd


def load_training_data():
    """Load the demo training data as a `pandas` `DataFrame`."""
    path = resources.files("hopus").joinpath("demo/training_data.csv")
    return pd.read_csv(path)


def load_test_data():
    """Load the demo test data as a `pandas` `DataFrame`."""
    path = resources.files("hopus").joinpath("demo/test_data.csv")
    return pd.read_csv(path)
