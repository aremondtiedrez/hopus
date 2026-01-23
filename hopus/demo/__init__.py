"""
This module contains routines used in the demo notebook.
"""

from importlib import resources

import pandas as pd

from .. import models


def load_training_data() -> pd.DataFrame:
    """Load the demo training data as a `pandas` `DataFrame`."""
    path = resources.files("hopus").joinpath("demo/training_data.csv")
    return pd.read_csv(path)


def load_test_data() -> pd.DataFrame:
    """Load the demo test data as a `pandas` `DataFrame`."""
    path = resources.files("hopus").joinpath("demo/test_data.csv")
    return pd.read_csv(path)


def load_trained_model(kind: str = "BoostedTrees") -> models.Model:
    """Load a trained demo model."""
    if kind == "Baseline":
        model = models.Baseline()
        path = resources.files("hopus").joinpath("demo/baseline.csv")
    elif kind == "LinearRegression":
        model = models.LinearRegression()
        path = resources.files("hopus").joinpath("demo/linear_regression.npz")
    elif kind == "BoostedTrees":
        model = models.BoostedTrees()
        path = resources.files("hopus").joinpath("demo/boosted_trees.json")
    else:
        raise ValueError(
            "`kind` must be either `Baseline`, `LinearRegression`, or `BoostedTrees`."
        )
    model.load(path)
    return model
