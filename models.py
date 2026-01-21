"""
This module contains methods used to build and train various models.
"""

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

from xgboost import XGBRegressor as _XGBRegressor


class Model(ABC):
    """
    This class provides common functionalities to various models.
    """

    @abstractmethod
    def fit(self, features, target):
        """
        Fit the model to the given features X and target y
        (typically the training data).
        """

    @abstractmethod
    def predict(self, features, **kwargs):
        """Returns a prediction obtained by mapping the features through the model."""

    def evaluate(self, features, target):
        """
        Computes the mean squared error between the true target an
        the predictions made by the model from the features."""
        predictions = self.predict(features)
        return mean_squared_error(target, predictions)

    @abstractmethod
    def save(self, filename: str):
        """Save the model."""

    @abstractmethod
    def load(self, filename: str):
        """Load the model."""


class Baseline(Model):
    """
    The baseline model proceeds as follows. It uses the training set to compute,
    for each ZIP code, the mean over that ZIP code of the time-normalized
    price-per-square-foot. Over the training set, these means over ZIP codes are then
    multiplied by the predicted value of the home price index and the square footage
    to produce the price prediction.
    """

    def __init__(self):
        """Initialize a baseline model."""
        self._zipcode_averages = None

    def fit(self, features, target):
        """
        Fit the baseline model. Crucially: the baseline model does not need the `target`
        `DataFrame` to be trained, however it does require that
        the `features` `DataFrame` contain the following columns:
        - `zipCode`,
        - `timeNormalizedPricePerSqFt` (see `preprocessing.property_listings` and
          its method `_compute_time_normalized_price_per_square_foot`),
        - `predictedValueHomePriceIndex` (see `preprocessing.home_price_index` and
          its method `_compute_seasonal_adjustment`), and
        - `sqFt`.

        This is a thin wrapper around `Baseline._fit`.
        """
        self._fit(features)

    def _fit(self, features):
        """
        Fit the baseline mode. This is called via `Baseline.fit`.
        """
        self._zipcode_averages = features.groupby("zipCode").agg(
            meanTimeNormalizedPricePerSqFtInZipcode=(
                "timeNormalizedPricePerSqFt",
                "mean",
            )
        )

    def predict(self, features, target_type="price", **kwargs):
        """
        Make a prediction using the baseline model.

        This method expects the input `DataFrame` `features` to have
        the following columns:
        - `zipCode`,
        - `meanTimeNormalizedPricePerSqFtInZipcode` (see `Baseline.fit`),
        - `predictedValueHomePriceIndex` (see `preprocessing.home_price_index` and
          its method `_compute_seasonal_adjustment`), and
        - `sqFt`.
        """
        # Input validation
        if target_type not in ("price", "log_price"):
            raise ValueError("The `target_type` must be either `price` or `log_price`.")

        merged_features = pd.merge(
            features, self._zipcode_averages, on="zipCode", how="left"
        )
        merged_features["predictedPrice"] = (
            merged_features["meanTimeNormalizedPricePerSqFtInZipcode"]
            * merged_features["predictedValueHomePriceIndex"]
            * merged_features["sqFt"]
        )

        # If the target type is `price`, return the prediction
        if target_type == "price":
            return merged_features["predictedPrice"]

        # Otherwise, compute the predicted log-price and return that
        merged_features["predictedLogPrice"] = np.log(merged_features["predictedPrice"])
        return merged_features["predictedLogPrice"]

    def save(self, filename: str):
        """
        Save the model by saving the means, over each ZIP code, of the time-normalized
        price-per-square-foot. Since that data is stored internally as a `pandas`
        `DataFramed` it is saved as a `csv` file.

        The filename should NOT include the `.csv` extension.
        """
        self._zipcode_averages.to_csv(filename + ".csv")

    def load(self, filename: str):
        """
        Load the model by loading the means, over each ZIP code, of the time-normalized
        price-per-square-foot. That data is expected to be stored externally as a `csv`
        file and is then loaded and stored internally as a `pandas` `DataFrame`.

        The filename should NOT include the `.csv` extension.
        """
        self._zipcode_averages = pd.read_csv(filename + ".csv")


class LinearRegression(Model):
    """
    This is essentially a thin wrapper around `sklearn.linear_model.LinearRegression`.
    The purpose of this wrapper is to provide homogeneous access to various models,
    whether that model relies in implementation provided by `sklearn` or on custom
    implementation.
    """

    def __init__(self):
        """Initialize a linear regression model."""
        self._model = _LinearRegression()

    def fit(self, features, target):
        """
        Fit the model to the given features X and target y
        (typically the training data).
        """
        self._model.fit(features, target)

    def predict(self, features, **kwargs):
        """Returns a prediction obtained by mapping the features through the model."""
        return self._model.predict(features)

    def save(self, filename: str):
        """
        Save the model by saving the parameters (the coefficients and the intercept)
        as `numpy` arrays.

        The filename should NOT include the `.npz` extension.
        """
        np.savez(
            filename + ".npz",
            coefficients=self._model.coef_,
            intercept=self._model.intercept_,
        )

    def load(self, filename: str):
        """
        Load the model by loading its parameters (the coefficients and the intercept)
        which are expected to be stored as numpy arrays in a single `npz` file.

        The filename should NOT include the `.npz` extension.
        """
        parameters = np.load(filename + ".npz")
        self._model.coef_ = parameters["coefficients"]
        self._model.intercept = parameters["intercept"]


class BoostedTrees(Model):
    """This is a thin wrapper around the 'XGBRegressor' of the 'xgboost' library."""

    def __init__(self, **hyperparameters):
        """Initialize a 'BoostedTrees' object."""
        self._model = _XGBRegressor(**hyperparameters)

    def fit(self, features, target):
        """
        Fit the model to the given features X and target y
        (typically the training data).
        """
        self._model.fit(features, target)

    def predict(self, features, **kwargs):
        """Returns a prediction obtained by mapping the features through the model."""
        return self._model.predict(features)

    def save(self, filename: str):
        """
        Save the model by using the built-in save function from `xgboost`.

        The filename should NOT include the `.model` extension.
        """
        self._model.save_model(filename + ".model")

    def load(self, filename: str):
        """
        Load the model by using the built-in load function from `xgboost`.

        The filename should NOT include the `.model` extension.
        """
        self._model.load_model(filename + ".model")
