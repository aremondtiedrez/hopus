"""
This module contains methods used to build and train various models.
"""

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.metrics import mean_squared_error

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
    def predict(self, features):
        """Returns a prediction obtained by mapping the features through the model."""

    def evaluate(self, features, target):
        """
        Computes the mean squared error between the true target an
        the predictions made by the model from the features."""
        predictions = self.predict(features)
        return mean_squared_error(target, predictions)


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

    def predict(self, features):
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
        merged_features = pd.merge(
            features, self._zipcode_averages, on="zipCode", how="left"
        )
        merged_features["predictedPrice"] = (
            merged_features["meanTimeNormalizedPricePerSqFtInZipcode"]
            * merged_features["predictedValueHomePriceIndex"]
            * merged_features["sqFt"]
        )
        return merged_features["predictedPrice"]


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

    def predict(self, features):
        """Returns a prediction obtained by mapping the features through the model."""
        return self._model.predict(features)


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

    def predict(self, features):
        """Returns a prediction obtained by mapping the features through the model."""
        return self._model.predict(features)
