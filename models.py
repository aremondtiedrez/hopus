"""
This module contains methods used to build and train various models.
"""

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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


class LinearRegressionModel(Model):
    """
    This is essentially a thin wrapper around `sklearn.linear_model.LinearRegression`.
    The purpose of this wrapper is to provide homogeneous access to various models,
    whether that model relies in implementation provided by `sklearn` or on custom
    implementation.
    """

    def __init__(self):
        """Initialize a linear regression model."""
        self._model = LinearRegression()

    def fit(self, features, target):
        """
        Fit the model to the given features X and target y
        (typically the training data).
        """
        self._model.fit(features, target)

    def predict(self, features):
        """Returns a prediction obtained by mapping the features through the model."""
        return self._model.predict(features)
