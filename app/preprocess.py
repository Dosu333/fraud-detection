import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _extract_time_features(self, step: int, type: str):
        day = step // 24
        if type == "hour":
            return step % 24
        elif type == "day":
            return day
        elif type == "weekend":
            return 1 if (day % 7) in [5, 6] else 0

    def transform(self, X):
        X = X.copy()

        for col in ["amount", "oldbalanceOrg"]:
            X[col] = np.log1p(X[col])

        if "step" in X.columns:
            X["hourOfDay"] = X["step"].apply(
                lambda x: self._extract_time_features(x, "hour")
            )
            X["dayOfMonth"] = X["step"].apply(
                lambda x: self._extract_time_features(x, "day")
            )
            X["isWeekend"] = X["step"].apply(
                lambda x: self._extract_time_features(x, "weekend")
            )
            X = X.drop(["step"], axis=1)

        if "timestamp" in X.columns:
            X["timestamp"] = pd.to_datetime(X["timestamp"], unit="s")
            X["hourOfDay"] = X["timestamp"].dt.hour
            X["dayOfMonth"] = X["timestamp"].dt.day
            X["isWeekend"] = X["timestamp"].dt.weekday.apply(
                lambda x: 1 if x >= 5 else 0
            )
            X = X.drop(["timestamp"], axis=1)
        return X
