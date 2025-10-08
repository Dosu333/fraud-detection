import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(['nameOrig', 'nameDest', ], axis=1)
        X["balanceDiffOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
        X["balanceDiffDest"] = X["newbalanceDest"] - X["oldbalanceDest"]
        X["orig_balance_ratio"] = (X["newbalanceOrig"] + 1) / (X["oldbalanceOrg"] + 1)
        X["dest_balance_ratio"] = (X["newbalanceDest"] + 1) / (X["oldbalanceDest"] + 1)
        return X
