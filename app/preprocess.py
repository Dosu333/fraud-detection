import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def _extract_time_features(self, step: int, type: str):
        if type == "hour":
            return step % 24
        elif type == "day":
            return step // 24
        elif type == "weekend":
            return 1 if step // 24 > 4 else 0
        
    def transform(self, X):
        X = X.copy()
        X["balanceDiffOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
        X["balanceDiffDest"] = X["newbalanceDest"] - X["oldbalanceDest"]
        X["origBalanceRatio"] = (X["newbalanceOrig"] + 1) / (X["oldbalanceOrg"] + 1)
        X["destBalanceRatio"] = (X["newbalanceDest"] + 1) / (X["oldbalanceDest"] + 1)
        
        for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            X[col] = np.log1p(X[col])
            
        X['hourOfDay'] = X['step'].apply(lambda x: self._extract_time_features(x, 'hour'))
        X['dayOfMonth'] = X['step'].apply(lambda x: self._extract_time_features(x, 'day'))
        X['isWeekend'] = X['step'].apply(lambda x: self._extract_time_features(x, 'weekend'))
        X = X.drop(['nameOrig', 'nameDest', 'step'], axis=1)
        return X
