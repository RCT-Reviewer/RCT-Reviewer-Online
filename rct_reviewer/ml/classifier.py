"""
Lightweight classifier class for linear models.
Modernized for Python 3.13 compatibility.
"""
import numpy as np
import scipy
import logging
from rct_reviewer import get_data_path

log = logging.getLogger(__name__)

class MiniClassifier:
    def __init__(self, filename: str):
        path = get_data_path(filename)
        log.debug(f"Loading model {path}...")
        with np.load(path, allow_pickle=True, encoding='latin1') as raw_data:
            
            self.coef = raw_data["coef"].item().todense().A1
            self.intercept = raw_data["intercept"].item()
        log.debug(f"Model {filename} loaded")

    def decision_function(self, X):
        return X.dot(self.coef.T) + self.intercept

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0).astype(np.int64)  

    def predict_proba(self, X):
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-1.0 * z))
        scores = self.decision_function(X)
        return sigmoid(scores)