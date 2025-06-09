import joblib
import numpy as np
from .base import BaseAdapter


class SklearnAdapter(BaseAdapter):
    def __init__(self, path: str):

        self.model = joblib.load(path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        preds = self.predict(X)

        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1
        return proba
