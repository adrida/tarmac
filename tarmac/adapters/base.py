from abc import ABC, abstractmethod
import numpy as np


class BaseAdapter(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return discrete predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates if available."""
        ...
