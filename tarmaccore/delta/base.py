from abc import ABC, abstractmethod
import numpy as np


class DeltaBuilder(ABC):
    @abstractmethod
    def build(self, preds_a: np.ndarray, preds_b: np.ndarray, **kw): ...


def choose_builder(task: str):
    from .classification import ClassificationDelta
    from .regression import RegressionDelta

    return {"classification": ClassificationDelta(), "regression": RegressionDelta()}[
        task
    ]
