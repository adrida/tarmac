import numpy as np
from .base import DeltaBuilder


class RegressionDelta(DeltaBuilder):
    def build(self, preds_a, preds_b, epsilon=0.05, **kw):
        diff = np.abs(preds_a - preds_b)
        if epsilon < 1:  # treat as fraction of target range
            epsilon *= diff.max() if diff.size else 1
        return (diff > epsilon).astype(int)
