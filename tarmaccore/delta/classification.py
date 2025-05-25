from .base import DeltaBuilder


class ClassificationDelta(DeltaBuilder):
    def build(self, preds_a, preds_b, **kw):
        delta_y = (preds_a != preds_b).astype(int)
        return delta_y  # features X unchanged, handled outside
