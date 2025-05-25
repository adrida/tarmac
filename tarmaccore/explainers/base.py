from abc import ABC, abstractmethod


class IExplainer(ABC):
    @abstractmethod
    def fit(self, X, delta_labels): ...
    @abstractmethod
    def explain(self): ...
