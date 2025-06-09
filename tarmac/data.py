from pathlib import Path
import numpy as np
import pandas as pd

from sklearn import datasets


def load_table(path: Path, feature_columns: list = None) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if feature_columns:
            df = df[feature_columns]
        return df.values
    elif ext in {".npy", ".npz"}:
        data = np.load(path)
        if feature_columns and len(feature_columns) < data.shape[1]:
            data = data[:, : len(feature_columns)]
        return data
    else:
        raise ValueError(f"Unsupported data format: {ext}")


def union_datasets(
    X_a: np.ndarray, X_b: np.ndarray, y_a: np.ndarray = None, y_b: np.ndarray = None
) -> tuple:
    """Union of two datasets, maintaining correspondence between X and y.

    Args:
        X_a: Features from dataset A
        X_b: Features from dataset B
        y_a: Optional targets from dataset A
        y_b: Optional targets from dataset B

    Returns:
        If y_a and y_b are provided: tuple(X_union, y_union)
        If only X_a and X_b are provided: X_union
    """
    X_combined = np.vstack([X_a, X_b])

    uniq_idx = np.unique(X_combined, axis=0, return_index=True)[1]
    uniq_idx.sort()

    X_union = X_combined[uniq_idx]

    if y_a is not None and y_b is not None:
        y_combined = np.concatenate([y_a, y_b])
        y_union = y_combined[uniq_idx]
        return X_union, y_union

    return X_union


def sample_datasets(
    a: np.ndarray,
    b: np.ndarray,
    strategy: str = "union",
    size: int = None,
    seed: int = 0,
) -> np.ndarray:
    X = union_datasets(a, b) if strategy == "union" else np.vstack([a, b])
    if strategy == "random":
        rng = np.random.default_rng(seed)
        n = min(size or len(X), len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx]
    return X  # for "union" or "full"


def load_builtin_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "iris":
        return datasets.load_iris(return_X_y=True)
    elif name == "diabetes":
        return datasets.load_diabetes(return_X_y=True)
    else:
        raise ValueError(f"Unsupported builtin dataset: {name}")
