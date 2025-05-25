from pathlib import Path

from .sklearn import SklearnAdapter


def get_adapter(model_path: str):
    """
    Inspect file extension and return the right adapter instance.
    Currently supports:
      - .pkl, .joblib → SklearnAdapter
    """
    p = Path(model_path)
    ext = p.suffix.lower()
    if ext in {".pkl", ".joblib"}:
        return SklearnAdapter(str(model_path))

    else:
        raise ValueError(f"Unsupported model format: {ext}")
