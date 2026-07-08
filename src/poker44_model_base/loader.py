from __future__ import annotations
import os
from pathlib import Path

class ModelArtifactNotFound(Exception):
    pass

def resolve_artifact_path(model_path=None) -> Path:
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    env = os.getenv("POKER44_MODEL_PATH") or os.getenv("POKER44_MODEL_ARTIFACT")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    candidates.append(root / "models" / "current.joblib")
    for c in candidates:
        if c and c.is_file():
            return c
    raise ModelArtifactNotFound("No model artifact found")

def load_model(model_path=None):
    import joblib
    return joblib.load(resolve_artifact_path(model_path))
