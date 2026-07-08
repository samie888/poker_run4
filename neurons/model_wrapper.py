from __future__ import annotations
import hashlib, os
from pathlib import Path
from typing import Any
from poker44_model_base.loader import ModelArtifactNotFound, resolve_artifact_path
from poker44_ml.inference import Poker44Model

REPO_URL = "https://github.com/samie888/poker_run1"
REPO_COMMIT = os.getenv("POKER44_MODEL_REPO_COMMIT", "")
MODEL_NAME = os.getenv("POKER44_MODEL_NAME", "poker44-agg-detector")
MODEL_VERSION = os.getenv("POKER44_MODEL_VERSION", "1")
DATA_ATTESTATION = (
    "No validator-private evaluation labels are used for training. Training uses released benchmark data and local runtime features."
)
IMPLEMENTATION_FILES = (
    "neurons/miner.py",
    "src/poker44_model_base/loader.py",
    "poker44_ml/inference.py",
    "poker44_ml/features.py",
    "poker44_ml/stacked.py",
    "poker44_ml/calibration.py",
    "poker44/validator/payload_view.py",
)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def build_model_manifest(model_path=None) -> dict[str, Any]:
    try:
        artifact_path = resolve_artifact_path(model_path)
        artifact_sha256 = _sha256_file(artifact_path)
        inference_mode = "local-joblib"
    except ModelArtifactNotFound:
        artifact_sha256 = ""
        inference_mode = "artifact-required"
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "framework": "mlp-runtime",
        "license": "MIT",
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "artifact_sha256": artifact_sha256,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "open_source": True,
        "inference_mode": inference_mode,
        "data_attestation": DATA_ATTESTATION,
        "training_data_statement": DATA_ATTESTATION,
    }

class MinerModel:
    def __init__(self, model_path=None):
        self.artifact_path = resolve_artifact_path(model_path)
        self.model = Poker44Model(self.artifact_path)
        self.model_manifest = build_model_manifest(model_path)
    def predict_chunk_scores(self, chunks):
        return [float(v) for v in self.model.predict_chunk_scores(chunks)] if chunks else []

def load_miner_model(model_path=None) -> MinerModel:
    return MinerModel(model_path)
