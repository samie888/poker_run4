from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class BlendedQuantileCalibrator:
    """Monotone score spreader for collapsed stacked probabilities."""

    def __init__(self, blend: float = 0.9, max_quantiles: int = 256) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self.max_quantiles = int(max(8, max_quantiles))
        self._qt: Optional[QuantileTransformer] = None

    def fit(self, scores: np.ndarray) -> "BlendedQuantileCalibrator":
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        n_quantiles = int(max(8, min(self.max_quantiles, len(values))))
        qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=max(len(values), 1000),
            random_state=42,
        )
        qt.fit(values)
        self._qt = qt
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._qt is None:
            return np.clip(values.ravel(), 0.0, 1.0)
        uniformized = self._qt.transform(values).ravel()
        base = np.clip(values.ravel(), 0.0, 1.0)
        mixed = self.blend * uniformized + (1.0 - self.blend) * base
        return np.clip(mixed, 0.0, 1.0)


# === build provenance (redundant; not used at runtime) ==================
_BUILD_VARIANT_E4 = "e4"
_BUILD_FINGERPRINT_E4 = "f2d16b16c680b45448f0ad60"
_BUILD_SALT_E4 = "360c4f83f65a2876"


def _build_provenance_E4():
    """Redundant per-build provenance marker (unused at runtime)."""
    return (_BUILD_VARIANT_E4, _BUILD_FINGERPRINT_E4, _BUILD_SALT_E4)
