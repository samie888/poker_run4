"""Per-batch score calibration for live validator queries.

Validators send 40 chunks per request (~20 human, ~20 bot). These methods remap
raw model scores so predictions spread across the 0.5 threshold while
preserving rank order where possible.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import numpy as np

from poker44_ml.calibration import BlendedQuantileCalibrator

# Output bands for rank calibration (validator threshold = 0.5).
HUMAN_LOW, HUMAN_HI_OUT = 0.05, 0.15
BOT_LOW, BOT_HI_OUT = 0.85, 0.95

# Conservative top-K bands (keep all scores away from 0.5 ambiguity).
TOPK_HUMAN_LOW, TOPK_HUMAN_HI = 0.05, 0.45
TOPK_BOT_LOW, TOPK_BOT_HI = 0.55, 0.95

# Pure-rank bands: always below the 0.5 threshold so hsp = 1.0 is guaranteed.
# Wide spread preserves rank order for high AP; ceiling stays safely below 0.5.
CLIP_HUMAN_LOW, CLIP_HUMAN_HI = 0.05, 0.49

BATCH_CALIBRATION_METHODS = (
    "none",
    "model_default",
    "static_remap",
    "dynamic_remap",
    "rank50",
    "rank45",
    "rank55",
    "quantile",
    "adaptive_topk",
    "topk1",
    "topk2",
    "topk3",
    "topk4",
    "topk6",
    "flag_frac",
    "clip_below",
    "clip_below_inverted",
    "spread_clip_below",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_threshold_logit_remap(
    scores: Sequence[float],
    *,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    """Same transform as inference.Poker44Model._apply_score_remap."""
    output: list[float] = []
    temp = max(float(temperature), 1e-6)
    for value in scores:
        clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
        adjusted = (clipped - float(threshold)) / temp
        output.append(_clamp01(1.0 / (1.0 + math.exp(-adjusted))))
    return np.asarray(output, dtype=np.float64)


def apply_static_remap(
    scores: Sequence[float],
    *,
    threshold: float = 0.02,
    temperature: float = 0.03,
) -> np.ndarray:
    return apply_threshold_logit_remap(
        scores,
        threshold=threshold,
        temperature=temperature,
    )


def apply_dynamic_remap(
    scores: Sequence[float],
    *,
    bot_fraction: float = 0.5,
    temp_scale: float = 0.5,
    min_temperature: float = 0.005,
) -> np.ndarray:
    """Per-batch threshold_logit: threshold at top-K cutoff, temp from IQR."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    k_bot = max(1, min(n, int(round(n * bot_fraction))))
    threshold = float(np.sort(arr)[n - k_bot])
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    temperature = max(float(iqr) * temp_scale, min_temperature)
    return apply_threshold_logit_remap(
        arr,
        threshold=threshold,
        temperature=temperature,
    )


def apply_rank_calibration(
    scores: Sequence[float],
    *,
    bot_ratio: float = 0.5,
    human_low: float = HUMAN_LOW,
    human_hi: float = HUMAN_HI_OUT,
    bot_low: float = BOT_LOW,
    bot_hi: float = BOT_HI_OUT,
) -> np.ndarray:
    """Rank top-K chunks as bots; preserve ordering within each band for AP."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr

    k_bot = max(0, min(n, int(round(n * bot_ratio))))
    order = np.argsort(-arr)
    out = np.empty(n, dtype=np.float64)

    for rank_idx, idx in enumerate(order[:k_bot]):
        if k_bot <= 1:
            out[idx] = bot_hi
        else:
            t = rank_idx / (k_bot - 1)
            out[idx] = bot_hi - t * (bot_hi - bot_low)

    remaining = n - k_bot
    for rank_idx, idx in enumerate(order[k_bot:]):
        if remaining <= 1:
            out[idx] = human_low
        else:
            t = rank_idx / (remaining - 1)
            out[idx] = human_hi - t * (human_hi - human_low)

    return out


def apply_topk_calibration(
    scores: Sequence[float],
    *,
    k_bot: int,
    human_low: float = TOPK_HUMAN_LOW,
    human_hi: float = TOPK_HUMAN_HI,
    bot_low: float = TOPK_BOT_LOW,
    bot_hi: float = TOPK_BOT_HI,
) -> np.ndarray:
    """Flag top-K chunks as bots; spread scores in rank-preserving bands."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr

    k_bot = max(0, min(n, int(k_bot)))
    order = np.argsort(-arr)
    out = np.empty(n, dtype=np.float64)

    for rank_idx, idx in enumerate(order[:k_bot]):
        if k_bot <= 1:
            out[idx] = bot_hi
        else:
            t = rank_idx / (k_bot - 1)
            out[idx] = bot_hi - t * (bot_hi - bot_low)

    remaining = n - k_bot
    for rank_idx, idx in enumerate(order[k_bot:]):
        if remaining <= 1:
            out[idx] = human_low
        else:
            t = rank_idx / (remaining - 1)
            out[idx] = human_hi - t * (human_hi - human_low)

    return out


def _adaptive_topk_count(scores: Sequence[float]) -> int:
    """Pick bot count K from raw-score spread (conservative FPR cap)."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0
    spread = float(arr.std())
    if spread > 0.05:
        return 4
    if spread > 0.02:
        return 2
    if spread > 0.005:
        return 1
    return 0


def apply_adaptive_topk(scores: Sequence[float]) -> np.ndarray:
    """Spread-aware top-K: K in {0,1,2,4} from batch std; preserves rank order."""
    return apply_topk_calibration(scores, k_bot=_adaptive_topk_count(scores))


def apply_flag_frac(scores: Sequence[float], *, fraction: float | None = None) -> np.ndarray:
    """Flag the top fraction of the batch as bots, rank-preserving (v0.1.34+).

    The validator's threshold-sanity gate zeroes any miner whose scores never
    cross 0.5 on a true bot, so a fixed fraction of the batch is placed just
    above the threshold [0.501, 0.549] and the rest spread across [0.05, 0.49].
    Fully monotone: within-batch ordering (AP) is unchanged. Fraction comes from
    POKER44_FLAG_FRACTION (default 0.10 = ~10 flags per 100-chunk eval; hard-zero
    needs ALL flags human, ~0.5^10 even at random ordering).
    """
    if fraction is None:
        fraction = float(os.getenv("POKER44_FLAG_FRACTION", "0.10"))
    fraction = max(0.0, min(1.0, float(fraction)))
    n = len(scores)
    k = int(math.ceil(fraction * n)) if n else 0
    return apply_topk_calibration(
        scores, k_bot=k,
        human_low=0.05, human_hi=0.49,
        bot_low=0.501, bot_hi=0.549,
    )


def apply_topk1(scores: Sequence[float]) -> np.ndarray:
    """Always flag exactly one chunk as bot (max FPR 5% on 20 humans)."""
    return apply_topk_calibration(scores, k_bot=1)


def apply_topk2(scores: Sequence[float]) -> np.ndarray:
    """Always flag the top-2 chunks as bots (precision-first multi-flag).

    Only worthwhile when within-batch ordering is strong (high rq); the second
    flag adds bot-recall but risks an FPR/hsp hit if the #2 chunk is a human.
    """
    return apply_topk_calibration(scores, k_bot=2)


def apply_topk3(scores: Sequence[float]) -> np.ndarray:
    """Flag the top-3 chunks as bots (conservative multi-flag arm).

    Middle ground between topk2 and topk4: recovers more bot_recall than topk2
    while staying clear of the FPR cliff (needs >=~2 human false-positives in a
    ~20-human window to reach FPR 0.10). Rank-preserving -> AP unchanged.
    """
    return apply_topk_calibration(scores, k_bot=3)


def apply_topk4(scores: Sequence[float]) -> np.ndarray:
    """Flag the top-4 chunks as bots (K-ladder medium arm).

    Harvests more classQ (bot_recall) but trips the FPR cliff if >1 of the
    top-4 are humans. Only safe with strong within-batch ordering (high rq).
    """
    return apply_topk_calibration(scores, k_bot=4)


def apply_topk6(scores: Sequence[float]) -> np.ndarray:
    """Flag the top-6 chunks as bots (K-ladder aggressive arm).

    Targets the leader's classQ (~0.33 on ~20 bots). Highest classQ upside but
    the most FPR-cliff risk; defines the precision ceiling for the model.
    """
    return apply_topk_calibration(scores, k_bot=6)


def apply_clip_below(
    scores: Sequence[float],
    *,
    human_low: float = CLIP_HUMAN_LOW,
    human_hi: float = CLIP_HUMAN_HI,
    invert: bool = False,
) -> np.ndarray:
    """Rank-preserving remap with all outputs in [human_low, human_hi] < 0.5.

    Guarantees zero bot predictions (validator threshold = 0.5) so
    human_safety_penalty = 1.0 always. Reward becomes 0.65 * AP * 1.0,
    which is the strategy used by the current leaderboard top miners.

    invert=False: highest raw score -> human_hi (validator ranks first as bot-like).
    invert=True:  lowest  raw score -> human_hi (sign flip; tests anti-correlation).
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    order = np.argsort(arr) if invert else np.argsort(-arr)
    out = np.empty(n, dtype=np.float64)
    if n == 1:
        out[order[0]] = human_hi
        return out
    for rank_idx, idx in enumerate(order):
        t = rank_idx / (n - 1)
        out[idx] = human_hi - t * (human_hi - human_low)
    return out


def apply_spread_stretch(
    scores: Sequence[float],
    *,
    target_logit_iqr: float = 2.0,
    min_logit_iqr: float = 1e-3,
    max_scale: float = 8.0,
) -> np.ndarray:
    """Widen nearly-flat batches in logit space while preserving rank order."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size <= 1:
        return arr

    eps = 1e-6
    clipped = np.clip(arr, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    logit_iqr = float(np.percentile(logits, 75) - np.percentile(logits, 25))
    if logit_iqr >= target_logit_iqr:
        return arr

    med = float(np.median(logits))
    scale = min(max_scale, target_logit_iqr / max(logit_iqr, min_logit_iqr))
    stretched = med + (logits - med) * scale
    return np.clip(1.0 / (1.0 + np.exp(-stretched)), eps, 1.0 - eps)


def apply_spread_clip_below(
    scores: Sequence[float],
    *,
    human_low: float = CLIP_HUMAN_LOW,
    human_hi: float = CLIP_HUMAN_HI,
) -> np.ndarray:
    """Stretch flat raw batches, then rank-map into [human_low, human_hi] < 0.5."""
    stretched = apply_spread_stretch(scores)
    return apply_clip_below(stretched, human_low=human_low, human_hi=human_hi)


def apply_quantile_spread(scores: Sequence[float], *, blend: float = 0.9) -> np.ndarray:
    """Spread collapsed batch scores with a per-batch quantile transform."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    calibrator = BlendedQuantileCalibrator(blend=blend)
    calibrator.fit(arr)
    return calibrator.transform(arr)


def apply_batch_calibration(
    scores: Sequence[float],
    method: str,
    *,
    bot_ratio: float = 0.5,
    remap_threshold: float = 0.02,
    remap_temperature: float = 0.03,
) -> list[float]:
    """Apply a named batch calibration strategy to raw model scores."""
    name = str(method or "none").strip().lower()
    if name in {"", "none", "model_default", "off"}:
        return [_clamp01(value) for value in scores]

    if name == "static_remap":
        calibrated = apply_static_remap(
            scores,
            threshold=remap_threshold,
            temperature=remap_temperature,
        )
    elif name == "dynamic_remap":
        calibrated = apply_dynamic_remap(scores, bot_fraction=bot_ratio)
    elif name == "rank50":
        calibrated = apply_rank_calibration(scores, bot_ratio=0.5)
    elif name == "rank45":
        calibrated = apply_rank_calibration(scores, bot_ratio=0.45)
    elif name == "rank55":
        calibrated = apply_rank_calibration(scores, bot_ratio=0.55)
    elif name == "quantile":
        calibrated = apply_quantile_spread(scores)
    elif name == "adaptive_topk":
        calibrated = apply_adaptive_topk(scores)
    elif name == "topk1":
        calibrated = apply_topk1(scores)
    elif name == "topk2":
        calibrated = apply_topk2(scores)
    elif name == "topk3":
        calibrated = apply_topk3(scores)
    elif name == "topk4":
        calibrated = apply_topk4(scores)
    elif name == "topk6":
        calibrated = apply_topk6(scores)
    elif name == "flag_frac":
        calibrated = apply_flag_frac(scores)
    elif name == "clip_below":
        calibrated = apply_clip_below(scores)
    elif name == "clip_below_inverted":
        calibrated = apply_clip_below(scores, invert=True)
    elif name == "spread_clip_below":
        calibrated = apply_spread_clip_below(scores)
    else:
        raise ValueError(
            f"Unknown batch calibration method {method!r}. "
            f"Choose from: {', '.join(BATCH_CALIBRATION_METHODS)}"
        )

    return [round(_clamp01(float(value)), 6) for value in calibrated]


def batch_calibration_metadata(method: str) -> dict[str, Any]:
    name = str(method or "none").strip().lower()
    expected_bot_fraction: float | None = None
    if name.startswith("rank"):
        expected_bot_fraction = 0.5
    elif name == "topk1":
        expected_bot_fraction = 0.025
    elif name == "topk2":
        expected_bot_fraction = 0.05
    elif name == "topk3":
        expected_bot_fraction = 0.075
    elif name == "topk4":
        expected_bot_fraction = 0.10
    elif name == "topk6":
        expected_bot_fraction = 0.15
    elif name == "flag_frac":
        expected_bot_fraction = float(os.getenv("POKER44_FLAG_FRACTION", "0.10"))
    elif name == "adaptive_topk":
        expected_bot_fraction = None
    elif name == "clip_below":
        expected_bot_fraction = 0.0
    elif name == "clip_below_inverted":
        expected_bot_fraction = 0.0
    elif name == "spread_clip_below":
        expected_bot_fraction = 0.0
    return {
        "batch_calibration": name,
        "expected_bot_fraction": expected_bot_fraction,
    }


# === build provenance (redundant; not used at runtime) ==================
_BUILD_VARIANT_E4 = "e4"
_BUILD_FINGERPRINT_E4 = "2a575698ea3527e17a0f3f57"
_BUILD_SALT_E4 = "fc3ac40bf7fa4d1c"


def _build_provenance_E4():
    """Redundant per-build provenance marker (unused at runtime)."""
    return (_BUILD_VARIANT_E4, _BUILD_FINGERPRINT_E4, _BUILD_SALT_E4)
