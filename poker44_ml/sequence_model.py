"""Chunk-level Set Transformer for Poker44 bot detection.

This module adds a sixth base learner to the stacked ensemble: a small
hierarchical Transformer that consumes the *raw* action tokens of each hand
instead of the aggregated per-chunk features.

Why this exists
---------------
The feature-based learners in ``train_model_v2`` operate on ~390 hand-aggregate
statistics that throw away action **order**. Bots are most distinguishable in
their temporal patterns: sizing rhythms, postflop response sequences, and
preflop priors. A small Transformer over the (action_type, street, actor,
amount-bucket, pot-after) tokens captures that signal cheaply.

Architecture
------------
For each chunk (list of hands):
    1. **Hand encoder**: action tokens are embedded as the sum of
       (action_type, street, hero/other actor role, full alias actor seat,
       amount bucket, pot-flow bucket, action position in hand, position
       within current street, first-action-in-street flag, learned
       continuous projection of [amount_bb, pot_after_bb, pot_delta_bb]),
       then passed through a Transformer encoder with key-padding masks.
       The hand embedding is the attention-pool over the action sequence
       (Set Transformer PMA with k=1).
    2. **Hand-meta fusion**: each hand embedding is augmented with a
       deepest-street-reached embedding plus a learned projection of
       per-hand continuous context (hero stack BB, distinct actor count,
       streets dealt, per-street action counts, hero action share). These
       are derived from validator-visible fields that do not depend on
       ``normalized_amount_bb`` or ``pot_before``.
    3. **Chunk encoder**: hand embeddings flow through a Transformer
       encoder with hand-level key-padding masks (permutation invariant -
       no chunk positional encoding). A second attention-pool produces the
       chunk embedding.
    4. **Head**: 2-layer MLP outputs one logit per chunk; sigmoid gives
       ``P(bot | chunk)``.

The exposed wrapper :class:`SequenceModelWrapper` is a sklearn-style estimator
with ``fit(chunks, y, sample_weight=None)`` and ``predict_proba(chunks)`` that
plays nicely with the stacked ensemble pipeline. It pickles cleanly via
joblib (state_dict + config) and runs CPU-only by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for the chunk-level sequence model. "
        "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


# --- tokenizer constants ----------------------------------------------------

_ACTION_TYPE_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    # Miner-visible canonical payload excludes blinds/ante and unknown labels.
    "check": 1,
    "call": 2,
    "bet": 3,
    "raise": 4,
    "fold": 5,
}
_STREET_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
    "": 5,
}
_ACTOR_ROLE_PAD = 0
_ACTOR_ROLE_HERO = 1
_ACTOR_ROLE_OTHER = 2

# Mirrors real competition payload canonicalization which coarsens amounts.
_AMOUNT_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)
_AMOUNT_BUCKET_VOCAB_SIZE = len(_AMOUNT_BUCKETS) + 1  # + pad
_POT_FLOW_VOCAB = {
    "<pad>": 0,
    "flat": 1,
    "small_up": 2,
    "medium_up": 3,
    "large_up": 4,
}

# Validator payload_view aliases seats to 1..N where N <= metadata.max_seats (<=10).
# Vocab is pad(0) + 1..10.
_ACTOR_ALIAS_VOCAB_SIZE = 11

# Per-action "first action on its street" indicator. 0=pad, 1=continuation, 2=first.
_FIRST_IN_STREET_VOCAB_SIZE = 3

# Hand-level "deepest street reached" derived from visible actions/streets
# (validator masks metadata.hand_ended_on_street to "").
_HAND_END_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}

# payload_view canonicalizer keeps <=12 actions (often 5-8). Defaults align to
# that range, but these are configurable via SequenceModelConfig.
DEFAULT_MAX_ACTIONS_PER_HAND = 12
# Validator competition requests often contain ~40-80 hands/chunk. Default 64
# captures most chunk-level sequence signal without O(N^2) blow-ups.
DEFAULT_MAX_HANDS_PER_CHUNK = 64
# Keep only robust numeric channels consistently present in live payloads.
CONT_DIM = 3  # amount_bb, pot_after_bb, pot_delta_bb

# Per-hand continuous channels derived from validator-visible fields that are
# independent of normalized_amount_bb / pot_before. Order is fixed and consumed
# by ChunkSetTransformer.hand_meta_proj; do not reorder without retraining.
HAND_META_DIM = 8
#   0: log1p(hero_starting_stack_bb)        - from players[hero].starting_stack / bb
#   1: n_distinct_actors / 10.0             - from unique actor_seat across actions
#   2: streets_dealt / 4.0                  - len(streets)
#   3: actions_per_street_preflop / cap
#   4: actions_per_street_flop / cap
#   5: actions_per_street_turn / cap
#   6: actions_per_street_river / cap
#   7: hero_action_share                    - hero actions / total actions


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_type_id(value: Any) -> int:
    raw = str(value or "").strip().lower()
    if raw in _ACTION_TYPE_VOCAB:
        return _ACTION_TYPE_VOCAB[raw]
    if "raise" in raw:
        return _ACTION_TYPE_VOCAB["raise"]
    if "bet" in raw:
        return _ACTION_TYPE_VOCAB["bet"]
    if "call" in raw:
        return _ACTION_TYPE_VOCAB["call"]
    if "check" in raw:
        return _ACTION_TYPE_VOCAB["check"]
    if "fold" in raw or raw == "muck":
        return _ACTION_TYPE_VOCAB["fold"]
    # payload_view canonicalizer should already emit known labels; for any
    # leftover/empty value, use the least committal non-pad action token.
    return _ACTION_TYPE_VOCAB["check"]


def _street_id(value: Any) -> int:
    raw = str(value or "").strip().lower()
    return _STREET_VOCAB.get(raw, _STREET_VOCAB[""])


def _actor_role(actor_seat: int, hero_seat: int) -> int:
    if actor_seat <= 0:
        return _ACTOR_ROLE_OTHER
    if hero_seat and actor_seat == hero_seat:
        return _ACTOR_ROLE_HERO
    return _ACTOR_ROLE_OTHER


def _to_bb(value: Any, bb: float) -> float:
    if bb <= 0:
        return _safe_float(value, 0.0)
    return _safe_float(value, 0.0) / bb


def _bucket_id(value_bb: float) -> int:
    value = max(0.0, float(value_bb))
    if value <= 0.0:
        return 1
    nearest = min(_AMOUNT_BUCKETS, key=lambda b: abs(b - value))
    return _AMOUNT_BUCKETS.index(nearest) + 1


def _pot_flow_id(pot_before_bb: float, pot_after_bb: float) -> int:
    delta = max(0.0, float(pot_after_bb) - float(pot_before_bb))
    if delta <= 1e-6:
        return _POT_FLOW_VOCAB["flat"]
    if delta <= 1.0:
        return _POT_FLOW_VOCAB["small_up"]
    if delta <= 4.0:
        return _POT_FLOW_VOCAB["medium_up"]
    return _POT_FLOW_VOCAB["large_up"]


def _actor_alias_id(actor_seat: Any) -> int:
    """Clamp aliased actor_seat (already 1..N from payload_view) into the vocab."""
    try:
        seat = int(actor_seat or 0)
    except (TypeError, ValueError):
        return 0
    if seat <= 0:
        return 0
    return min(seat, _ACTOR_ALIAS_VOCAB_SIZE - 1)


def _hand_end_id(street_name: Any) -> int:
    raw = str(street_name or "").strip().lower()
    return _HAND_END_VOCAB.get(raw, 0)


def _hero_starting_stack_bb(hand: Dict[str, Any]) -> float:
    """Hero stack at hand start in BB units, derived from validator-visible fields."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    try:
        hero_seat = int(metadata.get("hero_seat") or 0)
    except (TypeError, ValueError):
        hero_seat = 0
    bb = _safe_float(metadata.get("bb"), 0.02) or 0.02
    if hero_seat <= 0 or not isinstance(players, list):
        return 0.0
    for player in players:
        if not isinstance(player, dict):
            continue
        try:
            seat = int(player.get("seat") or 0)
        except (TypeError, ValueError):
            continue
        if seat == hero_seat:
            return _safe_float(player.get("starting_stack"), 0.0) / bb
    return 0.0


def _sample_hand_indices(total: int, limit: int) -> List[int]:
    """Keep order while covering the full chunk span."""
    if total <= limit:
        return list(range(total))
    if limit <= 1:
        return [total // 2]
    last = total - 1
    indices = {
        int(round(i * last / (limit - 1)))
        for i in range(limit)
    }
    while len(indices) < limit:
        # fill deterministically if rounding collapsed some positions
        candidate = len(indices) * last // max(limit - 1, 1)
        indices.add(int(candidate))
    return sorted(indices)[:limit]


def encode_hand(
    hand: Dict[str, Any],
    *,
    max_actions_per_hand: int = DEFAULT_MAX_ACTIONS_PER_HAND,
) -> Dict[str, np.ndarray]:
    """Convert a single hand payload into padded token tensors."""
    metadata = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    streets_list = hand.get("streets") or []
    try:
        hero_seat = int(metadata.get("hero_seat") or 0)
    except (TypeError, ValueError):
        hero_seat = 0
    bb = _safe_float(metadata.get("bb"), 0.02) or 0.02

    action_type = np.zeros(max_actions_per_hand, dtype=np.int64)
    street = np.zeros(max_actions_per_hand, dtype=np.int64)
    actor_role = np.zeros(max_actions_per_hand, dtype=np.int64)
    actor_alias = np.zeros(max_actions_per_hand, dtype=np.int64)
    amount_bucket = np.zeros(max_actions_per_hand, dtype=np.int64)
    pot_flow = np.zeros(max_actions_per_hand, dtype=np.int64)
    street_pos = np.zeros(max_actions_per_hand, dtype=np.int64)
    first_in_street = np.zeros(max_actions_per_hand, dtype=np.int64)
    cont = np.zeros((max_actions_per_hand, CONT_DIM), dtype=np.float32)
    mask = np.zeros(max_actions_per_hand, dtype=np.bool_)

    actions_per_street = np.zeros(4, dtype=np.int64)  # preflop/flop/turn/river
    distinct_actors: set = set()
    hero_action_count = 0
    last_street_token = 0
    street_counter: Dict[str, int] = {}
    prev_street_raw: Optional[str] = None

    n_actions = min(len(actions), max_actions_per_hand)
    for idx in range(n_actions):
        action = actions[idx]
        if not isinstance(action, dict):
            continue
        action_type[idx] = _action_type_id(action.get("action_type"))
        street_raw = str(action.get("street", "") or "").strip().lower()
        street[idx] = _street_id(street_raw)
        actor_seat_raw = action.get("actor_seat")
        actor_role[idx] = _actor_role(
            int(actor_seat_raw) if isinstance(actor_seat_raw, (int, float, str)) and str(actor_seat_raw).strip() else 0,
            hero_seat,
        )
        actor_alias[idx] = _actor_alias_id(actor_seat_raw)
        amount_bb = _safe_float(action.get("normalized_amount_bb"), 0.0)
        if amount_bb == 0.0:
            amount_bb = _to_bb(action.get("amount"), bb)
        pot_before_bb = _to_bb(action.get("pot_before"), bb)
        pot_after_bb = _to_bb(action.get("pot_after"), bb)
        amount_bucket[idx] = _bucket_id(amount_bb)
        pot_flow[idx] = _pot_flow_id(pot_before_bb, pot_after_bb)
        cont[idx, 0] = math.log1p(max(amount_bb, 0.0))
        cont[idx, 1] = math.log1p(max(pot_after_bb, 0.0))
        cont[idx, 2] = math.log1p(max(pot_after_bb - pot_before_bb, 0.0))
        mask[idx] = True

        within = street_counter.get(street_raw, 0)
        street_pos[idx] = min(within, max_actions_per_hand - 1)
        first_in_street[idx] = 2 if street_raw != prev_street_raw else 1
        street_counter[street_raw] = within + 1
        prev_street_raw = street_raw

        bucket = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street_raw)
        if bucket is not None:
            actions_per_street[bucket] += 1
            last_street_token = bucket + 1  # 1..4 maps to _HAND_END_VOCAB
        if actor_seat_raw is not None:
            try:
                seat = int(actor_seat_raw)
                if seat > 0:
                    distinct_actors.add(seat)
                    if hero_seat and seat == hero_seat:
                        hero_action_count += 1
            except (TypeError, ValueError):
                pass

    if last_street_token == 0 and isinstance(streets_list, list) and streets_list:
        for entry in reversed(streets_list):
            if not isinstance(entry, dict):
                continue
            candidate = _hand_end_id(entry.get("street"))
            if candidate > 0:
                last_street_token = candidate
                break

    total_actions = int(actions_per_street.sum())
    cap = float(max(max_actions_per_hand, 1))
    hero_stack_bb = _hero_starting_stack_bb(hand)
    hand_meta = np.zeros(HAND_META_DIM, dtype=np.float32)
    hand_meta[0] = math.log1p(max(hero_stack_bb, 0.0))
    hand_meta[1] = min(len(distinct_actors), 10) / 10.0
    streets_dealt = sum(
        1 for entry in streets_list if isinstance(entry, dict) and str(entry.get("street", "")).strip()
    )
    hand_meta[2] = min(streets_dealt, 4) / 4.0
    hand_meta[3] = min(actions_per_street[0], cap) / cap
    hand_meta[4] = min(actions_per_street[1], cap) / cap
    hand_meta[5] = min(actions_per_street[2], cap) / cap
    hand_meta[6] = min(actions_per_street[3], cap) / cap
    hand_meta[7] = (hero_action_count / total_actions) if total_actions > 0 else 0.0

    return {
        "action_type": action_type,
        "street": street,
        "actor_role": actor_role,
        "actor_alias": actor_alias,
        "amount_bucket": amount_bucket,
        "pot_flow": pot_flow,
        "street_pos": street_pos,
        "first_in_street": first_in_street,
        "cont": cont,
        "mask": mask,
        "hand_end": np.int64(last_street_token),
        "hand_meta": hand_meta,
    }


def encode_chunk(
    chunk: Sequence[Dict[str, Any]],
    *,
    max_hands_per_chunk: int = DEFAULT_MAX_HANDS_PER_CHUNK,
    max_actions_per_hand: int = DEFAULT_MAX_ACTIONS_PER_HAND,
) -> Dict[str, np.ndarray]:
    """Pad a chunk into fixed-size hand x action tensors with masks."""
    action_type = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    street = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    actor_role = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    actor_alias = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    amount_bucket = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    pot_flow = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    street_pos = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    first_in_street = np.zeros((max_hands_per_chunk, max_actions_per_hand), dtype=np.int64)
    cont = np.zeros(
        (max_hands_per_chunk, max_actions_per_hand, CONT_DIM), dtype=np.float32
    )
    action_mask = np.zeros(
        (max_hands_per_chunk, max_actions_per_hand), dtype=np.bool_
    )
    hand_mask = np.zeros(max_hands_per_chunk, dtype=np.bool_)
    hand_end = np.zeros(max_hands_per_chunk, dtype=np.int64)
    hand_meta = np.zeros((max_hands_per_chunk, HAND_META_DIM), dtype=np.float32)

    selected_indices = _sample_hand_indices(len(chunk), max_hands_per_chunk)
    for hand_idx, source_idx in enumerate(selected_indices):
        hand = chunk[source_idx]
        if not isinstance(hand, dict):
            continue
        encoded = encode_hand(hand, max_actions_per_hand=max_actions_per_hand)
        action_type[hand_idx] = encoded["action_type"]
        street[hand_idx] = encoded["street"]
        actor_role[hand_idx] = encoded["actor_role"]
        actor_alias[hand_idx] = encoded["actor_alias"]
        amount_bucket[hand_idx] = encoded["amount_bucket"]
        pot_flow[hand_idx] = encoded["pot_flow"]
        street_pos[hand_idx] = encoded["street_pos"]
        first_in_street[hand_idx] = encoded["first_in_street"]
        cont[hand_idx] = encoded["cont"]
        action_mask[hand_idx] = encoded["mask"]
        hand_mask[hand_idx] = bool(encoded["mask"].any())
        hand_end[hand_idx] = int(encoded["hand_end"]) if hand_mask[hand_idx] else 0
        hand_meta[hand_idx] = encoded["hand_meta"]

    return {
        "action_type": action_type,
        "street": street,
        "actor_role": actor_role,
        "actor_alias": actor_alias,
        "amount_bucket": amount_bucket,
        "pot_flow": pot_flow,
        "street_pos": street_pos,
        "first_in_street": first_in_street,
        "cont": cont,
        "action_mask": action_mask,
        "hand_mask": hand_mask,
        "hand_end": hand_end,
        "hand_meta": hand_meta,
    }


# --- torch dataset ---------------------------------------------------------


class _ChunkDataset(Dataset):
    def __init__(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        labels: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        *,
        max_hands_per_chunk: int = DEFAULT_MAX_HANDS_PER_CHUNK,
        max_actions_per_hand: int = DEFAULT_MAX_ACTIONS_PER_HAND,
    ) -> None:
        self.encoded: List[Dict[str, np.ndarray]] = [
            encode_chunk(
                chunk,
                max_hands_per_chunk=max_hands_per_chunk,
                max_actions_per_hand=max_actions_per_hand,
            )
            for chunk in chunks
        ]
        self.labels = (
            np.asarray(labels, dtype=np.float32) if labels is not None else None
        )
        self.weights = (
            np.asarray(weights, dtype=np.float32) if weights is not None else None
        )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], float, float]:
        item = self.encoded[idx]
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        weight = float(self.weights[idx]) if self.weights is not None else 1.0
        return item, label, weight


def _collate(
    batch: List[Tuple[Dict[str, np.ndarray], float, float]]
) -> Dict[str, torch.Tensor]:
    keys = (
        "action_type",
        "street",
        "actor_role",
        "actor_alias",
        "amount_bucket",
        "pot_flow",
        "street_pos",
        "first_in_street",
        "cont",
        "action_mask",
        "hand_mask",
        "hand_end",
        "hand_meta",
    )
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = np.stack([item[0][key] for item in batch], axis=0)
        if key in ("cont", "hand_meta"):
            out[key] = torch.from_numpy(stacked).float()
        elif key in ("action_mask", "hand_mask"):
            out[key] = torch.from_numpy(stacked).bool()
        else:
            out[key] = torch.from_numpy(stacked).long()
    out["label"] = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    out["weight"] = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return out


# --- model ------------------------------------------------------------------


@dataclass
class SequenceModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_action_layers: int = 2
    n_hand_layers: int = 1
    dropout: float = 0.1
    ff_mult: int = 2
    max_actions_per_hand: int = DEFAULT_MAX_ACTIONS_PER_HAND
    max_hands_per_chunk: int = DEFAULT_MAX_HANDS_PER_CHUNK
    # Schema 1: legacy (action_type/street/actor_role/amount_bucket/pot_flow + cont3).
    # Schema 2: schema 1 plus per-action {actor_alias, street_pos, first_in_street}
    #           and per-hand {hand_end, hand_meta(HAND_META_DIM)}.
    schema_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_action_layers": int(self.n_action_layers),
            "n_hand_layers": int(self.n_hand_layers),
            "dropout": float(self.dropout),
            "ff_mult": int(self.ff_mult),
            "max_actions_per_hand": int(self.max_actions_per_hand),
            "max_hands_per_chunk": int(self.max_hands_per_chunk),
            "schema_version": int(self.schema_version),
        }


class _AttentionPool(nn.Module):
    """Single-query attention pool (Set Transformer PMA with k=1)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch = x.size(0)
        query = self.query.expand(batch, 1, -1)
        all_padded = (
            key_padding_mask.all(dim=1) if key_padding_mask is not None else None
        )
        safe_mask = key_padding_mask
        if safe_mask is not None and all_padded is not None and all_padded.any():
            safe_mask = safe_mask.clone()
            safe_mask[all_padded, 0] = False
        attn_out, _ = self.attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        pooled = self.norm(attn_out.squeeze(1))
        if all_padded is not None and all_padded.any():
            pooled = pooled.masked_fill(all_padded.unsqueeze(-1), 0.0)
        return pooled


class ChunkSetTransformer(nn.Module):
    """Hierarchical action → hand → chunk Transformer."""

    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.action_type_emb = nn.Embedding(
            len(_ACTION_TYPE_VOCAB), d_model, padding_idx=0
        )
        self.street_emb = nn.Embedding(len(_STREET_VOCAB), d_model, padding_idx=0)
        self.actor_emb = nn.Embedding(3, d_model, padding_idx=_ACTOR_ROLE_PAD)
        self.actor_alias_emb = nn.Embedding(
            _ACTOR_ALIAS_VOCAB_SIZE, d_model, padding_idx=0
        )
        self.amount_bucket_emb = nn.Embedding(_AMOUNT_BUCKET_VOCAB_SIZE, d_model, padding_idx=0)
        self.pot_flow_emb = nn.Embedding(len(_POT_FLOW_VOCAB), d_model, padding_idx=0)
        self.action_pos_emb = nn.Embedding(
            int(config.max_actions_per_hand), d_model
        )
        self.street_pos_emb = nn.Embedding(
            int(config.max_actions_per_hand), d_model
        )
        self.first_in_street_emb = nn.Embedding(
            _FIRST_IN_STREET_VOCAB_SIZE, d_model, padding_idx=0
        )
        self.cont_proj = nn.Linear(CONT_DIM, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        # Per-hand augmentation: encodes deepest street reached + continuous
        # behavioral context (stack depth, actor count, action distribution,
        # hero engagement) using only fields that don't depend on
        # normalized_amount_bb / pot_before.
        self.hand_end_emb = nn.Embedding(
            len(_HAND_END_VOCAB), d_model, padding_idx=0
        )
        self.hand_meta_proj = nn.Linear(HAND_META_DIM, d_model)
        self.hand_meta_norm = nn.LayerNorm(d_model)

        action_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer, num_layers=config.n_action_layers
        )
        self.action_pool = _AttentionPool(d_model, config.n_heads, config.dropout)

        hand_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.hand_encoder = nn.TransformerEncoder(
            hand_layer, num_layers=config.n_hand_layers
        )
        self.chunk_pool = _AttentionPool(d_model, config.n_heads, config.dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
        )

    def encode(
        self,
        action_type: torch.Tensor,
        street: torch.Tensor,
        actor_role: torch.Tensor,
        actor_alias: torch.Tensor,
        amount_bucket: torch.Tensor,
        pot_flow: torch.Tensor,
        street_pos: torch.Tensor,
        first_in_street: torch.Tensor,
        cont: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_end: torch.Tensor,
        hand_meta: torch.Tensor,
    ) -> torch.Tensor:
        """Shared feature extractor: tokens -> chunk embedding (pre-head).

        Exposed so auxiliary heads can tap the chunk representation without
        changing ``forward`` behaviour (``head(encode(...))``).
        """
        batch, hands, actions = action_type.shape
        position_ids = (
            torch.arange(actions, device=action_type.device)
            .unsqueeze(0)
            .expand(batch * hands, actions)
        )
        flat_action_type = action_type.reshape(batch * hands, actions)
        flat_street = street.reshape(batch * hands, actions)
        flat_actor = actor_role.reshape(batch * hands, actions)
        flat_actor_alias = actor_alias.reshape(batch * hands, actions)
        flat_amount_bucket = amount_bucket.reshape(batch * hands, actions)
        flat_pot_flow = pot_flow.reshape(batch * hands, actions)
        flat_street_pos = street_pos.reshape(batch * hands, actions)
        flat_first_in_street = first_in_street.reshape(batch * hands, actions)
        flat_cont = cont.reshape(batch * hands, actions, -1)
        flat_action_mask = action_mask.reshape(batch * hands, actions)

        embed = (
            self.action_type_emb(flat_action_type)
            + self.street_emb(flat_street)
            + self.actor_emb(flat_actor)
            + self.actor_alias_emb(flat_actor_alias)
            + self.amount_bucket_emb(flat_amount_bucket)
            + self.pot_flow_emb(flat_pot_flow)
            + self.action_pos_emb(position_ids)
            + self.street_pos_emb(flat_street_pos)
            + self.first_in_street_emb(flat_first_in_street)
            + self.cont_proj(flat_cont)
        )
        embed = self.input_norm(embed)
        embed = self.input_dropout(embed)

        key_padding = ~flat_action_mask
        encoded = self.action_encoder(embed, src_key_padding_mask=key_padding)
        hand_emb = self.action_pool(encoded, key_padding_mask=key_padding)
        hand_emb = hand_emb.reshape(batch, hands, -1)

        hand_meta_emb = self.hand_meta_proj(hand_meta) + self.hand_end_emb(hand_end)
        hand_emb = self.hand_meta_norm(hand_emb + hand_meta_emb)
        hand_emb = hand_emb.masked_fill(~hand_mask.unsqueeze(-1), 0.0)

        hand_kp = ~hand_mask
        encoded_hands = self.hand_encoder(hand_emb, src_key_padding_mask=hand_kp)
        chunk_emb = self.chunk_pool(encoded_hands, key_padding_mask=hand_kp)
        return chunk_emb

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        chunk_emb = self.encode(**inputs)
        logit = self.head(chunk_emb).squeeze(-1)
        return logit


# --- sklearn-style wrapper -------------------------------------------------


def parse_learning_rate_schedule(
    spec: str | None,
    *,
    default_lr: float,
    n_epochs: int,
) -> list[float]:
    """Expand ``lr:epochs`` segments into one LR per training epoch.

    Example: ``"1.3e-3:4,1e-3:4"`` with ``n_epochs=8`` -> eight rates.
    If the schedule defines fewer epochs than ``n_epochs``, the last rate is
    repeated. If it defines more, it is truncated to ``n_epochs``.
    """
    total_epochs = max(1, int(n_epochs))
    fallback = float(default_lr)
    if fallback <= 0:
        raise ValueError(f"default_lr must be positive, got {default_lr}")
    raw = str(spec or "").strip()
    if not raw:
        return [fallback] * total_epochs

    per_epoch: list[float] = []
    for part in raw.split(","):
        segment = part.strip()
        if not segment:
            continue
        if ":" not in segment:
            raise ValueError(
                f"Invalid learning-rate schedule segment {segment!r}; "
                "use lr:epochs (e.g. 1.3e-3:4,1e-3:4)."
            )
        lr_text, count_text = segment.rsplit(":", 1)
        lr = float(lr_text.strip())
        count = int(count_text.strip())
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr!r}")
        if count <= 0:
            raise ValueError(f"Epoch count must be positive, got {count!r}")
        per_epoch.extend([lr] * count)

    if not per_epoch:
        return [fallback] * total_epochs
    if len(per_epoch) < total_epochs:
        per_epoch.extend([per_epoch[-1]] * (total_epochs - len(per_epoch)))
    return per_epoch[:total_epochs]


@dataclass
class SequenceModelWrapper:
    """sklearn-style wrapper around :class:`ChunkSetTransformer`.

    Use ``fit(chunks, y, sample_weight=...)`` with **raw chunk payloads** (lists
    of hand dicts), not feature rows. Use ``predict_proba(chunks)`` to get an
    ``Nx2`` array compatible with the rest of the stacking pipeline.
    """

    config: SequenceModelConfig = field(default_factory=SequenceModelConfig)
    n_epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    learning_rate_schedule: Optional[str] = None
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "cpu"
    verbose: bool = False
    verbose_metrics: bool = True
    _model_state: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.config = (
            self.config
            if isinstance(self.config, SequenceModelConfig)
            else SequenceModelConfig(**dict(self.config))
        )

    def _new_model(self) -> ChunkSetTransformer:
        torch.manual_seed(int(self.seed))
        return ChunkSetTransformer(self.config).to(self.device)

    def _model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = (
            "action_type",
            "street",
            "actor_role",
            "actor_alias",
            "amount_bucket",
            "pot_flow",
            "street_pos",
            "first_in_street",
            "cont",
            "action_mask",
            "hand_mask",
            "hand_end",
            "hand_meta",
        )
        return {key: batch[key].to(self.device) for key in keys}

    def fit(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        y: Sequence[int],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SequenceModelWrapper":
        labels = np.asarray(y, dtype=np.float32)
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(len(chunks), dtype=np.float32)
        )

        rng = np.random.default_rng(int(self.seed))
        order = rng.permutation(len(chunks))
        val_size = max(int(round(self.val_fraction * len(chunks))), 1)
        val_idx = order[:val_size]
        train_idx = order[val_size:]

        train_chunks = [chunks[i] for i in train_idx]
        train_labels = labels[train_idx]
        train_weights = weights[train_idx]
        val_chunks = [chunks[i] for i in val_idx]
        val_labels = labels[val_idx]
        val_weights = weights[val_idx]

        train_ds = _ChunkDataset(
            train_chunks,
            train_labels,
            train_weights,
            max_hands_per_chunk=int(self.config.max_hands_per_chunk),
            max_actions_per_hand=int(self.config.max_actions_per_hand),
        )
        val_ds = _ChunkDataset(
            val_chunks,
            val_labels,
            val_weights,
            max_hands_per_chunk=int(self.config.max_hands_per_chunk),
            max_actions_per_hand=int(self.config.max_actions_per_hand),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        model = self._new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience = 0
        epoch_learning_rates = parse_learning_rate_schedule(
            self.learning_rate_schedule,
            default_lr=float(self.learning_rate),
            n_epochs=int(self.n_epochs),
        )

        for epoch in range(int(self.n_epochs)):
            epoch_lr = float(epoch_learning_rates[epoch])
            for param_group in optimizer.param_groups:
                param_group["lr"] = epoch_lr
            model.train()
            total_train = 0.0
            n_train = 0
            for batch in train_loader:
                logits = model(**self._model_inputs(batch))
                label_t = batch["label"].to(self.device)
                weight_t = batch["weight"].to(self.device)
                raw_loss = loss_fn(logits, label_t)
                loss = (raw_loss * weight_t).sum() / weight_t.sum().clamp(min=1e-6)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train += float(loss.item()) * label_t.size(0)
                n_train += label_t.size(0)

            val_loss = self._evaluate_loss(model, val_loader, loss_fn)
            if self.verbose:
                print(
                    f"    seq epoch {epoch + 1}/{self.n_epochs} "
                    f"lr={epoch_lr:.6g} "
                    f"train_loss={total_train / max(n_train, 1):.4f} "
                    f"val_loss={val_loss:.4f}"
                )
            if self.verbose_metrics and len(val_chunks) > 0:
                from poker44_ml.chunk_score_metrics import print_chunk_score_diagnostics

                val_proba = self._predict_proba_model(model, val_chunks)[:, 1]
                print_chunk_score_diagnostics(
                    f"seq epoch {epoch + 1}/{self.n_epochs} val",
                    val_labels.tolist(),
                    val_proba.tolist(),
                    indent="    ",
                )
            if val_loss + 1e-5 < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: tensor.detach().clone() for key, tensor in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= int(self.early_stopping_patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model_state = {
            "state_dict": {key: tensor.cpu() for key, tensor in model.state_dict().items()},
            "config": self.config.to_dict(),
        }
        if self.verbose_metrics:
            from poker44_ml.chunk_score_metrics import print_chunk_score_diagnostics

            train_proba = self._predict_proba_model(model, train_chunks)[:, 1]
            print_chunk_score_diagnostics(
                "seq fit train (best checkpoint)",
                train_labels.tolist(),
                train_proba.tolist(),
                indent="    ",
            )
            if len(val_chunks) > 0:
                val_proba = self._predict_proba_model(model, val_chunks)[:, 1]
                print_chunk_score_diagnostics(
                    "seq fit val (best checkpoint)",
                    val_labels.tolist(),
                    val_proba.tolist(),
                    indent="    ",
                )
        return self

    def _predict_proba_model(
        self,
        model: ChunkSetTransformer,
        chunks: Sequence[Sequence[Dict[str, Any]]],
    ) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 2), dtype=np.float64)
        ds = _ChunkDataset(
            list(chunks),
            np.zeros(len(chunks), dtype=np.float32),
            np.ones(len(chunks), dtype=np.float32),
            max_hands_per_chunk=int(self.config.max_hands_per_chunk),
            max_actions_per_hand=int(self.config.max_actions_per_hand),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate)
        model.eval()
        logits: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch_logits = model(**self._model_inputs(batch))
                logits.extend(batch_logits.detach().cpu().tolist())
        arr = np.asarray(logits, dtype=np.float64)
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
        arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - arr, arr])

    def _evaluate_loss(
        self,
        model: ChunkSetTransformer,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        if len(loader.dataset) == 0:  # type: ignore[arg-type]
            return float("inf")
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                logits = model(**self._model_inputs(batch))
                raw_loss = loss_fn(logits, batch["label"].to(self.device))
                w = batch["weight"].to(self.device)
                total += float((raw_loss * w).sum().item())
                count += int(w.sum().item())
        return total / max(count, 1)

    def predict_proba(
        self, chunks: Sequence[Sequence[Dict[str, Any]]]
    ) -> np.ndarray:
        if self._model_state is None:
            raise RuntimeError("SequenceModelWrapper.predict_proba called before fit.")
        cfg = self._model_state.get("config") or {}
        saved_schema = int(cfg.get("schema_version", 1))
        if saved_schema != int(self.config.schema_version):
            raise RuntimeError(
                "SequenceModelWrapper artifact schema mismatch: saved "
                f"schema_version={saved_schema}, runtime expects "
                f"{int(self.config.schema_version)}. Retrain the sequence "
                "model with the current code (new actor_alias / street_pos / "
                "first_in_street / hand_end / hand_meta channels)."
            )
        model = ChunkSetTransformer(self.config).to(self.device)
        model.load_state_dict(self._model_state["state_dict"])
        return self._predict_proba_model(model, chunks)

    def predict_chunk_scores(
        self, chunks: Sequence[Sequence[Dict[str, Any]]]
    ) -> List[float]:
        return self.predict_proba(chunks)[:, 1].tolist()

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "learning_rate_schedule": self.learning_rate_schedule,
            "weight_decay": float(self.weight_decay),
            "val_fraction": float(self.val_fraction),
            "early_stopping_patience": int(self.early_stopping_patience),
            "seed": int(self.seed),
            "device": str(self.device),
            "verbose": bool(self.verbose),
            "verbose_metrics": bool(self.verbose_metrics),
            "_model_state": self._model_state,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.config = SequenceModelConfig(**state["config"])
        self.n_epochs = int(state["n_epochs"])
        self.batch_size = int(state["batch_size"])
        self.learning_rate = float(state["learning_rate"])
        self.learning_rate_schedule = state.get("learning_rate_schedule")
        self.weight_decay = float(state["weight_decay"])
        self.val_fraction = float(state["val_fraction"])
        self.early_stopping_patience = int(state["early_stopping_patience"])
        self.seed = int(state["seed"])
        device = str(state["device"])
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.verbose = bool(state.get("verbose", False))
        self.verbose_metrics = bool(state.get("verbose_metrics", True))
        self._model_state = state.get("_model_state")


# === build provenance (redundant; not used at runtime) ==================
_BUILD_VARIANT_E4 = "e4"
_BUILD_FINGERPRINT_E4 = "a974f40c7cda8f1dbb8b4b1f"
_BUILD_SALT_E4 = "c34e94f1fbb3a2be"


def _build_provenance_E4():
    """Redundant per-build provenance marker (unused at runtime)."""
    return (_BUILD_VARIANT_E4, _BUILD_FINGERPRINT_E4, _BUILD_SALT_E4)
