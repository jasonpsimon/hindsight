"""Request-local rerank predictions aligned with the original input pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

ScoreKind = Literal["logit", "ordinal", "calibrated"]


@dataclass(frozen=True)
class RerankPrediction:
    """keep=False removes a candidate; None scores leave survivors unscored."""

    scores: list[float | None]
    keep: list[bool]
    served_provider: str | None
    score_kind: ScoreKind = "calibrated"
    timed_out: bool = False
    timeout_seconds: float | None = None
    model_name: str | None = None
    # Original-input indices of every kept candidate, in stable ranking order.
    # Optional so existing native providers retain input-order tie semantics.
    survivor_order: list[int] | None = None
    # A restored score can come from a different provider than the timeout.
    # Append metadata to preserve the existing positional constructor contract.
    timeout_provider: str | None = None


def validate_prediction(pred: RerankPrediction, expected_len: int, origin: str) -> None:
    """Validate against the input length before sorting or padding."""
    if len(pred.scores) != expected_len or len(pred.keep) != expected_len:
        raise ValueError(
            f"{origin}: expected {expected_len} scores/keep entries, "
            f"got scores={len(pred.scores)} keep={len(pred.keep)}"
        )
    for i, keep in enumerate(pred.keep):
        if not isinstance(keep, bool):
            raise ValueError(f"{origin}: keep[{i}] must be bool, got {type(keep).__name__}")
    for i, score in enumerate(pred.scores):
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{origin}: scores[{i}] must be float|None, got {type(score).__name__}")
        if not math.isfinite(score):
            raise ValueError(f"{origin}: scores[{i}] is not finite: {score!r}")
    if pred.survivor_order is not None:
        order = pred.survivor_order
        if (
            not isinstance(order, list)
            or any(type(index) is not int for index in order)
            or len(order) != len(set(order))
            or set(order) != {i for i, keep in enumerate(pred.keep) if keep}
        ):
            raise ValueError(f"{origin}: survivor_order must contain every kept input index exactly once")
    if pred.score_kind not in get_args(ScoreKind):
        raise ValueError(f"{origin}: unsupported score_kind {pred.score_kind!r}")
    if not isinstance(pred.timed_out, bool):
        raise ValueError(f"{origin}: timed_out must be bool, got {type(pred.timed_out).__name__}")
    if pred.timeout_seconds is not None:
        if isinstance(pred.timeout_seconds, bool) or not isinstance(pred.timeout_seconds, (int, float)):
            raise ValueError(f"{origin}: timeout_seconds must be float|None, got {type(pred.timeout_seconds).__name__}")
        if pred.timeout_seconds < 0 or not math.isfinite(pred.timeout_seconds):
            raise ValueError(f"{origin}: timeout_seconds is invalid: {pred.timeout_seconds!r}")


def stable_sigmoid(x: float) -> float:
    """Avoid overflow for large finite negative logits."""
    if x >= 50.0:
        return 1.0
    if x <= -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
