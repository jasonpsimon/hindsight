"""Mask-aware caller for encoders that explicitly support structured predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..cross_encoder import RerankTimeoutError, _served_provider
from ..pipeline_rerank import timeout_to_structured
from ..rerank_result import RerankPrediction, stable_sigmoid, validate_prediction
from .types import MergedCandidate, ScoredResult

if TYPE_CHECKING:
    from .reranking import RerankResult


async def rerank_with_structured(
    cross_encoder: Any, pairs: list[tuple[str, str]], candidates: list[MergedCandidate]
) -> RerankResult:
    from .reranking import RerankResult

    token = _served_provider.set(None)
    try:
        try:
            pred = await cross_encoder.predict_structured(pairs)
        except RerankTimeoutError as exc:
            name = getattr(exc, "provider_name", None)
            if name is None:
                name = _served_provider.get()
            if name is None:
                name = cross_encoder.provider_name
            pred = timeout_to_structured(exc, len(pairs), name, score_kind=getattr(cross_encoder, "score_kind", None))
        if not isinstance(pred, RerankPrediction):
            raise ValueError("CrossEncoderReranker.rerank: expected RerankPrediction")
        validate_prediction(pred, len(candidates), origin="CrossEncoderReranker.rerank")
        served = pred.served_provider
        if served is None:
            served = _served_provider.get()
    finally:
        _served_provider.reset(token)
    if served is None:
        served = cross_encoder.provider_name
    scored: list[ScoredResult] = []
    unscored: list[ScoredResult] = []
    # Original-index reconstruction used to erase final-stage ties and timeout
    # tails. Use the prediction's validated order as the stable sort seed.
    order = pred.survivor_order if pred.survivor_order is not None else range(len(candidates))
    for index in order:
        candidate = candidates[index]
        if not pred.keep[index]:
            continue
        raw = pred.scores[index]
        norm = 0.0 if raw is None else stable_sigmoid(raw) if pred.score_kind == "logit" else float(raw)
        result = ScoredResult(
            candidate=candidate,
            cross_encoder_score=0.0 if raw is None else float(raw),
            cross_encoder_score_normalized=norm,
            weight=norm,
        )
        (unscored if raw is None else scored).append(result)
    scored.sort(key=lambda result: result.weight, reverse=True)
    scored.extend(unscored)
    return RerankResult(results=scored, provider_name=served, score_kind=pred.score_kind)
