"""Structured failover transport retaining the existing member lifecycle."""

import logging
from dataclasses import replace

from .cross_encoder import MultiCrossEncoder, RerankTimeoutError, _served_provider
from .pipeline_rerank import LegacyPredictAdapter, timeout_to_structured
from .rerank_result import RerankPrediction, validate_prediction

logger = logging.getLogger(__name__)


class StructuredMultiCrossEncoder(MultiCrossEncoder):
    """Only the last outcome can supply partial scores. Members own retries.

    Keep original member instances for provider APIs and readiness handling.
    Adapt at the call boundary, never mutate their predict methods.
    """

    retry_policy = None

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        last: Exception | RerankPrediction | None = None
        for index, member in enumerate(self._members):
            try:
                if not self._ready[index]:
                    await self._ensure_member_ready(index)
                structured = getattr(member, "predict_structured", None)
                encoder = member if callable(structured) else LegacyPredictAdapter(member)
                try:
                    pred = await encoder.predict_structured(pairs)
                except RerankTimeoutError as exc:
                    served = getattr(exc, "provider_name", None)
                    pred = timeout_to_structured(
                        exc,
                        len(pairs),
                        member.provider_name if served is None else served,
                        score_kind=getattr(member, "score_kind", None),
                    )
                if not isinstance(pred, RerankPrediction):
                    raise ValueError("member must return RerankPrediction")
                validate_prediction(pred, len(pairs), origin=f"failover member {index}")
                if pred.served_provider is None:
                    pred = replace(pred, served_provider=member.provider_name)
            except Exception as exc:
                last = exc
                logger.warning("Reranker member %d failed: %s", index, exc)
                continue
            if pred.timed_out:
                last = pred
                continue
            self._active = index
            _served_provider.set(pred.served_provider)
            return pred
        if isinstance(last, RerankPrediction):
            _served_provider.set(last.served_provider)
            return last
        assert last is not None
        raise last

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        pred = await self.predict_structured(pairs)
        if pred.timed_out:
            exc = RerankTimeoutError(
                pred.scores,
                pred.timeout_seconds if pred.timeout_seconds is not None else 0.0,
                pred.model_name if pred.model_name is not None else self.provider_name,
            )
            exc.provider_name = pred.served_provider
            exc.keep = pred.keep
            exc.survivor_order = pred.survivor_order
            exc.score_kind = pred.score_kind
            exc.timeout_provider = pred.timeout_provider
            raise exc
        return [score if keep and score is not None else 0.0 for score, keep in zip(pred.scores, pred.keep)]

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self.predict(pairs)
