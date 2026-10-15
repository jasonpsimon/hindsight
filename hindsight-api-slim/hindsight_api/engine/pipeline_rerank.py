"""Progressive reranking and adapters for the existing list-based providers.

Each backend owns its retry budget. Keep masks and timeout metadata belong to a
prediction, never an instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rerank_result import RerankPrediction, ScoreKind, validate_prediction

if TYPE_CHECKING:
    from .cross_encoder import CrossEncoderModel, RerankTimeoutError
    from .remote_retry import RetryPolicy


def default_score_kind(provider_name: str) -> ScoreKind:
    if provider_name == "local":
        return "logit"
    if provider_name in {"typesafe", "decision_model"}:
        return "ordinal"
    # The calibrated transport bucket means pass-through, not probability:
    # Jina MLX emits signed cosine similarity; FlashRank already applies
    # sigmoid/softmax. Neither should receive another sigmoid.
    return "calibrated"


def timeout_to_structured(
    exc: RerankTimeoutError,
    expected_len: int,
    provider_name: str,
    model_name: str | None = None,
    score_kind: ScoreKind | None = None,
) -> RerankPrediction:
    """Timeout is not rejection. Preserve explicit zero budgets and empty names."""
    if len(exc.scores) > expected_len:
        raise ValueError(f"{provider_name}: timeout returned {len(exc.scores)} scores for {expected_len} pairs")
    scores = list(exc.scores) + [None] * (expected_len - len(exc.scores))
    if exc.score_kind is not None:
        score_kind = exc.score_kind
    pred = RerankPrediction(
        scores=scores,
        keep=exc.keep if exc.keep is not None else [True] * expected_len,
        served_provider=provider_name,
        score_kind=default_score_kind(provider_name) if score_kind is None else score_kind,
        timed_out=True,
        timeout_seconds=exc.timeout,
        model_name=exc.model_name if model_name is None else model_name,
        survivor_order=getattr(exc, "survivor_order", None),
    )
    validate_prediction(pred, expected_len, origin=provider_name)
    return pred


def list_to_structured(
    raw: list[float], expected_len: int, provider_name: str, prunes_candidates: bool = False
) -> RerankPrediction:
    if len(raw) != expected_len:
        raise ValueError(f"{provider_name}: expected {expected_len} scores, got {len(raw)}")
    pred = RerankPrediction(
        scores=list(raw),
        keep=[s > 0.0 for s in raw] if prunes_candidates else [True] * expected_len,
        served_provider=provider_name,
        score_kind=default_score_kind(provider_name),
    )
    validate_prediction(pred, expected_len, origin=provider_name)
    return pred


class LegacyPredictAdapter:
    """Use the public backend contract so truncation and retry have one owner."""

    retry_policy: RetryPolicy | None = None

    def __init__(self, inner: CrossEncoderModel) -> None:
        self._inner = inner

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def primary_provider_name(self) -> str:
        return self._inner.primary_provider_name

    @property
    def score_provider_names(self) -> frozenset[str]:
        return self._inner.score_provider_names

    @property
    def blocking_init(self) -> bool:
        return self._inner.blocking_init

    @property
    def prunes_candidates(self) -> bool:
        return self._inner.prunes_candidates

    async def initialize(self) -> None:
        await self._inner.initialize()

    def _publish_served(self) -> None:
        from .cross_encoder import _served_provider

        _served_provider.set(self.provider_name)

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        pred = await self._predict_structured(pairs)
        self._publish_served()
        return pred

    async def _predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        from .cross_encoder import RerankTimeoutError

        try:
            raw = await self._inner.predict(pairs)
        except RerankTimeoutError as exc:
            return timeout_to_structured(exc, len(pairs), self.provider_name)
        if not isinstance(raw, list):
            raise ValueError(f"{self.provider_name}: expected list of scores, got {type(raw).__name__}")
        return list_to_structured(raw, len(pairs), self.provider_name, self.prunes_candidates)

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        from .cross_encoder import RerankTimeoutError

        pred = await self.predict_structured(pairs)
        if pred.timed_out:
            err = RerankTimeoutError(
                scores=pred.scores,
                timeout=pred.timeout_seconds if pred.timeout_seconds is not None else 0.0,
                model_name=pred.model_name if pred.model_name is not None else self.provider_name,
            )
            err.provider_name = self.provider_name
            err.keep = pred.keep
            err.survivor_order = pred.survivor_order
            err.score_kind = pred.score_kind
            raise err
        return [score if keep and score is not None else 0.0 for score, keep in zip(pred.scores, pred.keep)]


class PipelineCrossEncoder:
    """Stages receive only survivors. Scores overwrite prior scores, even None.

    A total timeout may restore the last usable scores, but never undo a later
    keep=False decision or a failed stage's shortlist limit.
    """

    prunes_candidates: bool = False
    retry_policy = None

    def __init__(self, stages: list[tuple[CrossEncoderModel, int | None]]) -> None:
        from .factory_pipeline import wrap_if_legacy

        if len(stages) < 2:
            raise ValueError("PipelineCrossEncoder requires at least two stages")
        for i, (_, limit) in enumerate(stages):
            if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
                raise ValueError(f"Pipeline stage {i} limit must be a positive int or None, got {limit!r}")
        # Direct construction has the same boundary as the factory. Native
        # structured stages stay native; legacy backends retain retry ownership.
        self._stages = [(wrap_if_legacy(encoder), limit) for encoder, limit in stages]

    @property
    def provider_name(self) -> str:
        return "pipeline:" + "→".join(enc.provider_name for enc, _ in self._stages)

    @property
    def primary_provider_name(self) -> str:
        # All passes run normally; the final stage supplies the normal scores.
        return self._stages[-1][0].primary_provider_name

    @property
    def score_provider_names(self) -> frozenset[str]:
        # A total timeout can restore an earlier stage's scores. A floor must
        # therefore be valid for every stage, not just the final provider.
        return frozenset(name for encoder, _ in self._stages for name in encoder.score_provider_names)

    @property
    def blocking_init(self) -> bool:
        return any(enc.blocking_init for enc, _ in self._stages)

    async def initialize(self) -> None:
        for enc, _ in self._stages:
            await enc.initialize()

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        from .cross_encoder import _served_provider

        pred = await self._predict_structured(pairs)
        # A total timeout may restore an earlier source. Publish that source,
        # rather than leaving the last adapter's provider in task-local state.
        _served_provider.set(pred.served_provider)
        return pred

    async def _predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        from .cross_encoder import RerankTimeoutError

        n = len(pairs)
        active = list(range(n))
        scores: list[float | None] = [None] * n
        keep = [True] * n
        last_provider: str | None = None
        last_kind: ScoreKind = "calibrated"
        last_timed_out = False
        last_timeout: float | None = None
        last_model: str | None = None
        previous_usable: RerankPrediction | None = None

        for encoder, limit in self._stages:
            if not active:
                break
            if any(scores[i] is not None for i in active):
                previous_usable = RerankPrediction(
                    list(scores),
                    list(keep),
                    last_provider,
                    last_kind,
                    last_timed_out,
                    last_timeout,
                    last_model,
                    list(active),
                )
            stage_pairs = [pairs[i] for i in active]
            try:
                stage_pred = await encoder.predict_structured(stage_pairs)
            except RerankTimeoutError as exc:
                # Exception scores refer to this stage's survivors, not the
                # original request. Convert before mapping back to original indices.
                served = getattr(exc, "provider_name", None)
                if served is None:
                    served = encoder.provider_name
                stage_pred = timeout_to_structured(
                    exc, len(active), served, score_kind=getattr(encoder, "score_kind", None)
                )
            validate_prediction(stage_pred, len(active), origin=encoder.provider_name)
            stage_order = stage_pred.survivor_order
            if stage_order is None:
                stage_order = [i for i, kept in enumerate(stage_pred.keep) if kept]
            for local_i, orig_i in enumerate(active):
                if not stage_pred.keep[local_i]:
                    keep[orig_i] = False
                    scores[orig_i] = None
                    continue
                scores[orig_i] = stage_pred.scores[local_i]
            new_active = [active[i] for i in stage_order]
            active = new_active
            # Always order survivors, even without a keep limit. Next-stage rank
            # providers and timeout tails depend on this exact input order.
            active.sort(key=lambda i: (scores[i] is not None, scores[i] or 0.0), reverse=True)
            if limit is not None and len(active) > limit:
                for orig_i in active[limit:]:
                    keep[orig_i] = False
                    scores[orig_i] = None
                active = active[:limit]
            last_provider = encoder.provider_name if stage_pred.served_provider is None else stage_pred.served_provider
            last_kind = stage_pred.score_kind
            last_timed_out = stage_pred.timed_out
            last_timeout = stage_pred.timeout_seconds
            last_model = stage_pred.model_name

        if active and all(scores[i] is None for i in active) and last_timed_out and previous_usable is not None:
            restored_scores = list(previous_usable.scores)
            restored_keep = list(previous_usable.keep)
            for i in range(n):
                if not keep[i]:
                    restored_keep[i] = False
                    restored_scores[i] = None
            restored = RerankPrediction(
                scores=restored_scores,
                keep=restored_keep,
                served_provider=previous_usable.served_provider,
                score_kind=previous_usable.score_kind,
                timed_out=True,
                # Restored score provenance is separate from the final timeout.
                # Explicit absence, zero duration and empty names are all meaningful.
                timeout_seconds=last_timeout,
                model_name=last_model,
                survivor_order=[i for i in previous_usable.survivor_order or [] if restored_keep[i]],
            )
            validate_prediction(restored, n, origin=self.provider_name)
            return restored
        pred = RerankPrediction(
            scores, keep, last_provider, last_kind, last_timed_out, last_timeout, last_model, list(active)
        )
        validate_prediction(pred, n, origin=self.provider_name)
        return pred

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        from .cross_encoder import RerankTimeoutError

        pred = await self.predict_structured(pairs)
        if pred.timed_out:
            error = RerankTimeoutError(
                scores=pred.scores,
                timeout=pred.timeout_seconds if pred.timeout_seconds is not None else 0.0,
                model_name=pred.model_name if pred.model_name is not None else self.provider_name,
            )
            error.provider_name = pred.served_provider
            error.keep = pred.keep
            error.survivor_order = pred.survivor_order
            error.score_kind = pred.score_kind
            raise error
        return [score if keep and score is not None else 0.0 for score, keep in zip(pred.scores, pred.keep)]
