"""Native raised timeouts retain explicit kinds at direct and failover boundaries."""

import math

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.rerank_result import RerankPrediction
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder


class NativeTimeout(CrossEncoderModel):
    def __init__(self, provider: str, kind: str | None, scores: list[float | None]):
        self.name = provider
        if kind is not None:
            self.score_kind = kind
        self.scores = scores

    @property
    def provider_name(self) -> str:
        return self.name

    async def initialize(self) -> None:
        pass

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        assert len(pairs) == 2
        raise RerankTimeoutError(self.scores, 0.0, "model")


class FailedPrimary(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("offline primary failure")


def candidates() -> list[MergedCandidate]:
    return [
        MergedCandidate(
            retrieval=RetrievalResult(
                id=str(i), text=str(i), fact_type="world", occurred_start=None, occurred_end=None
            ),
            rrf_score=0.5,
        )
        for i in range(2)
    ]


@pytest.mark.parametrize("route", ["direct", "failover"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize(
    "provider,kind,raw",
    [("custom-local", "logit", -2.0), ("local", "ordinal", -2.0), ("local", "calibrated", -0.8)],
)
async def test_native_timeout_keeps_explicit_kind(route: str, partial: bool, provider: str, kind: str, raw: float):
    # A short partial list also verifies padding preserves original score positions.
    scores: list[float | None] = [raw] if partial else [raw, 0.3]
    native = NativeTimeout(provider, kind, scores)
    encoder = native if route == "direct" else StructuredMultiCrossEncoder([FailedPrimary(), native])
    result = await CrossEncoderReranker(encoder).rerank("q", candidates())
    assert result.provider_name == provider
    assert len(result.results) == 2
    by_id = {item.retrieval.id: item for item in result.results}
    for i, score in enumerate(scores + [None] * (2 - len(scores))):
        expected = 0.0 if score is None else 1.0 / (1.0 + math.exp(-score)) if kind == "logit" else score
        assert by_id[str(i)].weight == pytest.approx(expected)
        assert by_id[str(i)].cross_encoder_score == (0.0 if score is None else score)
    if route == "failover":
        pred = await encoder.predict_structured([("q", "0"), ("q", "1")])
        assert pred.score_kind == kind
        assert pred.scores == scores + [None] * (2 - len(scores))
        assert pred.keep == [True, True]
        assert pred.served_provider == provider
        assert pred.timed_out is True
        assert pred.timeout_seconds == 0.0
        assert pred.model_name == "model"


@pytest.mark.parametrize("route", ["direct", "failover"])
async def test_native_timeout_rejects_malformed_kind(route: str):
    native = NativeTimeout("local", "invalid-kind", [-2.0, 0.3])
    encoder = native if route == "direct" else StructuredMultiCrossEncoder([FailedPrimary(), native])
    with pytest.raises(ValueError, match="unsupported score_kind"):
        await CrossEncoderReranker(encoder).rerank("q", candidates())


@pytest.mark.parametrize("route", ["direct", "failover"])
@pytest.mark.parametrize("provider", ["local", "custom-local"])
async def test_native_timeout_without_kind_uses_provider_default(route: str, provider: str):
    native = NativeTimeout(provider, None, [-2.0])
    assert not hasattr(native, "score_kind")
    encoder = native if route == "direct" else StructuredMultiCrossEncoder([FailedPrimary(), native])
    result = await CrossEncoderReranker(encoder).rerank("q", candidates())
    assert result.provider_name == provider
    assert len(result.results) == 2
    by_id = {item.retrieval.id: item for item in result.results}
    expected = 1.0 / (1.0 + math.exp(2.0)) if provider == "local" else -2.0
    assert by_id["0"].weight == pytest.approx(expected)
    assert by_id["0"].cross_encoder_score == -2.0
    assert by_id["1"].weight == 0.0
    assert by_id["1"].cross_encoder_score == 0.0
