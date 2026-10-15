"""Offline regressions for independently confirmed provider and pipeline defects."""

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.factory_pipeline import wrap_if_legacy
from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter, PipelineCrossEncoder
from hindsight_api.engine.rerank_result import RerankPrediction


class PublicStage(CrossEncoderModel):
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores
        self.calls = 0

    @property
    def provider_name(self):
        return self.name

    async def initialize(self):
        pass

    async def predict(self, pairs):
        self.calls += 1
        return self.scores


class NativeStage(PublicStage):
    async def predict_structured(self, pairs):
        return self.scores


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_direct_stage_timeout_alignment(native, partial):
    error = RerankTimeoutError([0.7] if partial else [], 0.0, "")

    class RaisedNative(NativeStage):
        score_kind = "ordinal"

        async def predict_structured(self, pairs):
            raise error

    class RaisedPublic(PublicStage):
        async def predict(self, pairs):
            raise error

    last = (RaisedNative if native else RaisedPublic)("typesafe", None)
    pipeline = PipelineCrossEncoder([(PublicStage("cohere", [0.1, 0.9, 0.8]), 2), (last, None)])
    pred = await pipeline.predict_structured([("q", str(i)) for i in range(3)])
    assert pred.keep == [False, True, True]
    assert pred.scores == ([None, 0.7, None] if partial else [None, 0.9, 0.8])
    assert pred.score_kind == ("ordinal" if partial else "calibrated")
    assert pred.served_provider == ("typesafe" if partial else "cohere")
    assert pred.timed_out is True
    assert pred.timeout_seconds == 0.0
    assert pred.model_name == ""


async def test_constructor_adaptation_is_idempotent_and_native_stays_native():
    adapted = wrap_if_legacy(PublicStage("cohere", [0.8, 0.2]))
    native = NativeStage("cohere", RerankPrediction([0.6, 0.4], [True, True], "cohere"))
    pipeline = PipelineCrossEncoder([(adapted, None), (native, None)])
    assert pipeline._stages[0][0] is adapted
    assert pipeline._stages[1][0] is native
    assert wrap_if_legacy(adapted) is adapted
    assert wrap_if_legacy(native) is native
    assert (await pipeline.predict_structured([("q", "a"), ("q", "b")])).scores == [0.6, 0.4]


@pytest.mark.parametrize("native", [False, True])
async def test_complete_raised_timeout_is_still_stage_local(native):
    error = RerankTimeoutError([0.7, -0.2], 0.0, "")

    class Raised(PublicStage):
        async def predict(self, pairs):
            assert pairs == [("q", "1"), ("q", "2")]
            raise error

    class NativeRaised(Raised):
        score_kind = "ordinal"

        async def predict_structured(self, pairs):
            return await self.predict(pairs)

    last = (NativeRaised if native else Raised)("local", None)
    pred = await PipelineCrossEncoder([(PublicStage("cohere", [0.1, 0.9, 0.8]), 2), (last, None)]).predict_structured(
        [("q", str(i)) for i in range(3)]
    )
    assert pred.keep == [False, True, True]
    assert pred.scores == [None, 0.7, -0.2]
    assert pred.served_provider == "local"
    assert pred.score_kind == ("ordinal" if native else "logit")
    assert pred.timed_out is True
    assert pred.timeout_seconds == 0.0
    assert pred.model_name == ""


async def test_direct_raw_hook_retries_have_one_owner(monkeypatch):
    from hindsight_api.engine.remote_retry import RetryPolicy

    budgets = []

    async def retry(call, *, policy, budget, provider):
        budgets.append(budget)
        try:
            return await call()
        except RuntimeError:
            return await call()

    monkeypatch.setattr("hindsight_api.engine.remote_retry.acall_with_retry", retry)
    monkeypatch.setattr("hindsight_api.engine.cross_encoder.acall_with_retry", retry)

    class RawStage(PublicStage):
        predict = CrossEncoderModel.predict
        retry_policy = RetryPolicy()

        async def _predict(self, pairs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry me")
            return self.scores

    raw = RawStage("cohere", [0.8, 0.2])
    public = PublicStage("cohere", [0.6, 0.4])
    public.retry_policy = RetryPolicy()
    pipeline = PipelineCrossEncoder([(raw, None), (public, None)])
    assert isinstance(pipeline._stages[0][0], LegacyPredictAdapter)
    await pipeline.predict_structured([("q", "a"), ("q", "b")])
    assert len(budgets) == 1
    assert raw.calls == 2
    assert public.calls == 1


@pytest.mark.parametrize("route", ["pipeline", "failover"])
async def test_signed_jina_cosine_passes_through(route):
    from hindsight_api.engine.search.reranking import CrossEncoderReranker
    from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
    from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder

    jina = PublicStage("jina-mlx", [-0.8, 0.2])
    if route == "pipeline":
        # Preserve this test's document identities into Jina. The order-changing
        # case is covered separately by ordered-survivor entry-point regressions.
        encoder = PipelineCrossEncoder([(PublicStage("cohere", [0.6, 0.5]), None), (jina, None)])
    else:

        class Down(PublicStage):
            async def predict(self, pairs):
                raise RuntimeError("down")

        encoder = StructuredMultiCrossEncoder([Down("cohere", None), jina])
    candidates = [
        MergedCandidate(
            retrieval=RetrievalResult(
                id=str(i), text=str(i), fact_type="world", occurred_start=None, occurred_end=None
            ),
            rrf_score=0.5,
        )
        for i in range(2)
    ]
    result = await CrossEncoderReranker(encoder).rerank("q", candidates)
    assert [item.retrieval.id for item in result.results] == ["1", "0"]
    assert [item.weight for item in result.results] == pytest.approx([0.2, -0.8])
    assert [item.cross_encoder_score for item in result.results] == pytest.approx([0.2, -0.8])
    assert result.provider_name == "jina-mlx"


@pytest.mark.parametrize("model", [None, "", "final-model"])
@pytest.mark.parametrize("duration", [None, 0.0])
async def test_total_timeout_metadata_is_final_stage_metadata(model, duration):
    first = NativeStage(
        "cohere",
        RerankPrediction([0.9, 0.2], [True, True], "cohere", "ordinal", True, 8.0, "first-model"),
    )
    final = NativeStage("local", RerankPrediction([None, None], [True, True], "local", "logit", True, duration, model))
    pred = await PipelineCrossEncoder([(first, None), (final, None)]).predict_structured([("q", "a"), ("q", "b")])
    assert pred.scores == [0.9, 0.2]
    assert pred.keep == [True, True]
    assert pred.served_provider == "cohere"
    assert pred.score_kind == "ordinal"
    assert pred.timed_out is True
    assert pred.timeout_seconds == duration
    assert pred.model_name == model


async def test_timeout_limit_applies_stable_stage_input_before_score_restoration():
    first = PublicStage("cohere", [0.1, 0.9, 0.8])

    class TimedOut(NativeStage):
        async def predict_structured(self, pairs):
            assert pairs == [("q", "1"), ("q", "2"), ("q", "0")]
            return self.scores

    final = TimedOut("local", RerankPrediction([None] * 3, [True] * 3, "local", "logit", True, 0.0))
    pred = await PipelineCrossEncoder([(first, None), (final, 1)]).predict_structured([("q", str(i)) for i in range(3)])
    assert pred.keep == [False, True, False]
    assert pred.scores == [None, 0.9, None]
    assert pred.served_provider == "cohere"
    assert pred.score_kind == "calibrated"
    assert pred.timed_out is True
    assert pred.timeout_seconds == 0.0
    assert pred.model_name is None


@pytest.mark.parametrize("scalar", ["float32", "float64", "int64"])
async def test_flashrank_numeric_scalars_become_python_floats(monkeypatch, scalar):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from hindsight_api.engine.cross_encoder import FlashRankCrossEncoder

    module = ModuleType("flashrank")
    module.RerankRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "flashrank", module)

    class Ranker:
        def rerank(self, request):
            return [{"id": 0, "score": getattr(np, scalar)(1)}]

    provider = FlashRankCrossEncoder()
    provider._ranker = Ranker()
    pred = await LegacyPredictAdapter(provider).predict_structured([("q", "a")])
    assert pred.scores == [1.0]
    assert type(pred.scores[0]) is float
    assert pred.keep == [True]
    assert pred.score_kind == "calibrated"


@pytest.mark.parametrize("bad", [True, "0.2", object(), float("nan"), float("inf")])
async def test_flashrank_invalid_scores_still_rejected(monkeypatch, bad):
    import sys
    from types import ModuleType, SimpleNamespace

    from hindsight_api.engine.cross_encoder import FlashRankCrossEncoder

    module = ModuleType("flashrank")
    module.RerankRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "flashrank", module)

    class Ranker:
        def rerank(self, request):
            return [{"id": 0, "score": bad}]

    provider = FlashRankCrossEncoder()
    provider._ranker = Ranker()
    with pytest.raises(ValueError):
        await LegacyPredictAdapter(provider).predict_structured([("q", "a")])


@pytest.mark.parametrize("prune", [False, True])
async def test_typesafe_public_singleton_keeps_relevance_cut(monkeypatch, prune):
    from hindsight_api.engine.cross_encoder import TypeSafeCrossEncoder

    provider = TypeSafeCrossEncoder(api_key="offline-dummy", prune_candidates=prune)
    bodies = []

    async def ask(body):
        bodies.append(body)
        assert "rank" not in body["questions"]
        return {"answers": {"depth": {"score": 0}}}

    monkeypatch.setattr(provider, "_ask", ask)
    assert await provider.predict([("q", "single doc")]) == [1.0]
    assert len(bodies) == int(prune)
    if prune:
        assert "[1] single doc" in bodies[0]["state"]
        assert len(bodies[0]["questions"]["depth"]["criteria"]) >= 2


async def test_typesafe_singleton_cut_still_controls_pruning(monkeypatch):
    from hindsight_api.engine.cross_encoder import TypeSafeCrossEncoder

    provider = TypeSafeCrossEncoder(api_key="offline-dummy", prune_candidates=True)
    cuts = []

    async def cut(query, docs, order):
        cuts.append(list(order))
        return 0

    async def ask(body):
        pytest.fail("singleton rank must not call the endpoint")

    monkeypatch.setattr(provider, "_cut", cut)
    monkeypatch.setattr(provider, "_ask", ask)
    pred = await LegacyPredictAdapter(provider).predict_structured([("q", "a")])
    assert cuts == [[0]]
    assert pred.scores == [0.0]
    assert pred.keep == [False]
