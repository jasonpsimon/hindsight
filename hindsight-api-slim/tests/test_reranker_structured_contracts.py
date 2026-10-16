"""Offline contracts exercised through the real factory and recall caller."""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError, create_cross_encoder_from_env
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder


class PublicProvider(CrossEncoderModel):
    def __init__(self, name, scores=None, error=None, prune=False):
        self.name = name
        self.scores = scores
        self.error = error
        self.prunes_candidates = prune
        self.calls = []
        self.init_calls = 0

    @property
    def provider_name(self):
        return self.name

    async def initialize(self):
        self.init_calls += 1

    async def predict(self, pairs):
        self.calls.append(list(pairs))
        if self.error is not None:
            raise self.error
        assert self.scores is not None
        return list(self.scores)


def candidates(n):
    return [
        MergedCandidate(
            retrieval=RetrievalResult(
                id=str(i),
                text=f"doc {i}",
                context="context",
                fact_type="world",
                occurred_start=datetime(2022, 6, 5, tzinfo=timezone.utc),
                occurred_end=None,
            ),
            rrf_score=1 / (i + 1),
        )
        for i in range(n)
    ]


def failover_factory(monkeypatch, primary, backup):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_1_PROVIDER", "rrf")
    members = iter([primary, backup])
    monkeypatch.setattr("hindsight_api.engine.cross_encoder.create_cross_encoder", lambda member: next(members))
    return create_cross_encoder_from_env()


async def test_factory_failover_keeps_member_owned_pruning(monkeypatch):
    primary = PublicProvider("local", error=RuntimeError("down"))
    backup = PublicProvider("typesafe", [0.0, 2.0, 1.0], prune=True)
    encoder = failover_factory(monkeypatch, primary, backup)
    result = await CrossEncoderReranker(encoder).rerank("q", candidates(3))
    assert [item.retrieval.id for item in result.results] == ["1", "2"]
    assert [item.weight for item in result.results] == [2.0, 1.0]
    assert result.provider_name == "typesafe"


@pytest.mark.parametrize("backup_scores", [[-2.0, 0.0, 1.0], [0.0, 0.2, 0.8]])
async def test_mixed_failover_does_not_prune_ordinary_scores(monkeypatch, backup_scores):
    primary = PublicProvider("typesafe", error=RuntimeError("down"), prune=True)
    backup = PublicProvider("local", backup_scores)
    result = await CrossEncoderReranker(failover_factory(monkeypatch, primary, backup)).rerank("q", candidates(3))
    assert len(result.results) == 3
    assert result.provider_name == "local"
    assert next(item for item in result.results if item.cross_encoder_score == 0.0).weight == 0.5


@pytest.mark.parametrize("scores", [[-2.0, None, 0.0], [None, None, None], [-2.0, 0.0, 1.0]])
@pytest.mark.parametrize("model", ["", "not-a-provider"])
async def test_final_backup_timeout_attribution_and_metadata(monkeypatch, scores, model):
    primary = PublicProvider("cohere", error=RuntimeError("down"))
    backup = PublicProvider("local", error=RerankTimeoutError(scores, 0.0, model))
    encoder = failover_factory(monkeypatch, primary, backup)
    pred = await encoder.predict_structured([("q", str(i)) for i in range(3)])
    assert pred.scores == scores
    assert pred.keep == [True] * 3
    assert pred.served_provider == "local"
    assert pred.score_kind == "logit"
    assert pred.timed_out is True
    assert pred.timeout_seconds == 0.0
    assert pred.model_name == model
    with pytest.raises(RerankTimeoutError) as caught:
        await encoder.predict([("q", str(i)) for i in range(3)])
    assert caught.value.model_name == model
    assert caught.value.timeout == 0.0
    assert caught.value.scores == scores
    result = await CrossEncoderReranker(encoder).rerank("q", candidates(3))
    assert result.provider_name == "local"
    assert len(result.results) == 3
    assert [
        item.retrieval.id
        for item in result.results
        if item.cross_encoder_score == 0 and scores[int(item.retrieval.id)] is None
    ] == [str(i) for i, score in enumerate(scores) if score is None]
    for item in result.results:
        raw = scores[int(item.retrieval.id)]
        if raw is None:
            assert item.weight == 0.0
        elif raw == 0.0:
            assert item.weight == 0.5


@pytest.mark.parametrize("scores", [[0.9, None], [0.9, 0.1]])
async def test_timeout_even_with_complete_scores_advances_to_backup(monkeypatch, scores):
    primary = PublicProvider("local", error=RerankTimeoutError(scores, 1, "model"))
    backup = PublicProvider("cohere", [0.2, 0.8])
    result = await CrossEncoderReranker(failover_factory(monkeypatch, primary, backup)).rerank("q", candidates(2))
    assert result.provider_name == "cohere"
    assert [item.weight for item in result.results] == [0.8, 0.2]


async def test_ordinary_final_error_never_restores_earlier_timeout(monkeypatch):
    primary = PublicProvider("local", error=RerankTimeoutError([0.9, None], 1, "model"))
    backup = PublicProvider("cohere", error=RuntimeError("final failure"))
    with pytest.raises(RuntimeError, match="final failure"):
        await CrossEncoderReranker(failover_factory(monkeypatch, primary, backup)).rerank("q", candidates(2))


class NativeProvider(PublicProvider):
    def __init__(self, prediction):
        super().__init__(prediction.served_provider or "native")
        self.prediction = prediction

    async def predict_structured(self, pairs):
        self.calls.append(list(pairs))
        return self.prediction


def prediction(scores, keep=None, **kwargs):
    from hindsight_api.engine.rerank_result import RerankPrediction

    return RerankPrediction(scores, [True] * len(scores) if keep is None else keep, "native", **kwargs)


def pipeline(*stages):
    from hindsight_api.engine.factory_pipeline import wrap_if_legacy
    from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder

    return PipelineCrossEncoder([(wrap_if_legacy(provider), limit) for provider, limit in stages])


@pytest.mark.parametrize("timed_out", [False, True])
async def test_pipeline_none_overwrites_and_timeout_only_fallback(timed_out):
    first = NativeProvider(prediction([0.9, 0.2, 0.1]))
    last = NativeProvider(
        prediction([None, None, None], [True, False, True], timed_out=timed_out, timeout_seconds=0.0, model_name="")
    )
    pred = await pipeline((first, None), (last, None)).predict_structured([("q", str(i)) for i in range(3)])
    assert pred.keep == [True, False, True]
    assert pred.scores == ([0.9, None, 0.1] if timed_out else [None, None, None])
    assert pred.timed_out is timed_out
    assert pred.timeout_seconds == 0.0
    assert pred.model_name == ""


async def test_pipeline_pruning_and_original_indices_reach_caller():
    first = NativeProvider(prediction([1.0, 3.0, 2.0, 0.0], [True, True, True, False]))
    last = NativeProvider(prediction([None, -1000.0], score_kind="logit"))
    encoder = pipeline((first, 2), (last, None))
    result = await CrossEncoderReranker(encoder).rerank("q", candidates(4))
    assert [item.retrieval.id for item in result.results] == ["2", "1"]
    assert [item.weight for item in result.results] == [0.0, 0.0]
    assert last.calls == [
        [
            ("q", "[Date: June 05, 2022 (2022-06-05)] context: doc 1"),
            ("q", "[Date: June 05, 2022 (2022-06-05)] context: doc 2"),
        ]
    ]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_pipeline_constructor_rejects_bad_limits(limit):
    from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder

    with pytest.raises(ValueError, match="positive int"):
        PipelineCrossEncoder([(PublicProvider("rrf", []), limit), (PublicProvider("rrf", []), None)])


@pytest.mark.parametrize(
    "bad",
    [
        prediction([0.1]),
        prediction([0.1, 0.2], [True]),
        prediction([0.1, 0.2], [True, 1]),
        prediction([float("nan"), 0.2]),
        prediction([float("inf"), 0.2]),
        prediction([True, 0.2]),
    ],
)
async def test_native_malformed_results_fail_closed_in_caller(bad):
    with pytest.raises(ValueError):
        await CrossEncoderReranker(NativeProvider(bad)).rerank("q", candidates(2))


@pytest.mark.parametrize("bad", [prediction([0.1]), prediction([0.1, 0.2], [True])])
async def test_malformed_member_advances_to_backup(monkeypatch, bad):
    encoder = failover_factory(monkeypatch, NativeProvider(bad), PublicProvider("cohere", [0.3, 0.7]))
    result = await CrossEncoderReranker(encoder).rerank("q", candidates(2))
    assert result.provider_name == "cohere"
    assert [item.weight for item in result.results] == [0.7, 0.3]


async def test_timeout_oversized_scores_rejected():
    from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter

    encoder = LegacyPredictAdapter(PublicProvider("local", error=RerankTimeoutError([0.1, None, None], 1, "model")))
    with pytest.raises(ValueError, match="3 scores for 2 pairs"):
        await encoder.predict_structured([("q", "a"), ("q", "b")])


async def test_timeout_short_scores_padded_without_pruning():
    from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter

    encoder = LegacyPredictAdapter(PublicProvider("local", error=RerankTimeoutError([-2.0], 1, "model"), prune=True))
    pred = await encoder.predict_structured([("q", "a"), ("q", "b")])
    assert pred.scores == [-2.0, None]
    assert pred.keep == [True, True]


@pytest.mark.parametrize("model", [None, ""])
async def test_native_optional_model_preserved(model):
    pred = prediction([None, None], timed_out=True, timeout_seconds=0, model_name=model)
    result = await pipeline(
        (NativeProvider(prediction([0.2, 0.8])), None), (NativeProvider(pred), None)
    ).predict_structured([("q", "a"), ("q", "b")])
    assert result.model_name == model
    assert result.timeout_seconds == 0


async def test_empty_caller_does_not_predict():
    encoder = NativeProvider(prediction([]))
    result = await CrossEncoderReranker(encoder).rerank("q", [])
    assert result.results == []
    assert encoder.calls == []


async def test_lazy_failed_readiness_retries_then_recovers(monkeypatch):
    class Recovering(PublicProvider):
        async def initialize(self):
            self.init_calls += 1
            if self.init_calls == 1:
                raise RuntimeError("init down")

    first = Recovering("cohere", [0.2, 0.8])
    backup = PublicProvider("rrf", [0.5, 0.5])
    encoder = failover_factory(monkeypatch, first, backup)
    await encoder.initialize()
    pred = await encoder.predict_structured([("q", "a"), ("q", "b")])
    assert pred.served_provider == "cohere"
    assert first.init_calls == 2
    assert backup.init_calls == 1


async def test_concurrent_readiness_initializes_member_once(monkeypatch):
    import asyncio

    started, release = asyncio.Event(), asyncio.Event()

    class Held(PublicProvider):
        async def initialize(self):
            self.init_calls += 1
            started.set()
            await release.wait()

    first = Held("cohere", [0.2, 0.8])
    encoder = failover_factory(monkeypatch, first, PublicProvider("rrf", [0.5, 0.5]))
    task1 = asyncio.create_task(encoder.predict_structured([("q", "a"), ("q", "b")]))
    await started.wait()
    task2 = asyncio.create_task(encoder.predict_structured([("q", "a"), ("q", "b")]))
    release.set()
    await asyncio.wait_for(asyncio.gather(task1, task2), 5)
    assert first.init_calls == 1


async def test_deterministic_concurrent_failover_metadata_and_masks(monkeypatch):
    import asyncio

    first_done, second_done = asyncio.Event(), asyncio.Event()

    class Primary(PublicProvider):
        async def predict(self, pairs):
            if pairs[0][0] == "backup":
                raise RuntimeError("down")
            return [-2.0, 0.0]

    class Interleaved(StructuredMultiCrossEncoder):
        async def predict_structured(self, pairs):
            pred = await super().predict_structured(pairs)
            if pairs[0][0] == "primary":
                first_done.set()
                await second_done.wait()
            else:
                second_done.set()
            return pred

    base = failover_factory(monkeypatch, Primary("local"), PublicProvider("typesafe", [0.0, 1.0], prune=True))
    encoder = Interleaved(base._members)
    reranker = CrossEncoderReranker(encoder)

    async def second():
        await first_done.wait()
        return await reranker.rerank("backup", candidates(2))

    result1, result2 = await asyncio.wait_for(asyncio.gather(reranker.rerank("primary", candidates(2)), second()), 5)
    assert result1.provider_name == "local"
    assert len(result1.results) == 2
    assert result2.provider_name == "typesafe"
    assert [item.retrieval.id for item in result2.results] == ["1"]
    assert encoder.provider_name == "typesafe"


async def test_adapter_member_owns_exact_retry_budget():
    from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter
    from hindsight_api.engine.remote_retry import RetryPolicy

    class RateLimited(Exception):
        status_code = 429

    class Raw(CrossEncoderModel):
        retry_policy = RetryPolicy(max_retries=2, initial_backoff=0.001, max_backoff=0.001, budget_seconds=2)

        def __init__(self):
            self.calls = []

        @property
        def provider_name(self):
            return "cohere"

        async def initialize(self):
            pass

        async def _predict(self, pairs):
            self.calls.append(list(pairs))
            raise RateLimited("429")

    raw = Raw()
    with pytest.raises(RateLimited):
        await LegacyPredictAdapter(raw).predict_structured([("q", "a")])
    assert len(raw.calls) == 3


async def test_public_only_provider_does_not_gain_outer_retries():
    from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter
    from hindsight_api.engine.remote_retry import RetryPolicy

    class RateLimited(Exception):
        status_code = 429

    provider = PublicProvider("cohere", error=RateLimited("429"))
    provider.retry_policy = RetryPolicy(max_retries=2, initial_backoff=0.001, max_backoff=0.001, budget_seconds=2)
    with pytest.raises(RateLimited):
        await LegacyPredictAdapter(provider).predict_structured([("q", "a")])
    assert len(provider.calls) == 1


async def test_adapter_honors_public_override_on_raw_hook_subclass():
    from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter

    class Raw(PublicProvider):
        async def _predict(self, pairs):
            return [0.1] * len(pairs)

    class PublicOverride(Raw):
        async def predict(self, pairs):
            return [0.9] * len(pairs)

    provider = PublicOverride("cohere")
    pred = await LegacyPredictAdapter(provider).predict_structured([("q", "a")])
    assert pred.scores == [0.9]
