"""Final caller ordering keeps the last stage's stable survivor identities."""

import asyncio

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder, timeout_to_structured
from hindsight_api.engine.rerank_result import RerankPrediction, validate_prediction
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder


class Native(CrossEncoderModel):
    def __init__(self, scores, keep=None, timed_out=False):
        self.scores = scores
        self.keep = keep
        self.timed_out = timed_out
        self.calls = []

    @property
    def provider_name(self):
        return "cohere"

    async def initialize(self):
        pass

    async def predict_structured(self, pairs):
        self.calls.append(list(pairs))
        return RerankPrediction(
            self.scores,
            self.keep if self.keep is not None else [True] * len(pairs),
            "cohere",
            timed_out=self.timed_out,
        )


def candidates(duplicate=False):
    return [
        MergedCandidate(RetrievalResult(id=name, text="same" if duplicate else name, fact_type="world"), 0.5)
        for name in "ABC"
    ]


@pytest.mark.parametrize(
    "scores,keep,limit,timed_out,expected",
    [
        ([0.7, 0.7, 0.7], None, 2, False, ["C", "B"]),
        ([None, None, None], None, None, False, ["C", "B", "A"]),
        ([None, 0.4, None], None, None, True, ["B", "C", "A"]),
        ([0.7, 0.7, 0.7], [True, False, True], None, False, ["C", "A"]),
    ],
)
async def test_final_caller_retains_stage_input_order(scores, keep, limit, timed_out, expected):
    first = Native([0.2, 0.5, 0.9])
    final = Native(scores, keep, timed_out)
    pipeline = PipelineCrossEncoder([(first, None), (final, limit)])
    result = await CrossEncoderReranker(pipeline).rerank("q", candidates())
    assert first.calls == [[("q", x) for x in "ABC"]]
    assert final.calls == [[("q", x) for x in "CBA"]]
    assert [item.id for item in result.results] == expected


async def test_three_stage_restoration_keeps_masks_limits_and_snapshot_order():
    first = Native([0.2, 0.5, 0.9])
    middle = Native([None, None, None], [True, False, True])
    final = Native([None, None], timed_out=True)
    pipeline = PipelineCrossEncoder([(first, None), (middle, None), (final, 1)])
    result = await CrossEncoderReranker(pipeline).rerank("q", candidates())
    assert middle.calls == [[("q", x) for x in "CBA"]]
    assert final.calls == [[("q", x) for x in "CA"]]
    assert [item.id for item in result.results] == ["C"]
    assert [item.weight for item in result.results] == [0.9]


async def test_duplicate_text_retains_distinct_ids():
    pipeline = PipelineCrossEncoder([(Native([0.2, 0.5, 0.9]), None), (Native([0.7] * 3), 2)])
    result = await CrossEncoderReranker(pipeline).rerank("q", candidates(duplicate=True))
    assert [item.id for item in result.results] == ["C", "B"]
    assert all(item.retrieval.text == "same" for item in result.results)


async def test_failover_carries_native_order():
    class Ordered(Native):
        async def predict_structured(self, pairs):
            return RerankPrediction([0.5] * 3, [True] * 3, "cohere", survivor_order=[2, 0, 1])

    result = await CrossEncoderReranker(StructuredMultiCrossEncoder([Ordered([]), Native([])])).rerank(
        "q", candidates()
    )
    assert [item.id for item in result.results] == ["C", "A", "B"]


@pytest.mark.parametrize("order", [[2, 2, 0], [2], [2, 0, 3], [True, 0, 2], [2, 0, 1]])
def test_survivor_order_validation(order):
    pred = RerankPrediction([0.5] * 3, [True, False, True], "cohere", survivor_order=order)
    with pytest.raises(ValueError, match="survivor_order"):
        validate_prediction(pred, 3, "test")


async def test_cancellation_does_not_change_next_request_order():
    entered = asyncio.Event()

    class Cancellable(Native):
        async def predict_structured(self, pairs):
            if pairs[0][0] == "cancel":
                entered.set()
                await asyncio.Future()
            return await super().predict_structured(pairs)

    pipeline = PipelineCrossEncoder([(Native([0.2, 0.5, 0.9]), None), (Cancellable([0.7] * 3), 2)])
    reranker = CrossEncoderReranker(pipeline)
    task = asyncio.create_task(reranker.rerank("cancel", candidates()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await reranker.rerank("q", candidates())
    assert [item.id for item in result.results] == ["C", "B"]


async def test_concurrent_masks_and_restoration_are_request_local():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Interleaved(Native):
        async def predict_structured(self, pairs):
            if pairs[0][0] == "restore":
                entered.set()
                await release.wait()
                return RerankPrediction([None] * 3, [True, False, True], "cohere", timed_out=True)
            return RerankPrediction([0.7] * 3, [False, True, True], "cohere")

    pipeline = PipelineCrossEncoder([(Native([0.2, 0.5, 0.9]), None), (Interleaved([]), None)])
    reranker = CrossEncoderReranker(pipeline)
    pending = asyncio.create_task(reranker.rerank("restore", candidates()))
    await entered.wait()
    other = await reranker.rerank("other", candidates())
    release.set()
    restored = await pending
    assert [item.id for item in other.results] == ["B", "A"]
    assert [item.id for item in restored.results] == ["C", "A"]
    assert [item.weight for item in restored.results] == [0.9, 0.2]


async def test_native_order_is_mapped_into_next_stage_input():
    class Ordered(Native):
        async def predict_structured(self, pairs):
            return RerankPrediction([0.5] * 3, [True, False, True], "cohere", survivor_order=[2, 0])

    final = Native([None, None])
    result = await CrossEncoderReranker(PipelineCrossEncoder([(Ordered([]), None), (final, None)])).rerank(
        "q", candidates()
    )
    assert final.calls == [[("q", "C"), ("q", "A")]]
    assert [item.id for item in result.results] == ["C", "A"]


async def test_public_timeout_roundtrip_keeps_order_mask_and_kind():
    class Ordinal(Native):
        async def predict_structured(self, pairs):
            return RerankPrediction([0.2, 0.5, 0.9], [True] * 3, "local", "ordinal")

    pipeline = PipelineCrossEncoder([(Ordinal([]), None), (Native([None] * 3, timed_out=True), 2)])
    with pytest.raises(RerankTimeoutError) as caught:
        await pipeline.predict([("q", x) for x in "ABC"])
    pred = timeout_to_structured(caught.value, 3, caught.value.provider_name)
    assert pred.survivor_order == [2, 1]
    assert pred.keep == [False, True, True]
    assert pred.scores == [None, 0.5, 0.9]
    assert pred.score_kind == "ordinal"


def test_primary_only_factory_returns_exact_backend_instance(monkeypatch):
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine import cross_encoder

    monkeypatch.delenv("HINDSIGHT_API_RERANKER_PIPELINE", raising=False)
    config = HindsightConfig.from_env()
    backend = Native([])
    calls = []

    def construct(member):
        calls.append(member)
        return backend

    monkeypatch.setattr("hindsight_api.config.get_config", lambda: config)
    monkeypatch.setattr(cross_encoder, "create_cross_encoder", construct)
    assert cross_encoder.create_cross_encoder_from_env() is backend
    assert len(calls) == 1
    assert calls[0] == config.reranker_chain()[0]
