"""Production stage events must describe the actual recall boundaries."""

import asyncio
import json
import logging

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder
from hindsight_api.engine.rerank_result import RerankPrediction
from hindsight_api.engine.rerank_telemetry import rerank_candidate_ids
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


class FixedEncoder(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.2, 0.9, 0.5][: len(pairs)]


async def test_real_candidate_ids_at_ordered_stage_boundaries(caplog) -> None:
    candidates = [
        MergedCandidate(
            retrieval=RetrievalResult(id=id_, text="duplicate private text", fact_type="world"), rrf_score=1.0
        )
        for id_ in ["memory-a", "memory-b", "memory-c"]
    ]
    pipeline = PipelineCrossEncoder([(FixedEncoder(), 2), (FixedEncoder(), None)])
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        result = await CrossEncoderReranker(pipeline).rerank("private query", candidates)
    events = [
        json.loads(r.message.removeprefix("rerank_stage "))
        for r in caplog.records
        if r.message.startswith("rerank_stage ")
    ]
    assert len(events) == 2
    assert events[0]["identity_scope"] == "memory_id"
    assert events[0]["input_ids"] == ["memory-a", "memory-b", "memory-c"]
    assert events[0]["ranked_ids"] == ["memory-b", "memory-c", "memory-a"]
    assert events[0]["output_ids"] == events[1]["input_ids"] == ["memory-b", "memory-c"]
    assert events[1]["output_ids"] == [r.id for r in result.results] == ["memory-c", "memory-b"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert [e["stage"] for e in events] == [1, 2]
    assert all(e["latency_ms"] >= 0 for e in events)
    assert "private query" not in caplog.text
    assert "duplicate private text" not in caplog.text


class NativeEncoder(FixedEncoder):
    def __init__(self, prediction: RerankPrediction) -> None:
        self.prediction = prediction

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        return self.prediction


async def test_timeout_mask_limit_and_restoration_events(caplog) -> None:
    first = NativeEncoder(RerankPrediction([0.1, 0.9, 0.6], [True] * 3, "earlier", "ordinal"))
    last = NativeEncoder(
        RerankPrediction([None] * 3, [True, False, True], "timeout", timed_out=True, timeout_seconds=0)
    )
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        pred = await PipelineCrossEncoder([(first, None), (last, 1)]).predict_structured([("q", "same")] * 3)
    events = [
        json.loads(r.message.removeprefix("rerank_stage "))
        for r in caplog.records
        if r.message.startswith("rerank_stage ")
    ]
    completed = [
        json.loads(r.message.removeprefix("rerank_complete "))
        for r in caplog.records
        if r.message.startswith("rerank_complete ")
    ]
    assert events[1]["input_ids"] == ["1", "2", "0"]
    assert events[1]["ranked_ids"] == ["1", "0"]
    assert events[1]["provider_pruned_ids"] == ["2"]
    assert events[1]["output_ids"] == ["1"]
    assert events[1]["identity_scope"] == "input_index"
    assert pred.survivor_order == [1]
    assert pred.scores == [None, 0.9, None]
    assert completed[0]["restored"] is True
    assert completed[0]["output_indices"] == pred.survivor_order
    assert completed[0]["score_provider"] == "earlier"
    assert completed[0]["timeout_provider"] == "timeout"
    assert completed[0]["timeout_seconds"] == 0


async def test_concurrent_recall_ids_and_cancellation_are_isolated(caplog) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Blocking(FixedEncoder):
        async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            if pairs[0][0] == "cancel":
                entered.set()
                await release.wait()
            return [0.2, 0.9][: len(pairs)]

    pipeline = PipelineCrossEncoder([(Blocking(), None), (FixedEncoder(), None)])

    async def recall(prefix: str):
        candidates = [MergedCandidate(RetrievalResult(f"{prefix}-{i}", "same", "world"), 1.0) for i in range(2)]
        return await CrossEncoderReranker(pipeline).rerank(prefix, candidates)

    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        cancelled = asyncio.create_task(recall("cancel"))
        await entered.wait()
        responses = await asyncio.gather(recall("one"), recall("two"))
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await pipeline.predict_structured([("q", "same")])
    assert rerank_candidate_ids.get() is None
    assert [[r.id for r in response.results] for response in responses] == [["one-0", "one-1"], ["two-0", "two-1"]]
    events = [
        json.loads(r.message.removeprefix("rerank_stage "))
        for r in caplog.records
        if r.message.startswith("rerank_stage ")
    ]
    assert len({e["request_id"] for e in events}) == 3
    assert not any("cancel" in id_ for e in events for id_ in e["input_ids"])
    for event in events:
        assert len({id_.split("-")[0] for id_ in event["input_ids"]}) == 1
    assert events[-1]["identity_scope"] == "input_index"


async def test_ordinary_failure_is_logged_without_restoring_or_exception_text(caplog) -> None:
    class Broken(FixedEncoder):
        async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("private provider body")

    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        with pytest.raises(RuntimeError, match="private provider body"):
            await PipelineCrossEncoder([(FixedEncoder(), None), (Broken(), None)]).predict_structured([("q", "d")])
    failures = [
        json.loads(r.message.removeprefix("rerank_stage_failed "))
        for r in caplog.records
        if r.message.startswith("rerank_stage_failed ")
    ]
    assert len(failures) == 1
    assert failures[0]["stage"] == 2
    assert failures[0]["input_ids"] == ["0"]
    assert failures[0]["error_type"] == "RuntimeError"
    assert "private provider body" not in caplog.text
    assert not any(r.message.startswith("rerank_complete ") for r in caplog.records)


async def test_partial_timeout_records_unscored_tail_without_old_scores(caplog) -> None:
    first = NativeEncoder(RerankPrediction([900.0, 100.0, 600.0], [True] * 3, "first", "logit"))
    last = NativeEncoder(RerankPrediction([None, 0.1, 0.9], [True] * 3, "second", timed_out=True))
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        result = await PipelineCrossEncoder([(first, None), (last, 2)]).predict_structured([("q", "same")] * 3)
    events = [
        json.loads(r.message.removeprefix("rerank_stage "))
        for r in caplog.records
        if r.message.startswith("rerank_stage ")
    ]
    assert events[1]["input_ids"] == ["0", "2", "1"]
    assert events[1]["ranked_ids"] == ["1", "2", "0"]
    assert events[1]["unscored_ids"] == ["0"]
    assert events[1]["output_ids"] == ["1", "2"]
    assert result.scores == [None, 0.9, 0.1]
    assert result.served_provider == "second"
