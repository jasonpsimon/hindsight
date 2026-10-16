"""Parent regression probes for native nested pipeline identity handling."""

import asyncio
import logging

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder
from hindsight_api.engine.rerank_result import RerankPrediction
from hindsight_api.engine.rerank_telemetry import RerankStageEvent, rerank_candidate_ids, rerank_stage_binding
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


class FixedNative(CrossEncoderModel):
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    @property
    def provider_name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        pass

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        assert len(pairs) == len(self.scores)
        return RerankPrediction(list(self.scores), [True] * len(pairs), "cohere")


def make_candidates() -> list[MergedCandidate]:
    return [
        MergedCandidate(RetrievalResult(identity, "same text", "world"), 1.0) for identity in ["id-a", "id-b", "id-c"]
    ]


async def test_nested_after_pruning_still_returns_original_ids() -> None:
    nested = PipelineCrossEncoder([(FixedNative([0.9, 0.2]), None), (FixedNative([0.9, 0.2]), None)])
    outer = PipelineCrossEncoder([(FixedNative([0.1, 0.9, 0.6]), 2), (nested, None)])
    result = await CrossEncoderReranker(outer).rerank("q", make_candidates())
    assert [row.id for row in result.results] == ["id-b", "id-c"]


async def test_nested_reordering_keeps_correct_persistent_ids_in_events(caplog) -> None:
    nested = PipelineCrossEncoder([(FixedNative([0.9, 0.6, 0.1]), None), (FixedNative([0.9, 0.6, 0.1]), None)])
    outer = PipelineCrossEncoder([(FixedNative([0.1, 0.9, 0.6]), None), (nested, None)])
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        result = await CrossEncoderReranker(outer).rerank("q", make_candidates())
    assert [row.id for row in result.results] == ["id-b", "id-c", "id-a"]
    events = [
        RerankStageEvent.model_validate_json(record.message.removeprefix("rerank_stage "))
        for record in caplog.records
        if record.message.startswith("rerank_stage ")
    ]
    assert events[0].output_ids == ["id-b", "id-c", "id-a"]
    assert events[1].input_ids == events[0].output_ids
    assert events[1].input_indices == events[0].output_indices == [1, 2, 0]
    assert events[1].parent_request_id == events[0].request_id
    assert events[1].parent_stage == 2
    assert events[1].request_id != events[0].request_id
    assert rerank_stage_binding.get() is None


@pytest.mark.parametrize("persistent", [False, True])
async def test_nested_duplicate_occurrences_and_direct_scope(persistent, caplog) -> None:
    child = PipelineCrossEncoder([(FixedNative([0.9, 0.1]), None), (FixedNative([0.9, 0.1]), None)])
    outer = PipelineCrossEncoder([(FixedNative([0.1, 0.9, 0.6]), 2), (child, None)])
    token = rerank_candidate_ids.set(("duplicate", "duplicate", "other") if persistent else None)
    try:
        with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
            pred = await outer.predict_structured([("q", "same")] * 3)
    finally:
        rerank_candidate_ids.reset(token)
    events = [
        RerankStageEvent.model_validate_json(r.message.removeprefix("rerank_stage "))
        for r in caplog.records
        if r.message.startswith("rerank_stage ")
    ]
    assert events[1].input_ids == (["duplicate", "other"] if persistent else ["1", "2"])
    assert events[1].input_indices == [1, 2]
    assert events[1].identity_scope == ("memory_id" if persistent else "input_index")
    assert pred.survivor_order == [1, 2]
    assert rerank_stage_binding.get() is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_nested_error_resets_bindings_in_same_task(cancel) -> None:
    class Broken(FixedNative):
        async def predict_structured(self, pairs):
            assert rerank_stage_binding.get() is not None
            if cancel:
                raise asyncio.CancelledError()
            raise RuntimeError("deliberate")

    child = PipelineCrossEncoder([(Broken([]), None), (FixedNative([0.9]), None)])
    outer = PipelineCrossEncoder([(FixedNative([0.9]), None), (child, None)])
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await outer.predict_structured([("q", "d")])
    assert rerank_stage_binding.get() is None
    assert rerank_candidate_ids.get() is None
