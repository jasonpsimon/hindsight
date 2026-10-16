"""Malformed native outputs fail closed and leave a text-free failure event."""

import logging

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder
from hindsight_api.engine.rerank_result import RerankPrediction
from hindsight_api.engine.rerank_telemetry import RerankStageFailureEvent, rerank_stage_binding


class Malformed(CrossEncoderModel):
    def __init__(self, raised: bool) -> None:
        self.raised = raised

    @property
    def provider_name(self) -> str:
        return "malformed"

    async def initialize(self) -> None:
        pass

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        if self.raised:
            raise RerankTimeoutError([0.5, 0.4], 0, "private model body")
        return RerankPrediction([0.5], [], "malformed")


@pytest.mark.parametrize("raised", [False, True])
async def test_invalid_native_result_logs_failure(raised: bool, caplog: pytest.LogCaptureFixture) -> None:
    encoder = Malformed(raised)
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        with pytest.raises(ValueError, match="timeout returned|scores/keep"):
            await PipelineCrossEncoder([(encoder, None), (encoder, None)]).predict_structured([("q", "d")])
    events = [
        RerankStageFailureEvent.model_validate_json(record.message.removeprefix("rerank_stage_failed "))
        for record in caplog.records
        if record.message.startswith("rerank_stage_failed ")
    ]
    assert len(events) == 1
    assert events[0].stage == 1
    assert events[0].input_ids == ["0"]
    assert events[0].error_type == "ValueError"
    assert events[0].latency_ms >= 0
    assert "private model body" not in caplog.text
    assert not any(record.message.startswith("rerank_complete ") for record in caplog.records)
    assert rerank_stage_binding.get() is None
