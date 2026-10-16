"""Native provider doubles distinguish restored scores from timeout provenance."""

import logging
from typing import cast

import pytest

from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter, PipelineCrossEncoder, timeout_to_structured
from hindsight_api.engine.rerank_result import RerankPrediction
from hindsight_api.engine.rerank_telemetry import RerankCompletionEvent
from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder


class Native(CrossEncoderModel):
    def __init__(self, name: str, timeout: bool = False) -> None:
        self.name = name
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return self.name

    async def initialize(self) -> None:
        pass

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        return RerankPrediction(
            [None if self.timeout else 0.9] * len(pairs),
            [True] * len(pairs),
            self.name,
            timed_out=self.timeout,
            timeout_seconds=0.0 if self.timeout else None,
            model_name="" if self.timeout else None,
        )


async def test_nested_restoration_retains_distinct_timeout_provider(caplog: pytest.LogCaptureFixture) -> None:
    child = PipelineCrossEncoder([(Native("A"), None), (Native("B", timeout=True), None)])
    outer = PipelineCrossEncoder([(Native("start"), None), (child, None)])
    with caplog.at_level(logging.DEBUG, logger="hindsight_api.engine.pipeline_rerank"):
        pred = await outer.predict_structured([("q", "d")])
    events = [
        RerankCompletionEvent.model_validate_json(record.message.removeprefix("rerank_complete "))
        for record in caplog.records
        if record.message.startswith("rerank_complete ")
    ]
    assert events[-1].score_provider == "A"
    assert events[-1].timeout_provider == "B"
    assert pred.timeout_provider == "B"
    assert pred.served_provider == "A"
    assert pred.timeout_seconds == 0.0
    assert pred.model_name == ""
    with pytest.raises(RerankTimeoutError) as caught:
        await outer.predict([("q", "d")])
    replay = timeout_to_structured(caught.value, 1, caught.value.provider_name)
    assert replay.served_provider == "A"
    assert replay.timeout_provider == "B"
    assert replay.timeout_seconds == 0.0
    assert replay.model_name == ""


@pytest.mark.parametrize("duration", [None, 0.0])
@pytest.mark.parametrize("model", [None, ""])
async def test_nested_timeout_metadata_does_not_inherit_score_metadata(duration, model) -> None:
    class MetadataTimeout(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            return RerankPrediction(
                [None],
                [True],
                "B",
                timed_out=True,
                timeout_seconds=duration,
                model_name=model,
                timeout_provider="",
            )

    child = PipelineCrossEncoder([(Native("A"), None), (MetadataTimeout("B"), None)])
    outer = PipelineCrossEncoder([(Native("initial"), None), (child, None)])
    pred = await outer.predict_structured([("q", "d")])
    assert pred.served_provider == "A"
    assert pred.timeout_provider == ""
    assert pred.timeout_seconds == duration
    assert pred.model_name == model


@pytest.mark.parametrize("route", ["adapter", "failover", "native-raised-failover"])
async def test_restored_provenance_survives_composition_roundtrip(route: str) -> None:
    child = PipelineCrossEncoder([(Native("A"), None), (Native("B", timeout=True), None)])

    class PublicBridge(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            await child.predict(pairs)
            raise AssertionError("child must raise its public native timeout")

    wrapper = (
        LegacyPredictAdapter(child)
        if route == "adapter"
        else StructuredMultiCrossEncoder(
            [
                Native("initial-timeout", timeout=True),
                # The inherited failover annotation names the base class, but
                # its public member contract also accepts native pipelines.
                PublicBridge("bridge") if route == "native-raised-failover" else cast(CrossEncoderModel, child),
            ]
        )
    )
    pred = await wrapper.predict_structured([("q", "d")])
    assert pred.served_provider == "A"
    assert pred.timeout_provider == "B"
    with pytest.raises(RerankTimeoutError) as caught:
        await wrapper.predict([("q", "d")])
    replay = timeout_to_structured(caught.value, 1, caught.value.provider_name)
    assert replay.served_provider == "A"
    assert replay.timeout_provider == "B"
    assert replay.timeout_seconds == 0.0
    assert replay.model_name == ""
