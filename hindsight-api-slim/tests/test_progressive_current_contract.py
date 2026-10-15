"""Current public entry-point contracts for progressive reranking."""

from dataclasses import dataclass, fields, replace

import pytest

from hindsight_api.config import HindsightConfig, RerankerMemberConfig
from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.pipeline_rerank import LegacyPredictAdapter, PipelineCrossEncoder
from hindsight_api.engine.remote_retry import RetryPolicy
from hindsight_api.engine.token_encoding import truncate_to_tokens


class RecordingEncoder(CrossEncoderModel):
    def __init__(self, scores, name="cohere"):
        self.scores = scores
        self.name = name
        self.calls = []

    @property
    def provider_name(self):
        return self.name

    async def initialize(self):
        pass

    async def _predict(self, pairs):
        self.calls.append(list(pairs))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores[: len(pairs)]


@pytest.mark.parametrize("limit", [None, 3, 2])
async def test_survivors_always_ordered_before_next_stage(limit):
    first = RecordingEncoder([0.2, 0.9, 0.5])
    second = RecordingEncoder([0.3, 0.8, 0.1])
    pairs = [("untouched query", text) for text in ["CHESS", "CELLO", "BERLIN"]]
    result = await PipelineCrossEncoder([(first, limit), (second, None)]).predict_structured(pairs)
    expected = [pairs[1], pairs[2], pairs[0]][:limit]
    assert second.calls == [expected]
    assert result.scores[1] == 0.3
    assert result.scores[2] == 0.8
    assert pairs == [("untouched query", text) for text in ["CHESS", "CELLO", "BERLIN"]]


async def test_adapter_calls_public_predict_and_member_caps_without_mutation():
    first = RecordingEncoder([0.2, 0.9])
    second = RecordingEncoder([0.3, 0.8])
    first.max_tokens_per_candidate = 2
    second.max_tokens_per_candidate = 5
    docs = ["alpha beta gamma delta epsilon zeta", "one two three four five six"]
    pairs = [("query stays whole", doc) for doc in docs]
    await PipelineCrossEncoder([(first, None), (second, None)]).predict_structured(pairs)
    assert first.calls == [[(q, truncate_to_tokens(d, 2).text) for q, d in pairs]]
    assert second.calls == [[(q, truncate_to_tokens(d, 5).text) for q, d in reversed(pairs)]]
    assert [d for q, d in pairs] == docs


async def test_adapter_has_exactly_one_retry_owner():
    class Broken(RecordingEncoder):
        async def _predict(self, pairs):
            self.calls.append(list(pairs))
            raise ConnectionError("deterministic outage")

    inner = Broken([])
    inner.retry_policy = RetryPolicy(max_retries=2, initial_backoff=0, max_backoff=0, budget_seconds=10)
    with pytest.raises(ConnectionError):
        await LegacyPredictAdapter(inner).predict_structured([("q", "d")])
    assert len(inner.calls) == 3


@pytest.mark.parametrize(
    "initial,raw,alias,expected",
    [
        (None, None, None, None),
        (8, None, None, 8),
        (None, "4", None, 4),
        (8, "  ", None, 8),
        (None, None, "3", 3),
        (8, "5", "3", 5),
        (8, " ", "3", 3),
    ],
)
def test_stage_token_cap_overrides(monkeypatch, initial, raw, alias, expected):
    from hindsight_api.config import build_pipeline_stage_member

    primary = replace(HindsightConfig.from_env().reranker_chain()[0], max_tokens_per_candidate=initial)
    if raw is not None:
        monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_MAX_TOKENS_PER_CANDIDATE", raw)
    if alias is not None:
        monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_LITELLM_MAX_TOKENS_PER_DOC", alias)
    stage = build_pipeline_stage_member(1, "rrf", primary)
    assert stage.max_tokens_per_candidate == expected
    assert stage.max_tokens_per_candidate is None or type(stage.max_tokens_per_candidate) is int


def test_stage_cap_invalid_is_indexed_error(monkeypatch):
    from hindsight_api.config import build_pipeline_stage_member

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_MAX_TOKENS_PER_CANDIDATE", "nope")
    with pytest.raises(ValueError, match="PASS2_MAX_TOKENS_PER_CANDIDATE"):
        build_pipeline_stage_member(2, "rrf", HindsightConfig.from_env().reranker_chain()[0])


def test_stage_clones_subclass_and_inherited_defaults(monkeypatch):
    from hindsight_api.config import build_pipeline_stage_member

    @dataclass
    class CustomMember(RerankerMemberConfig):
        unrelated: str = "preserved"
        extra_budget: int = 17

    primary = HindsightConfig.from_env().reranker_chain()[0]
    custom = CustomMember(**{f.name: getattr(primary, f.name) for f in fields(primary)})
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_PROVIDER", "RRF")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_MAX_TOKENS_PER_CANDIDATE", "7")
    stage = build_pipeline_stage_member(2, "local", custom)
    assert type(stage) is CustomMember
    for f in fields(custom):
        assert getattr(stage, f.name) == {"index": 2, "provider": "RRF", "max_tokens_per_candidate": 7}.get(
            f.name, getattr(custom, f.name)
        )


async def test_pipeline_identity_and_rank_floor_providers_cover_restoration():
    ordinal = RecordingEncoder([0.9], "typesafe")
    final = RecordingEncoder(RerankTimeoutError([None], 0, ""), "local")
    pipeline = PipelineCrossEncoder([(ordinal, None), (final, None)])
    assert pipeline.primary_provider_name == "local"
    assert pipeline.score_provider_names == frozenset({"local", "typesafe"})
    pred = await pipeline.predict_structured([("q", "doc")])
    assert pred.served_provider == "typesafe"
    assert pred.score_kind == "ordinal"
    assert pred.model_name == ""
    assert pred.timeout_seconds == 0


@pytest.mark.parametrize("names", [("typesafe", "cohere"), ("cohere", "typesafe"), ("cohere", "cohere")])
async def test_real_recall_entry_checks_every_pipeline_score_source(names):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from hindsight_api import MemoryEngine, RequestContext
    from hindsight_api.engine.response_models import MinScores
    from hindsight_api.engine.search.reranking import CrossEncoderReranker
    from hindsight_api.extensions.operation_validator import OperationValidationError

    # The entry guard runs after auth, before any storage read. Stop accepted
    # calls at the operation validator so the test needs no external database.
    class ReachedValidator(Exception):
        pass

    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    pipeline = PipelineCrossEncoder([(RecordingEncoder([0.5], name), None) for name in names])
    engine._cross_encoder_reranker = CrossEncoderReranker(pipeline)
    validator = AsyncMock(side_effect=ReachedValidator)
    engine._operation_validator = SimpleNamespace(validate_recall=validator)
    expected = OperationValidationError if "typesafe" in names else ReachedValidator
    with pytest.raises(expected) as exc:
        await engine.recall_async(
            bank_id="only-this-test", query="q", min_scores=MinScores(reranker=0.5), request_context=RequestContext()
        )
    if "typesafe" in names:
        assert exc.value.status_code == 400
        assert "typesafe" in str(exc.value)
        validator.assert_not_called()
    else:
        validator.assert_awaited_once()


@pytest.mark.parametrize(
    "provider,setting,message",
    [
        ("unknown-provider", None, "Unknown reranker provider"),
        ("tei", None, "RERANKER_2_TEI_URL"),
        ("cohere", None, "RERANKER_2_COHERE_API_KEY"),
    ],
)
def test_effective_pass_provider_is_validated(monkeypatch, provider, setting, message):
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_PROVIDER", provider)
    with pytest.raises(ValueError, match=message):
        create_cross_encoder_from_env()


def test_effective_case_insensitive_pass_and_own_credentials(monkeypatch):
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_PROVIDER", "TEI")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_TEI_URL", "http://127.0.0.1:1")
    encoder = create_cross_encoder_from_env()
    assert encoder.primary_provider_name == "tei"
    assert encoder._stages[1][0]._inner.base_url == "http://127.0.0.1:1"


async def test_structured_failover_members_apply_own_caps_to_original_pairs():
    from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder

    first = RecordingEncoder(RuntimeError("down"))
    second = RecordingEncoder([0.8], "siliconflow")
    first.max_tokens_per_candidate = 2
    second.max_tokens_per_candidate = 5
    pairs = [("long unchanged question", "one two three four five six seven eight")]
    pred = await StructuredMultiCrossEncoder([first, second]).predict_structured(pairs)
    assert first.calls == [[(pairs[0][0], truncate_to_tokens(pairs[0][1], 2).text)]]
    assert second.calls == [[(pairs[0][0], truncate_to_tokens(pairs[0][1], 5).text)]]
    assert pairs == [("long unchanged question", "one two three four five six seven eight")]
    assert pred.served_provider == "siliconflow"


def test_structured_failover_preserves_primary_floor_policy():
    from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder

    encoder = StructuredMultiCrossEncoder([RecordingEncoder([], "cohere"), RecordingEncoder([], "typesafe")])
    assert encoder.primary_provider_name == "cohere"
    assert encoder.score_provider_names == frozenset({"cohere"})


async def test_public_pipeline_prediction_publishes_restored_score_provenance():
    from hindsight_api.engine.cross_encoder import _served_provider

    first = RecordingEncoder([0.9], "typesafe")
    final = RecordingEncoder(RerankTimeoutError([None], 0, ""), "local")
    pipeline = PipelineCrossEncoder([(first, None), (final, None)])
    token = _served_provider.set(None)
    try:
        pred = await pipeline.predict_structured([("q", "doc")])
        assert pred.served_provider == "typesafe"
        assert _served_provider.get() == "typesafe"
        with pytest.raises(RerankTimeoutError) as caught:
            await pipeline.predict([("q", "doc")])
        assert caught.value.provider_name == "typesafe"
        assert caught.value.model_name == ""
        assert caught.value.scores == [0.9]
    finally:
        _served_provider.reset(token)
