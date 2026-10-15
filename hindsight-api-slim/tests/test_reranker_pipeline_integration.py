"""Offline integration of progressive reranking through the public config loader."""

from dataclasses import asdict, fields

import pytest

from hindsight_api.config import HindsightConfig


def test_from_env_builds_pipeline_from_complete_primary(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "local,typesafe")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_LIMIT", "20")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_LIMIT", "5")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_LOCAL_BATCH_SIZE", "7")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_TYPESAFE_API_KEY", "dummy-stage-key")
    config = HindsightConfig.from_env()
    stages = getattr(config, "reranker_pipeline", None)
    assert stages is not None, "from_env ignores the pipeline environment"
    assert [stage.provider for stage in stages] == ["local", "typesafe"]
    assert [stage.limit for stage in stages] == [20, 5]
    primary = config.reranker_chain()[0]
    expected = asdict(primary)
    expected.update(index=1, provider="local")
    assert asdict(stages[0].member) == expected
    assert stages[1].member.local_batch_size == 7
    assert stages[1].member.typesafe_api_key == "dummy-stage-key"
    assert stages[0].member is not primary
    assert "reranker_pipeline" in {field.name for field in fields(config)}
    assert "reranker_pipeline" not in config.get_configurable_fields()


async def test_env_factory_runs_progressive_pipeline(monkeypatch):
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_LIMIT", "2")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_LIMIT", "1")
    encoder = create_cross_encoder_from_env()
    assert callable(getattr(encoder, "predict_structured", None)), "factory ignores pipeline"
    prediction = await encoder.predict_structured([("q", "a"), ("q", "b"), ("q", "c")])
    assert prediction.keep == [True, False, False]


async def test_recall_caller_consumes_pipeline_keep_mask(monkeypatch):
    from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
    from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_LIMIT", "2")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS2_LIMIT", "1")
    candidates = [
        MergedCandidate(
            retrieval=RetrievalResult(
                id=str(i), text=f"document {i}", fact_type="world", occurred_start=None, occurred_end=None
            ),
            rrf_score=1.0 / (i + 1),
        )
        for i in range(3)
    ]
    result = await CrossEncoderReranker().rerank("q", candidates)
    assert isinstance(result, RerankResult)
    assert [item.retrieval.id for item in result.results] == ["0"]
    assert result.provider_name == "rrf"


def test_unset_pipeline_does_not_parse_primary_only_factory_fields(monkeypatch):
    # Flashrank is historically parsed by reranker_chain, not from_env. Loading
    # unrelated configuration must not begin parsing it because of this feature.
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_FLASHRANK_BATCH_SIZE", "not-a-number")
    config = HindsightConfig.from_env()
    assert config.reranker_pipeline is None


@pytest.mark.parametrize("raw", ["", "  "])
def test_empty_pipeline_is_disabled(monkeypatch, raw):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", raw)
    assert HindsightConfig.from_env().reranker_pipeline is None


@pytest.mark.parametrize("raw", ["rrf", ",rrf,", ",,"])
def test_pipeline_needs_two_providers(monkeypatch, raw):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", raw)
    with pytest.raises(ValueError, match="at least two"):
        HindsightConfig.from_env()


@pytest.mark.parametrize("raw", ["0", "-1", "word", "1.5"])
def test_env_stage_limit_rejected(monkeypatch, raw):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_LIMIT", raw)
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_PASS1_LIMIT"):
        HindsightConfig.from_env()


def test_pipeline_plus_failover_rejected_before_building_any_provider(monkeypatch):
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_1_PROVIDER", "rrf")

    def unexpected(member):
        pytest.fail("provider constructed before unsupported combination rejected")

    monkeypatch.setattr("hindsight_api.engine.cross_encoder.create_cross_encoder", unexpected)
    with pytest.raises(ValueError, match="cannot be combined"):
        create_cross_encoder_from_env()


def test_optional_fields_and_namespace_isolation(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "rrf,rrf")
    monkeypatch.setenv("HINDSIGHT_RERANKER_PASS1_LIMIT", "1")
    config = HindsightConfig.from_env()
    assert config.reranker_pipeline[0].limit is None
    assert config.reranker_pipeline[0].member.max_tokens_per_candidate is None
    assert config.reranker_pipeline[0].member.cohere_api_key is None
    assert config.reranker_members == []
    config.reranker_pipeline[0].member.local_batch_size = 1
    assert config.reranker_pipeline[1].member.local_batch_size != 1
    assert config.reranker_local_batch_size != 1
    assert "reranker_pipeline" in config._CREDENTIAL_FIELDS
    assert asdict(config)["reranker_pipeline"][0]["member"]["provider"] == "rrf"


def test_pass_overrides_every_supported_primary_field(monkeypatch):
    # Native dataclass field enumeration guards newly added provider settings.
    primary = HindsightConfig.from_env().reranker_chain()[0]
    expected = {}
    for member_field in fields(primary):
        name = member_field.name
        if name == "index":
            continue
        current = getattr(primary, name)
        if isinstance(current, bool):
            value = not current
            text = str(value).lower()
        elif isinstance(current, int) or name == "max_tokens_per_candidate":
            value, text = 13, "13"
        elif isinstance(current, float):
            value, text = 2.5, "2.5"
        else:
            value = text = "rrf" if name == "provider" else "dummy-value"
        expected[name] = value
        monkeypatch.setenv("HINDSIGHT_API_RERANKER_PASS1_" + name.upper(), text)
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PIPELINE", "local,rrf")
    stage = HindsightConfig.from_env().reranker_pipeline[0]
    expected["index"] = 1
    assert asdict(stage.member) == expected
    assert stage.provider == "rrf"
