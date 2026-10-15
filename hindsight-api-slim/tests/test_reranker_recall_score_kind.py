"""Request-local score semantics through real recall, without services or models."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine import memory_engine
from hindsight_api.engine.cross_encoder import CrossEncoderModel, RerankTimeoutError
from hindsight_api.engine.pipeline_rerank import PipelineCrossEncoder
from hindsight_api.engine.rerank_result import RerankPrediction, ScoreKind
from hindsight_api.engine.response_models import MinScores, RecallResult
from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.structured_failover import StructuredMultiCrossEncoder
from hindsight_api.models import RequestContext


class Native(CrossEncoderModel):
    def __init__(self, provider: str = "local", kind: ScoreKind = "ordinal", outcome: str = "success") -> None:
        self.name = provider
        self.score_kind = kind
        self.outcome = outcome

    @property
    def provider_name(self) -> str:
        return self.name

    async def initialize(self) -> None:
        pass

    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        scores = [0.2, 0.8, 0.4][: len(pairs)]
        if self.outcome != "success":
            raise RerankTimeoutError(scores[:1] if self.outcome == "partial" else scores, 0.0, "offline-model")
        return RerankPrediction(scores, [True] * len(pairs), self.name, self.score_kind)


class FailedPrimary(Native):
    async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
        raise RuntimeError("deterministic offline primary failure")


class ConfigResolver:
    async def get_bank_config(self, bank_id: str, request_context: RequestContext) -> dict[str, object]:
        return {}


@dataclass
class RecallHarness:
    engine: memory_engine.MemoryEngine

    async def recall(
        self, query: str = "q", floor: float | None = 0.6, final: float | None = None, max_tokens: int = 4096
    ) -> RecallResult:
        return await self.engine.recall_async(
            bank_id="offline-bank",
            query=query,
            max_tokens=max_tokens,
            fact_type=["world"],
            request_context=RequestContext(),
            min_scores=MinScores(reranker=floor, final=final),
            include_entities=False,
            include_chunks=False,
            _quiet=True,
        )

    def use(self, encoder: CrossEncoderModel | PipelineCrossEncoder) -> None:
        self.engine._cross_encoder_reranker = CrossEncoderReranker(encoder)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> RecallHarness:
    config = HindsightConfig.from_env()
    config.recall_strategy_boosts = {}
    config.recency_decay_function = "none"
    config.reranker_max_candidates = 10
    config.reranker_max_candidates_low = 0
    monkeypatch.setattr(memory_engine, "get_config", lambda: config)
    monkeypatch.setattr("hindsight_api.config.get_config", lambda: config)
    monkeypatch.setattr(memory_engine, "count_tokens", len)
    monkeypatch.setattr(memory_engine, "_truncate_query_to_token_limit", lambda query, *args, **kwargs: query)
    engine = memory_engine.MemoryEngine.__new__(memory_engine.MemoryEngine)
    engine._operation_validator = None
    engine._config_resolver = ConfigResolver()
    engine._search_semaphore = asyncio.Semaphore(2)
    engine._initialized = True
    engine._read_backend = object()
    engine.embeddings = object()
    engine.query_analyzer = object()
    engine._authenticate_tenant = AsyncMock()
    engine._require_bank_exists = AsyncMock()
    engine._tag_scope = AsyncMock(return_value=None)
    monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", AsyncMock(return_value=[[0.1]]))
    retrieval = MultiFactTypeRetrievalResult(
        results_by_fact_type={
            "world": ParallelRetrievalResult(
                semantic=[
                    RetrievalResult(
                        id=f"00000000-0000-0000-0000-{i:012d}", text=f"fact {i}", fact_type="world", similarity=1.0 / i
                    )
                    for i in range(1, 4)
                ],
                bm25=[],
                graph=[],
                temporal=None,
                timings={},
            )
        }
    )
    monkeypatch.setattr(
        "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel", AsyncMock(return_value=retrieval)
    )

    # Fail closed for every store method except the two in-memory boundaries.
    class OfflineStore:
        def store_owned_for(self, bank_id: str) -> bool:
            return False

        async def full_recall(self, request: object) -> None:
            return None

        async def hydrate_results(self, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: OfflineStore())
    result = RecallHarness(engine)
    result.use(Native())
    return result


@pytest.mark.parametrize("route", ["direct", "pipeline", "failover"])
@pytest.mark.parametrize("outcome", ["success", "partial", "complete"])
async def test_ordinal_local_survives_recall_floor_without_relevance_publication(
    harness: RecallHarness,
    route: str,
    outcome: str,
) -> None:
    native = Native(outcome=outcome)
    encoder = native
    if route == "pipeline":
        encoder = PipelineCrossEncoder([(Native("cohere", "calibrated"), None), (native, None)])
    elif route == "failover":
        encoder = StructuredMultiCrossEncoder([FailedPrimary("cohere", "calibrated"), native])
    harness.use(encoder)
    result = await harness.recall()
    assert len(result.results) == 3
    assert all(fact.scores.reranker is None for fact in result.results)


@pytest.mark.parametrize("kind", ["calibrated", "logit"])
@pytest.mark.parametrize("outcome", ["success", "partial", "complete"])
async def test_pointwise_kind_overrides_rank_named_backup_after_prediction(
    harness: RecallHarness,
    kind: ScoreKind,
    outcome: str,
) -> None:
    harness.use(
        StructuredMultiCrossEncoder(
            [
                FailedPrimary("cohere", "calibrated"),
                Native("typesafe", kind, outcome),
            ]
        )
    )
    result = await harness.recall(floor=None)
    expected = [0.8, 0.4, 0.2] if outcome != "partial" else [0.2, None, None]
    if kind == "logit":
        from hindsight_api.engine.rerank_result import stable_sigmoid

        expected = [None if score is None else stable_sigmoid(score) for score in expected]
    assert [fact.scores.reranker for fact in result.results] == pytest.approx(
        [0.0 if score is None else score for score in expected]
    )
    filtered = await harness.recall()
    assert len(filtered.results) == (0 if outcome == "partial" else 1)
    assert all(fact.scores.reranker >= 0.6 for fact in filtered.results)


async def test_configured_rank_primary_floor_is_still_rejected(harness: RecallHarness) -> None:
    from hindsight_api.extensions.operation_validator import OperationValidationError

    harness.use(Native("typesafe", "calibrated"))
    with pytest.raises(OperationValidationError, match="min_scores.reranker is not supported"):
        await harness.recall()


async def test_ordinal_final_floor_still_filters(harness: RecallHarness) -> None:
    result = await harness.recall(final=0.6)
    assert len(result.results) == 1
    assert result.results[0].scores.final == pytest.approx(0.8)
    assert result.results[0].scores.reranker is None


async def test_rank_named_ordinal_backup_keeps_floor_suppression(harness: RecallHarness) -> None:
    harness.use(
        StructuredMultiCrossEncoder(
            [
                FailedPrimary("cohere", "calibrated"),
                Native("typesafe", "ordinal"),
            ]
        )
    )
    result = await harness.recall()
    assert len(result.results) == 3
    assert all(fact.scores.reranker is None for fact in result.results)


@pytest.mark.parametrize("kind", ["ordinal", "calibrated"])
async def test_total_timeout_restores_previous_score_kind_through_recall(
    harness: RecallHarness,
    kind: ScoreKind,
) -> None:
    class EmptyTimeout(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            raise RerankTimeoutError([], 0.0, "offline-empty")

    harness.use(
        PipelineCrossEncoder(
            [
                (Native("local", kind), None),
                (EmptyTimeout("local", "ordinal" if kind == "calibrated" else "logit"), None),
            ]
        )
    )
    result = await harness.recall()
    assert len(result.results) == (3 if kind == "ordinal" else 1)
    assert [fact.scores.reranker for fact in result.results] == (
        [None] * 3 if kind == "ordinal" else pytest.approx([0.8])
    )


@pytest.mark.parametrize("first_query", ["ordinal", "pointwise"])
async def test_concurrent_recalls_consume_their_own_score_kind(
    harness: RecallHarness,
    first_query: str,
) -> None:
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()

    class PerRequest(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            kind = "ordinal" if pairs[0][0] == "ordinal" else "calibrated"
            return RerankPrediction([0.2, 0.8, 0.4], [True] * 3, "local", kind)

    class Interleaved(CrossEncoderReranker):
        async def rerank(self, query: str, candidates: list[MergedCandidate]) -> RerankResult:
            result = await super().rerank(query, candidates)
            if query == first_query:
                first_ready.set()
                await second_ready.wait()
            else:
                second_ready.set()
            return result

    harness.engine._cross_encoder_reranker = Interleaved(PerRequest())
    second_query = "pointwise" if first_query == "ordinal" else "ordinal"

    async def second_recall() -> RecallResult:
        await first_ready.wait()
        return await harness.recall(second_query)

    results = await asyncio.wait_for(asyncio.gather(harness.recall(first_query), second_recall()), timeout=5)
    by_query = dict(zip([first_query, second_query], results))
    assert len(by_query["ordinal"].results) == 3
    assert all(fact.scores.reranker is None for fact in by_query["ordinal"].results)
    assert len(by_query["pointwise"].results) == 1
    assert by_query["pointwise"].results[0].scores.reranker == pytest.approx(0.8)


async def test_final_ties_reach_real_recall_token_budget(harness: RecallHarness) -> None:
    class Reverse(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            assert [text for _, text in pairs] == ["fact 1", "fact 2", "fact 3"]
            return RerankPrediction([0.2, 0.5, 0.9], [True] * 3, "cohere")

    class Tied(Native):
        async def predict_structured(self, pairs: list[tuple[str, str]]) -> RerankPrediction:
            assert [text for _, text in pairs] == ["fact 3", "fact 2", "fact 1"]
            return RerankPrediction([0.7] * 3, [True] * 3, "cohere")

    harness.use(PipelineCrossEncoder([(Reverse(), None), (Tied(), 2)]))
    result = await harness.recall(floor=None, max_tokens=len("fact 3"))
    assert [fact.id for fact in result.results] == ["00000000-0000-0000-0000-000000000003"]
    assert [fact.text for fact in result.results] == ["fact 3"]
    assert [fact.scores.reranker for fact in result.results] == [0.7]
