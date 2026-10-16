"""Ordered shortlists cross real provider HTTP, PostgreSQL and the public client.

The first model deliberately prefers CELLO to BERLIN. The second prefers CHESS
if it leaks through, otherwise BERLIN. Keeping the original pool, forgetting to
sort survivors, or stopping after the first model therefore cannot pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException
from pydantic import BaseModel

from hindsight_system_tests import start_hindsight_server, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import LLMStub
from hindsight_system_tests.server import HindsightServer, StubServer, free_port
from hindsight_system_tests.validation import validate_rerank

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
BERLIN = "Alice lives in Berlin | Involving: Alice"
CELLO = "Alice plays the cello professionally | Involving: Alice"
CHESS = "Alice plays chess on Tuesdays | Involving: Alice"
ALL_FACTS = [BERLIN, CELLO, CHESS]


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]


class RankedDocument(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RankedDocument]


class StageEvent(BaseModel):
    request_id: str
    stage: int
    identity_scope: str
    input_ids: list[str]
    ranked_ids: list[str]
    output_ids: list[str]
    latency_ms: float
    limit: int | None
    provider_pruned_ids: list[str] = []
    unscored_ids: list[str] = []
    served_provider: str | None = None
    score_kind: str = "calibrated"
    timed_out: bool = False
    timeout_seconds: float | None = None


class CompletionEvent(BaseModel):
    request_id: str
    output_ids: list[str]
    restored: bool
    score_provider: str | None
    timeout_provider: str | None
    score_kind: str
    timed_out: bool
    timeout_seconds: float | None


class StageFailureEvent(BaseModel):
    request_id: str
    stage: int
    input_ids: list[str]
    latency_ms: float
    error_type: str


@dataclass
class RankingProvider:
    calls: list[RerankRequest] = field(default_factory=list)
    responses: list[RerankResponse] = field(default_factory=list)
    guard: threading.Lock = field(default_factory=threading.Lock)
    url: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    pipeline: str
    first_limit: int
    second_model: str
    survivors: list[str]
    final: list[str]


SCENARIOS = [
    Scenario("progressive", "siliconflow,siliconflow", 2, "final", [CELLO, BERLIN], [BERLIN]),
    # Negative controls change configuration, not engine code or assertions.
    Scenario("first_ranking_only", "siliconflow,siliconflow", 2, "shortlist", [CELLO, BERLIN], [CELLO, BERLIN]),
    Scenario("unpruned", "siliconflow,siliconflow", 3, "final", [CELLO, BERLIN, CHESS], [CHESS]),
    Scenario("reversed_final", "siliconflow,siliconflow", 2, "reverse", [CELLO, BERLIN], [CELLO]),
    Scenario("primary_only", "", 2, "final", [], [CELLO, BERLIN, CHESS]),
    # Equal final scores must preserve the preceding pass's order, not restore
    # the initial retrieval order when publishing through the recall endpoint.
    Scenario("final_tie", "siliconflow,siliconflow", 2, "tie", [CELLO, BERLIN], [CELLO, BERLIN]),
]


@pytest.fixture
def ranking_provider() -> Iterator[RankingProvider]:
    provider = RankingProvider()
    app = FastAPI()

    @app.post("/rerank")
    async def rerank(request: Request) -> RerankResponse:
        body = await request.json()
        validate_rerank(body)
        call = RerankRequest.model_validate(body)
        if call.model in {"remote-error", "remote-timeout"}:
            with provider.guard:
                provider.calls.append(call)
            if call.model == "remote-timeout":
                await asyncio.sleep(0.3)
            raise HTTPException(status_code=503, detail="scripted public provider outage")
        # Each document is scored independently of its incoming index. This
        # catches a pipeline that treats a backend-local index as an original id.
        order = {
            "shortlist": [CELLO, BERLIN, CHESS],
            "final": [CHESS, BERLIN, CELLO],
            "reverse": [CELLO, BERLIN, CHESS],
            "tie": [CELLO, BERLIN, CHESS],
        }[call.model]
        results = [
            RankedDocument(
                index=i, relevance_score=0.6 if call.model == "tie" else {0: 0.9, 1: 0.6, 2: 0.1}[order.index(text)]
            )
            for i, text in enumerate(call.documents)
        ]
        results.sort(key=lambda row: row.relevance_score, reverse=True)
        response = RerankResponse(results=results)
        with provider.guard:
            provider.calls.append(call)
            provider.responses.append(response)
        return response

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert server.started, "ranking HTTP stub did not start"
        provider.url = f"http://127.0.0.1:{port}"
        yield provider
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "ranking HTTP stub did not stop"


@pytest.fixture(params=SCENARIOS, ids=lambda scenario: scenario.name)
def scenario(request: pytest.FixtureRequest) -> Scenario:
    return request.param


@pytest.fixture
def pipeline_server(
    scenario: Scenario, ranking_provider: RankingProvider, stub_server: StubServer, tmp_path: Path
) -> Iterator[HindsightServer]:
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "pipeline-server.log",
        extra_env={
            "HINDSIGHT_API_LOG_LEVEL": "debug",
            "HINDSIGHT_API_RERANKER_PIPELINE": scenario.pipeline,
            "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL": ranking_provider.url,
            "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL": "shortlist",
            "HINDSIGHT_API_RERANKER_PASS1_LIMIT": str(scenario.first_limit),
            "HINDSIGHT_API_RERANKER_PASS2_LIMIT": "2" if scenario.second_model in {"shortlist", "tie"} else "1",
            "HINDSIGHT_API_RERANKER_PASS2_SILICONFLOW_MODEL": scenario.second_model,
            # Tiny timestamp differences otherwise break the final tie and can
            # hide original-index reconstruction in downstream score sorting.
            **({"HINDSIGHT_API_RECENCY_DECAY_FUNCTION": "none"} if scenario.second_model == "tie" else {}),
        },
    )
    try:
        yield server
    finally:
        server.stop()


async def test_progressive_shortlist_is_the_client_result(
    pipeline_server: HindsightServer, ranking_provider: RankingProvider, scenario: Scenario, llm: LLMStub
) -> None:
    bank = f"systest-progressive-{uuid.uuid4().hex[:12]}"
    client = Hindsight(base_url=pipeline_server.url)
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
            fact("Alice plays chess on Tuesdays", who="Alice", entities=["Alice", "chess"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    try:
        await client.aretain(
            bank_id=bank,
            document_id="alice-profile",
            content="Alice lives in Berlin, plays cello professionally, and plays chess on Tuesdays.",
        )
        await wait_until_settled(client, bank)
        stored = await client.memory.list_memories(bank, limit=100)
        assert Counter(row.text for row in stored.items) == Counter(ALL_FACTS)
        identities = {row.text: row.id for row in stored.items}
        assert len(set(identities.values())) == 3
        # Worker calls, if any, belong to retain. Only recall's requests prove
        # the stage-to-stage contract below.
        with ranking_provider.guard:
            ranking_provider.calls.clear()
            ranking_provider.responses.clear()
        response = await client.arecall(bank_id=bank, query=QUERY, types=["world"])
        calls = ranking_provider.calls
        assert len(calls) == (2 if scenario.survivors else 1)
        assert calls[0].model == "shortlist"
        assert calls[0].query == QUERY
        assert sorted(calls[0].documents) == sorted(ALL_FACTS)
        if scenario.second_model == "tie":
            # A tie that accidentally starts in the original order would miss
            # the regression. BERLIN has the strongest retrieval score here.
            assert [text for text in calls[0].documents if text in scenario.survivors] != scenario.survivors
        first_ranking = [calls[0].documents[row.index] for row in ranking_provider.responses[0].results]
        assert first_ranking == [CELLO, BERLIN, CHESS]
        if scenario.survivors:
            assert calls[1].model == scenario.second_model
            assert calls[1].query == QUERY
            assert calls[1].documents == scenario.survivors
            if scenario.first_limit == 2:
                assert CHESS not in calls[1].documents
        if scenario.second_model == "tie":
            assert len(response.results) == 2
            for row in response.results:
                assert row.scores is not None
                assert row.scores.final == 0.6
                assert row.scores.reranker == 0.6
        assert [row.text for row in response.results] == scenario.final
        for row in response.results:
            assert row.id == identities[row.text]
            assert row.type == "world"
            assert row.context == ""
            assert row.tags == []
            assert row.metadata == {}
            assert row.document_id == "alice-profile"
            assert row.chunk_id == f"{bank}_alice-profile_0"
            assert row.occurred_start is None
            assert row.occurred_end is None
            assert row.scores is not None
            last_order = (
                [CHESS, BERLIN, CELLO]
                if scenario.survivors and scenario.second_model == "final"
                else [CELLO, BERLIN, CHESS]
            )
            expected_score = (
                0.6 if scenario.second_model == "tie" else {0: 0.9, 1: 0.6, 2: 0.1}[last_order.index(row.text)]
            )
            assert row.scores.reranker == pytest.approx(expected_score)
            assert row.scores.semantic == pytest.approx(
                0.4903616753328348 if row.text == BERLIN else 0.4146140043322447, abs=1e-7
            )
            assert row.scores.keyword == pytest.approx(
                0.4000000059604645 if row.text == BERLIN else 0.30000001192092896, abs=1e-12
            )
            multiplier = 1.0 if scenario.second_model == "tie" else 1.1
            assert row.scores.reranker is not None
            assert row.scores.final == pytest.approx(row.scores.reranker * multiplier, abs=1e-6)
            assert row.entities is None
            assert row.source_fact_ids is None
            assert row.attachments is None
            assert row.mentioned_at is not None
        assert response.trace is None
        assert response.entities is None
        assert response.chunks is None
        assert response.source_facts is None
        # Read the API process's production logs, not the HTTP fixture's records.
        # Tie the persistent identities back to the client's stored memories.
        stage_events = [
            StageEvent.model_validate_json(line.split("rerank_stage ", 1)[1])
            for line in pipeline_server.logs().splitlines()
            if "rerank_stage " in line
        ]
        if scenario.survivors:
            assert len(stage_events) == 2
            first_event, last_event = stage_events
            assert first_event.input_ids == [identities[text] for text in calls[0].documents]
            assert first_event.ranked_ids == [identities[text] for text in first_ranking]
            assert first_event.output_ids == last_event.input_ids == [identities[text] for text in scenario.survivors]
            assert last_event.output_ids == [row.id for row in response.results]
            assert first_event.request_id == last_event.request_id
            assert [event.stage for event in stage_events] == [1, 2]
            assert all(event.identity_scope == "memory_id" and event.latency_ms >= 0 for event in stage_events)
            assert first_event.limit == scenario.first_limit
        else:
            assert stage_events == []
        evidence = os.environ.get("HINDSIGHT_SYSTEM_TEST_STORY_EVIDENCE")
        if evidence:
            with Path(evidence).open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "scenario": asdict(scenario),
                            "bank": bank,
                            "calls": [call.model_dump() for call in calls],
                            "provider_responses": [reply.model_dump() for reply in ranking_provider.responses],
                            "recall": response.model_dump(mode="json"),
                            "production_stage_events": [event.model_dump() for event in stage_events],
                        }
                    )
                    + "\n"
                )
    finally:
        # Do not suppress cleanup errors. A failed delete is not clean isolation.
        await client.banks.delete_bank(bank)
        banks = await client.banks.list_banks()
        assert bank not in [row.bank_id for row in banks.banks]
        await client.aclose()


@pytest.mark.parametrize("failure", ["remote-error", "remote-timeout"])
async def test_later_remote_failure_is_not_a_successful_snapshot(
    failure: str, ranking_provider: RankingProvider, stub_server: StubServer, tmp_path: Path, llm: LLMStub
) -> None:
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "error-server.log",
        extra_env={
            "HINDSIGHT_API_LOG_LEVEL": "debug",
            "HINDSIGHT_API_RERANKER_PIPELINE": "siliconflow,siliconflow",
            "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL": ranking_provider.url,
            "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL": "shortlist",
            "HINDSIGHT_API_RERANKER_MAX_RETRIES": "0",
            "HINDSIGHT_API_RERANKER_PASS1_LIMIT": "2",
            "HINDSIGHT_API_RERANKER_PASS2_SILICONFLOW_MODEL": failure,
            "HINDSIGHT_API_RERANKER_PASS2_SILICONFLOW_TIMEOUT": "0.03",
        },
    )
    client = Hindsight(base_url=server.url)
    bank = f"systest-progressive-error-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
            fact("Alice plays chess on Tuesdays", who="Alice", entities=["Alice", "chess"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    try:
        await client.aretain(bank_id=bank, content="Alice lives in Berlin and plays cello and chess.")
        await wait_until_settled(client, bank)
        stored = await client.memory.list_memories(bank, limit=100)
        identities = {row.text: row.id for row in stored.items}
        with pytest.raises(ApiException) as exc:
            await client.arecall(bank_id=bank, query=QUERY, types=["world"])
        assert exc.value.status == 500
        logs = server.logs().splitlines()
        events = [
            StageEvent.model_validate_json(line.split("rerank_stage ", 1)[1])
            for line in logs
            if "rerank_stage " in line
        ]
        failures = [
            StageFailureEvent.model_validate_json(line.split("rerank_stage_failed ", 1)[1])
            for line in logs
            if "rerank_stage_failed " in line
        ]
        assert len(events) == len(failures) == 1
        assert events[0].output_ids == failures[0].input_ids == [identities[CELLO], identities[BERLIN]]
        assert events[0].request_id == failures[0].request_id
        assert failures[0].stage == 2
        assert failures[0].latency_ms >= 0
        assert not any("rerank_complete " in line for line in logs)
    finally:
        try:
            await client.banks.delete_bank(bank)
            assert bank not in [row.bank_id for row in (await client.banks.list_banks()).banks]
        finally:
            await client.aclose()
            server.stop()


@pytest.mark.parametrize("mode", ["provider-mask", "native-partial", "native-total"])
async def test_native_timeout_and_provider_mask_boundaries(
    mode: str, ranking_provider: RankingProvider, stub_server: StubServer, tmp_path: Path, llm: LLMStub, stubs
) -> None:
    """No factory/prediction patching: use supported TypeSafe and local settings."""
    extra = {
        "HINDSIGHT_API_LOG_LEVEL": "debug",
        "HINDSIGHT_API_RECENCY_DECAY_FUNCTION": "none",
        "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL": ranking_provider.url,
        "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL": "shortlist",
    }
    if mode == "provider-mask":
        stubs.rerank.cut_level = 1
        extra.update(
            {
                "HINDSIGHT_API_RERANKER_PIPELINE": "typesafe,siliconflow",
                "HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY": "stub-key",
                "HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL": stub_server.url,
                "HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES": "true",
                "HINDSIGHT_API_RERANKER_PASS2_LIMIT": "1",
            }
        )
    else:
        model = os.environ.get("HINDSIGHT_SYSTEM_TEST_LOCAL_RERANKER_MODEL")
        if not model:
            pytest.skip("set HINDSIGHT_SYSTEM_TEST_LOCAL_RERANKER_MODEL to a public pretrained model snapshot")
        assert model is not None
        assert Path(model).is_dir()
        extra.update(
            {
                "HINDSIGHT_API_RERANKER_PIPELINE": "siliconflow,local",
                "HINDSIGHT_API_RERANKER_PASS2_LOCAL_MODEL": model,
                "HINDSIGHT_API_RERANKER_PASS2_LOCAL_FORCE_CPU": "true",
                "HINDSIGHT_API_RERANKER_PASS2_LOCAL_BATCH_SIZE": "1",
                # The deadline is checked before each real batch. A tiny positive
                # budget expires before the first; 1ms allows one CPU batch only.
                "HINDSIGHT_API_RERANKER_PASS2_LOCAL_TIMEOUT": "1e-300" if mode == "native-total" else "0.001",
                "HINDSIGHT_API_RERANKER_PASS2_LIMIT": "2" if mode == "native-total" else "3",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }
        )
    server = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "native-server.log", extra_env=extra)
    client = Hindsight(base_url=server.url)
    bank = f"systest-native-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
            fact("Alice plays chess on Tuesdays", who="Alice", entities=["Alice", "chess"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    try:
        await client.aretain(bank_id=bank, content="Alice lives in Berlin, plays cello professionally and plays chess.")
        await wait_until_settled(client, bank)
        stored = await client.memory.list_memories(bank, limit=100)
        identities = {row.text: row.id for row in stored.items}
        assert set(identities) == set(ALL_FACTS)
        with ranking_provider.guard:
            ranking_provider.calls.clear()
            ranking_provider.responses.clear()
        response = await client.arecall(bank_id=bank, query=QUERY, types=["world"])
        events = [
            StageEvent.model_validate_json(line.split("rerank_stage ", 1)[1])
            for line in server.logs().splitlines()
            if "rerank_stage " in line
        ]
        completed = [
            CompletionEvent.model_validate_json(line.split("rerank_complete ", 1)[1])
            for line in server.logs().splitlines()
            if "rerank_complete " in line
        ]
        assert len(events) == 2 and len(completed) == 1
        first, last = events
        final = completed[0]
        assert set(first.input_ids) == set(identities.values())
        assert first.output_ids == last.input_ids
        assert first.request_id == last.request_id == final.request_id
        assert final.output_ids == [row.id for row in response.results]
        published_scores: list[float] = []
        for row in response.results:
            assert row.scores is not None and row.scores.reranker is not None
            published_scores.append(row.scores.reranker)
        assert all(event.identity_scope == "memory_id" and event.latency_ms >= 0 for event in events)
        assert len(ranking_provider.calls) == 1
        call = ranking_provider.calls[0]
        assert call.query == QUERY
        if mode == "provider-mask":
            assert first.limit is None
            assert first.provider_pruned_ids == [identities[CHESS]]
            assert first.output_ids == [identities[BERLIN], identities[CELLO]]
            assert len(first.ranked_ids) == len(first.output_ids) == 2
            assert set(first.output_ids).isdisjoint(first.provider_pruned_ids)
            assert last.input_ids == [identities[text] for text in call.documents]
            assert last.provider_pruned_ids == []
            assert len(last.ranked_ids) == 2 and len(last.output_ids) == 1
            assert last.limit == 1
            assert not final.timed_out and not final.restored
            assert final.score_provider == "siliconflow" and final.timeout_provider is None
            assert [row.text for row in response.results] == [CELLO]
            assert response.results[0].scores is not None
            assert response.results[0].scores.reranker == 0.9
        else:
            assert first.input_ids == [identities[text] for text in call.documents]
            assert first.output_ids == [identities[text] for text in [CELLO, BERLIN, CHESS]]
            assert last.timed_out and final.timed_out
            assert final.timeout_provider == "local"
            assert last.provider_pruned_ids == []
            if mode == "native-total":
                assert last.unscored_ids == last.input_ids
                assert last.output_ids == first.output_ids[:2]
                assert final.restored and final.score_provider == "siliconflow"
                assert [row.text for row in response.results] == [CELLO, BERLIN]
                assert published_scores == [0.9, 0.6]
                assert final.timeout_seconds == 1e-300
            else:
                assert 0 < len(last.unscored_ids) < len(last.input_ids)
                assert not final.restored and final.score_provider == "local"
                assert final.score_kind == "logit"
                assert last.output_ids == final.output_ids
                assert set(final.output_ids) == set(identities.values())
                scored_count = len(last.input_ids) - len(last.unscored_ids)
                assert all(score > 0 for score in published_scores[:scored_count])
                assert [row.id for row in response.results[scored_count:]] == last.unscored_ids
                assert all(score == 0 for score in published_scores[scored_count:])
                assert final.timeout_seconds == 0.001
        evidence = os.environ.get("HINDSIGHT_SYSTEM_TEST_STORY_EVIDENCE")
        if evidence:
            with Path(evidence).open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "mode": mode,
                            "stored_ids": identities,
                            "calls": [call.model_dump() for call in ranking_provider.calls],
                            "events": [event.model_dump() for event in events],
                            "completion": final.model_dump(),
                            "recall": response.model_dump(mode="json"),
                        }
                    )
                    + "\n"
                )
    finally:
        try:
            await client.banks.delete_bank(bank)
            assert bank not in [row.bank_id for row in (await client.banks.list_banks()).banks]
        finally:
            await client.aclose()
            server.stop()
