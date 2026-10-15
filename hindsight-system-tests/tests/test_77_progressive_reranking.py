"""Ordered shortlists cross real provider HTTP, PostgreSQL and the public client.

The first model deliberately prefers CELLO to BERLIN. The second prefers CHESS
if it leaks through, otherwise BERLIN. Keeping the original pool, forgetting to
sort survivors, or stopping after the first model therefore cannot pass.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from hindsight_client import Hindsight
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
        assert sorted(row.text for row in stored.items) == sorted(ALL_FACTS)
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
            assert row.scores.final == pytest.approx(row.scores.reranker * multiplier, abs=1e-6)
            assert row.entities is None
            assert row.source_fact_ids is None
            assert row.attachments is None
            assert row.mentioned_at is not None
        assert response.trace is None
        assert response.entities is None
        assert response.chunks is None
        assert response.source_facts is None
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
