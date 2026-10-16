"""Text-free, request-local identity and typed pipeline log payloads."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .rerank_result import ScoreKind

# Recall supplies persistent IDs. Direct pair callers have only positional identity.
# Immutable context values and token reset keep concurrent/cancelled calls isolated.
rerank_candidate_ids: ContextVar[tuple[str, ...] | None] = ContextVar("rerank_candidate_ids", default=None)


@dataclass(frozen=True)
class RerankStageBinding:
    candidate_ids: tuple[str, ...]
    root_indices: tuple[int, ...]
    identity_scope: Literal["memory_id", "input_index"]
    parent_request_id: str
    parent_stage: int


rerank_stage_binding: ContextVar[RerankStageBinding | None] = ContextVar("rerank_stage_binding", default=None)


class RerankEvent(BaseModel):
    parent_request_id: str | None = None
    parent_stage: int | None = None


class RerankStageEvent(RerankEvent):
    request_id: str
    stage: int
    configured_provider: str
    served_provider: str | None
    identity_scope: Literal["memory_id", "input_index"]
    input_ids: list[str]
    input_indices: list[int]
    ranked_ids: list[str]
    ranked_indices: list[int]
    output_ids: list[str]
    output_indices: list[int]
    provider_pruned_ids: list[str]
    unscored_ids: list[str]
    limit: int | None
    latency_ms: float
    score_kind: ScoreKind
    timed_out: bool
    timeout_seconds: float | None


class RerankCompletionEvent(RerankEvent):
    request_id: str
    output_ids: list[str]
    output_indices: list[int]
    restored: bool
    score_provider: str | None
    score_kind: ScoreKind
    timed_out: bool
    timeout_provider: str | None
    timeout_seconds: float | None


class RerankStageFailureEvent(RerankEvent):
    request_id: str
    stage: int
    configured_provider: str
    identity_scope: Literal["memory_id", "input_index"]
    input_ids: list[str]
    input_indices: list[int]
    latency_ms: float
    error_type: str
