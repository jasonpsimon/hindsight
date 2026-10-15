"""Factory routing for progressive reranking, without changing primary identity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pipeline_rerank import LegacyPredictAdapter, PipelineCrossEncoder

if TYPE_CHECKING:
    from ..config import HindsightConfig, StaticConfigProxy


def needs_legacy_adapter(enc: Any) -> bool:
    if isinstance(enc, (LegacyPredictAdapter, PipelineCrossEncoder)):
        return False
    return not any("predict_structured" in cls.__dict__ for cls in type(enc).mro())


def wrap_if_legacy(enc: Any) -> Any:
    return LegacyPredictAdapter(enc) if needs_legacy_adapter(enc) else enc


def build_encoder_from_config(config: HindsightConfig | StaticConfigProxy) -> Any:
    from .cross_encoder import create_cross_encoder
    from .structured_failover import StructuredMultiCrossEncoder

    chain = config.reranker_chain()
    if config.reranker_pipeline:
        if len(chain) > 1:
            raise ValueError(
                "HINDSIGHT_API_RERANKER_PIPELINE cannot be combined with a failover chain "
                "(HINDSIGHT_API_RERANKER_1_*). Use the pipeline alone, or failover alone."
            )
        return PipelineCrossEncoder(
            [(wrap_if_legacy(create_cross_encoder(stage.member)), stage.limit) for stage in config.reranker_pipeline]
        )
    if len(chain) == 1:
        # The primary-only public factory has historically returned the provider
        # itself. Adapting it here would break isinstance checks and provider APIs.
        return create_cross_encoder(chain[0])
    return StructuredMultiCrossEncoder([create_cross_encoder(member) for member in chain])
