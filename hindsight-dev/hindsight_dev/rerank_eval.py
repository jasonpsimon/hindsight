"""Offline relevance preservation, independent of provider score scales."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CutoffRecall:
    gold_count: int
    limit: int | None
    input_recall: float
    recall_before: float
    recall_after: float
    gold_pruned: list[str]
    gold_lost: list[str]

    def require_recall(self, minimum: float) -> None:
        """Require recall >= minimum, a finite real number in [0, 1]."""
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1:
            raise ValueError("minimum recall must be between 0 and 1")
        if self.recall_after < minimum:
            raise ValueError(f"recall {self.recall_after} below {minimum}; gold lost at cut: {self.gold_lost}")


def evaluate_rerank_cutoff(
    *,
    gold_ids: set[str],
    input_ids: list[str],
    ranked_ids: list[str],
    output_ids: list[str],
    limit: int | None,
) -> CutoffRecall:
    """Use the INITIAL/global qrels for every stage, never the stage's surviving gold.

    ranked_ids are post-provider-mask, pre-limit. Provider losses are separate
    from shortlist losses. IDs, not document text, determine relevance. Repeated
    IDs count once, distinct IDs with duplicate text remain distinct. No gold is
    defined as perfect preservation, including an empty input.
    Output must be the exact ordered prefix, including repeated occurrences.
    """
    inputs, ranked, outputs = set(input_ids), set(ranked_ids), set(output_ids)
    if not outputs <= ranked <= inputs:
        raise ValueError("stage outputs must be subsets of ranking and input")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or len(output_ids) > limit
    ):
        raise ValueError("invalid candidate cutoff")
    if Counter(ranked_ids) - Counter(input_ids):
        raise ValueError("ranking contains more occurrences than input")
    if output_ids != ranked_ids[:limit]:
        raise ValueError("stage output must equal the ordered ranked prefix for its cutoff")
    denominator = len(gold_ids)
    return CutoffRecall(
        gold_count=denominator,
        limit=limit,
        input_recall=len(inputs & gold_ids) / denominator if denominator else 1.0,
        recall_before=len(ranked & gold_ids) / denominator if denominator else 1.0,
        recall_after=len(outputs & gold_ids) / denominator if denominator else 1.0,
        gold_pruned=sorted((inputs - ranked) & gold_ids),
        gold_lost=sorted((ranked - outputs) & gold_ids),
    )
