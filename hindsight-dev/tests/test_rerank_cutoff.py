"""Cutoff reports use the original qrels denominator, including earlier losses."""

from enum import IntEnum

import pytest

import hindsight_dev


def test_cutoff_guard_rejects_irrecoverable_gold_loss() -> None:
    evaluate = getattr(hindsight_dev, "evaluate_rerank_cutoff", lambda **kwargs: None)
    report = evaluate(
        gold_ids={"a", "b"}, input_ids=["a", "b", "c"], ranked_ids=["c", "a", "b"], output_ids=["c", "a"], limit=2
    )
    assert report is not None, "No relevance-preservation report was produced"
    assert report.recall_before == 1.0
    assert report.recall_after == 0.5
    assert report.gold_lost == ["b"]
    with pytest.raises(ValueError, match="recall"):
        report.require_recall(1.0)


@pytest.mark.parametrize(
    "gold,inputs,ranked,outputs,before,after,pruned,lost",
    [
        ({"a", "b"}, ["a"], ["a"], ["a"], 0.5, 0.5, [], []),
        ({"a", "b"}, ["a", "b"], ["b"], ["b"], 0.5, 0.5, ["a"], []),
        ({"a"}, ["a", "a"], ["a", "a"], ["a", "a"], 1.0, 1.0, [], []),
        (set(), [], [], [], 1.0, 1.0, [], []),
        ({"a"}, [], [], [], 0.0, 0.0, [], []),
    ],
)
def test_fixed_denominator_and_empty_or_duplicate_ids(gold, inputs, ranked, outputs, before, after, pruned, lost):
    report = hindsight_dev.evaluate_rerank_cutoff(
        gold_ids=gold, input_ids=inputs, ranked_ids=ranked, output_ids=outputs, limit=None
    )
    assert report.recall_before == before
    assert report.recall_after == after
    assert report.gold_pruned == pruned
    assert report.gold_lost == lost


@pytest.mark.parametrize("limit", [True, False, 1.5, float("inf"), float("nan"), 0, -1])
def test_cutoff_rejects_non_positive_integer_limit(limit) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        hindsight_dev.evaluate_rerank_cutoff(
            gold_ids={"a"}, input_ids=["a"], ranked_ids=["a"], output_ids=["a"], limit=limit
        )


@pytest.mark.parametrize(
    "inputs,ranked,outputs,limit",
    [
        (["x", "gold"], ["x", "gold"], ["gold"], 1),
        (["x", "gold"], ["x", "gold"], ["gold", "x"], None),
        (["x", "gold"], ["x", "gold"], ["x"], None),
        (["x", "gold"], ["x", "gold"], [], 1),
        (["gold"], ["gold", "gold"], ["gold"], 1),
    ],
)
def test_rejects_misleading_ranked_prefix(inputs, ranked, outputs, limit) -> None:
    with pytest.raises(ValueError, match="prefix|occurrence"):
        hindsight_dev.evaluate_rerank_cutoff(
            gold_ids={"gold"}, input_ids=inputs, ranked_ids=ranked, output_ids=outputs, limit=limit
        )


@pytest.mark.parametrize("minimum", [True, False, "0.8", None, float("nan"), float("inf"), -0.1, 1.1])
def test_guard_rejects_invalid_threshold(minimum) -> None:
    report = hindsight_dev.evaluate_rerank_cutoff(
        gold_ids={"a"}, input_ids=["a"], ranked_ids=["a"], output_ids=["a"], limit=1
    )
    with pytest.raises(ValueError, match="minimum recall"):
        report.require_recall(minimum)


def test_guard_boundary_and_duplicate_prefix_keep_global_denominator() -> None:
    report = hindsight_dev.evaluate_rerank_cutoff(
        gold_ids={"a", "lost-earlier"}, input_ids=["a", "a"], ranked_ids=["a", "a"], output_ids=["a"], limit=1
    )
    assert report.gold_count == 2
    assert report.input_recall == report.recall_before == report.recall_after == 0.5
    report.require_recall(0.5)
    report.require_recall(0)
    with pytest.raises(ValueError, match="below"):
        report.require_recall(0.8)


def test_cutoff_accepts_integer_subclass_like_pipeline() -> None:
    class Limit(IntEnum):
        TWO = 2

    report = hindsight_dev.evaluate_rerank_cutoff(
        gold_ids={"a"}, input_ids=["a", "b", "c"], ranked_ids=["a", "b", "c"], output_ids=["a", "b"], limit=Limit.TWO
    )
    assert report.recall_after == 1.0
