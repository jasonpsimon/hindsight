# Progressive reranking evaluation

`relevance-evaluation.tar.gz` contains the reviewed public fixtures, fixed relevance labels, model and dataset pins, inference runner, reports and license attributions for the progressive-reranking cutoff smoke.

Archive SHA256:

```text
2f867710e9f75d86cd58458f8a0810c8b1a47963319aa9bbccbb2c9b2933e931
```

## Scope and results

The five-query TinyBERT/MiniLM baseline at limits 8/4 still fails the unchanged 0.8 macro-recall floor. It returns 0.8 after the first cut and 0.4 after the second.

Those queries became development data. The selected reference configuration runs MiniLM in both stages with limits 12/10. A disjoint, prospectively frozen set of twenty SciFact queries returns 1.0 macro recall after each cut. Both stages remove candidates. Every stage uses the initial gold denominator, including relevant documents lost earlier. Synthetic early-gold-loss metric controls reject at both cuts.

This is a small conditioned-candidate smoke. All labeled relevant documents are initially present. It does not measure retrieval quality, establish cross-family cascade efficiency or guarantee quality for other providers. Repeating MiniLM is a conservative preservation check, not a speed claim.

## Reproduce

Use an owned workspace and Python 3.11 or newer. Build the `hindsight-api-slim`, `hindsight-api`, `hindsight-embed` and `hindsight-dev` distributions from this branch. The repository uses `python -m build`; the build frontend and each project's build requirements must be installed in the chosen environment.

Place the wheels in `api-slim/`, `api/`, `embed/` and `dev/` under the reproduction workspace. Extract this archive into that workspace. It creates `evaluation/`.

Follow `evaluation/README.txt` for the pinned model downloads and inference commands. When installing into a new environment, include all four locally built wheels, not just the two in its abbreviated example. This supplies the source workspace's matching package versions without relying on a published release:

```sh
python -m pip install ../api-slim/*.whl ../api/*.whl ../embed/*.whl ../dev/*.whl -r requirements-evaluation.txt
```

Run that command from `evaluation/` inside the chosen environment. The archive's validation command must exit 0. The rejected-baseline command must write its report and exit 1. Keep every report, including failures. Do not edit labels, lower the guard or overwrite an earlier measurement.

The download, hash verification, extracted runner and inference steps were exercised against the built distributions. Interpreter and dependency trees were reused during the reported matrix, so these results do not claim fresh dependency resolution on every platform.

Model weights are downloaded separately and are not included. The archive includes dataset and model attribution. Preserve those terms when redistributing the public fixture text.
