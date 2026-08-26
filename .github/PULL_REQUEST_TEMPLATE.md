## What this changes

## Checks

The same ones CI runs:

- [ ] `pytest -q`
- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy`
- [ ] `python benchmarks/reversal.py` — SPOTIS still measures 0.0%

## If this adds or changes a method

- [ ] `python -m mcdakit.testing <module>:<Method>` passes
- [ ] Tested against values verified outside this codebase (paper, hand
      computation, or another library — say which in the test's docstring)
- [ ] Docstring names the source
- [ ] Added to the README method table

## If this changes a published number

Benchmark rates, worked-example scores and documented outputs appear in the
README, the docs and the changelog. A stale number there is a correctness bug.

- [ ] README, docs and `CHANGELOG.md` updated together
