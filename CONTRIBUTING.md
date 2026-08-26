# Contributing

## Setup

```bash
git clone https://github.com/aitmlouk/mcdakit
cd mcdakit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,compare,docs]"
```

## The checks CI runs

```bash
pytest -q                        # 269 tests, includes doctests
ruff check .                     # lint
ruff format --check .            # formatting
mypy                             # types
python benchmarks/reversal.py    # SPOTIS must measure 0.0%
```

All five must pass. `ruff format .` fixes formatting in place.

## House rules

These are not style preferences; each one is load-bearing.

**1. numpy only at runtime.** No pandas, no scipy, no sklearn. A single tiny
dependency is a large part of why someone picks this over a heavier
alternative. If something seems to need scipy, write the twenty lines. There
is a test that enforces this.

**2. Never silently degrade.** Missing bounds, zero-variance criteria, ties,
all-zero weights, an argument nobody reads: warn or raise, never quietly
produce a number that looks authoritative. This package's whole claim is
trustworthiness, and a confident wrong answer is worse than an error.

**3. Cite the maths.** Every method's docstring names its source. This library
will be read by people who know the papers.

**4. Verify numbers before publishing them.** Any figure in a docstring,
README or docs page must be produced by running the code, not recalled.
Recalled figures were wrong three times out of three during the initial build.

**5. Higher is always better.** Every method returns scores where larger ranks
higher, whatever its native convention. Negate in the method, document it, and
test it.

## Adding a method

Please read the extending guide (`docs/extending.md`) first — you may not need to change this
package at all. A method distributed as a plugin is a first-class citizen.

If it does belong here, it needs:

- a `Method` subclass with `name`, `summary` and `citation`;
- the algorithm as a plain function over numpy arrays, so it can be tested and
  imported on its own;
- tests against values verified **outside this codebase** — a published paper,
  a hand computation, or a cross-check against another library;
- an entry in the README's method table.

## Tests

New behaviour needs a test that would fail without it. Beyond that:

- Prefer values checked independently over values the code happened to
  produce. A fixture recording current output passes forever, including after
  a regression.
- When a result is surprising but correct — VIKOR picking a different winner
  from every additive method, say — write the test that documents *why*, not a
  test that hides it.
- Property-based tests live in `tests/test_properties.py`. Invariants
  (dominance, scale invariance, permutation invariance) belong there.

## Changing a published number

If a benchmark figure moves, update the README, the docs, and `CHANGELOG.md`
together, and say in the commit message what changed and why. A stale number
in the README is a correctness bug.
