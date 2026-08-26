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
pytest -q                        # the suite, including doctests
pytest -q --cov=mcdakit          # must stay at 100% (line and branch)
ruff check .                     # lint
ruff format --check .            # formatting
mypy                             # types
vulture src/ .vulture-allowlist.py --min-confidence 60   # dead code
python benchmarks/reversal.py    # SPOTIS must measure 0.0%
```

All of these must pass. `ruff format .` fixes formatting in place.

While developing you will usually want a subset. The whole suite takes about
four seconds, so reach for these mainly to read the output, not for speed:

```bash
pytest tests/test_normalization.py          # one file
pytest tests/test_normalization.py::TestDefinitions   # one class
pytest -k reversal                          # anything matching a keyword
pytest -x -q                                # stop at the first failure
pytest --lf                                 # only what failed last time
```

If `pytest` reports `ModuleNotFoundError: No module named 'mcdakit'`, the
virtualenv is not active — activate it, or call it explicitly with
`.venv/bin/python -m pytest`. The `src/` layout means the package is only
importable once installed, which is deliberate: it stops a stale working-tree
copy shadowing the one under test.

Coverage is enforced at **100%, line and branch**. A line nothing exercises is
a line nobody has checked. Where code is genuinely unreachable — a guard
against the library's own plumbing breaking, a fallback a validated type makes
impossible — mark it `# pragma: no cover` **with a comment saying why**, and
never by lowering the threshold. There are 14 such pragmas today; each names
its reason.

If vulture flags something that *is* used — via a test, a doc example, or a
plugin — add it to `.vulture-allowlist.py` **with the reason**. An allowlist
of unexplained names stops being a check.

## Documentation

The site is Sphinx with MyST, so pages are Markdown. Build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W docs docs/_build/html
open docs/_build/html/index.html
```

`-W` turns warnings into errors, which is what CI and the hosting build both
use. A broken cross-reference fails the build rather than shipping a dead link.

Where things go:

| Page | For |
|---|---|
| `docs/quickstart.md` | Someone who has just installed it |
| `docs/stability.md` | Rank reversal, SPOTIS, sensitivity — the *why* |
| `docs/extending.md` | Writing and shipping a method |
| `docs/api.md` | Generated reference; add an `automodule` entry for new modules |
| `docs/adr/` | Decisions with consequences, and their costs |

Two rules the tests enforce: every name in `mcdakit.__all__` must appear in
`docs/api.md`, and every public callable needs a docstring. A name nobody can
find in the docs is a name nobody uses.

### Hosting

Two configurations are committed, and they are **alternatives — pick one**:

- `.readthedocs.yaml` — Read the Docs. Gives versioned docs (one build per
  git tag), a search index, and PDF output. The convention in scientific
  Python, and what a researcher will look for first. Needs an account at
  readthedocs.org and the repository imported there.
- `.github/workflows/docs.yml` — GitHub Pages. No third-party account; enable
  under Settings → Pages → Source: GitHub Actions. Publishes `main` only, so
  there is no version switcher.

Running both means two URLs that drift apart, and readers finding whichever
ranks higher in search. Delete the one you are not using.

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
package at all. A method distributed as a plugin is a first-class citizen:
reachable by name, included in `compare_methods()`, analysable by
`sensitivity()`.

**Whether it lives here or in your own package, check it first:**

```bash
python -m mcdakit.testing my_package.methods:MyMethod
```

That runs the conformance suite — thirteen properties every sound method
holds. It catches the failures that are otherwise silent: a score convention
that ranks everything backwards, a NaN on a criterion every option scores
alike, criterion direction handled twice. Call it from your own tests too:

```python
from mcdakit.testing import check_method

def test_my_method_conforms():
    check_method(MyMethod())
```

Every method shipped here passes it, and CI enforces that — holding
contributors to a standard the library itself failed would be indefensible.

Passing is a floor, not a ceiling: it says your method is well-behaved, not
that its arithmetic matches the paper.

If it does belong here, it needs:

- a `Method` subclass with `name`, `summary` and `citation`;
- the algorithm as a plain function over numpy arrays, so it can be tested and
  imported on its own;
- `python -m mcdakit.testing` passing;
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
