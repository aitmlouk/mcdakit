# mcdakit — build plan

A standalone Python package for multi-criteria decision analysis. **No Odoo.**
Nothing in this project imports from, depends on, or knows about Odoo. Source
code is *read* from the Odoo modules as a reference during the port, then the
connection ends.

Written 2026-08-26. This document is the brief for a fresh session; it assumes
no memory of the conversation that produced it.

---

## 1. Why this exists

The field is crowded. Before writing a line, understand what we are **not**
doing, because "another TOPSIS implementation" has no audience:

| Package | Status |
|---|---|
| `pymcdm` v1.4 | Mature, the reference implementation |
| `scikit-criteria` v0.10 | Mature, sklearn-style API |
| `pydecision` v5.1.1 | Active, already has the LLM angle |

**Our position: rankings that hold.** Most libraries answer "what is the
ranking?". `mcdakit` also answers **"would the ranking survive someone
disagreeing with my inputs?"** Two features carry that, and they are the
reason the package is worth publishing:

1. **Rank-reversal-free ranking** (SPOTIS) — the ranking does not change when
   an option nobody chose is removed from the shortlist.
2. **Sensitivity analysis** — how far each weight can move before the winner
   changes, reported as a tolerance and a fragile/moderate/robust band.

### The evidence behind that positioning

Measured on the existing engine, not assumed. 400 random 5x4 problems,
removing the **last-placed** option and re-ranking the survivors:

| Method | Rank reversal rate |
|---|---|
| SAW | 16.8% |
| ELECTRE | 31.0% |
| TOPSIS | 31.8% |
| Weighted sum (min-max) | 32.0% |
| PROMETHEE | 34.2% |
| VIKOR | 45.2% |
| **SPOTIS** | **0.0%** |

In roughly a third of cases, deleting an option that lost changes the order of
the ones that remain. A worked case from real supplier data:

```
WITH the rejected option:     Kestrel 0.6000  >  Nordpack 0.5953
WITHOUT it:                   Nordpack 0.6511 >  Kestrel 0.6000
```

Kestrel's score never moved. Nordpack's rose because dropping the rejected
option shrank the price span (1.2 -> 0.55), inflating its normalised price
score from 0.25 to 0.545. **A rejected option was defining the scale.**

These numbers must be reproduced by the package's own benchmark (Phase 6). If
they do not reproduce, the positioning is wrong and this document needs
revisiting before release — do not paper over a mismatch.

### Reading

- Dezert, Tchamova, Han, Tacnet, *The SPOTIS Rank Reversal Free Method for
  Multi-Criteria Decision-Making Support* — the method we implement.
- Li & Abbas, *A Trend Analysis of Rank Reversal in Widely Used Decision-Making
  Methods*, J. Multi-Criteria Decision Analysis 33(1), Jan 2026.
- *Entropy and Normalization in MCDA: A Data-Driven Perspective on Ranking
  Stability*, Jan 2026 — normalisation is "a substantive modeling decision
  rather than a purely technical preprocessing step".
- Sałabun, *The COMET method* — first fully reversal-immune method; needs far
  more input than SPOTIS, which is why we implement SPOTIS.

---

## 2. Decisions already made

| Decision | Value | Note |
|---|---|---|
| Package name | `mcdakit` | Verified available on PyPI 2026-08-26; no GitHub collision |
| Import name | `mcdakit` | Identical to distribution name. Do not split them |
| Python | >= 3.9 | Dev on 3.12 |
| Runtime deps | `numpy` only | Hard rule, see below |
| Layout | `src/` | Prevents accidental imports of the working tree |
| Build backend | hatchling | |
| Test | pytest | |
| Lint | ruff | |

**The numpy-only rule.** No pandas, no scipy, no sklearn. Every algorithm being
ported is already pure numpy. A single tiny dependency is a large part of why
someone would pick this over a heavier alternative. If something seems to need
scipy, write the twenty lines instead.

### Open — decide before Phase 1 ends

- **Licence.** Recommendation: **Apache-2.0** (permissive, patent grant, good
  for adoption). Note this differs from the Odoo modules (OPL-1 / LGPL-3) —
  that is intentional and fine, they are separate projects.
- **Does SPOTIS ship in v0.1?** Recommendation: **yes**. It is the
  differentiator; shipping v0.1 without it is shipping the generic half.

---

## 3. Source material

Read-only reference. Paths are relative to
`/Users/addiait-mlouk/VscodeProjects/odoo/custom-addons/omcda/`.

### Already pure — 335 lines, zero Odoo references. Port nearly as-is.

`omcda/models/mcda.py`:

| Line | Function |
|---|---|
| 1025 | `_orient_matrix(data, directions)` — cost/benefit via `max + min - x` |
| 1156 | `preference_function(x, y, threshold=0)` |
| 1162 | `preference_degree(difference, shape, q, p, s)` — 6 PROMETHEE shapes |
| 1216 | `calculate_preference_matrix(data, weights, thresholds, shapes)` |
| 1251 | `calculate_flows(preference_matrix)` |
| 1259 | `calculate_simple_scoring(data)` |
| 1264 | `calculate_weighted_scoring(data, weights)` — min-max normalised |
| 1293 | `calculate_topsis(data, weights)` |
| 1327 | `calculate_saw(data, weights)` |
| 1348 | `calculate_vikor(data, weights, v=0.5)` |
| 1388 | `calculate_electre(data, weights)` |

### Needs real extraction — algorithm is sound, but it walks ORM records

`omcda/wizard/ahp_weights.py`: `SAATY_SCALE` (19), `RANDOM_INDEX` (35),
`CONSISTENCY_LIMIT = 0.10` (40), `_priorities` (105, row geometric mean),
`_consistency` (118).

`omcda/models/dashboard.py`: `FRAGILE_THRESHOLD = 0.10` (458),
`ROBUST_THRESHOLD = 0.25` (459), `_winner_index` (461), `_flip_point` (469,
bisection on the real scoring function), `get_sensitivity` (509).

`_flip_point` uses bisection deliberately: the methods are not all linear in
the weights (TOPSIS and VIKOR normalise, PROMETHEE is a step function), so an
analytic shortcut would be wrong for most of them. Keep the bisection.

### Existing tests worth mining

`omcda/tests/`: `test_ranking.py`, `test_direction.py`,
`test_preference_functions.py`, `test_ahp.py`, `test_sensitivity.py`. These
encode known-good expected values (e.g. weighted scoring 0.6000 / 0.5250 /
0.2833 on the fixture). Port the numbers; drop the Odoo scaffolding.

### Do not port

`compute_ranking`, `compute_group_ranking`, `_get_score_matrix`, e-mail,
record sync, dashboards — 694 lines of ORM plumbing that belong in Odoo.

---

## 4. Phases

Each phase ends green: tests pass, ruff clean. Do not start the next phase
with the previous one broken.

### Phase 1 — skeleton

```
mcdakit/
  pyproject.toml
  README.md
  LICENSE
  CHANGELOG.md
  .gitignore
  src/mcdakit/__init__.py
  tests/
```

`git init` (this is its own repo, unrelated to the Odoo repo). Configure ruff
and pytest in `pyproject.toml`. `pip install -e ".[dev]"` works, empty test
suite passes.

### Phase 2 — core types

`Criterion` (name, weight, direction `"benefit"|"cost"`, optional
`bounds=(lo, hi)`), and a `Decision`/`Result` pair. Validate at construction:
weights non-negative and not all zero, `direction` a known value, `bounds`
with `lo < hi`, matrix shape matching the criteria count. Raise clear errors —
a ragged matrix should say so, not produce a numpy traceback.

Port `_orient_matrix` here. Test cost/benefit inversion against the known
supplier case: the cheap-but-poor option must not win on a cost criterion.

### Phase 3 — the seven classical methods

Port all eleven functions from the table above into
`src/mcdakit/methods/`. One module per method, each a plain function over
`(data, weights)` returning a score vector where higher is better.

Port the expected values from `test_ranking.py` as the first tests. This phase
is a move, not a rewrite — resist redesigning the maths while relocating it.

**Watch:** `_score_with_method` in the Odoo code calls these through
`self.env['promethee.rank']` even though they are `@staticmethod`s. That
indirection is Odoo-only and disappears here.

### Phase 4 — SPOTIS (the differentiator)

```
d_ij       = |S_ij - S*_j| / |S_j^max - S_j^min|
d(A_i, s*) = sum_j w_j * d_ij            # lowest distance wins
```

`S*` is the ideal point built from the **fixed bounds**, not from the observed
data: `S*_j = hi_j` for benefit criteria, `lo_j` for cost criteria. Because
every option is measured against the ideal alone and never against other
options, adding or removing an option cannot move anyone.

Return scores negated so "higher is better" matches the other methods, and say
so in the docstring — this is exactly the kind of sign convention that causes
silent bugs later.

**Bounds are required.** When a criterion has no bounds, fall back to observed
min/max and **emit a warning naming the criteria involved**: the result is then
no longer reversal-free, and silently pretending otherwise would be the worst
possible failure for this package. Surface it in the result object too, so a
caller can check programmatically rather than scraping warnings.

Tests: the 4-supplier case must keep `Kestrel > Nordpack` with and without the
rejected option; the paper's own car example (SPOTIS agrees with AHP and SAW
where TOPSIS ranks the worst car first).

### Phase 5 — AHP and sensitivity

**AHP:** Saaty 1-9 scale, row-geometric-mean priorities, consistency ratio
with the random-index table, `CONSISTENCY_LIMIT = 0.10`. Return the CR and let
the caller decide; do not raise on an inconsistent matrix.

**Sensitivity:** port `_flip_point` bisection and the fragile/moderate/robust
bands. Public API roughly:

```python
sensitivity(result) -> {
    "overall": float | None,     # smallest tolerance across criteria
    "level": "fragile" | "moderate" | "robust" | "immovable",
    "weakest": str,              # criterion name
    "rows": [ {name, weight, increase, decrease, tolerance, flips_to}, ... ],
}
```

Works for every method, since it bisects over whatever scoring function it is
given.

### Phase 6 — validation

This is where credibility comes from, and it is slower than it looks.

1. **Worked examples from the literature.** At minimum the SPOTIS paper's car
   problem. Each as a test citing its source, so results are checkable against
   a published paper rather than against ourselves.
2. **Property-based tests** (hypothesis, dev-only): scale invariance where
   claimed, permutation invariance, weights summing differently but
   proportionally give identical order.
3. **The reversal benchmark**, as a reproducible script, regenerating the table
   in section 1. If SPOTIS is not 0.0%, the implementation is wrong.
4. **Cross-check against `pymcdm`** in dev extras. Where we disagree,
   understand why before shipping — a documented, deliberate difference is
   fine; an unexplained one is a bug.

### Phase 7 — docs and release

README leading with the differentiator, not the method list. The first code
block a reader sees should be the sensitivity or the stability story.

Docstrings on every public function with the formula and a citation.
CHANGELOG. Then TestPyPI, install into a clean venv, verify, and only then
PyPI `0.1.0`.

**Claim the name early.** `mcdakit` has been free for a while but there is no
reason to assume it stays free. Consider a `0.0.1` placeholder at the end of
Phase 1.

---

## 5. Target API

The whole package should be usable without reading docs:

```python
from mcdakit import Criterion, rank, sensitivity

criteria = [
    Criterion("Price", weight=0.40, direction="cost", bounds=(2.00, 4.00)),
    Criterion("Quality", weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost", bounds=(5, 35)),
    Criterion("Support", weight=0.15, direction="benefit", bounds=(0, 10)),
]

result = rank(matrix, criteria, method="spotis", labels=[...])
result.winner  # "Kestrel Supply"
result.ranking  # [("Kestrel Supply", 0.72), ...]

sensitivity(result).level  # "fragile" — a 4% weight change flips it
```

Method names are strings, matching the Odoo engine's vocabulary:
`simple_scoring`, `weighted_scoring`, `saw`, `topsis`, `vikor`, `electre`,
`promethee`, `spotis`.

Also expose `compare_methods(matrix, criteria)` returning every method's
ranking — the cross-method agreement check. When methods disagree 4-3, that
disagreement *is* the finding, and hiding it behind one number would be
dishonest.

---

## 6. Rules

1. **No Odoo.** No imports, no references, no path assumptions. If `mcdakit`
   only works inside Odoo, it has failed.
2. **numpy only** at runtime.
3. **Never silently degrade.** Missing bounds, zero-variance criteria, ties,
   all-zero weights: warn or raise, never quietly produce a number that looks
   authoritative. This package's whole claim is trustworthiness.
4. **Cite the maths.** Every method's docstring names its source. This is a
   research-adjacent library and will be read by people who know the papers.
5. **Port before improving.** Get the seven methods moved and passing their
   existing expected values first. Improve afterwards, deliberately, with a
   test showing what changed.
6. **The Odoo modules are not touched by this work.** They keep their 120
   passing tests and their own copy of the engine. Rewiring Odoo to depend on
   `mcdakit` is a *later, separate* decision — do not start it here, and do not
   let it shape the API.

---

## 7. Definition of done for v0.1.0

- [ ] Installs from PyPI into a clean venv, `import mcdakit` works
- [ ] Eight methods, each tested against known-good values
- [ ] SPOTIS measured at 0.0% reversal by our own benchmark
- [ ] Missing bounds warns, and the warning is tested
- [ ] AHP with consistency ratio; sensitivity with tolerance bands
- [ ] At least one worked example from a published paper, cited
- [ ] README leads with stability, not with a method list
- [ ] Apache-2.0 (or chosen licence) present and referenced in `pyproject.toml`
- [ ] ruff clean, pytest green, no Odoo string anywhere in `src/`

---

## 8. Honest risks

- **The field is crowded.** Without SPOTIS and sensitivity, this is a worse
  `pymcdm`. The differentiator is the product; do not let it slip to v0.2.
- **Bounds are a real cost.** SPOTIS needs per-criterion bounds, and the
  authors call automatic bound selection "a challenging open question". In
  procurement the bounds usually exist (budget ceiling, acceptable lead time,
  rating scale). Where they do not, we degrade loudly (rule 3).
- **Validation is the slow part.** Phase 6 done properly takes longer than
  Phases 1-5 combined. It is also the only thing that makes the package
  trustworthy rather than merely present.
- **Two release cadences later.** If Odoo ever depends on this, a scoring fix
  becomes: fix library, release, bump the pin. That is a real cost and is the
  main argument for keeping the projects separate for now.
