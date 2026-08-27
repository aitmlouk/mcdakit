# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Group decisions** (`mcdakit.group`). `group_rank()` combines several
  stakeholders and reports how much they disagreed. Two aggregations are run
  because they answer different questions: averaging the scores treats the
  group as one better-informed judge, while Borda-combining each person's
  ranking treats them as voters and stops one outlying score dragging the
  mean. When the two pick different winners, `agree` is `False` — that
  divergence is a finding about the group, not a defect to hide.
- `disagreement()` separates a mean of 5.5 everyone agreed on from one that
  averages 9 and 2 — identical numbers, opposite findings — and names the most
  contested option and criterion.
- `Participant` carries a weight, so a chair with a casting vote is expressed
  as a weight rather than by entering them twice. Spread is measured
  unweighted, because the question is how far the people differ, not how much
  the chair outweighs them.
- Participants scoring different options or criteria are refused rather than
  averaged over uneven evidence.

## [0.1.0] - 2026-08-27

First release.

### The package

- **Extensible methods.** A ranking method is a `Method` object that
  declares what it needs, registered by name. A method defined outside the
  package is a first-class citizen: reachable from `rank()`, included in
  `compare_methods()`, analysable by `sensitivity()`. See `docs/extending.md`.
- `Method`, `ScoringContext`, `MethodResult` and `Wants` — the extension
  contract. `Wants.RAW` lets a method ask for the un-oriented matrix, which
  is how SPOTIS avoids needing a special case.
- `register()`, `unregister()`, `available()`, `get_method()`, `has_method()`
  and `method_names()` — the registry. `METHODS` is a live view of it, so a
  method registered at any point is immediately visible, while still behaving
  like the tuple it reads as (`in`, iteration, indexing, `len`, `==`).
- Entry-point discovery under the `mcdakit.methods` group: an installed
  package's methods are found automatically, with no import by the user. A
  plugin that fails to load warns (`PluginLoadError`) rather than taking the
  registry down with it.
- **`mcdakit.testing` — a conformance suite for method authors.**
  `check_method(MyMethod())`, or
  `python -m mcdakit.testing my_package:MyMethod`, checks thirteen properties
  every sound method holds. It catches the failures that are otherwise
  silent: a smallest-is-best measure returned without negating (every ranking
  upside down), a NaN on a zero-variance criterion, criterion direction
  handled twice, a `reversal_free` claim that does not hold. Every built-in
  passes it and CI enforces that.
- **Pluggable normalisation.** `rank(..., normalization="minmax")` swaps the
  scheme a method uses; `vector`, `minmax`, `max` and `sum` ship, and
  `@register_normalization` or a bare callable adds more. Each method declares
  the scheme it is defined with, so defaults reproduce the textbook method.
  This is a modelling axis, not a detail: on the same data the choice changes
  the winner *and* whether the ranking survives removing a loser. Methods
  whose guarantee depends on their own scaling — SPOTIS — declare
  `normalization = None` and refuse the argument rather than silently
  discarding it.
- `py.typed`, so downstream type checkers see the annotations.
- `CITATION.cff`, issue templates for bug reports and method proposals, and a
  pull-request checklist.
- Sphinx documentation, a CI matrix over Python 3.9-3.13 on Linux, macOS and
  Windows, `CONTRIBUTING.md`, and an architecture decision record.

### Refuses to guess

Nothing here degrades quietly. Each of these was a real defect during
development, caught and turned into an error rather than left to surprise
someone downstream.

- A method reading `ctx.directions` while declaring `Wants.ORIENTED` raises.
  The matrix arrives with cost columns already mirrored, so handling direction
  again flips them back and the cheapest option ranks last — a confident,
  exactly reversed ranking with no error. Found by writing a method (COPRAS)
  against the extension API and getting a plausible-looking wrong answer.
- A method returning `NaN` or infinity is refused, naming the options
  affected. NaN sorts unpredictably, so a ranking containing one is not a
  ranking.
- A keyword argument no method reads raises rather than being ignored, so
  `rank(..., v=0.5)` on a method without a `v` cannot look like it worked.
- SPOTIS without bounds warns and reports `reversal_free = False` rather than
  returning a number that looks guaranteed.
- A flat list where a matrix was meant — `[1, 2]` rather than `[[1, 2]]` —
  explains the nesting instead of raising numpy's "object of type float has
  no len()".
- A ragged matrix, unknown direction, all-zero weights or reversed bounds are
  each refused at construction, where the mistake was made.

### Core, from the initial port

- `Criterion`, `Decision` and `Result` core types with validation at
  construction — ragged matrices, unknown directions, and degenerate weights
  are refused with an explanatory error rather than a numpy traceback.
- Eight ranking methods: `simple_scoring`, `weighted_scoring`, `saw`,
  `topsis`, `vikor`, `electre`, `promethee`, `spotis`.
- `spotis` — rank-reversal-free ranking against fixed criterion bounds. Warns
  (and records `Result.reversal_free is False`) when a criterion has no bounds
  and observed min/max have to stand in.
- PROMETHEE with all six of Brans and Vincke's preference shapes.
- `sensitivity()` — per-criterion weight tolerance by bisection on the real
  scoring function, with fragile / moderate / robust / immovable bands.
- `ahp_weights()` — Saaty pairwise comparison to weights, with the
  consistency ratio reported rather than enforced.
- `compare_methods()` — every method's ranking side by side, so cross-method
  disagreement is visible instead of hidden behind one number.
- Reversal benchmark under `benchmarks/`, measuring SPOTIS at 0.0% against
  18-34% for the comparison-based methods, and pinned as a test so the
  headline claim cannot regress unnoticed.
- SPOTIS and TOPSIS cross-validated against `pymcdm` 1.4.0 to 1e-12. TOPSIS
  differs from pymcdm's *default* only because pymcdm normalises min-max where
  this package uses Hwang and Yoon's vector normalisation; told to use vector
  normalisation, the two agree exactly.
