# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Extensible methods.** A ranking method is now a `Method` object that
  declares what it needs, registered by name. A method defined outside the
  package is a first-class citizen: reachable from `rank()`, included in
  `compare_methods()`, analysable by `sensitivity()`. See `docs/extending.md`.
- `Method`, `ScoringContext`, `MethodResult` and `Wants` — the extension
  contract. `Wants.RAW` lets a method ask for the un-oriented matrix, which is
  what SPOTIS needed a hardcoded special case for previously.
- `register()`, `unregister()`, `available()`, `get_method()`, `has_method()`
  and `method_names()` — the registry.
- Entry-point discovery under the `mcdakit.methods` group: an installed
  package's methods are found automatically, with no import by the user. A
  plugin that fails to load warns (`PluginLoadError`) rather than taking the
  registry down with it.
- `py.typed`, so downstream type checkers see the annotations.
- Sphinx documentation, a CI matrix over Python 3.9-3.13 on Linux, macOS and
  Windows, `CONTRIBUTING.md`, and an architecture decision record.

### Changed

- `METHODS` is a live view of the registry rather than a frozen tuple, so a
  method registered at any point is immediately visible. It still supports
  everything a tuple did (`in`, iteration, indexing, `len`, `==`).
- Passing a keyword argument no method reads now raises `McdaError` instead of
  being silently ignored — the same class of bug as the `v=` defect fixed
  before 0.1.0, caught generically this time.
- `Result.ranks` is a property; it was always populated, but was typed as
  optional.

## [0.1.0] - 2026-08-26

First release.

### Added

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
