# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Reversal benchmark under `benchmarks/`, reproducing the published table.
