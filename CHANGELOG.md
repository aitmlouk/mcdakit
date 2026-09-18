# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-09-18

The first published release. Version 0.1.0 below was tagged in the changelog
during the initial port but never released, so this is what a user can
actually install.

### Changed

- `agreement()` returns `consensus: None` when two or more options tie for the
  lead, and names the joint leaders in a new `tied` field. It previously
  returned whichever tied option happened to be counted first, which reported
  a split vote as though it were a decision — concealing the disagreement the
  function exists to surface. Adding the seventeenth method turned the
  documented example into a 7--7 tie and exposed this.
- One module per method under `mcdakit.methods`, replacing the grouped
  `reference_point` module. Import paths change for direct function imports
  (`from mcdakit.methods.aras import aras`); `rank(method="aras")` and every
  other public entry point are unaffected.

### Added

- **Provider adapters** (`mcdakit.ai_providers`): `openai()`, `anthropic()`,
  `google()`, `ollama()` and `openai_compatible()` return ready-made `ask`
  callables, reading `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and
  `GOOGLE_API_KEY`/`GEMINI_API_KEY` from the environment. `openai_compatible`
  covers any server exposing `/chat/completions` — vLLM, LM Studio,
  llama.cpp, Groq, Together, OpenRouter — so a local model needs no key and
  nothing leaves the machine.

  **No provider SDK is required.** Each adapter speaks the vendor's HTTP API
  through the standard library, so NumPy remains the only runtime dependency
  and the one-dependency comparison in the paper still holds. A test parses
  the module's imports and fails if any non-stdlib name appears.

  Rate limits and transient 5xx are retried with exponential backoff
  honouring the provider's `Retry-After`; a 400 is not retried, since it will
  fail identically however often it is sent. A missing key names the
  environment variable to set instead of surfacing a vendor 401, and an
  unexpected reply shape names the provider and shows what arrived.

  `mcdakit.ai` still takes any `Callable[[str], str]`, so an in-house gateway
  or a test fixture works exactly as before. The adapters are a convenience,
  not a requirement.
- `load_env()` reads `KEY=value` lines from a `.env` file into the
  environment, so keys live in a gitignored file rather than in source. It
  searches parent directories, understands `export`, quotes and comments, and
  leaves an already-set variable alone unless `override=True` — a file in a
  checkout should not silently beat what a deployment set. No dependency is
  added; `python-dotenv` would cost more than the dozen lines it replaces.
  A `.env.example` template ships with the repository, and `.env` is
  gitignored.

- **WPM** (`wpm`), the weighted product model, also called MEW. Where a
  weighted sum lets a strength pay for a weakness, the product does not: one
  near-zero value drags the whole score down however good the rest are, which
  is the reason to choose it when a criterion is disqualifying rather than
  merely undesirable. It is `waspas(lambda_=0.0)` and is implemented as that
  call so the two cannot drift apart, but is exposed under its own name
  because nobody searching for WPM would think to look inside a WASPAS
  parameter. Agrees with `pymcdm` to 1e-12 once both use the same
  normalisation — `pymcdm` defaults to sum normalisation where mcdakit uses
  linear, which rescales the values without changing their order — and with
  the published formula written out independently. This completes the set of
  methods common to the surveyed libraries; seventeen are now provided.
- **Full AHP** (`ahp_rank`), ranking alternatives from pairwise comparisons
  alone — no decision matrix. Alternatives are compared against each other
  under each criterion, so criteria nobody can measure numerically can still
  drive the ranking. Reproduces Saaty's car example, and reports a consistency
  ratio for every comparison matrix separately rather than averaging them,
  since a hierarchy is only as sound as its least coherent judgement set.
  `AhpResult.inconsistent` names the offenders. Cross-checked against
  `pyrepo-mcda`'s `_classic_ahp`, which agrees on weights, global priorities
  and consistency ratios to fourteen significant figures.

  Note that the *public* `AHP` of `pyrepo-mcda` is a min-max weighted sum over
  a numeric matrix (verified identical to `weighted_scoring` here, as its own
  docstring concedes), and `pyDecision`'s `ahp_method` is the weighting step
  alone (identical to `ahp_weights` here). Full AHP exists in `pyrepo-mcda`
  only as a private method.
- **ARAS, COCOSO, CODAS, EDAS, MABAC and MARCOS**, completing the set of
  methods common to every other Python MCDA library. Each reproduces values
  that `pymcdm` and `pyrepo-mcda` independently agree on, and matches `pymcdm`
  on 40 random problems spanning different shapes, weightings and direction
  mixtures. The package now provides sixteen methods.
- **WASPAS** (`waspas`), blending the weighted sum and weighted product
  models under a `lambda_` parameter. Agrees with `pymcdm` and `pyrepo-mcda`
  to machine precision and with the published formula of Zavadskas et al.
  (2012) written out independently.
- Methods may declare `requires_positive`, which the conformance suite honours
  by not demanding a score for a column of zeros — and then checks that the
  method genuinely refuses such input rather than returning NaN, so the
  exemption has to be earned.
- **COPRAS** (`copras`), the first of the methods common to every other Python
  MCDA library. Validated against the published formula of Zavadskas,
  Kaklauskas and Sarka (1994) written out independently, and agreeing with
  `pyrepo-mcda` and `pyDecision`. It resolves criterion direction itself and
  so takes the matrix as measured; negative values are refused, since
  sum-normalisation is defined for ratio-scale data.

- **Explaining a ranking** (`mcdakit.explain`). `explain()` decomposes an
  option's score by criterion; `compare()` decomposes the margin between two
  options and identifies a criterion as *decisive* when reversing it alone
  would reverse the outcome. Transparency is treated in the recent
  decision-analysis literature as a precondition for trusting a computed
  result, and no other Python MCDA library provides it.
- The attribution is honest about its own validity. For an additive method the
  contributions sum exactly to the score, and a test asserts it. For TOPSIS,
  VIKOR, ELECTRE and PROMETHEE no exact decomposition exists, so each
  criterion is attributed by muting it and re-scoring; `basis` reports which
  was used, and the printed output carries the caveat.

- **Language-model assistance** (`mcdakit.ai`), opt-in and designed so that
  the model proposes and the package verifies. `propose_criteria()` and
  `propose_weights()` turn a described problem into validated
  `Criterion` objects; anything malformed or out of range is rejected with a
  reason rather than absorbed. Proposed weights are run through
  `sensitivity()`, so the caller learns whether the resulting recommendation
  would survive the model being somewhat wrong, and are compared against the
  objective schemes in `mcdakit.weighting`, so a model that orders importance
  unlike every data-driven method is flagged. Asking more than once measures
  the model's self-consistency, which a single reply cannot reveal. Nothing is
  applied automatically.
- `propose_comparisons()` elicits *pairwise* judgements rather than a weight
  vector, because a weight vector cannot contradict itself and a set of
  ratios can. The replies are assembled into a reciprocal matrix and put
  through `ahp_weights()`, so a model that answers inconsistently is caught by
  Saaty's consistency ratio rather than by inspection.
- `simulate_panel()` scores a problem once per expert persona and returns
  `Participant` objects, so a simulated panel is scrutinised by the same
  `group_rank()` and `disagreement()` machinery as a real one. A persona whose
  reply is malformed is discarded with its reason, and `spread` reports
  whether the personas differed at all — one that agrees exactly added nothing
  over a single opinion.
- `narrate()` writes a result up for a decision report from computed figures
  only, then checks every numeric literal in the reply against those figures
  and lists any the model invented.
- `critique_weights()` applies the same scrutiny to weights from any source —
  elicited from a person, taken from a previous study, or produced by a model
  this package never saw.
- Nothing in `mcdakit.ai` is re-exported from the package namespace, so using
  it requires an explicit `from mcdakit.ai import ...` that remains visible in
  the importing module. An MCDA result is frequently used to justify a
  decision to a third party, and whether a model shaped the inputs belongs in
  that record. Importing `mcdakit` does not load the module.
- The provider is injected as a plain callable, so **no LLM dependency is
  added**: NumPy remains the only runtime requirement, there is no optional
  extra to install, and the module is deterministic under test.

- **Weights derived from the data** (`mcdakit.weighting`). `derive_weights()`
  with `entropy`, `critic`, `std` and `equal`, plus a registry so a scheme
  written elsewhere is reachable by name. AHP asks a person what matters;
  these read it off the decision matrix, for when nobody has a view or the
  analyst wants a starting point that is not an opinion. `entropy` is
  scale-invariant, `std` is not, and `critic` additionally discounts criteria
  that duplicate each other so a doubled dimension is not counted twice.
  Documented as *computed*, not *correct*: a criterion can be uninformative in
  the current shortlist and still be the one that matters.

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

The initial port from the Odoo module. Recorded here for continuity;
never tagged and never published to PyPI.

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
