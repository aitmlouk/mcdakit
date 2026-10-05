# Limitations

`mcdakit` is 0.x software; the API may change between releases. These are the
things worth knowing before you rely on it, and the cases where another
library is the better choice.

## Scope

- **No fuzzy extensions.** Fuzzy MCDA is not implemented and is not planned.
  [`pyfdm`](https://pypi.org/project/pyfdm/) specialises in it and is the
  appropriate choice.
- **Seventeen methods, not a catalogue.** `pymcdm` has 28 and `pyDecision`
  76. If you need COMET, MOORA, MAIRCA or another method not listed in the
  [API reference](api.md), use one of those; both operate on plain NumPy
  arrays, so the two can be combined in one analysis.
- **Four objective weighting schemes** (entropy, CRITIC, standard deviation,
  equal) against `pymcdm`'s eleven.

## The sensitivity analysis

- **One weight at a time.** The analysis moves a single criterion's weight
  and rescales the rest to preserve a unit sum. It answers how far one
  disagreement can go, not how the result behaves under simultaneous
  uncertainty in several inputs. Monte-Carlo robustness over uncertain scores
  would complement it and is not implemented.
- **The classification bands are conventions.** Fragile below 10%, robust at
  or above 25%. These are reporting conventions, not derived quantities. The
  threshold itself is the measured result; the bands exist so that it can be
  stated to a reader without MCDA training.

## Rank reversal

- **SPOTIS is reversal-free only when bounds are supplied.** Without them the
  package falls back to observed minima and maxima, warns, and sets
  `reversal_free = False`. The guarantee is a property of the bounds, not of
  the method name.
- **Measured rates are setup-dependent.** The benchmark's figures come from
  one generator: values uniform on a bounded scale with all-benefit criteria.
  Methods that normalise by the column maximum alone look markedly more
  stable under it than they would under a different setup. Only the SPOTIS
  result holds by construction. See [Stability](stability.md).

## Language-model assistance

- **Optional, unverified by default.** `mcdakit.ai` checks what a model
  proposes, but a model that passes those checks is not thereby correct. The
  checks catch malformed output, self-contradiction and invented figures;
  they cannot catch a plausible but poor judgement.
- **Adapters are tested against vendor-shaped replies, not live APIs.** The
  OpenAI, Anthropic and Google adapters are exercised against a local server
  answering in each vendor's documented response format. The Ollama path has
  been run end to end against a live local model; the three hosted paths have
  not been verified against a real round trip.

## Scale

Ranking is vectorised and sub-millisecond for problems of the size MCDA is
applied to. The sensitivity analysis bisects independently per criterion, so
its cost tracks the number of criteria rather than the number of
alternatives: around 50 ms at 200 alternatives and 10 criteria, 142 ms at 20
criteria. Problems with hundreds of criteria are outside what has been
measured.
