# mcdakit

Multi-criteria decision analysis whose rankings hold.

Most MCDA libraries answer *what is the ranking?* `mcdakit` also answers
**would the ranking survive someone disagreeing with my inputs?** Seventeen
ranking methods, one of them reversal-free by construction, and numpy as the
only runtime dependency.

```{toctree}
:maxdepth: 2
:caption: Guides

quickstart
stability
ai
extending
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
adr/index
changelog
```

## Install

```bash
pip install mcdakit
```

## At a glance

```python
from mcdakit import Criterion, rank, sensitivity

criteria = [
    Criterion("Price",     weight=0.40, direction="cost",    bounds=(2.00, 4.00)),
    Criterion("Quality",   weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost",    bounds=(5, 35)),
    Criterion("Support",   weight=0.15, direction="benefit", bounds=(0, 10)),
]
matrix = [
    [2.75, 7.0, 14, 8.0],
    [2.90, 8.5, 16, 8.0],
    [3.40, 9.0, 11, 7.0],
    [2.20, 3.0, 32, 2.0],
]
labels = ["Option 1", "Option 2", "Option 3", "Option 4"]

result = rank(matrix, criteria, method="spotis", labels=labels)
result.winner                 # 'Option 1'
sensitivity(result)["level"]  # 'fragile'
```

## Where to start

| If you want to | Read |
|---|---|
| Rank some alternatives | [Quickstart](quickstart.md) |
| Know whether the answer is trustworthy | [Stability](stability.md) |
| Have a model help set the problem up | [AI assistance](ai.md) |
| Add your own method | [Extending](extending.md) |
| Look up a function | [API reference](api.md) |
| Know why it is built this way | [Design decisions](adr/index.md) |

## What is here

**Seventeen ranking methods** — TOPSIS, VIKOR, ELECTRE I, PROMETHEE II with
all six preference shapes, SPOTIS, COPRAS, WASPAS, WPM, ARAS, COCOSO, CODAS,
EDAS, MABAC, MARCOS, SAW and two scoring baselines — plus full AHP, which
ranks from pairwise comparisons with no decision matrix at all.

**Answers about the answer.** How far each weight can move before the winner
changes, solved for rather than sampled. Whether removing a rejected
alternative reorders the rest. Which criterion decided a close margin, and
whether reversing it alone would reverse the outcome.

**Several stakeholders**, aggregated two ways, with the disagreement between
them reported rather than averaged away.

**Optional language-model assistance**, where the model proposes and the
package checks: suggestions are parsed into validated types, proposed weights
are put through the sensitivity analysis, pairwise judgements are tested for
self-contradiction, and every number in a generated write-up is compared
against the computed figures.

**A documented extension contract.** A method written in your own package is
a first-class citizen — reachable by name, included in cross-method
comparison, analysable by the sensitivity machinery — and a conformance suite
you run before publishing it.

Runnable scripts are in
[`examples/`](https://github.com/aitmlouk/mcdakit/tree/main/examples),
including one for the AI workflow that needs no API key.

## Indices

- {ref}`genindex`
- {ref}`modindex`
