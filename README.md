# mcdakit

Multi-criteria decision analysis for Python. Seventeen ranking methods, one
dependency (numpy), and two features most MCDA libraries leave out:

**Rankings that hold.** Most libraries answer *what is the ranking?*
`mcdakit` also answers *would the ranking survive someone disagreeing with my
inputs?*

```python
from mcdakit import Criterion, rank, sensitivity

criteria = [
    Criterion("Price", weight=0.40, direction="cost", bounds=(2.00, 4.00)),
    Criterion("Quality", weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost", bounds=(5, 35)),
    Criterion("Support", weight=0.15, direction="benefit", bounds=(0, 10)),
]
matrix = [
    [2.75, 7.0, 14, 8.0],  # Option 1
    [2.90, 8.5, 16, 8.0],  # Option 2
    [3.40, 9.0, 11, 7.0],  # Option 3
    [2.20, 3.0, 32, 2.0],  # Option 4  — cheapest, but nobody would buy it
]
labels = ["Option 1", "Option 2", "Option 3", "Option 4"]

result = rank(matrix, criteria, method="spotis", labels=labels)
result.winner  # 'Option 1'

report = sensitivity(result)
report["level"]  # 'fragile'
report["weakest"]  # 'Quality' — the weight with the least room to move
```

`sensitivity` is the part worth reading twice. It reports, for every
criterion, how far its weight can move before a *different* option wins:

```
Price       0.40   flips at  7.8% (down)  ->  Option 2
Quality     0.25   flips at  3.9% (up)    ->  Option 2
Lead time   0.20   flips at  8.8% (down)  ->  Option 2
Support     0.15   never flips
```

A winner that survives a 30% shift in every weight is a different finding
from one that flips at 4%, even though both print as "first".

## Install

```bash
pip install mcdakit
```

Python 3.9+. numpy is the only runtime dependency — no pandas, no scipy.

## Rank reversal, and why SPOTIS is here

Remove an option nobody was going to choose, re-rank the ones that remain, and
their order can change. This is not a bug in any implementation; it is a
property of methods that measure options against each other. Measured over 400
random 5x4 problems, removing the last-placed option each time:

| Method | Rank reversal rate |
|---|---|
| `simple_scoring` | 0.0% † |
| `wpm` | 0.0% † |
| **`spotis`** | **0.0%** |
| `saw`, `waspas`, `marcos` | 1.8% |
| `aras` | 12.0% |
| `copras` | 15.0% |
| `codas` | 17.0% |
| `electre` | 18.5% |
| `edas` | 18.8% |
| `topsis` | 19.2% |
| `weighted_scoring`, `mabac` | 27.5% |
| `promethee` | 29.8% |
| `vikor` | 33.8% |
| `cocoso` | 49.0% |

† `simple_scoring` is stable only because it ignores weights *and*
normalisation; `wpm` divides by the column maximum, which the last-placed
option rarely sets in this all-benefit setup. Neither is reversal-free by
construction. That makes it unusable on mixed units, not trustworthy — the
two ways to avoid rank reversal are to use no information, or to use SPOTIS.

Reproduce it yourself, on any problem size:

```bash
python benchmarks/reversal.py --trials 400 --options 5 --criteria 4
```

Rates depend on the setup and the numbers above are specific to it: values are
uniform on a bounded scale with all-benefit criteria, so the last-placed option
is a column *minimum* far more often than a column *maximum*. That is why `saw`
— which normalises by the column maximum alone — looks so stable here, while
methods reading the observed *range* (`weighted_scoring`, `vikor`, `topsis`)
are hit hardest. The figures are stable across seeds; only SPOTIS's 0.0% is
guaranteed rather than measured.

In the supplier data above, dropping the rejected option flips the top two:

```
WITH the rejected option:     Option 1  0.7048  >  Option 2  0.6982
WITHOUT it:                   Option 2  0.6452  >  Option 1  0.6300
```

Option 2 overtook Option 1 because dropping Option 4 — the cheapest option, and
the one nobody would buy — shrank the price span from 1.20 to 0.65, lifting
Option 2's normalised price score from 0.417 to 0.769. **A rejected option was
defining the scale.**

SPOTIS (Dezert et al., FUSION 2020) measures every option against a fixed
ideal point built from bounds you set in advance, never against the other
options. Nothing an option does can move another option's score, so no
addition or removal can reorder the rest.

That guarantee depends entirely on those bounds. If a criterion has none,
`mcdakit` falls back to observed min/max, **warns**, and sets
`result.reversal_free = False` — it will not quietly hand back a number that
looks authoritative:

```python
import warnings
from mcdakit import BoundsWarning

warnings.simplefilter("error", BoundsWarning)  # in a pipeline that relies on it
```

Check any method for reversal on your own data:

```python
from mcdakit import reversal_check

reversal_check(matrix, criteria, method="topsis", labels=labels)
# {'reversed': True, 'cases': [{'removed': 'Option 4', ...}]}
```

## Methods

| Name | Method | Reference |
|---|---|---|
| `simple_scoring` | Unweighted sum | — |
| `weighted_scoring` | Weighted sum, min-max normalised | — |
| `saw` | Simple Additive Weighting | Churchman & Ackoff, 1954 |
| `topsis` | Distance to ideal / anti-ideal | Hwang & Yoon, 1981 |
| `vikor` | Compromise ranking | Opricovic & Tzeng, 2004 |
| `electre` | ELECTRE I outranking | Roy, 1968 |
| `promethee` | PROMETHEE II net flows, six preference shapes | Brans & Vincke, 1985 |
| `copras` | Complex proportional assessment | Zavadskas et al., 1994 |
| `waspas` | Weighted sum and product blended | Zavadskas et al., 2012 |
| `wpm` | Weighted product; a weak criterion cannot be paid for | Bridgman, 1922 |
| `aras` | Additive ratio against a constructed optimum | Zavadskas & Turskis, 2010 |
| `cocoso` | Combined compromise of three strategies | Yazdani et al., 2019 |
| `codas` | Distance from the worst point | Keshavarz Ghorabaee et al., 2016 |
| `edas` | Distance from the average solution | Keshavarz Ghorabaee et al., 2015 |
| `mabac` | Border approximation area comparison | Pamucar & Cirovic, 2015 |
| `marcos` | Utility against ideal and anti-ideal | Stevic et al., 2020 |
| `spotis` | Rank-reversal-free | Dezert et al., 2020 |

Every method returns scores where **higher is better**, including VIKOR and
SPOTIS whose native measures are smallest-is-best and are negated for you.

`topsis` uses vector normalisation, Hwang and Yoon's original choice. Worth
knowing if you compare numbers with another library: `pymcdm` defaults to
min-max here, and the two give different scores on the same data. Told to use
vector normalisation it reproduces ours to machine precision — the test suite
pins both facts.

### When methods disagree, that *is* the finding

```python
from mcdakit import compare_methods, agreement

results = compare_methods(matrix, criteria, labels=labels)
agreement(results)
# {'winners': {'Option 3': 7, 'Option 1': 7, 'Option 2': 3},
#  'consensus': None, 'tied': ('Option 3', 'Option 1'),
#  'votes': 7, 'of': 17, 'unanimous': False}
```

Seven methods to seven is a tie, so `consensus` is `None` and `tied` names
both leaders. Reporting one of them as *the* answer would conceal the split.
The options are close enough that the modelling choice decides the outcome,
and that is the most useful thing on the table.

## Why did this option win?

A ranking is a conclusion, and a conclusion nobody can interrogate is hard to
act on or defend:

```python
from mcdakit.explain import explain, compare

print(compare(result, "Option 1", "Option 2"))
# Option 1 beats Option 2 by 0.0065 under weighted_scoring
#   attribution: exact
#   Quality       -0.0625  favours Option 2
#   Price         +0.0500  favours Option 1
#   Lead time     +0.0190  favours Option 1
#   Support       +0.0000  no difference
#   Lead time is decisive: without it the result reverses
```

Where the method is a weighted sum — `weighted_scoring`, `saw`,
`simple_scoring`, `spotis` — the contributions **sum exactly to the score**,
and the tests assert it. Where it is not — TOPSIS is a ratio of distances,
VIKOR takes a maximum, ELECTRE and PROMETHEE compare pairwise — no exact
decomposition exists, so each criterion is instead attributed by removing it
and re-scoring. Every report says which of the two it used:

```python
explain(result).basis     # 'exact' or 'leave_one_out'
```

Presenting an approximation as a decomposition would be the overstatement the
rest of this package exists to avoid.

## Several stakeholders

Two people scoring the same options rarely agree, and the useful question is
not only *what did the group decide* but *how much did they disagree, and does
the answer survive it*:

```python
from mcdakit import Criterion
from mcdakit.group import Participant, group_rank

criteria = [Criterion("Price", 0.5, "cost"), Criterion("Quality", 0.5)]
out = group_rank(
    [
        Participant("Alice", [[2.75, 8.0], [2.90, 6.0]]),
        Participant("Bob",   [[2.75, 5.0], [2.90, 9.0]]),
        Participant("Chair", [[2.75, 7.0], [2.90, 7.0]], weight=2.0),
    ],
    criteria,
    labels=["Option 1", "Option 2"],
)

out["result"].winner        # ranking the averaged scores
out["borda"]["order"][0]    # combining each person's ranking
out["agree"]                # did the two agree?
out["unanimous"]            # did every participant, alone, pick the winner?
out["disagreement"]["most_contested_option"]
```

Two aggregations are reported because they answer different questions.
**Averaging the scores** treats the group as one better-informed judge — right
when the differences are noise. **Combining the rankings** (Borda) treats each
person as a voter — right when the differences are real preferences, and it
stops one outlying score dragging the mean.

When they disagree, `agree` is `False`, and *that* is the finding: the answer
depends on which model of the group you accept.

`disagreement()` separates a mean of 5.5 everyone agreed on from one that
averages 9 and 2 — the same number, completely different findings.

Participants must score the same options and criteria; a mismatch is refused
rather than averaged over uneven evidence.

## Language-model assistance, verified

An LLM is useful for turning a described problem into a structured one. It is
not an authority: a model returns weights with the same confidence whether
they are considered or invented. So the model proposes and the package checks.

**It is opt-in.** Nothing in `mcdakit.ai` is re-exported from the package, so
using it takes an explicit import that stays visible in your source. An MCDA
result is often used to justify a decision to somebody else, and whether a
model shaped the inputs belongs in the record. There is no extra install: the
module imports only the standard library and NumPy.

```python
from mcdakit.ai import propose_criteria, propose_weights

# `ask` is any callable mapping a prompt to a reply — no vendor is assumed
# and no dependency is added.
proposal = propose_criteria("selecting a component supplier", ask=my_model,
                            samples=2)
print(proposal)
# 3 criteria proposed:
#   Price (cost, weight 0.50)
#   ! discarded 'Colour': direction must be one of benefit, cost
#   agreement across replies: 100%
#   not applied — construct a Decision to accept

weights = propose_weights("selecting a supplier", proposal.criteria,
                          ask=my_model, matrix=matrix)
weights.stability["level"]   # 'fragile' — flips at a 3.2% weight change
weights.disagrees_with       # ('entropy', 'std') — objective schemes differ
weights.accepted             # False. Applying it is your decision.
```

Every suggestion is parsed into validated types, so a malformed answer is
rejected with a reason. Proposed weights are put through `sensitivity()`, so
you learn whether the recommendation survives the model being somewhat wrong.
Asking twice measures self-consistency, which one reply cannot reveal.

`critique_weights()` applies the same checks to weights from any source.

Three further entry points, each with its own check:

```python
from mcdakit.ai import propose_comparisons, simulate_panel, narrate
```

**`propose_comparisons()`** asks the model to compare criteria *pairwise* —
*is price more important than quality, and by how much?* — rather than for a
weight vector. The difference is that pairwise judgements can contradict each
other, and Saaty's consistency ratio detects it:

```
3 pairwise judgements, consistency ratio 2.759 (INCONSISTENT)
  ! the judgements contradict each other; ask again or revise them by hand
```

**`simulate_panel()`** scores the problem once per expert persona and returns
ordinary `Participant` objects, so a simulated panel goes through the same
`group_rank()` and `disagreement()` machinery as a real one. Human panels are
costly and cannot be re-run; a simulated panel is reproducible but is a
model's impression of an expert, so `spread` reports whether the personas
actually differed — a panel that agrees exactly added nothing over asking
once.

**`narrate()`** turns a result into a paragraph for a report. The model is
given only computed figures, so its one available failure is misstating a
number — and every numeric literal in the reply is checked against what it was
given:

```python
narrate(result, ask=model)["unsupported_numbers"]   # ('97.3',)
```

## Weights from the data

AHP below asks a person what matters. When nobody has a view — or you want a
starting point that is not an opinion — derive the weights from the decision
matrix instead:

```python
from mcdakit import derive_weights

derive_weights(matrix, "entropy", directions)   # also: critic, std, equal
```

They share one premise: **a criterion on which all the options score alike
cannot separate them, so it carries no information.** `entropy` measures that
in proportions and is scale-invariant; `std` measures raw spread and is not;
`critic` additionally discounts criteria that duplicate each other, so two
columns measuring nearly the same thing do not count that dimension twice.
`equal` is the baseline — if an objective scheme cannot beat it on your
problem, it is adding complexity rather than information.

Objective here means *computed*, not *correct*. A criterion can be
uninformative in the current shortlist and still be the one that matters: a
budget everyone happens to meet is not thereby unimportant.

Register your own with `@register_weighting("name")`, or pass any callable.

## Weights from pairwise comparisons

People cannot reliably say a criterion is worth 0.35 rather than 0.40, but
they can say price matters somewhat more than quality. AHP turns those
judgements into weights and measures whether they contradict each other:

```python
from mcdakit import ahp_weights

out = ahp_weights(
    {(0, 1): 3, (0, 2): 9, (1, 2): 3},  # price 3x quality, 9x delivery...
    names=["Price", "Quality", "Delivery"],
)
out["weights"]  # [0.6923, 0.2308, 0.0769]
out["consistency_ratio"]  # 0.0 — perfectly consistent
out["consistent"]  # True (Saaty's 0.10 rule of thumb)
```

The consistency ratio is reported, never enforced. An inconsistent matrix is a
signal to revisit the judgements, not an error.

### Ranking without a decision matrix

Sometimes there are no numbers to put in a matrix. Nobody can score *style* out
of ten, but anyone can say this car looks better than that one. `ahp_rank`
compares alternatives pairwise under each criterion as well, which is AHP as
Saaty defined it — the decision matrix disappears entirely:

```python
from mcdakit import ahp_rank

result = ahp_rank(
    {(0, 1): 1 / 2, (0, 2): 3, (1, 2): 4},   # how the criteria compare
    [style, reliability, fuel],              # how the cars compare, per criterion
    names=["Style", "Reliability", "Fuel economy"],
    labels=["Civic", "Saturn", "Escort"],
)
print(result)
```

```
AHP ranking:
  1. Saturn        0.4440
  2. Civic         0.4263
  3. Escort        0.1297
  consistency ratios:
    criteria        0.0158
    Style           0.1874  EXCEEDS 0.10
    Reliability     0.0032
    Fuel economy    0.0079
  ! Style contradicts itself; revisit those judgements before relying on the ranking
```

Every matrix gets its own consistency ratio, reported separately. An average
would hide the one that matters: here the style judgements contradict
themselves, and Saturn leads Civic by less than that inconsistency is worth.
The ranking is still returned — whether the contradiction is disqualifying is a
judgement for whoever made the comparisons.

## PROMETHEE preference functions

By default PROMETHEE's *usual* criterion treats any difference however small as
total preference — so 4200 beats 4201 exactly as decisively as it beats 9000.
Fine for a rating out of ten, wrong for money:

```python
Criterion("Price", 0.5, "cost", preference_shape="linear", q=100, p=500)
# indifferent below 100 EUR, decisive above 500
```

All six of Brans and Vincke's shapes are available: `usual`, `ushape`,
`vshape`, `level`, `linear`, `gaussian`.

## Adding your own method

A method written outside this package is a first-class citizen: reachable by
name, included in `compare_methods`, analysable by `sensitivity`. Nothing in
the library branches on a method's name — what a method needs, it declares.

```python
import numpy as np
from mcdakit import Method, ScoringContext, register, rank

class Median(Method):
    name = "median"
    summary = "Median score across criteria."

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return np.median(ctx.data, axis=1)

register(Median())

rank(matrix, criteria, method="median", labels=labels).winner
```

Methods that resolve criterion direction themselves — as SPOTIS does — ask for
the un-oriented matrix:

```python
from mcdakit import Wants

class MyMethod(Method):
    name = "mine"
    wants = Wants.RAW      # give me the figures as measured
```

To ship one as a package, declare an entry point and it is discovered on
install, with no import needed by the user:

```toml
[project.entry-points."mcdakit.methods"]
my_method = "my_package.methods:MyMethod"
```

Before publishing a method, check it:

```bash
python -m mcdakit.testing my_package.methods:MyMethod
```

That runs thirteen properties every sound method holds, catching the failures
that are otherwise silent — a smallest-is-best measure returned without
negating (every ranking upside down), a NaN on a criterion every option scores
alike, criterion direction handled twice. Every built-in passes it and CI
enforces that.

### Normalisation is swappable too

Putting a price spanning 900 and a rating spanning 3 onto one scale is a
modelling decision, not preprocessing — it can change the winner *and* how
stable the ranking is:

```python
rank(matrix, criteria, method="topsis", normalization="minmax")
```

Built in: `vector`, `minmax`, `max`, `sum`. Each method defaults to the scheme
it was defined with, so you get the textbook method unless you ask otherwise.
Register your own with `@register_normalization("name")`, or pass any callable.

The full contract — reporting caveats, refusing problems, accepting options,
what is *not* yet extensible — is in [`docs/extending.md`](docs/extending.md).

## Examples

Two runnable scripts in [`examples/`](examples):

```bash
python examples/supplier_selection.py          # the full workflow
python examples/same_problem_every_library.py  # the same problem, four libraries
```

The second is built around a deliberate anticlimax: all four libraries agree on
the winner. The differences appear in what comes next — whether the answer
holds, why it won, and what happens on input that cannot honestly be ranked.

## Compared with other libraries

`benchmarks/comparison.py` runs `mcdakit` against `pymcdm`, `pyDecision` and
`scikit-criteria` on the same problem and reports four things: whether shared
methods agree, what each library can do (established by calling it), what each
does with input it cannot honestly rank, and how they scale.

Summarising the run:

- **Shared methods agree.** SPOTIS and VIKOR match `pymcdm` exactly; TOPSIS
  matches once both use vector normalisation, and differs from its min-max
  default — a modelling choice, not a defect, and the tests pin both facts.
- **Degenerate input** — all weights zero, or a missing score — is refused
  here and returns `NaN` there, with a warning in one case and silently in
  the other.
- **Rankings agree at every problem size** tested up to 500×15.

Install the comparison libraries first; they are never a runtime dependency:

```bash
pip install pymcdm pyDecision scikit-criteria
python benchmarks/comparison.py
```

## What this is not

Not a replacement for [`pymcdm`](https://pypi.org/project/pymcdm/) or
[`scikit-criteria`](https://pypi.org/project/scikit-criteria/) if you want
breadth — they carry more methods and normalisation schemes. `mcdakit`
carries seventeen and spends its surface area on stability instead.

## Documentation

- [Quickstart](docs/quickstart.md) — describing a decision, ranking, reading a result
- [Stability](docs/stability.md) — rank reversal, SPOTIS, sensitivity
- [Extending](docs/extending.md) — writing and shipping your own method
- [Contributing](CONTRIBUTING.md) — house rules and the checks CI runs
- [Design decisions](docs/adr/) — why the architecture is the way it is

Build them locally with `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html`.

## Licence

Apache-2.0.
