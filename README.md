# mcdakit

Multi-criteria decision analysis for Python. Eight ranking methods, one
dependency (numpy), and two features most MCDA libraries leave out:

**Rankings that hold.** Most libraries answer *what is the ranking?*
`mcdakit` also answers *would the ranking survive someone disagreeing with my
inputs?*

```python
from mcdakit import Criterion, rank, sensitivity

criteria = [
    Criterion("Price",     weight=0.40, direction="cost",    bounds=(2.00, 4.00)),
    Criterion("Quality",   weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost",    bounds=(5, 35)),
    Criterion("Support",   weight=0.15, direction="benefit", bounds=(0, 10)),
]
matrix = [
    [2.75, 7.0, 14, 8.0],   # Kestrel Supply
    [2.90, 8.5, 16, 8.0],   # Nordpack
    [3.40, 9.0, 11, 7.0],   # Meridian
    [2.20, 3.0, 32, 2.0],   # Bytharm  — cheapest, but nobody would buy it
]
labels = ["Kestrel Supply", "Nordpack", "Meridian", "Bytharm"]

result = rank(matrix, criteria, method="spotis", labels=labels)
result.winner            # 'Kestrel Supply'

report = sensitivity(result)
report["level"]          # 'fragile'
report["weakest"]        # 'Quality' — the weight with the least room to move
```

`sensitivity` is the part worth reading twice. It reports, for every
criterion, how far its weight can move before a *different* option wins:

```
Price       0.40   flips at  7.8% (down)  ->  Nordpack
Quality     0.25   flips at  3.9% (up)    ->  Nordpack
Lead time   0.20   flips at  8.8% (down)  ->  Nordpack
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
| **`spotis`** | **0.0%** |
| `saw` | 1.8% |
| `electre` | 18.5% |
| `topsis` | 19.2% |
| `weighted_scoring` | 27.5% |
| `promethee` | 29.8% |
| `vikor` | 33.8% |

† `simple_scoring` is stable only because it ignores weights *and*
normalisation. That makes it unusable on mixed units, not trustworthy — the
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
WITH the rejected option:     Kestrel 0.7048  >  Nordpack 0.6982
WITHOUT it:                   Nordpack 0.6452 >  Kestrel 0.6300
```

Nordpack overtook Kestrel because dropping Bytharm — the cheapest option, and
the one nobody would buy — shrank the price span from 1.20 to 0.65, lifting
Nordpack's normalised price score from 0.417 to 0.769. **A rejected option was
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
warnings.simplefilter("error", BoundsWarning)   # in a pipeline that relies on it
```

Check any method for reversal on your own data:

```python
from mcdakit import reversal_check
reversal_check(matrix, criteria, method="topsis", labels=labels)
# {'reversed': True, 'cases': [{'removed': 'Bytharm', ...}]}
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
# {'winners': {'Kestrel Supply': 5, 'Nordpack': 2, 'Meridian': 1},
#  'consensus': 'Kestrel Supply', 'votes': 5, 'of': 8, 'unanimous': False}
```

A 5-2-1 split says the options are close enough that the modelling choice
decides the outcome. Collapsing that into one number would hide the most
useful thing on the table.

## Weights from pairwise comparisons

People cannot reliably say a criterion is worth 0.35 rather than 0.40, but
they can say price matters somewhat more than quality. AHP turns those
judgements into weights and measures whether they contradict each other:

```python
from mcdakit import ahp_weights

out = ahp_weights(
    {(0, 1): 3, (0, 2): 9, (1, 2): 3},          # price 3x quality, 9x delivery...
    names=["Price", "Quality", "Delivery"],
)
out["weights"]             # [0.6923, 0.2308, 0.0769]
out["consistency_ratio"]   # 0.0 — perfectly consistent
out["consistent"]          # True (Saaty's 0.10 rule of thumb)
```

The consistency ratio is reported, never enforced. An inconsistent matrix is a
signal to revisit the judgements, not an error.

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

## What this is not

Not a replacement for [`pymcdm`](https://pypi.org/project/pymcdm/) or
[`scikit-criteria`](https://pypi.org/project/scikit-criteria/) if you want
breadth — they carry many more methods and normalisation schemes. `mcdakit`
carries eight methods and spends its surface area on stability instead.

## Licence

Apache-2.0.
