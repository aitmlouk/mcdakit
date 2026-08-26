# Rank reversal and stability

## The problem

Remove an option nobody was going to choose, re-rank the ones that remain, and
their order can change. This is not a bug in any implementation; it is a
property of methods that measure options against each other.

Measured over 400 random 5×4 problems, removing the last-placed option:

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
normalisation — unusable on mixed units, not trustworthy.

Reproduce with `python benchmarks/reversal.py`. Rates are setup-dependent; see
the README for why `saw` looks so stable in this particular experiment.

## Why it happens

```
WITH the rejected option:     Kestrel 0.7048  >  Nordpack 0.6982
WITHOUT it:                   Nordpack 0.6452 >  Kestrel 0.6300
```

Dropping Bytharm — the cheapest option, and the one nobody would buy — shrank
the price span from 1.20 to 0.65, lifting Nordpack's normalised price score
from 0.417 to 0.769. **A rejected option was defining the scale.**

## SPOTIS

SPOTIS measures every option against a fixed ideal point built from `bounds`
you set in advance, never against the other options:

```{math}
d_{ij} = \frac{|S_{ij} - S^*_j|}{|S_j^{max} - S_j^{min}|}
\qquad
d(A_i, S^*) = \sum_j w_j \, d_{ij}
```

Because `d_ij` depends only on option *i*, nothing another option does can
move it. No addition or removal can reorder the rest.

## The guarantee needs bounds

Without bounds, SPOTIS falls back to observed min/max — which move when the
option set moves, defeating the point. `mcdakit` warns and reports it:

```python
result = rank(matrix, criteria, method="spotis")
result.reversal_free   # False when any criterion lacked bounds
result.warnings        # says which
```

In a pipeline that depends on the guarantee, insist:

```python
import warnings
from mcdakit import BoundsWarning
warnings.simplefilter("error", BoundsWarning)
```

## Checking any method

```python
from mcdakit import reversal_check

reversal_check(matrix, criteria, method="topsis", labels=labels)
# {'reversed': True, 'cases': [{'removed': ..., 'before': [...], 'after': [...]}]}
```
