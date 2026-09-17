# Quickstart

## Describing a decision

A decision is a matrix of options (rows) scored on criteria (columns), plus a
{class}`~mcdakit.Criterion` per column saying what that column means.

```python
from mcdakit import Criterion

criteria = [
    Criterion("Price",     weight=0.40, direction="cost",    bounds=(2.00, 4.00)),
    Criterion("Quality",   weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost",    bounds=(5, 35)),
    Criterion("Support",   weight=0.15, direction="benefit", bounds=(0, 10)),
]
```

`direction="cost"` says smaller is better. **Enter figures as measured** —
real euros, real days — and let the library handle direction. Pre-inverting
them yourself will double-invert and silently produce the wrong answer.

`bounds` is the range the criterion *could* take, independent of the options
on the table: a budget ceiling, a rating scale, an acceptable delivery window.
Optional everywhere except SPOTIS, whose guarantee depends on it.

`weight` is relative. `40/30/20/10` and `0.4/0.3/0.2/0.1` are the same input.

## Ranking

```python
from mcdakit import rank

matrix = [
    [2.75, 7.0, 14, 8.0],
    [2.90, 8.5, 16, 8.0],
    [3.40, 9.0, 11, 7.0],
    [2.20, 3.0, 32, 2.0],
]
labels = ["Option 1", "Option 2", "Option 3", "Option 4"]

result = rank(matrix, criteria, method="spotis", labels=labels)

result.winner        # 'Option 1'
result.order         # best first
result.ranking       # [(label, score), ...]
result.scores        # in the original row order
result.rank_of("Option 2")
print(result)        # a readable table
```

Scores always read **higher is better**, including for VIKOR and SPOTIS whose
native measures are smallest-is-best.

## Asking how stable it is

```python
from mcdakit import sensitivity

report = sensitivity(result)
report["level"]      # 'fragile' | 'moderate' | 'robust' | 'immovable'
report["weakest"]    # the criterion with least room to move
report["overall"]    # smallest tolerance across criteria

for row in report["rows"]:
    print(row["name"], row["tolerance"], row["direction"], row["flips_to"])
```

Each row says how far that weight must move — and **which way** — before a
different option wins. "Quality can move by 7.5%" is unusable without knowing
that means down.

Only one weight moves at a time, so this is an upper bound on stability: a
simultaneous shift could unseat the winner sooner.

## Comparing methods

```python
from mcdakit import compare_methods, agreement

results = compare_methods(matrix, criteria, labels=labels)
agreement(results)
# {'winners': {...}, 'consensus': ..., 'tied': (...), 'votes': 7, 'of': 17,
#  'unanimous': False}
```

When methods disagree, that disagreement *is* the finding: the options are
close enough that the modelling choice decides the outcome.

## Weights from pairwise comparison

```python
from mcdakit import ahp_weights

out = ahp_weights(
    {(0, 1): 3, (0, 2): 9, (1, 2): 3},
    names=["Price", "Quality", "Delivery"],
)
out["weights"]             # [0.6923, 0.2308, 0.0769]
out["consistency_ratio"]   # 0.0
out["consistent"]          # True
```

The consistency ratio is reported, never enforced.
