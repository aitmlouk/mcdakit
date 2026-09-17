# mcdakit

Multi-criteria decision analysis whose rankings hold.

Most MCDA libraries answer *what is the ranking?* `mcdakit` also answers
*would the ranking survive someone disagreeing with my inputs?* — through
rank-reversal-free ranking (SPOTIS) and weight sensitivity analysis.

numpy is the only runtime dependency.

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

## Indices

- {ref}`genindex`
- {ref}`modindex`
