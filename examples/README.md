# Examples

Two runnable scripts. Neither hard-codes a number: every figure printed is
computed when you run it, so they cannot drift from the package.

## `supplier_selection.py`

The complete workflow on a small procurement problem — four suppliers, two
cost criteria and two benefit criteria.

```bash
python examples/supplier_selection.py
```

It works through the five questions a decision report has to answer:

1. **What is the ranking?**
2. **Would it survive someone disagreeing?** — the weight tolerance, its
   direction, and which supplier takes over
3. **Why did it win?** — the margin decomposed by criterion
4. **Does it depend on a rejected alternative?** — the same removal under
   three methods, two of which reverse
5. **Do the methods agree?** — where they split, that split is the finding

The last section prints the paragraph a report could actually contain.

## `same_problem_every_library.py`

The same problem put to `mcdakit`, `pymcdm`, `pyrepo-mcda` and `pyDecision`
side by side.

```bash
pip install pymcdm pyDecision pyrepo-mcda     # development extras only
python examples/same_problem_every_library.py
```

It is built around a deliberate anticlimax. All four libraries rank the
problem and **all four agree on the winner** — nothing there distinguishes
them. The differences appear afterwards:

| | |
|---|---|
| Says whether the answer holds | one of four |
| Says why it won | one of four |
| Refuses to answer on unrankable input | one of four |

Missing libraries are reported rather than skipped, so a partial installation
produces a partial comparison rather than a misleading one.

## Related scripts

Under [`../benchmarks/`](../benchmarks):

- `crosscheck.py` — every method against every library implementing it,
  treating an unexplained difference as a defect until attributed
- `comparison.py` — capability, degenerate input, API complexity, memory and
  scaling
- `reversal.py` — the rank-reversal rate behind the README's table
