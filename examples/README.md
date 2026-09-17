# Examples

Three runnable scripts. None hard-codes a number: every figure printed is
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

## `ai_assisted.py`

Language-model assistance end to end, and what the package does with it.

```bash
python examples/ai_assisted.py                       # no key needed
ANTHROPIC_API_KEY=... python examples/ai_assisted.py # or a live model
MCDAKIT_OLLAMA_MODEL=llama3 python examples/ai_assisted.py   # or a local one
```

With no API key it replays recorded replies from a real model, so it runs
offline and deterministically. With a key, or a local Ollama model, it asks
for real.

Six sections, of which the interesting ones are the failures:

1. **Naming criteria** — asked twice, so a model that improvises is
   distinguishable from one that is consistent; malformed suggestions are
   rejected with a reason
2. **The measurements are yours** — a proposal that does not match what was
   measured is refused, never guessed at
3. **Weights, checked against the data** — a sensitivity analysis run *with*
   the model's weights, plus the objective schemes that disagree
4. **A model contradicting itself** — pairwise judgements caught by Saaty's
   consistency ratio, which a weight vector could never reveal
5. **A simulated panel** — with the spread reported, so a panel that agreed
   exactly is visible as having added nothing
6. **The write-up, verified** — every numeric literal compared against the
   computed figures

Section 6 is worth running twice. On the recorded path the model asserts a
confident, fluent, entirely wrong result — the wrong winner and four invented
figures — and `faithful` comes back `False` with each one named.

## Related scripts

Under [`../benchmarks/`](../benchmarks):

- `crosscheck.py` — every method against every library implementing it,
  treating an unexplained difference as a defect until attributed
- `comparison.py` — capability, degenerate input, API complexity, memory and
  scaling
- `reversal.py` — the rank-reversal rate behind the README's table
