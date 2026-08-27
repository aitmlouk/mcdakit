# Extending mcdakit

A method written outside this package is a first-class citizen. Once
registered it is reachable by name from {func}`~mcdakit.rank`, included in
{func}`~mcdakit.compare_methods`, and analysable by
{func}`~mcdakit.sensitivity` — with no change to the library.

Nothing in `mcdakit` branches on a method's name. What a method needs, it
declares.

## A method in ten lines

```python
import numpy as np
from mcdakit import Method, ScoringContext, register

class Median(Method):
    name = "median"
    summary = "Median score across criteria, ignoring weights."

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return np.median(ctx.data, axis=1)

register(Median())
```

That is the whole contract. It now works everywhere a built-in works:

```python
from mcdakit import rank, sensitivity

result = rank(matrix, criteria, method="median", labels=labels)
sensitivity(result)["level"]
```

## The one rule: higher must be better

`score` returns one number per option, and **larger must mean better**. If
your method's natural measure is smallest-is-best — a distance, a regret, a
cost — negate it before returning, and say so in the docstring.

Getting this backwards produces a confident, exactly inverted ranking. It is
the worst failure mode available here, and no test will catch it for you
unless you write one.

## What `score` receives

Everything arrives in a {class}`~mcdakit.ScoringContext`:

| Attribute | What it is |
|---|---|
| `ctx.data` | The decision matrix, oriented or raw per `wants` |
| `ctx.weights` | Weights **as given**, unnormalised |
| `ctx.normalized_weights` | Weights scaled to sum to one |
| `ctx.directions` | `"benefit"` / `"cost"` per criterion |
| `ctx.bounds` | `(lo, hi)` or `None` per criterion |
| `ctx.criteria` | The {class}`~mcdakit.Criterion` objects |
| `ctx.labels` | Option names |
| `ctx.n_options`, `ctx.n_criteria` | Shape |
| `ctx.opts` | Keyword arguments the caller passed to `rank()` |
| `ctx.decision` | The whole problem — a last resort |

Weights arrive unnormalised because some methods want the raw scale. If yours
needs them to sum to one, use `ctx.normalized_weights`; it also handles a
total of zero.

## Oriented or raw data

By default a method receives the matrix with **cost columns already mirrored**
(`max + min - x`), so it can assume larger is better throughout. That is what
`Wants.ORIENTED` means, and it is what almost every classical method wants.

A method that resolves direction itself declares otherwise:

```python
from mcdakit import Method, Wants

class MyBoundedMethod(Method):
    name = "my_bounded"
    wants = Wants.RAW          # give me the figures as measured

    def score(self, ctx):
        # ctx.data is un-oriented; read ctx.directions yourself
        ...
```

SPOTIS is the built-in that does this: it compares each option against a fixed
ideal point derived from `bounds`, so it must see real prices and real
delivery times, not mirrored ones.

:::{danger}
**If you read `ctx.directions`, you must set `Wants.RAW`.**

Under the default `Wants.ORIENTED` the cost columns have *already* been
flipped for you. Consulting directions and flipping again inverts them back,
and the cheapest option comes last — a confident, exactly reversed ranking
with no error.

`rank()` raises if it sees that combination, so the mistake fails loudly
rather than silently. This is the single easiest way to write a method that
looks right and is wrong.
:::

:::{warning}
Orientation depends on which options are present — the midpoint moves when the
option set moves. That dependence is one mechanism behind rank reversal. A
method that wants to be reversal-free almost certainly wants `Wants.RAW` and
fixed bounds.
:::

## Reporting caveats

Return a {class}`~mcdakit.MethodResult` instead of a bare array when there is
something the caller should know:

```python
from mcdakit import Method, MethodResult

class Careful(Method):
    name = "careful"

    def score(self, ctx):
        missing = [c.name for c in ctx.criteria if c.bounds is None]
        if missing:
            return MethodResult(
                scores=self._scores(ctx),
                reversal_free=False,
                warnings=(f"No bounds for {', '.join(missing)}; "
                          f"the guarantee does not hold.",),
            )
        return MethodResult(self._scores(ctx), reversal_free=True)
```

`reversal_free` is the strongest claim this library makes. Set it only if
adding or removing an option genuinely cannot change the order of the others.
Never set it optimistically.

## Refusing a problem

Implement `validate` to reject inputs your method cannot handle. It runs
before `score`:

```python
from mcdakit import McdaError

class NeedsSeveral(Method):
    name = "needs_several"

    def validate(self, decision):
        if decision.n_options < 3:
            raise McdaError(
                f"{self.name} needs at least three options to compare, "
                f"got {decision.n_options}."
            )
```

Use `validate` for what is fatal, and a `MethodResult` warning for what is
merely regrettable.

## Accepting options

Anything extra passed to `rank()` arrives in `ctx.opts`:

```python
class Tunable(Method):
    name = "tunable"

    def score(self, ctx):
        alpha = ctx.opts.get("alpha", 0.5)
        ...

rank(matrix, criteria, method="tunable", alpha=0.8)
```

Read them through `ctx.opts`, not `ctx.options` — `opts` marks them consumed.
If a method never reads its options, `rank()` raises rather than letting a
misspelled argument vanish silently:

```python
rank(matrix, criteria, method="median", alpha=0.8)
# McdaError: Method 'median' takes no keyword arguments, but got alpha.
```

## Shipping a method as a package

Register directly for a method defined in your own program. To distribute one,
declare an entry point instead — it is then discovered automatically on
install, with no import or call needed by the user:

```toml
# your pyproject.toml
[project.entry-points."mcdakit.methods"]
my_method = "my_package.methods:MyMethod"
```

The target may be a `Method` subclass (instantiated with no arguments) or an
already-built instance.

A plugin that fails to import warns rather than raising, so one broken package
cannot stop the library — or every other plugin — from working. Promote it if
you would rather know loudly:

```python
import warnings
from mcdakit import PluginLoadError
warnings.simplefilter("error", PluginLoadError)
```

## Inspecting the registry

```python
from mcdakit import METHODS, available, get_method, has_method

list(METHODS)             # every registered name, a live view
available()               # {name: summary}
has_method("median")      # True
get_method("median")      # the instance
```

`METHODS` reads the registry on every access, so a method registered at any
point is immediately visible.

## Replacing a built-in

Registering a name twice is refused, because silently shadowing a method would
make `method="topsis"` mean different things in different processes. Override
deliberately:

```python
register(MyBetterTopsis(), replace=True)
```

## Checking your method

Writing a ranking method is easy; writing a *correct* one is not, and the
failure modes are quiet. Run the conformance suite before you publish:

```bash
python -m mcdakit.testing my_package.methods:MyMethod
```

```
ok    declares a name
ok    describes itself
ok    cites its source
ok    returns a supported type
ok    scores every option
FAIL  higher is better    the option that is best on every criterion did not
                          score highest. If your measure is smallest-is-best
                          (a distance, a regret), negate it before returning.
...
```

Or from your own test suite:

```python
from mcdakit.testing import check_method

def test_my_method_conforms():
    check_method(MyMethod())
```

It checks thirteen properties every sound method holds, including the three
that are otherwise invisible:

| Property | The failure it catches |
|---|---|
| higher is better | A distance or regret returned without negating — every ranking upside down |
| handles cost criteria | Direction ignored, or handled twice |
| scores stay finite | NaN from a zero-variance criterion, a zero column, or identical options |
| weight scale does not matter | Raw weights used where normalised ones were needed |
| option order does not matter | A result that depends on how the rows were listed |
| reversal-freedom holds if claimed | `reversal_free = True` that does not survive removing a loser |

Every method shipped in `mcdakit` passes this suite, and CI enforces it.

Pass `strict=False` (or `--no-strict`) while prototyping to tolerate a missing
`summary` or `citation`.

**Passing is a floor, not a ceiling.** It says your method is well-behaved, not
that its arithmetic matches the paper. For that you still need a worked
example with values verified outside your own code — a published table, a hand
computation, or a cross-check against another library:

```python
def test_matches_the_published_example():
    """Table 3 of Author (2019); distances 0.1996, 0.3622, 0.3169."""
    result = rank(PAPER_MATRIX, PAPER_CRITERIA, method="mine")
    assert result.score_of("A1") == pytest.approx(-0.19956052, abs=1e-8)
```

If your method claims `reversal_free`, prove it on your own data too:

```python
from mcdakit import reversal_check

def test_it_is_reversal_free():
    assert reversal_check(matrix, criteria, method="mine")["reversed"] is False
```

`benchmarks/reversal.py` measures the same property across many random
problems, and takes any registered method.

## Extending beyond methods

A ranking method is the most common extension, but not the only one.

### Normalisation schemes

Putting a price spanning 900 and a rating spanning 3 onto a common scale is a
*modelling* decision, not preprocessing. Which scheme you pick changes the
numbers, sometimes the winner, and — as the rank-reversal literature shows —
how stable the ranking is. So schemes are named and swappable:

```python
from mcdakit import rank, available_normalizations

available_normalizations()
# {'vector': ..., 'minmax': ..., 'max': ..., 'sum': ...}

rank(matrix, criteria, method="topsis", normalization="minmax")
```

Each method declares the scheme it is *defined* with — `topsis` is `vector`,
`saw` is `max`, `weighted_scoring` is `minmax` — so the default reproduces the
textbook method, and overriding is an explicit choice.

Register your own the same way you register a method:

```python
import numpy as np
from mcdakit import register_normalization

@register_normalization("logistic", summary="Logistic squash about the mean.")
def logistic(data):
    spread = np.where(data.std(axis=0) == 0, 1.0, data.std(axis=0))
    return 1 / (1 + np.exp(-(data - data.mean(axis=0)) / spread))

rank(matrix, criteria, method="topsis", normalization="logistic")
```

A scheme receives an already-oriented matrix and returns one of the same
shape. Give a column that cannot discriminate — every option alike, or all
zeros — a constant rather than dividing by zero; the library raises if a
scheme returns NaN, because one poisoned column would corrupt the ranking of
every other criterion too.

A one-off scheme needs no registration at all — pass the callable:

```python
rank(matrix, criteria, method="topsis", normalization=my_function)
```

Inside a method, reach it through the context rather than normalising inline,
so your method stays swappable too:

```python
def score(self, ctx):
    return ctx.normalize() @ ctx.normalized_weights
```

Methods that must *not* be swappable say so by leaving `normalization = None`.
SPOTIS does: its reversal-freedom rests on scaling against fixed bounds rather
than the observed options, so substituting another scheme would silently
destroy the guarantee. Passing `normalization=` to such a method raises rather
than being ignored.

### What is not yet extensible

Being straight about the limits, since the answer shapes what you can build:

| Extension point | Status |
|---|---|
| Ranking methods | Registry + entry points |
| Normalisation schemes | Registry, or a bare callable |
| Weighting schemes | `ahp_weights()` only; no registry — compute weights yourself and pass them on `Criterion` |
| Group aggregation | `mcdakit.group` — several stakeholders, with a disagreement report. Not pluggable: two aggregations ship, adding a third means a PR |
| Sensitivity analyses | `sensitivity()` only; not pluggable |

Weights and aggregation are plain data going in, so you are not blocked —
you just do not get a name, discovery, or the conformance suite. If you want
either as a real extension point, open an issue saying what you are building.

## Contributing a method upstream

A plugin needs nothing from us. If a method belongs in `mcdakit` itself —
because it is widely used, or needs something the contract cannot express —
open an issue using the "Propose a method" template, and see
[`CONTRIBUTING.md`](https://github.com/aitmlouk/mcdakit/blob/main/CONTRIBUTING.md).
