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

## Testing your method

Two properties are worth asserting whatever your method does:

```python
import numpy as np
from mcdakit import rank, reversal_check

def test_a_dominant_option_wins():
    """The one result no method may get wrong."""
    matrix = [[1, 1], [9, 9]]
    assert rank(matrix, criteria, method="mine").winner == "Best"

def test_scores_are_finite():
    result = rank(matrix, criteria, method="mine")
    assert np.all(np.isfinite(result.scores))
```

If your method claims `reversal_free`, prove it:

```python
def test_it_is_reversal_free():
    assert reversal_check(matrix, criteria, method="mine")["reversed"] is False
```

`benchmarks/reversal.py` measures the same property across many random
problems, and takes any registered method.
