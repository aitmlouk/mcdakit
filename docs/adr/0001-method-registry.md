# 1. Methods are a registry of objects, not a hardcoded dispatch

**Status:** accepted, 2026-08-26

## Context

v0.1.0 dispatched methods through an if/elif chain over a frozen `METHODS`
tuple:

```python
if method == "spotis":
    return spotis(decision.matrix, weights, decision.directions, bounds)
data = orient(decision.matrix, decision.directions)
if method == "promethee":
    return promethee(data, weights, _shapes(decision))
return _ORIENTED[method](data, weights)
```

Two methods needed special handling that the others did not — SPOTIS wants the
un-oriented matrix and the bounds, PROMETHEE wants per-criterion preference
shapes — and `sensitivity.py` carried a third special case, suppressing
SPOTIS's warning by checking the method name.

A method written outside the package could not express any of that. It could
not say "give me raw data", could not report a caveat, and could not be added
without editing `ranking.py`. Three name checks in the library meant three
places where an outside method was second-class by construction.

## Decision

A method is an object implementing a `Method` protocol. It declares what it
needs (`wants = Wants.RAW | Wants.ORIENTED`), receives everything through a
`ScoringContext`, and may return either a bare score array or a `MethodResult`
carrying `reversal_free` and `warnings`. A registry maps names to instances,
and `importlib.metadata` entry points let a package advertise methods that are
discovered on install.

`METHODS` became a live view of the registry rather than a frozen tuple,
because a snapshot taken at import would not see a plugin registered later.

The three name checks are gone. `ranking.py` no longer mentions any method by
name; `sensitivity.py` suppresses warnings generically around the bisection,
which works for methods nobody has written yet.

## Consequences

**Good.** Third-party methods are indistinguishable from built-ins: reachable
by name, included in `compare_methods`, analysable by `sensitivity`. Adding a
built-in no longer touches the dispatcher. Declared capabilities are
self-documenting — `wants = Wants.RAW` says what a comment used to.

**Cost.** More surface area: `Method`, `ScoringContext`, `MethodResult`,
`Wants` and the registry are all public API now, and public API is a promise.
An indirection sits between `rank()` and the arithmetic.

**A bug it surfaced.** With options passed as opaque `**kwargs`, a method that
ignored them let a misspelled argument vanish silently — the same class of bug
as the original dropped-`kwargs` defect. `ScoringContext.opts` tracks whether
options were read, and `rank()` raises if nobody read them.

**Not changed.** The eight algorithms are byte-for-byte the same functions,
still importable directly and still covered by the tests that verified them
against hand computations and pymcdm. The refactor moved dispatch, not
arithmetic — 236 existing tests passed unchanged, which is the evidence that
it was behaviour-preserving.

## Alternatives considered

**A `@method` decorator over plain functions.** Lighter, and closer to the
existing code. Rejected because metadata in decorator arguments is harder to
discover than attributes on a class, and a function has nowhere natural to put
`validate`.

**Passing `(data, weights)` and keeping capabilities implicit.** Rejected: it
is exactly what made SPOTIS need a special case, and the next method with an
unusual need would need another one.
