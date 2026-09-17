"""A conformance suite for method authors.

Writing a ranking method is easy; writing one that is *correct* is not, and
the failure modes are quiet. A method whose scores read smallest-is-best
produces a confident, exactly inverted ranking. One that divides by a
zero-variance criterion produces NaN on some inputs and not others. Neither
announces itself.

This module is the check to run before publishing a method. It is deliberately
about properties every sound method shares, not about any particular
algorithm — so it can tell you that something is wrong without knowing what
you implemented.

Use it in your own test suite::

    from mcdakit.testing import check_method

    def test_my_method_conforms():
        check_method(MyMethod())

or from a terminal::

    python -m mcdakit.testing my_package.methods:MyMethod

The suite is a floor, not a ceiling. Passing it means your method is
well-behaved, not that its arithmetic matches the paper — for that you still
need a worked example with values verified outside your own code.
"""

from __future__ import annotations

import numpy as np

from .methods.base import Method, Wants, _as_method_result
from .ranking import rank
from .types import Criterion, Decision, McdaError

__all__ = ["ConformanceError", "check_method", "conformance_report"]


class ConformanceError(AssertionError):
    """A method violated a property every sound method should hold."""


def _problem(matrix, directions=None, weights=None, bounds=(0.0, 100.0)):
    """Build a Decision with bounds set, so bounded methods are testable."""
    n = len(matrix[0])
    directions = directions or ["benefit"] * n
    weights = weights if weights is not None else [1.0] * n
    criteria = [
        Criterion(f"C{j + 1}", float(weights[j]), directions[j], bounds=bounds)
        for j in range(n)
    ]
    return Decision(matrix, criteria)


def _scores(method: Method, decision: Decision) -> np.ndarray:
    """Score a problem through the public path, so the guards apply."""
    from .methods.registry import _METHODS

    # Registered under the method's own name where possible, so any error the
    # library raises names the method the author recognises rather than an
    # internal placeholder. Falls back to a unique key if the name is taken.
    name = method.name or f"__unnamed_{id(method)}"
    key = name if name not in _METHODS else f"__conformance_{id(method)}"
    previous = _METHODS.get(key)
    original_name = method.name
    try:
        method.name = key
        _METHODS[key] = method
        return rank(decision, method=key).scores
    finally:
        method.name = original_name
        if previous is None:
            _METHODS.pop(key, None)
        else:  # pragma: no cover - only when a name collides mid-check
            _METHODS[key] = previous


# --------------------------------------------------------------------------
# Individual checks. Each returns None on success or a message on failure.
# --------------------------------------------------------------------------


def _check_declares_a_name(method: Method):
    if not getattr(method, "name", "").strip():
        return "`name` is empty; that is what callers pass as method=."
    return None


def _check_describes_itself(method: Method):
    if not getattr(method, "summary", "").strip():
        return (
            "`summary` is empty. It is what available() shows a user "
            "choosing between methods."
        )
    return None


def _check_cites_its_source(method: Method):
    if not getattr(method, "citation", "").strip():
        return (
            "`citation` is empty. This library is read by people who know "
            "the papers; name the source."
        )
    return None


def _check_scores_every_option(method: Method):
    decision = _problem([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scores = _scores(method, decision)
    # Belt and braces: Result refuses a wrong-shaped score vector before this
    # can look, so a method is rejected earlier with a clearer message.
    if scores.shape != (3,):  # pragma: no cover
        return f"returned {scores.shape} scores for 3 options."
    return None


def _check_scores_are_finite(method: Method):
    """Degenerate but legal inputs: a flat criterion, zeros, huge and tiny."""
    cases = {
        "a criterion every option scores identically": [
            [5.0, 1.0],
            [5.0, 9.0],
        ],
        "identical options": [[3.0, 3.0], [3.0, 3.0]],
        "a very wide range": [[1e-6, 1.0], [1e6, 9.0]],
    }
    if not getattr(method, "requires_positive", False):
        # A method defined only for positive values has no meaning here, and
        # demanding a score would be asking it to invent one. Such a method
        # must still refuse the input explicitly, which is checked below.
        cases["a column of zeros"] = [[0.0, 1.0], [0.0, 9.0]]
    for description, matrix in cases.items():
        try:
            scores = _scores(method, _problem(matrix))
        except McdaError as exc:
            if "non-finite" in str(exc):
                return f"produced a non-finite score on {description}."
            raise
        # Result raises on a non-finite score, so the except branch above is
        # the live path; this catches a method that somehow returns one
        # without going through Result.
        if not np.all(np.isfinite(scores)):  # pragma: no cover
            return f"produced a non-finite score on {description}."
    return None


def _check_a_declared_domain_is_enforced(method: Method):
    """A method declaring `requires_positive` must actually refuse zero.

    The declaration exempts it from the finiteness check above, so it has to
    earn that exemption: a method that quietly returned NaN instead would get
    the exemption without the guarantee.
    """
    if not getattr(method, "requires_positive", False):
        return None
    try:
        _scores(method, _problem([[0.0, 1.0], [0.0, 9.0]]))
    except McdaError:
        return None
    return (
        "declares requires_positive but accepted a column of zeros; it must "
        "refuse input outside its domain rather than return a score for it."
    )


def _check_higher_is_better(method: Method):
    """An option better on every criterion must score highest.

    The single most valuable check here. A method whose natural measure is
    smallest-is-best — a distance, a regret, a cost — must negate before
    returning, and forgetting to is invisible until someone notices the
    rankings are upside down.
    """
    matrix = [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]]
    scores = _scores(method, _problem(matrix))
    if int(np.argmax(scores)) != 2:
        return (
            "the option that is best on every criterion did not score "
            "highest. If your measure is smallest-is-best (a distance, a "
            "regret), negate it before returning: every method here reads "
            "higher-is-better."
        )
    return None


def _check_cost_criteria_are_handled(method: Method):
    """The cheapest option must win when price is the only criterion."""
    matrix = [[100.0], [900.0]]
    scores = _scores(method, _problem(matrix, directions=["cost"]))
    if scores[0] <= scores[1]:
        return (
            "on a single cost criterion the cheaper option did not score "
            "higher. If you read ctx.directions yourself, set "
            "`wants = Wants.RAW`; otherwise let the library orient the "
            "matrix and do not consult directions."
        )
    return None


def _check_weight_scale_does_not_matter(method: Method):
    """Weights are relative; doubling them all is the same input."""
    matrix = [[1.0, 9.0], [9.0, 1.0], [5.0, 4.0]]
    one = _scores(method, _problem(matrix, weights=[0.3, 0.7]))
    two = _scores(method, _problem(matrix, weights=[3.0, 7.0]))
    if list(np.argsort(-one)) != list(np.argsort(-two)):
        return (
            "scaling every weight by the same factor changed the order. "
            "Weights are relative; normalise with ctx.normalized_weights."
        )
    return None


def _check_option_order_does_not_matter(method: Method):
    """Row order is presentation, not information."""
    matrix = [[1.0, 9.0], [9.0, 2.0], [5.0, 4.0]]
    forward = _scores(method, _problem(matrix))
    backward = _scores(method, _problem(matrix[::-1]))
    if not np.allclose(np.sort(forward), np.sort(backward), atol=1e-9):
        return (
            "reordering the rows changed the set of scores. A method must "
            "not depend on the order options are listed in."
        )
    return None


def _check_single_option(method: Method):
    """A one-option problem is degenerate but legal."""
    scores = _scores(method, _problem([[5.0, 5.0]]))
    # As above: Result validates shape and finiteness first, so reaching this
    # would mean Result stopped doing so.
    if scores.shape != (1,) or not np.all(np.isfinite(scores)):  # pragma: no cover
        return "failed on a single-option problem."
    return None


def _check_reversal_freedom_if_claimed(method: Method):
    """Only checked when the method claims it — and then it must hold."""
    if not getattr(method, "reversal_free", False):
        return None
    matrix = [[8.0, 2.0], [6.0, 7.0], [4.0, 9.0], [1.0, 1.0]]
    decision = _problem(matrix)
    full = _scores(method, decision)
    order = list(np.argsort(-full))
    loser = order[-1]
    reduced = _scores(method, decision.without(loser))
    kept = [i for i in range(len(full)) if i != loser]
    if list(np.argsort(-reduced)) != list(np.argsort(-full[kept])):
        return (
            "declares reversal_free = True, but removing the last-placed "
            "option reordered the rest. reversal_free is the strongest claim "
            "this library makes; set it only if it holds by construction."
        )
    return None


def _check_raw_methods_see_raw_data(method: Method):
    """A RAW method must not be handed an oriented matrix, and vice versa."""
    seen = {}
    original = method.score
    # A method may carry `score` on the instance rather than the class. `del`
    # would destroy it instead of restoring it, so remember which it was.
    had_own = "score" in vars(method)

    def spy(ctx):
        seen["data"] = np.asarray(ctx.data).copy()
        return original(ctx)

    try:
        method.score = spy  # type: ignore[method-assign]
        decision = _problem([[100.0], [900.0]], directions=["cost"])
        _scores(method, decision)
    finally:
        if had_own:
            method.score = original  # type: ignore[method-assign]
        else:
            vars(method).pop("score", None)

    got = seen.get("data")
    if got is None:  # pragma: no cover - score() raised, so there is nothing to judge
        return None
    raw = np.allclose(got.ravel(), [100.0, 900.0])
    # These two only fire if rank() hands a method the matrix it did not ask
    # for — a break in the library's own plumbing, not in the method. Not
    # reachable from a test without first breaking orientation, which is the
    # thing being guarded.
    if method.wants is Wants.RAW and not raw:  # pragma: no cover
        return "declares Wants.RAW but did not receive the raw matrix."
    if method.wants is Wants.ORIENTED and raw:  # pragma: no cover
        return "declares Wants.ORIENTED but received an un-oriented matrix."
    return None


def _check_returns_a_supported_type(method: Method):
    decision = _problem([[1.0, 2.0], [3.0, 4.0]])
    from .methods.base import ScoringContext

    data = np.asarray(decision.matrix, dtype=float)
    if method.wants is not Wants.RAW:
        from .orientation import orient

        data = orient(decision.matrix, decision.directions)
    ctx = ScoringContext(
        data=data,
        weights=decision.weights,
        decision=decision,
        _normalization=method.normalization,
    )
    try:
        _as_method_result(method.score(ctx))
    except Exception as exc:  # pragma: no cover - defensive
        return f"score() raised {type(exc).__name__}: {exc}"
    return None


#: Every check, as ``(label, function)``. Ordered cheapest and most
#: fundamental first, so the first failure reported is usually the root cause.
CHECKS = (
    ("declares a name", _check_declares_a_name),
    ("describes itself", _check_describes_itself),
    ("cites its source", _check_cites_its_source),
    ("returns a supported type", _check_returns_a_supported_type),
    ("scores every option", _check_scores_every_option),
    ("higher is better", _check_higher_is_better),
    ("handles cost criteria", _check_cost_criteria_are_handled),
    ("receives the data form it asked for", _check_raw_methods_see_raw_data),
    ("scores stay finite", _check_scores_are_finite),
    ("a declared domain is enforced", _check_a_declared_domain_is_enforced),
    ("weight scale does not matter", _check_weight_scale_does_not_matter),
    ("option order does not matter", _check_option_order_does_not_matter),
    ("handles a single option", _check_single_option),
    ("reversal-freedom holds if claimed", _check_reversal_freedom_if_claimed),
)

#: Checks that describe good practice rather than correctness. Failing one
#: does not make a method wrong, so ``strict=False`` lets them pass.
ADVISORY = frozenset({"describes itself", "cites its source"})


def conformance_report(method: Method, *, strict: bool = True) -> dict:
    """Run every check and return ``{label: None | message}``.

    Unlike :func:`check_method` this never raises, so it can be used to show
    a table of what passed and what did not.
    """
    if not isinstance(method, Method):
        raise McdaError(f"check takes a Method instance, got {type(method).__name__}.")
    report = {}
    for label, check in CHECKS:
        if not strict and label in ADVISORY:
            continue
        try:
            report[label] = check(method)
        except Exception as exc:
            report[label] = f"raised {type(exc).__name__}: {exc}"
    return report


def check_method(method: Method, *, strict: bool = True) -> None:
    """Assert that ``method`` upholds every property a sound method should.

    Parameters
    ----------
    method:
        The instance to check.
    strict:
        When ``True`` (the default) a missing ``summary`` or ``citation`` is a
        failure. Set ``False`` while prototyping.

    Raises
    ------
    ConformanceError
        Listing every property that failed, not merely the first.

    Examples
    --------
    >>> import numpy as np
    >>> from mcdakit import Method
    >>> from mcdakit.testing import check_method
    >>> class Total(Method):
    ...     name = "doctest_total"
    ...     summary = "Weighted total."
    ...     citation = "n/a"
    ...     def score(self, ctx):
    ...         return ctx.data @ ctx.normalized_weights
    >>> check_method(Total())
    """
    report = conformance_report(method, strict=strict)
    failures = {label: why for label, why in report.items() if why}
    if failures:
        lines = "\n".join(f"  - {label}: {why}" for label, why in failures.items())
        raise ConformanceError(
            f"{type(method).__name__} failed "
            f"{len(failures)} of {len(report)} conformance checks:\n{lines}"
        )


def _main(argv=None) -> int:
    """``python -m mcdakit.testing package.module:MethodClass``"""
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        prog="python -m mcdakit.testing",
        description="Check that a method upholds the mcdakit contract.",
    )
    parser.add_argument(
        "target",
        help="import path, as module:attribute (a Method class or instance)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="treat a missing summary or citation as acceptable",
    )
    args = parser.parse_args(argv)

    if ":" not in args.target:
        parser.error("target must look like my_package.methods:MyMethod")
    module_name, _, attribute = args.target.partition(":")
    loaded = getattr(importlib.import_module(module_name), attribute)
    method = loaded() if isinstance(loaded, type) else loaded

    report = conformance_report(method, strict=not args.no_strict)
    width = max(len(label) for label in report)
    failures = 0
    for label, why in report.items():
        if why:
            failures += 1
            print(f"FAIL  {label:<{width}}  {why}")
        else:
            print(f"ok    {label}")

    print()
    if failures:
        print(f"{failures} of {len(report)} checks failed.")
        return 1
    print(f"All {len(report)} checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
