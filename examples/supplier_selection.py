"""A complete decision analysis, from raw figures to a defensible report.

This is the workflow the package is built around, on a small procurement
problem: four suppliers scored on price, quality, lead time and support.

Run it with::

    python examples/supplier_selection.py

Every number printed is computed here; nothing is hard-coded. The point of the
example is not the ranking --- any MCDA library produces one --- but everything
that follows it.
"""

from mcdakit import (
    Criterion,
    Decision,
    agreement,
    compare_methods,
    rank,
    reversal_check,
    sensitivity,
)
from mcdakit.explain import compare, explain

# --------------------------------------------------------------------------
# 1. Describe the problem
# --------------------------------------------------------------------------
# Figures are entered as measured --- real euros, real days. Cost criteria are
# marked as such and the library handles the direction; inverting them by hand
# is the classic way to get a confidently reversed ranking.
#
# `bounds` is the range a criterion *could* take, independent of the suppliers
# on the table: a budget ceiling, a rating scale, an acceptable delivery
# window. It is optional everywhere except SPOTIS, whose guarantee depends on
# it.

criteria = [
    Criterion("Price", weight=0.40, direction="cost", bounds=(2.00, 4.00)),
    Criterion("Quality", weight=0.25, direction="benefit", bounds=(0, 10)),
    Criterion("Lead time", weight=0.20, direction="cost", bounds=(5, 35)),
    Criterion("Support", weight=0.15, direction="benefit", bounds=(0, 10)),
]

matrix = [
    [2.75, 7.0, 14, 8.0],  # Option 1
    [2.90, 8.5, 16, 8.0],  # Option 2
    [3.40, 9.0, 11, 7.0],  # Option 3
    [2.20, 3.0, 32, 2.0],  # Option 4 — cheapest, but last on everything else
]
labels = ["Option 1", "Option 2", "Option 3", "Option 4"]
decision = Decision(matrix, criteria, labels)


def heading(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


# --------------------------------------------------------------------------
heading("1. The ranking")
# --------------------------------------------------------------------------
# SPOTIS is used because it cannot reverse rank: each alternative is measured
# against a fixed ideal point built from the bounds above, never against the
# other alternatives.

result = rank(decision, method="spotis")
print(result)
print(f"\n  reversal_free: {result.reversal_free}")


# --------------------------------------------------------------------------
heading("2. Would the ranking survive someone disagreeing?")
# --------------------------------------------------------------------------
# Weights are the softest input in any MCDA model. This reports how far each
# one can move before a different supplier wins --- and which way.

report = sensitivity(result)
print(f"  overall tolerance: {report['overall']:.1%}  ({report['level']})")
print(f"  weakest criterion: {report['weakest']}\n")

for row in report["rows"]:
    if row["tolerance"] is None:
        print(f"  {row['name']:<12} never unseats the winner")
    else:
        print(
            f"  {row['name']:<12} {row['tolerance']:>6.1%} "
            f"{row['direction']:<5} -> {row['flips_to']}"
        )

print(
    "\n  Read that as: the recommendation holds only while nobody weights "
    f"\n  {report['weakest']} more than {report['overall']:.1%} differently."
)


# --------------------------------------------------------------------------
heading("3. Why did it win?")
# --------------------------------------------------------------------------
# A ranking nobody can interrogate is hard to act on or to defend.

runner_up = result.order[1]
print(compare(result, result.winner, runner_up))

print()
print(explain(result))


# --------------------------------------------------------------------------
heading("4. Does the ranking depend on a rejected alternative?")
# --------------------------------------------------------------------------
# Option 4 is last under every method. Removing it should not reorder the
# others --- but under most methods it can, because it helps define the scale
# the others are measured against.

for method in ("spotis", "weighted_scoring", "topsis"):
    check = reversal_check(decision, method=method)
    if check["reversed"]:
        case = check["cases"][0]
        print(f"  {method:<18} REVERSES when {case['removed']} is removed")
        print(f"  {'':<18}   {case['before']}")
        print(f"  {'':<18}   {case['after']}")
    else:
        print(f"  {method:<18} stable")


# --------------------------------------------------------------------------
heading("5. Do the methods agree?")
# --------------------------------------------------------------------------
# When they disagree, that disagreement is the finding: it means the
# alternatives are close enough for the modelling choice to decide the
# outcome.

results = compare_methods(decision)
for name, outcome in results.items():
    print(f"  {name:<18} {outcome.winner}")

summary = agreement(results)
if summary["consensus"] is None:
    joint = " and ".join(summary["tied"])
    print(
        f"\n  no consensus: {joint} tie on "
        f"{summary['votes']} of {summary['of']} methods"
    )
else:
    print(
        f"\n  consensus: {summary['consensus']} "
        f"({summary['votes']} of {summary['of']} methods)"
    )
print(f"  unanimous: {summary['unanimous']}")


# --------------------------------------------------------------------------
heading("What a decision report can now say")
# --------------------------------------------------------------------------
print(
    f"""
  {result.winner} is recommended.

  The result is {report["level"]}: weighting {report["weakest"]}
  {report["overall"]:.1%} differently would favour
  {report["rows"][0]["flips_to"] or "another supplier"} instead.

  {summary["votes"]} of {summary["of"]} methods agree on the winner.

  Under SPOTIS the ranking does not depend on which alternatives were
  shortlisted; under weighted scoring it does.
"""
)
