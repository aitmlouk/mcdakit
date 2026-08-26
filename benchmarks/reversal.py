"""Measure how often each method reverses rank.

The experiment behind the table in the README, and behind this package's
reason to exist. On each trial: generate a random decision problem, rank it,
remove the **last-placed** option — one nobody would have chosen — re-rank the
survivors, and check whether their relative order changed.

A change means the ranking of the options you were actually choosing between
depended on an option you had already rejected.

Run with::

    python benchmarks/reversal.py
    python benchmarks/reversal.py --trials 2000 --options 7 --criteria 5

Every run is seeded, so the numbers are reproducible rather than anecdotal.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

sys.path.insert(0, "src")

from mcdakit import METHODS, BoundsWarning, Criterion, Decision, rank

#: Bounds wide enough to contain any generated value, so SPOTIS is measured
#: with the fixed bounds it is designed for rather than the fallback.
BOUNDS = (0.0, 10.0)


def random_problem(rng, n_options: int, n_criteria: int) -> Decision:
    """A random problem with fixed, known bounds on every criterion."""
    matrix = rng.uniform(BOUNDS[0], BOUNDS[1], size=(n_options, n_criteria))
    weights = rng.dirichlet(np.ones(n_criteria))
    criteria = [
        Criterion(f"C{j + 1}", float(weights[j]), "benefit", bounds=BOUNDS)
        for j in range(n_criteria)
    ]
    return Decision(matrix, criteria)


def reverses(decision: Decision, method: str) -> bool:
    """Does removing the last-placed option reorder the survivors?"""
    full = rank(decision, method=method)
    loser = full.order[-1]
    loser_index = decision.labels.index(loser)

    before = [label for label in full.order if label != loser]
    after = rank(decision.without(loser_index), method=method).order
    return before != after


def run(trials: int, n_options: int, n_criteria: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(METHODS, 0)

    # SPOTIS is given real bounds here, so the fallback warning should never
    # fire. Promote it to an error rather than trusting that silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundsWarning)
        for _ in range(trials):
            decision = random_problem(rng, n_options, n_criteria)
            for method in METHODS:
                if reverses(decision, method):
                    counts[method] += 1

    return {method: counts[method] / trials for method in METHODS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--options", type=int, default=5)
    parser.add_argument("--criteria", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    rates = run(args.trials, args.options, args.criteria, args.seed)

    print(
        f"\nRank reversal after removing the last-placed option\n"
        f"{args.trials} random {args.options}x{args.criteria} problems, "
        f"seed {args.seed}\n"
    )
    print(f"{'Method':<20} {'Reversal rate':>14}")
    print("-" * 35)
    for method, rate in sorted(rates.items(), key=lambda kv: kv[1]):
        print(f"{method:<20} {rate:>13.1%}")

    if rates["spotis"] != 0.0:
        print(
            f"\nFAIL: SPOTIS reversed on {rates['spotis']:.1%} of trials. It "
            f"is reversal-free by construction, so this is a bug.",
            file=sys.stderr,
        )
        return 1

    print("\nSPOTIS: 0.0%, as the method guarantees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
