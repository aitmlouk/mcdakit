"""Shared fixtures.

The supplier case is the same one the Odoo module's tests used, whose weighted
result was verified by hand: with min-max normalisation before weighting, the
scores are B=0.6000, A=0.5250, C=0.2833. Asserting against arithmetic checked
independently is the point — a fixture that merely records whatever the code
produced would pass forever, including after a regression.
"""

import numpy as np
import pytest

from mcdakit import Criterion, Decision

SUPPLIER_MATRIX = [
    [9, 5, 7, 6],   # Supplier A
    [6, 9, 8, 9],   # Supplier B
    [7, 7, 6, 5],   # Supplier C
]
SUPPLIER_LABELS = ["Supplier A", "Supplier B", "Supplier C"]


@pytest.fixture
def supplier_criteria():
    return [
        Criterion("Price", weight=0.40),
        Criterion("Quality", weight=0.30),
        Criterion("Delivery", weight=0.20),
        Criterion("Support", weight=0.10),
    ]


@pytest.fixture
def supplier(supplier_criteria):
    return Decision(SUPPLIER_MATRIX, supplier_criteria, SUPPLIER_LABELS)


@pytest.fixture
def procurement():
    """The four-supplier case from the README, with real units and bounds.

    Kestrel wins on the full shortlist. Nordpack overtakes it under most
    methods once the rejected option is removed, because that option was
    defining the price scale — the worked example behind SPOTIS.
    """
    criteria = [
        Criterion("Price", 0.40, "cost", bounds=(2.00, 4.00)),
        Criterion("Quality", 0.25, "benefit", bounds=(0, 10)),
        Criterion("Lead time", 0.20, "cost", bounds=(5, 35)),
        Criterion("Support", 0.15, "benefit", bounds=(0, 10)),
    ]
    matrix = [
        [2.75, 7.0, 14, 8.0],   # Kestrel Supply
        [2.90, 8.5, 16, 8.0],   # Nordpack
        [3.40, 9.0, 11, 7.0],   # Meridian
        [2.20, 3.0, 32, 2.0],   # Bytharm  (rejected: last under every method,
                                #  and the cheapest, so it defines the price scale)
    ]
    labels = ["Kestrel Supply", "Nordpack", "Meridian", "Bytharm"]
    return Decision(matrix, criteria, labels)


@pytest.fixture
def rng():
    """Seeded generator: a benchmark that cannot be reproduced is an anecdote."""
    return np.random.default_rng(20260826)
