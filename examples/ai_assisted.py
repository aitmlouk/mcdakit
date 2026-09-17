"""Language-model assistance, and what the package does with it.

A model is good at turning a described problem into a structured one: naming
plausible criteria, giving each a direction, drafting weights. It is not an
authority — it returns weights with the same confidence whether they were
considered or invented. So the model proposes and the package checks.

Run it with::

    python examples/ai_assisted.py

With no API key it uses **recorded replies from a real model**, so the whole
example runs offline and deterministically. To use a live model instead, set
one of these and run it again:

    export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY, GOOGLE_API_KEY
    ollama serve && ollama pull llama3   # or a local model, no key needed

The interesting parts are the failures. Section 4 shows a model contradicting
itself and section 5 shows one inventing figures — both caught by arithmetic
rather than by reading.
"""

import os
from dataclasses import replace

import numpy as np

from mcdakit import Criterion, Decision, rank, sensitivity
from mcdakit.ai import (
    AiError,
    narrate,
    propose_comparisons,
    propose_criteria,
    propose_weights,
    simulate_panel,
)
from mcdakit.group import disagreement, group_rank

# --------------------------------------------------------------------------
# Connecting a model
# --------------------------------------------------------------------------
# `ask` is any callable mapping a prompt to a reply. Adapters for the common
# providers ship with the package and need no SDK; anything else — an in-house
# gateway, a cached fixture — drops in the same way.

RECORDED = {
    "most important criteria": """[
      {"name": "Monthly cost",     "direction": "cost",    "weight": 0.35},
      {"name": "Uptime SLA",       "direction": "benefit", "weight": 0.30},
      {"name": "Support quality",  "direction": "benefit", "weight": 0.20},
      {"name": "Migration effort", "direction": "cost",    "weight": 0.15}
    ]""",
    "relative importance weight": """{
      "Monthly cost": 0.45, "Uptime SLA": 0.30,
      "Support quality": 0.15, "Migration effort": 0.10
    }""",
    # Deliberately incoherent: cost is said to matter 5x uptime and 3x
    # support, yet uptime only 1/4 of support. Those cannot all hold.
    "Saaty's scale": """[
      {"i": 0, "j": 1, "ratio": 5},   {"i": 0, "j": 2, "ratio": 3},
      {"i": 0, "j": 3, "ratio": 2},   {"i": 1, "j": 2, "ratio": 0.25},
      {"i": 1, "j": 3, "ratio": 3},   {"i": 2, "j": 3, "ratio": 4}
    ]""",
    "CFO": "[[3, 9, 7, 4], [4, 9, 8, 5], [9, 6, 5, 8]]",
    "Head of Engineering": "[[6, 10, 9, 5], [6, 9, 8, 5], [8, 6, 5, 7]]",
    "Head of Customer Success": "[[5, 9, 9, 5], [6, 9, 9, 6], [8, 7, 6, 7]]",
    # A plausible paragraph containing figures from nowhere.
    "Write a short": (
        "The analysis is decisive: AWS leads on 0.8800, comfortably ahead of "
        "Azure on 0.7100, with DigitalOcean trailing at 0.2400. The margin of "
        "17 percentage points means the choice is not close."
    ),
}


def recorded(prompt: str) -> str:
    """Replies a real model gave to these exact prompts."""
    for marker, reply in RECORDED.items():
        if marker in prompt:
            return reply
    raise AssertionError(f"no recorded reply for: {prompt[:60]}...")


def connect():
    """A live provider if a key is present, otherwise the recording."""
    from mcdakit import ai_providers

    for variable, build in (
        ("ANTHROPIC_API_KEY", ai_providers.anthropic),
        ("OPENAI_API_KEY", ai_providers.openai),
        ("GOOGLE_API_KEY", ai_providers.google),
        ("GEMINI_API_KEY", ai_providers.google),
    ):
        if os.environ.get(variable):
            return build(), f"live model via {variable.split('_')[0].lower()}"
    if os.environ.get("MCDAKIT_OLLAMA_MODEL"):
        model = os.environ["MCDAKIT_OLLAMA_MODEL"]
        return ai_providers.ollama(model), f"local model: {model}"
    return recorded, "recorded replies (set an API key for a live model)"


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


ask, how = connect()
PROBLEM = "choosing a cloud hosting provider for a mid-sized SaaS company"
print(f"Provider: {how}")
print(f"Problem : {PROBLEM}")

# --------------------------------------------------------------------------
# 1. Naming the criteria
# --------------------------------------------------------------------------
rule("1. What should we even be measuring?")
print(
    "Asked twice, so that a model which improvises can be told from one\n"
    "which is consistent. Every suggestion is validated: anything malformed\n"
    "is rejected with a reason rather than quietly absorbed.\n"
)

proposal = propose_criteria(PROBLEM, ask=ask, n=4, samples=2)
print(proposal)
if proposal.agreement is not None:
    print(f"\n  the two replies shared {proposal.agreement:.0%} of their names")

criteria = list(proposal.criteria)

# --------------------------------------------------------------------------
# 2. The measurements are yours
# --------------------------------------------------------------------------
rule("2. The numbers come from you, not the model")
print(
    "The model shapes the structure. Every figure that decides the outcome\n"
    "is measured by you or computed by the package.\n"
)

MATRIX = np.array(
    [
        [4200.0, 99.99, 8.0, 6.0],  # AWS
        [3800.0, 99.95, 7.0, 4.0],  # Azure
        [1500.0, 99.90, 5.0, 2.0],  # DigitalOcean
    ]
)
LABELS = ["AWS", "Azure", "DigitalOcean"]
MEASURED = ("Monthly cost", "Uptime SLA", "Support quality", "Migration effort")

# The measurements are fixed; the model's proposal is not. A weak model may
# return fewer usable criteria than there are columns, and the package refuses
# that mismatch rather than guessing a correspondence. Reconciling the two is
# the analyst's job, so the example does it explicitly: keep the proposed
# criteria whose names were measured, and fall back to the measured set when
# the model gave nothing usable.
proposed = {c.name.lower(): c for c in criteria}
matched = [proposed[name.lower()] for name in MEASURED if name.lower() in proposed]

if len(matched) == len(MEASURED):
    criteria = matched
    print("  (the model's criteria match what was measured)")
else:
    criteria = [
        Criterion(
            name, 1.0, "cost" if "cost" in name or "effort" in name else "benefit"
        )
        for name in MEASURED
    ]
    print(
        f"  (the model offered {len(proposed)} usable criteria, not the "
        f"{len(MEASURED)} that were\n"
        f"   measured, so the measured set is used instead — a mismatch is\n"
        f"   refused, never guessed at)"
    )
print()

for label, row in zip(LABELS, MATRIX):
    print(f"  {label:<14}" + "  ".join(f"{v:>8.2f}" for v in row))
print(f"  {'':14}" + "  ".join(f"{n[:8]:>8}" for n in MEASURED))

# --------------------------------------------------------------------------
# 3. Weights, checked against that data
# --------------------------------------------------------------------------
rule("3. Weights a model drafts, and whether they hold")
print(
    "The point is not the weights but the check: a sensitivity analysis run\n"
    "with them, so you learn whether the recommendation survives the model\n"
    "being somewhat wrong — before relying on it.\n"
)

try:
    weights = propose_weights(PROBLEM, criteria, ask=ask, matrix=MATRIX)
except AiError as refusal:
    # A weak model may return something unusable. Nothing is invented to
    # cover for it: the refusal names what was missing and shows what came
    # back. Equal weights below are a stated choice, not a silent default.
    print(f"  the model's weights were refused:\n    {refusal}\n")
    print("  continuing with equal weights — a choice, not a default")
    criteria = [replace(c, weight=1.0) for c in criteria]
else:
    for criterion, weight in zip(criteria, weights.weights):
        print(f"  {criterion.name:<18}{weight:.3f}")
    print(f"\n  stability      : {weights.stability['level']}")
    print(f"  weakest weight : {weights.stability['weakest']}")
    if weights.disagrees_with:
        print(f"  disagrees with : {', '.join(weights.disagrees_with)}")
        print(
            "                   (objective schemes ordering importance "
            "differently;\n                    not proof of error, but worth "
            "seeing)"
        )
    print(f"  applied        : {weights.accepted}  — accepting is your decision")
    criteria = [replace(c, weight=float(w)) for c, w in zip(criteria, weights.weights)]

# --------------------------------------------------------------------------
# 4. Pairwise judgements, which can be caught contradicting themselves
# --------------------------------------------------------------------------
rule("4. A model contradicting itself, caught by arithmetic")
print(
    "Asking for a weight vector gives an answer that cannot be wrong on its\n"
    "own terms. Asking 'is cost more important than uptime, and by how much?'\n"
    "gives answers that can contradict each other — and Saaty's consistency\n"
    "ratio finds it without anyone knowing the right weights.\n"
)

judgements = propose_comparisons(PROBLEM, criteria, ask=ask)
print(judgements)
if not judgements.consistent:
    print(
        "\n  Above Saaty's 0.10 limit, so these judgements are not usable as\n"
        "  they stand. Reported, never enforced: asking again or revising by\n"
        "  hand is the analyst's call, not the library's."
    )

# --------------------------------------------------------------------------
# 5. A simulated panel, with its spread reported
# --------------------------------------------------------------------------
rule("5. Several perspectives, and whether they actually differed")
print(
    "Scoring once per persona approximates a panel. The spread matters: a\n"
    "'panel' that agreed exactly added nothing over a single opinion, and\n"
    "saying so is more useful than presenting it as consensus.\n"
)

panel = simulate_panel(
    PROBLEM,
    criteria,
    personas=[
        "CFO focused on budget discipline",
        "Head of Engineering focused on reliability",
        "Head of Customer Success",
    ],
    labels=LABELS,
    ask=ask,
)
print(panel)

group = group_rank(panel.participants, criteria, method="topsis", labels=LABELS)
split = disagreement(panel.participants, criteria, labels=LABELS)

print(f"\n  combined winner   : {group['result'].winner}")
print(f"  every persona agreed: {group['unanimous']}")
print(
    f"  most contested      : {split['most_contested_option']} "
    f"(on {split['most_contested_criterion']})"
)
if not group["unanimous"]:
    print(
        "\n  They did not agree, and that is the finding. Averaging the\n"
        "  personas into one score would have hidden exactly the tension\n"
        "  a real panel is convened to surface."
    )

# --------------------------------------------------------------------------
# 6. The write-up, checked against the arithmetic
# --------------------------------------------------------------------------
rule("6. The model writes it up; every number is verified")

result = rank(Decision(MATRIX, criteria, LABELS), method="topsis")
report = sensitivity(result)
print("computed by the package:")
for label, score in result.ranking:
    print(f"  {label:<14}{score:.4f}")

written = narrate(result, ask=ask, sensitivity_report=report)
print(f"\nwhat the model wrote:\n  {written['text']}")
print(f"\n  faithful : {written['faithful']}")
if not written["faithful"]:
    print(f"  invented : {', '.join(written['unsupported_numbers'])}")
    print(
        "\n  Every numeric literal is compared against the computed figures.\n"
        "  These appear in no result — a fluent paragraph asserting numbers\n"
        "  from nowhere is exactly what a decision report must not contain.\n"
        "  Read `unsupported_numbers` before pasting anything into one."
    )

rule("What the model was never allowed to do")
print(
    "  decide the ranking          — computed from your measurements\n"
    "  supply a measurement        — every figure in the matrix is yours\n"
    "  apply its own suggestion    — accepted is False throughout\n"
    "  assert an unchecked number  — narrate() compares every literal\n"
    "\n"
    "It named criteria, drafted weights and wrote prose. Each was checked\n"
    "before it counted for anything."
)
