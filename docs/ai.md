# Language-model assistance

Turning a described problem into a structured one is the part of MCDA that is
genuinely hard to automate. Naming plausible criteria, giving each a
direction, drafting a first set of weights — a model's breadth helps here, and
a wrong answer is cheap to spot.

A model is *not* useful as an authority. It returns weights with the same
confidence whether they were considered or invented, and nothing in the output
distinguishes the two. So this module inverts the usual arrangement: **the
model proposes, the package verifies**, and tells you what the proposal is
worth.

Nothing here is re-exported from the package namespace. Using it takes an
explicit `from mcdakit.ai import ...` that stays visible in your source —
an MCDA result is often used to justify a decision to someone else, and
whether a model shaped the inputs belongs in the record.

## Connecting a model

`ask` is **any callable that takes a prompt string and returns the reply
string**. No vendor is assumed, nothing is installed, and `mcdakit` acquires
no LLM dependency.

```python
Ask = Callable[[str], str]
```

That is the entire interface, and it commits the package to no vendor.

**Adapters ship with the package**, so for the common providers you need
write nothing at all. They speak each vendor's HTTP API through the standard
library — **no SDK to install**, and NumPy remains the only dependency:

```python
from mcdakit.ai_providers import openai, anthropic, google, ollama

ask = anthropic()                    # reads ANTHROPIC_API_KEY
ask = openai()                       # reads OPENAI_API_KEY
ask = google()                       # reads GOOGLE_API_KEY or GEMINI_API_KEY
ask = ollama("llama3")               # local, no key, nothing leaves the machine

ask = anthropic(model="claude-opus-5", api_key="sk-...")   # explicit
```

### Local and self-hosted models

`ollama()` covers the usual local case. For anything exposing the
OpenAI-compatible `/chat/completions` endpoint — **vLLM, LM Studio,
llama.cpp's server, Groq, Together, OpenRouter, DeepSeek, Mistral** — one
adapter reaches all of them:

```python
from mcdakit.ai_providers import openai_compatible

ask = openai_compatible("mistral-7b", "http://localhost:8000/v1")   # vLLM
ask = openai_compatible("llama-3.3-70b", "https://api.groq.com/openai/v1",
                        api_key="gsk_...")
```

Keeping the problem description, criteria and measurements on your own
machine matters for commercially sensitive decisions, which is a large share
of real MCDA work.

### What the adapters handle

* **Rate limits.** A 429 or a transient 5xx is retried with exponential
  backoff, honouring the provider's own `Retry-After`. A 400 is not retried:
  it will fail identically however often it is sent.
* **Missing keys.** `ProviderError: No API key for Anthropic. Set the
  ANTHROPIC_API_KEY environment variable...` rather than a vendor 401.
* **Changed reply shapes.** An unexpected response names the provider and
  shows what arrived, instead of an `IndexError` from inside an adapter.
* **Temperature 0 by default**, because `samples=2` measures whether the model
  is consistent — sampling noise would corrupt that measurement.

### Writing your own

Nothing requires you to use them. Any callable works, so an in-house gateway,
a cached fixture or an SDK you already trust drops straight in:

```python
def ask(prompt: str) -> str:
    return my_company_gateway.complete(prompt)
```

Replies arrive as real models write them — wrapped in markdown fences, with a
sentence of preamble before the JSON. The parsers expect that and extract the
JSON regardless.

## Proposing criteria

```python
from mcdakit.ai import propose_criteria

proposal = propose_criteria(
    "choosing a cloud hosting provider for a mid-sized SaaS company",
    ask=ask,
    samples=2,     # ask twice; the agreement between replies is reported
)
print(proposal)
```

```
10 criteria proposed:
  Monthly cost (cost, weight 0.18)
  Uptime / reliability SLA (benefit, weight 0.16)
  Performance (benefit, weight 0.12)
  ...
  ! discarded 'Colour': direction must be one of benefit, cost
  agreement across replies: 80%
  not applied — construct a Decision to accept
```

Each suggestion is parsed into a validated
{class}`~mcdakit.types.Criterion`, so a malformed or out-of-range answer never
reaches `proposal.criteria` — it lands in `proposal.rejected` **with the
reason**. Six proposed criteria of which two were malformed is a different
signal from four clean ones, and dropping the difference would hide it.

`samples=2` asks the same question twice and reports the share of names common
to every reply. Low agreement means the model is improvising, which is worth
more than any single answer.

```python
proposal.criteria    # validated Criterion objects
proposal.rejected    # ((name, reason), ...)
proposal.agreement   # 0.8
proposal.accepted    # always False — see below
```

## Proposing weights, and checking them against the data

```python
from mcdakit.ai import propose_weights

weights = propose_weights(
    "choosing a cloud hosting provider", criteria, ask=ask, matrix=matrix
)

weights.weights          # array, normalised to sum to 1
weights.stability        # the sensitivity() report USING these weights
weights.disagrees_with   # ('entropy', 'critic', 'std')
```

This is the part worth reading twice. `stability` is a full
{func}`~mcdakit.sensitivity` report computed *with the model's weights*:

```python
weights.stability["level"]    # 'moderate'
weights.stability["weakest"]  # 'Uptime SLA'
```

A model's weights can be entirely plausible and still produce a
recommendation that a small correction overturns. You learn that **before**
relying on them.

`disagrees_with` names the objective schemes that order the criteria
differently. A model that ranks importance unlike entropy, CRITIC *and*
standard deviation is not necessarily wrong — but the divergence is worth
seeing.

To apply the same checks to weights from any source, human included:

```python
from mcdakit.ai import critique_weights

critique_weights(my_weights, criteria, matrix)   # no model involved
```

## Pairwise comparison, which can be checked

Asking for a weight vector gives you an answer that cannot be wrong on its own
terms. Asking *is price more important than quality, and by how much?* gives
you answers that **can contradict each other** — and Saaty's consistency ratio
finds the contradiction by arithmetic:

```python
from mcdakit.ai import propose_comparisons

judgements = propose_comparisons("choosing a host", criteria, ask=ask)
print(judgements)
```

```
6 pairwise judgements, consistency ratio 0.022 (consistent)
  Monthly cost      0.2394
  Uptime SLA        0.5215
  Support quality   0.1530
  Migration effort  0.0860
  not applied — construct a Decision to accept
```

If the model says price matters three times quality, quality three times
delivery, and price only twice delivery, that is incoherent. The ratio detects
it without anyone knowing what the right weights were. Above 0.10 the
proposal prints `INCONSISTENT` and says to ask again.

## Simulating a panel

Human expert panels are expensive, slow and hard to reproduce. Scoring the
problem once per persona gives a cheap approximation — provided the spread
between personas is reported, so a "panel" that agreed exactly is visible as
having added nothing:

```python
from mcdakit.ai import simulate_panel

panel = simulate_panel(
    "choosing a cloud hosting provider",
    criteria,
    personas=["CFO focused on budget discipline",
              "VP of Engineering focused on reliability",
              "Head of Customer Success"],
    labels=["AWS", "Azure", "DigitalOcean"],
    ask=ask,
)
print(panel)
```

```
3 simulated participants:
  CFO focused on budget discipline
  VP of Engineering focused on reliability
  Head of Customer Success
  spread across personas: 0.447
  not applied — pass to group_rank() to combine
```

The personas come back as ordinary
{class}`~mcdakit.group.Participant` objects and go through exactly the same
{func}`~mcdakit.group.group_rank` machinery as a real panel. A simulated panel
is not a substitute for asking people; it is a way to see whether the outcome
is sensitive to perspective before you spend their time.

## Writing it up, with the figures checked

```python
from mcdakit.ai import narrate

written = narrate(result, ask=ask, sensitivity_report=report)

written["text"]                 # the paragraph
written["faithful"]             # False
written["unsupported_numbers"]  # ('0.7212', '0.6845', '0.5130')
```

The model is given the computed facts and asked only to write them up.
Nothing is left for it to analyse, so the single failure available to it is
**misstating a number** — and that is checkable. Every numeric literal in the
reply is compared against the figures supplied, and any that does not appear
is listed.

This is not hypothetical. Asked to describe a result where DigitalOcean won
with 0.8049, a model produced a fluent paragraph awarding the win to AWS at
0.7212. `faithful` was `False` and every invented figure was named.

**Read `unsupported_numbers` before pasting the text into a report.**

## What the module will not do

```python
proposal.accepted   # False. Always.
```

Every function returns a *proposal*. Nothing is applied automatically, and
there is no setting that changes this. Accepting a suggestion means
constructing a {class}`~mcdakit.types.Decision` from it yourself — a line of
code you write, in a file someone can read.

A package whose stated purpose is refusing answers it cannot stand behind does
not get to make an exception because the answer came from a model.

## A complete run

```python
from dataclasses import replace

from mcdakit import Decision, rank, sensitivity
from mcdakit.ai import propose_criteria, propose_weights, narrate

# 1. The model suggests a structure; you read it.
proposal = propose_criteria("choosing a cloud host", ask=ask, samples=2)
criteria = [c for c in proposal.criteria if c.name != "Vendor lock-in risk"]

# 2. You supply the data. The model never sees or invents measurements.
matrix = [[4200, 99.99, 8, 6], [3800, 99.95, 7, 4], [1500, 99.90, 5, 2]]

# 3. The model drafts weights; the package checks them against that data.
weights = propose_weights("choosing a cloud host", criteria, ask=ask,
                          matrix=matrix)
if weights.stability["level"] == "fragile":
    print("the recommendation turns on a small weight change — revisit")

# 4. You accept, explicitly.
criteria = [replace(c, weight=w) for c, w in zip(criteria, weights.weights)]
result = rank(Decision(matrix, criteria, ["AWS", "Azure", "DigitalOcean"]),
              method="topsis")

# 5. The write-up is checked against the arithmetic.
written = narrate(result, ask=ask, sensitivity_report=sensitivity(result))
assert written["faithful"], written["unsupported_numbers"]
```

The model shapes the *structure*. Every number that decides the outcome is
either measured by you or computed by the package.
