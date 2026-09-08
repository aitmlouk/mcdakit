# API reference

## Core types

```{eval-rst}
.. automodule:: mcdakit.types
   :members: Criterion, Decision, Result, McdaError
```

## Ranking

```{eval-rst}
.. automodule:: mcdakit.ranking
   :members: rank, score, compare_methods, agreement, reversal_check, as_decision
```

## Language-model assistance

Opt-in: reached as ``from mcdakit.ai import propose_criteria``, not from the
package namespace, so that using a model is a visible choice in the importing
module. No extra dependency is required.

```{eval-rst}
.. automodule:: mcdakit.ai
   :members: propose_criteria, propose_weights, critique_weights, AiProposal, WeightProposal, AiError
```

## Explaining a result

```{eval-rst}
.. automodule:: mcdakit.explain
   :members: explain, compare, Explanation, Margin, Contribution
```

## Group decisions

```{eval-rst}
.. automodule:: mcdakit.group
   :members: Participant, group_rank, aggregate_scores, aggregate_rankings, disagreement
```

## Sensitivity analysis

```{eval-rst}
.. automodule:: mcdakit.sensitivity
   :members: sensitivity
```

## Weighting

```{eval-rst}
.. automodule:: mcdakit.weighting
   :members: derive_weights, entropy_weights, critic_weights, std_weights, equal_weights, register_weighting, get_weighting, available_weightings
```

```{eval-rst}
.. py:data:: mcdakit.WEIGHTINGS

   Registered weighting schemes, as ``{name: (function, summary)}``. Use
   :func:`~mcdakit.available_weightings` for the readable form.
```

## AHP

```{eval-rst}
.. automodule:: mcdakit.ahp
   :members: ahp_weights, priorities, consistency_ratio, comparison_matrix
```

## The extension contract

```{eval-rst}
.. automodule:: mcdakit.methods.base
   :members: Method, ScoringContext, MethodResult, Wants
```

## The registry

```{eval-rst}
.. automodule:: mcdakit.methods.registry
   :members: register, unregister, get, has, names, available, DuplicateMethodError, PluginLoadError
```

At the top level these are re-exported under names that read better next to
the rest of the API:

```{eval-rst}
.. py:data:: mcdakit.METHODS

   Every registered method name — built-in and plugin alike. A live view of
   the registry rather than a snapshot, so a method registered at any point is
   immediately visible. Behaves like the tuple it reads as: ``in``, iteration,
   indexing, ``len`` and ``==`` all work.

.. py:function:: mcdakit.get_method(name)

   Alias of :func:`mcdakit.methods.registry.get`.

.. py:function:: mcdakit.has_method(name)

   Alias of :func:`mcdakit.methods.registry.has`.

.. py:function:: mcdakit.method_names()

   Alias of :func:`mcdakit.methods.registry.names`.

.. py:data:: mcdakit.BUILTIN_METHODS

   The eight :class:`~mcdakit.Method` classes this package ships, in reading
   order: simplest first, the reversal-free one last.
```

## Conformance testing

```{eval-rst}
.. automodule:: mcdakit.testing
   :members: check_method, conformance_report, ConformanceError
```

## Methods

```{eval-rst}
.. automodule:: mcdakit.methods.spotis
   :members: spotis, Spotis, BoundsWarning

.. automodule:: mcdakit.methods.topsis
   :members: topsis, Topsis

.. automodule:: mcdakit.methods.promethee
   :members: promethee, preference_degree, preference_matrix, flows, Promethee

.. automodule:: mcdakit.methods.vikor
   :members: vikor, Vikor

.. automodule:: mcdakit.methods.electre
   :members: electre, Electre

.. automodule:: mcdakit.methods.scoring
   :members: simple_scoring, weighted_scoring, saw
```

## Normalisation

```{eval-rst}
.. automodule:: mcdakit.normalization
   :members: normalize, register_normalization, get_normalization, available_normalizations, vector_normalization, minmax_normalization, max_normalization, sum_normalization
```

## Constants

```{eval-rst}
.. py:data:: mcdakit.FRAGILE_THRESHOLD

   ``0.10``. Below this weight tolerance, :func:`~mcdakit.sensitivity` calls a
   decision *fragile*: a re-weighting smaller than a tenth of the total
   unseats the winner.

.. py:data:: mcdakit.ROBUST_THRESHOLD

   ``0.25``. At or above this, the winner survives any plausible
   single-criterion disagreement and is reported as *robust*.

.. py:data:: mcdakit.CONSISTENCY_LIMIT

   ``0.10``. Saaty's rule of thumb: above this consistency ratio the pairwise
   judgements contradict each other badly enough that the derived weights
   should not be relied on. Reported, never enforced.

.. py:data:: mcdakit.RANDOM_INDEX

   Saaty's random consistency index by matrix order — the average consistency
   index of a randomly filled *n* × *n* reciprocal matrix. Dividing by it is
   what makes the consistency ratio comparable across problem sizes.

.. py:data:: mcdakit.SAATY_SCALE

   The fundamental 1–9 comparison scale, as ``{ratio: meaning}``.

.. py:data:: mcdakit.NORMALIZATIONS

   Registered normalisation schemes, as ``{name: (function, summary)}``. Use
   :func:`~mcdakit.available_normalizations` for the readable form.
```

## Orientation

```{eval-rst}
.. automodule:: mcdakit.orientation
   :members: orient
```
