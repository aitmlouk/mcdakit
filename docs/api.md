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

## Sensitivity analysis

```{eval-rst}
.. automodule:: mcdakit.sensitivity
   :members: sensitivity
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

## Orientation

```{eval-rst}
.. automodule:: mcdakit.orientation
   :members: orient
```
