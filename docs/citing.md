# Citing mcdakit

If you use `mcdakit` in academic work, please cite it. The repository's
`CITATION.cff` holds the metadata, and GitHub renders a "Cite this
repository" button from it:

```{literalinclude} ../CITATION.cff
:language: yaml
```

`preferred-citation` names the paper describing the software. It is recorded
as submitted and not yet published, so the journal volume, pages and DOI are
still to be announced; citation tools reading this file will reproduce that
status rather than implying a publication that does not yet exist.

## Citing a method, not the package

Every method names its source in its docstring, and those are the citations a
methods paper needs. The package implements them; it did not invent them.
SPOTIS is due to Dezert et al. (2020), TOPSIS to Hwang and Yoon (1981), VIKOR
to Opricovic and Tzeng (2004), PROMETHEE to Brans and Vincke (1985), ELECTRE
to Roy (1968), and AHP to Saaty (1980). The remainder are listed in the
[API reference](api.md) with their references.

`mcdakit` is released under the Apache License 2.0.
