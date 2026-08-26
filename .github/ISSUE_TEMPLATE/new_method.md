---
name: Propose a method
about: Suggest an MCDA method to add, or announce one you have implemented
title: "Method: "
labels: method
---

## The method

**Name:**
**Reference:** (paper, with year — this library is read by people who know them)

## Does it need to live here?

A method distributed as a plugin is a first-class citizen: reachable by name
from `rank()`, included in `compare_methods()`, analysable by `sensitivity()`.
See `docs/extending.md`. Adding it to core is worth it when the method is
widely used, or when it needs something the extension contract cannot express.

- [ ] I have read `docs/extending.md`
- [ ] This cannot be done as a plugin, because:

## Conformance

- [ ] `python -m mcdakit.testing my_module:MyMethod` passes

If it does not, paste the output — a failing check usually names the fix.

## Verifiable values

Every method here is tested against numbers verified **outside** this
codebase. Which do you have?

- [ ] A worked example from a published paper (cite table/section)
- [ ] A hand computation
- [ ] A cross-check against another library (name it and its version)
