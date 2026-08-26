"""The method registry: how names resolve to implementations.

Methods are looked up by name at call time rather than hardcoded into
:func:`~mcdakit.ranking.rank`, so a method defined outside this package is a
first-class citizen — indistinguishable from a built-in once registered.

Two ways in. Register directly, for a method defined in your own program::

    from mcdakit import register
    register(MyMethod())

Or declare an entry point, for a method you distribute as a package. It is
then discovered automatically the first time anything asks the registry a
question, with no import or call needed by the user::

    # in your pyproject.toml
    [project.entry-points."mcdakit.methods"]
    my_method = "my_package.methods:MyMethod"

The entry point may name either a :class:`~mcdakit.methods.base.Method`
subclass (instantiated with no arguments) or an already-built instance.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

from ..types import McdaError
from .base import Method

#: The entry-point group third-party packages advertise methods under.
ENTRY_POINT_GROUP = "mcdakit.methods"

# Name -> instance. Insertion-ordered, so `available()` lists the built-ins in
# the order they were registered and plugins after them.
_METHODS: dict = {}

# Entry-point scanning is deferred to first use rather than done at import, so
# `import mcdakit` stays fast and a broken plugin cannot break the import.
_PLUGINS_LOADED = False


class DuplicateMethodError(McdaError):
    """A second method tried to claim a name already in use."""


class PluginLoadError(UserWarning):
    """A package advertised a method that could not be loaded.

    A warning rather than an exception: one broken plugin must not stop the
    library — and every other plugin — from working. Its own category so a
    strict caller can promote it with
    ``warnings.simplefilter("error", PluginLoadError)``.
    """


def register(method: Method, *, replace: bool = False) -> Method:
    """Add a method to the registry, and return it.

    Parameters
    ----------
    method:
        An instance, not a class — methods may carry configuration, and
        registering an instance keeps that possibility open.
    replace:
        Overwrite an existing registration of the same name. Off by default:
        silently shadowing a method would make ``rank(..., method="topsis")``
        mean different things in different processes, which is exactly the
        kind of ambiguity this library exists to remove.

    Raises
    ------
    McdaError
        If ``method`` is not a :class:`~mcdakit.methods.base.Method`, has no
        name, or the name is taken and ``replace`` is false.
    """
    if not isinstance(method, Method):
        raise McdaError(
            f"register() takes a Method instance, got "
            f"{type(method).__name__}. Subclass Method and pass an instance, "
            f"not the class itself."
        )
    name = getattr(method, "name", "")
    if not name or not isinstance(name, str) or not name.strip():
        raise McdaError(
            f"{type(method).__name__} needs a non-empty `name` attribute to "
            f"be registered; that is what callers pass as method=."
        )

    if name in _METHODS and not replace:
        existing = _METHODS[name]
        raise DuplicateMethodError(
            f"A method named {name!r} is already registered "
            f"({type(existing).__name__}). Choose another name, or pass "
            f"replace=True to override it deliberately."
        )

    _METHODS[name] = method
    return method


def unregister(name: str) -> None:
    """Remove a method. Raises if it was not registered."""
    _load_plugins()
    if name not in _METHODS:
        raise McdaError(f"No method named {name!r} to unregister.")
    del _METHODS[name]


def get(name: str) -> Method:
    """Look up a method by name.

    Raises
    ------
    McdaError
        With the available names listed, since a typo is the overwhelmingly
        likely cause.
    """
    _load_plugins()
    try:
        return _METHODS[name]
    except KeyError:
        raise McdaError(
            f"Unknown method {name!r}. Available: {', '.join(sorted(_METHODS))}."
        ) from None


def has(name: str) -> bool:
    """Whether a method of this name is registered."""
    _load_plugins()
    return name in _METHODS


def names() -> tuple:
    """Every registered method name, in registration order."""
    _load_plugins()
    return tuple(_METHODS)


def available() -> dict:
    """``{name: summary}`` for every registered method.

    The human-facing counterpart to :func:`names` — what to print when someone
    asks what this installation can do.
    """
    _load_plugins()
    return {
        name: method.summary or method.__doc__ or ""
        for name, method in _METHODS.items()
    }


def methods() -> Iterator:
    """Iterate over the registered method instances."""
    _load_plugins()
    return iter(_METHODS.values())


def _load_plugins() -> None:
    """Discover entry-point methods, once per process."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    # Set before scanning: a plugin that imports mcdakit while being loaded
    # would otherwise recurse into this function forever.
    _PLUGINS_LOADED = True

    for entry_point in _entry_points():
        try:
            loaded = entry_point.load()
            method = loaded() if isinstance(loaded, type) else loaded
            register(method)
        except Exception as exc:
            # One bad plugin must not take down the registry, so every failure
            # mode is caught and reported rather than raised.
            warnings.warn(
                f"Could not load the method advertised by "
                f"{entry_point.name!r} ({entry_point.value}): "
                f"{type(exc).__name__}: {exc}",
                PluginLoadError,
                stacklevel=2,
            )


def _entry_points():
    """Entry points in our group, across the supported Python versions."""
    from importlib.metadata import entry_points

    try:
        # Python 3.10+ selectable API.
        return tuple(entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:  # pragma: no cover - Python 3.9
        return tuple(entry_points().get(ENTRY_POINT_GROUP, ()))


def _reset_for_testing() -> None:
    """Clear the registry and re-arm plugin discovery.

    Test-support only; not part of the public API.
    """
    global _PLUGINS_LOADED
    _METHODS.clear()
    _PLUGINS_LOADED = False


__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateMethodError",
    "PluginLoadError",
    "available",
    "get",
    "has",
    "methods",
    "names",
    "register",
    "unregister",
]
