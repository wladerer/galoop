"""Tests for the pluggable calculator backend registry."""

from __future__ import annotations

import pytest
from ase.calculators.calculator import Calculator

from galoop.engine import backends


# Module-level factories so import-path resolution has something to grab.

def _good_factory(params):
    class _C(Calculator):
        implemented_properties = ["energy"]
    return _C()


def _tuple_factory_target(params):
    class _C(Calculator):
        implemented_properties = ["energy"]
    return _C()


_TUPLE_BACKEND = (_tuple_factory_target, True)


def _not_callable_target():
    return 42  # not called; the module attr itself is looked at


_BAD_ATTR = 42  # deliberately not a callable or tuple


class TestResolveBuiltins:
    def test_mace_builtin_registered(self):
        factory, drives = backends.resolve("mace")
        assert callable(factory)
        assert drives is False

    def test_vasp_builtin_registered(self):
        factory, drives = backends.resolve("vasp")
        assert callable(factory)
        assert drives is True

    def test_case_insensitive_builtin(self):
        factory1, _ = backends.resolve("MACE")
        factory2, _ = backends.resolve("mace")
        assert factory1 is factory2

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown calculator type"):
            backends.resolve("uma")


class TestResolveImportPath:
    def test_bare_callable_import(self):
        factory, drives = backends.resolve("tests.test_backends:_good_factory")
        assert factory is _good_factory
        assert drives is False

    def test_tuple_target_import(self):
        factory, drives = backends.resolve("tests.test_backends:_TUPLE_BACKEND")
        assert factory is _tuple_factory_target
        assert drives is True

    def test_missing_module_raises(self):
        with pytest.raises(ImportError, match="Failed to import backend module"):
            backends.resolve("galoop._definitely_not_a_module:anything")

    def test_missing_attribute_raises(self):
        with pytest.raises(ValueError, match="has no attribute"):
            backends.resolve("tests.test_backends:no_such_symbol")

    def test_bad_attribute_shape_raises(self):
        with pytest.raises(ValueError, match="must resolve to a callable"):
            backends.resolve("tests.test_backends:_BAD_ATTR")

    def test_empty_module_side_raises(self):
        with pytest.raises(ValueError, match="must be 'pkg.mod:callable'"):
            backends.resolve(":func")

    def test_empty_attr_side_raises(self):
        with pytest.raises(ValueError, match="must be 'pkg.mod:callable'"):
            backends.resolve("pkg.mod:")


class TestRegisterMonkeypatch:
    def test_register_adds_entry(self, monkeypatch):
        def fake(params):
            return None  # type: ignore[return-value]
        monkeypatch.setitem(backends._BUILTIN, "fake", (fake, False))
        factory, drives = backends.resolve("fake")
        assert factory is fake
        assert drives is False
