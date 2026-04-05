"""
tests/test_import_paths.py

Verify that the codebase uses 'galoop' everywhere, and every module imports.
"""

import ast
from pathlib import Path

import pytest


def _galoop_root() -> Path:
    return Path(__file__).resolve().parent.parent / "galoop"


def _python_files(root: Path):
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in str(p)
    )


class TestNoGociaImports:
    """The old package name 'gocia' must not appear in any import."""

    def test_no_gocia_imports(self):
        root = _galoop_root()
        violations = []
        for pyfile in _python_files(root):
            source = pyfile.read_text()
            tree = ast.parse(source, filename=str(pyfile))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gocia"):
                    violations.append(f"  {pyfile.relative_to(root.parent)}:{node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("gocia"):
                            violations.append(f"  {pyfile.relative_to(root.parent)}:{node.lineno}")
        assert not violations, "Found 'gocia' imports:\n" + "\n".join(violations)

    def test_no_gocia_in_source_strings(self):
        """Even deferred imports inside f-strings or comments shouldn't reference gocia."""
        root = _galoop_root()
        for pyfile in _python_files(root):
            source = pyfile.read_text()
            assert "from gocia" not in source, f"'from gocia' found in {pyfile.name}"


class TestAllModulesImport:

    def test_individual(self):
        from galoop.individual import Individual, STATUS, OPERATOR
        assert STATUS.PENDING == "pending"
        assert OPERATOR.INIT == "init"

    def test_database(self):
        # database.py is now a shim; GaloopProject is the live implementation
        from galoop.database import diff_configs
        from galoop.project import GaloopProject, diff_configs as dc2
        assert callable(diff_configs)

    def test_fingerprint(self):
        from galoop.fingerprint import compute_soap, tanimoto_similarity, classify_postrelax

    def test_config(self):
        from galoop.config import load_config, GaloopConfig

    def test_cli(self):
        from galoop.cli import cli

    def test_calculator(self):
        from galoop.engine.calculator import Pipeline, CalculatorStage, build_pipeline

    def test_scheduler(self):
        # scheduler.py is now a stub; row handles job submission via workflow.toml
        import galoop.engine.scheduler  # must at least import cleanly

    def test_surface(self):
        from galoop.science.surface import (
            load_slab, load_adsorbate, place_adsorbate,
            parse_formula, check_clash, detect_desorption, orient_upright,
        )

    def test_energy(self):
        from galoop.science.energy import grand_canonical_energy

    def test_reproduce(self):
        from galoop.science.reproduce import splice, merge, mutate_add, mutate_remove, mutate_displace
