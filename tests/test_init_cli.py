"""Tests for the ``galoop init`` scaffolding command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from galoop.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def slab_file(tmp_path):
    """A real POSCAR so --slab has something to copy."""
    from ase.build import fcc111
    from ase.io import write

    path = tmp_path / "source_slab.vasp"
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    write(str(path), slab, format="vasp")
    return path


class TestInitBasic:
    def test_writes_minimal_yaml(self, runner, tmp_path):
        target = tmp_path / "new_run"
        result = runner.invoke(cli, ["init", str(target)])
        assert result.exit_code == 0, result.output
        yaml_path = target / "galoop.yaml"
        assert yaml_path.exists()
        text = yaml_path.read_text()
        # Required top-level blocks present
        for key in ("slab:", "adsorbates:", "calculator_stages:",
                    "scheduler:", "ga:", "conditions:", "fingerprint:"):
            assert key in text, f"missing '{key}' in generated yaml"

    def test_warns_when_no_slab_given(self, runner, tmp_path):
        target = tmp_path / "no_slab"
        result = runner.invoke(cli, ["init", str(target)])
        assert result.exit_code == 0
        assert "warning" in result.output.lower()
        assert "<PATH_TO_YOUR_SLAB_FILE>" in (target / "galoop.yaml").read_text()

    def test_slab_copied_and_linked(self, runner, tmp_path, slab_file):
        target = tmp_path / "with_slab"
        result = runner.invoke(cli, ["init", str(target), "-s", str(slab_file)])
        assert result.exit_code == 0, result.output
        dst = target / "slab.vasp"
        assert dst.exists()
        # YAML points at the copied file, not the source
        text = (target / "galoop.yaml").read_text()
        assert str(dst) in text
        assert str(slab_file) not in text

    def test_existing_yaml_without_force_errors(self, runner, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        (target / "galoop.yaml").write_text("existing content")
        result = runner.invoke(cli, ["init", str(target)])
        assert result.exit_code == 1
        assert "already exists" in result.output
        # Content was preserved, not overwritten
        assert (target / "galoop.yaml").read_text() == "existing content"

    def test_force_overwrites(self, runner, tmp_path):
        target = tmp_path / "force"
        target.mkdir()
        (target / "galoop.yaml").write_text("existing content")
        result = runner.invoke(cli, ["init", str(target), "--force"])
        assert result.exit_code == 0
        assert (target / "galoop.yaml").read_text() != "existing content"


class TestInitBackends:
    def test_mace_backend_default(self, runner, tmp_path):
        target = tmp_path / "mace"
        runner.invoke(cli, ["init", str(target)])
        text = (target / "galoop.yaml").read_text()
        assert "type: mace" in text
        assert "model: small" in text
        assert "device: cuda" in text

    def test_vasp_backend(self, runner, tmp_path):
        target = tmp_path / "vasp"
        runner.invoke(cli, ["init", str(target), "-b", "vasp"])
        text = (target / "galoop.yaml").read_text()
        assert "type: vasp" in text
        assert "incar:" in text
        assert "ENCUT" in text

    def test_custom_backend_auto_writes_calc_template(self, runner, tmp_path):
        target = tmp_path / "custom"
        result = runner.invoke(cli, ["init", str(target), "-b", "custom"])
        assert result.exit_code == 0
        yaml_text = (target / "galoop.yaml").read_text()
        assert "type: calc:make_calculator" in yaml_text
        # calc.py written automatically for custom backend
        calc_path = target / "calc.py"
        assert calc_path.exists()
        calc_text = calc_path.read_text()
        assert "def make_calculator(params: dict)" in calc_text
        assert "NotImplementedError" in calc_text  # placeholder safeguard

    def test_calc_template_flag_explicit(self, runner, tmp_path):
        target = tmp_path / "explicit_calc"
        result = runner.invoke(cli, ["init", str(target), "--calc-template"])
        assert result.exit_code == 0
        assert (target / "calc.py").exists()


class TestInitProducesValidYaml:
    def test_generated_yaml_has_new_style_params(self, runner, tmp_path, slab_file):
        """The scaffolded config must use the new params-dict shape, not
        legacy top-level mace_model fields."""
        target = tmp_path / "validate"
        runner.invoke(cli, ["init", str(target), "-s", str(slab_file)])
        text = (target / "galoop.yaml").read_text()
        # Must not reintroduce the legacy fields
        for legacy in ("mace_model:", "mace_device:", "mace_dtype:"):
            assert legacy not in text, f"scaffolded yaml contains legacy {legacy}"
        # Must expose the new params: block
        assert "params:" in text
