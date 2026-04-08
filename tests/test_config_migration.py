"""Tests for the config schema after the pluggable-backends refactor.

The refactor was a clean break: top-level ``mace_model`` / ``mace_device`` /
``mace_dtype`` are no longer accepted. Users must put them in the stage's
``params`` dict. These tests verify both the rejection of old configs and
the acceptance of new-style ones.
"""

from __future__ import annotations

import pytest


def _minimal_valid(tmp_path):
    """Build a minimal but valid new-style config dict."""
    from ase.build import fcc111
    from ase.io import write

    slab_path = tmp_path / "slab.vasp"
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    write(str(slab_path), slab, format="vasp")
    top_z = float(slab.get_positions()[:, 2].max())

    return {
        "slab": {
            "geometry": str(slab_path),
            "energy": -1000.0,
            "sampling_zmin": top_z + 0.5,
            "sampling_zmax": top_z + 3.5,
        },
        "adsorbates": [
            {"symbol": "H", "chemical_potential": -3.4, "min_count": 0, "max_count": 2},
        ],
        "calculator_stages": [{
            "name": "preopt",
            "type": "mace",
            "params": {"model": "small", "device": "cpu", "dtype": "float32"},
        }],
    }


class TestLegacyFieldsRejected:
    def test_top_level_mace_model_rejected(self, tmp_path):
        from galoop.config import GaloopConfig
        data = _minimal_valid(tmp_path)
        data["mace_model"] = "small"
        with pytest.raises(ValueError, match="no longer supported"):
            GaloopConfig.model_validate(data)

    def test_top_level_mace_device_rejected(self, tmp_path):
        from galoop.config import GaloopConfig
        data = _minimal_valid(tmp_path)
        data["mace_device"] = "cuda"
        with pytest.raises(ValueError, match="no longer supported"):
            GaloopConfig.model_validate(data)


class TestNewStyleAccepted:
    def test_mace_stage_with_params_validates(self, tmp_path):
        from galoop.config import GaloopConfig
        cfg = GaloopConfig.model_validate(_minimal_valid(tmp_path))
        assert cfg.calculator_stages[0].type == "mace"
        assert cfg.calculator_stages[0].params["model"] == "small"

    def test_import_path_type_validates(self, tmp_path):
        """StageConfig should accept any string for `type` — backend
        resolution happens at CalculatorStage construction, not at
        config-validate time."""
        from galoop.config import GaloopConfig
        data = _minimal_valid(tmp_path)
        data["calculator_stages"][0]["type"] = "my_pkg.mod:my_factory"
        data["calculator_stages"][0]["params"] = {"checkpoint": "/tmp/x"}
        cfg = GaloopConfig.model_validate(data)
        assert cfg.calculator_stages[0].type == "my_pkg.mod:my_factory"

    def test_snap_stage_optional(self, tmp_path):
        from galoop.config import GaloopConfig
        data = _minimal_valid(tmp_path)
        cfg = GaloopConfig.model_validate(data)
        assert cfg.snap_stage is None

    def test_snap_stage_can_be_set(self, tmp_path):
        from galoop.config import GaloopConfig
        data = _minimal_valid(tmp_path)
        data["snap_stage"] = {
            "name": "snap",
            "type": "mace",
            "params": {"model": "small", "device": "cpu", "dtype": "float32"},
        }
        cfg = GaloopConfig.model_validate(data)
        assert cfg.snap_stage is not None
        assert cfg.snap_stage.params["model"] == "small"
