"""Tests for galoop.calibrate.

The calibration code drives the full calculator pipeline (MACE/VASP) on
reference molecules and the bare slab. We don't want tests that load MACE,
so we monkeypatch ``_run_pipeline`` to return canned energies and exercise
the surrounding multi-element resolution and config-mutation logic.
"""

from __future__ import annotations

import pytest

from galoop import calibrate as calibrate_mod
from galoop.calibrate import (
    _decompose_formula,
    _resolve_elemental_potentials,
    calibrate,
)


# ---------------------------------------------------------------------------
# _decompose_formula
# ---------------------------------------------------------------------------

class TestDecomposeFormula:

    def test_single_element(self):
        assert _decompose_formula("O") == {"O": 1}
        assert _decompose_formula("H") == {"H": 1}

    def test_diatomic_same_element(self):
        # parse_formula spells "H2" as ["H", "H"]
        assert _decompose_formula("H2") == {"H": 2}

    def test_simple_compound(self):
        assert _decompose_formula("OH") == {"O": 1, "H": 1}
        assert _decompose_formula("H2O") == {"H": 2, "O": 1}

    def test_polyatomic(self):
        result = _decompose_formula("NH3")
        assert result == {"N": 1, "H": 3}

    def test_mixed_composition(self):
        # CH3OH = C(1) H(3) O(1) H(1) = C(1) H(4) O(1)
        result = _decompose_formula("CH3OH")
        assert result == {"C": 1, "H": 4, "O": 1}


# ---------------------------------------------------------------------------
# _resolve_elemental_potentials with a stubbed pipeline runner
# ---------------------------------------------------------------------------

class _StubConfig:
    """Minimal config object used for the resolver — only the resolver
    only inspects ``calculator_stages`` indirectly via the (stubbed)
    pipeline call, so we don't need a real GaloopConfig here."""


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Patch _run_pipeline to return canned energies per molecule directory.

    The directory's last path component is the molecule formula
    (cal_dir / formula), so we use that as the lookup key.
    """
    canned = {
        "H2": -6.7,    # mu_H = -3.35
        "N2": -16.0,   # mu_N = -8.0
        "H2O": -14.5,  # mu_O = -14.5 - 2*-3.35 = -7.8
        "CH4": -24.0,  # mu_C = -24 - 4*-3.35 = -10.6
    }

    def fake_run_pipeline(atoms, struct_dir, config, n_slab_atoms=0):
        formula = struct_dir.name
        if formula not in canned:
            raise RuntimeError(f"unexpected formula {formula}")
        return canned[formula]

    monkeypatch.setattr(calibrate_mod, "_run_pipeline", fake_run_pipeline)
    return canned


class TestResolveElementalPotentials:

    def test_single_element_h_only(self, stub_pipeline, tmp_path):
        mu, mol_e = _resolve_elemental_potentials(
            {"H"}, _StubConfig(), tmp_path,
        )
        assert "H" in mu
        assert mu["H"] == pytest.approx(-3.35)
        assert mol_e["H2"] == pytest.approx(-6.7)
        # Other molecules should NOT have been relaxed
        assert "H2O" not in mol_e
        assert "CH4" not in mol_e

    def test_n_only(self, stub_pipeline, tmp_path):
        mu, mol_e = _resolve_elemental_potentials(
            {"N"}, _StubConfig(), tmp_path,
        )
        assert mu["N"] == pytest.approx(-8.0)

    def test_o_requires_h_solved_first(self, stub_pipeline, tmp_path):
        # H2O has both O and H — resolver must do H2 before H2O
        mu, mol_e = _resolve_elemental_potentials(
            {"O"}, _StubConfig(), tmp_path,
        )
        # H2 was relaxed even though only O was requested, because H2O
        # depends on mu_H. mu_H must be in the result for the math to work.
        assert "H" in mu
        assert "O" in mu
        # mu_O = -14.5 - 2*(-3.35) = -7.8
        assert mu["O"] == pytest.approx(-7.8)

    def test_c_requires_h_solved_first(self, stub_pipeline, tmp_path):
        mu, mol_e = _resolve_elemental_potentials(
            {"C"}, _StubConfig(), tmp_path,
        )
        # mu_C = -24 - 4*(-3.35) = -10.6
        assert mu["C"] == pytest.approx(-10.6)
        assert "H" in mu

    def test_all_four_elements(self, stub_pipeline, tmp_path):
        mu, mol_e = _resolve_elemental_potentials(
            {"H", "N", "O", "C"}, _StubConfig(), tmp_path,
        )
        assert set(mu.keys()) == {"H", "N", "O", "C"}
        assert set(mol_e.keys()) == {"H2", "N2", "H2O", "CH4"}

    def test_unknown_element_raises(self, stub_pipeline, tmp_path):
        with pytest.raises(ValueError, match="Cannot auto-derive"):
            _resolve_elemental_potentials(
                {"Pt"}, _StubConfig(), tmp_path,
            )


# ---------------------------------------------------------------------------
# Top-level calibrate() with stubbed pipeline
# ---------------------------------------------------------------------------

class TestCalibrate:

    def test_short_circuit_when_nothing_needs_calibration(
        self, monkeypatch, tmp_path, minimal_config,
    ):
        # minimal_config already has slab.energy and chemical_potentials set
        called = []

        def fake_run_pipeline(*args, **kwargs):
            called.append(args)
            return -1.0

        monkeypatch.setattr(calibrate_mod, "_run_pipeline", fake_run_pipeline)
        result = calibrate(minimal_config, run_dir=tmp_path)
        assert result == {}
        assert called == []  # nothing relaxed

    def test_calibrates_slab_when_energy_missing(
        self, monkeypatch, tmp_path, minimal_config,
    ):
        minimal_config.slab.energy = None
        # Keep chemical potentials set so we only test the slab branch
        for ads in minimal_config.adsorbates:
            ads.chemical_potential = -1.0

        def fake_run_pipeline(atoms, struct_dir, config, n_slab_atoms=0):
            assert struct_dir.name == "slab"
            return -123.456

        monkeypatch.setattr(calibrate_mod, "_run_pipeline", fake_run_pipeline)
        result = calibrate(minimal_config, run_dir=tmp_path)
        assert result["slab_energy"] == -123.456
        assert minimal_config.slab.energy == -123.456
        assert (tmp_path / "calibration" / "reference_energies.txt").exists()

    def test_calibrates_adsorbate_chemical_potentials(
        self, monkeypatch, tmp_path, minimal_config, stub_pipeline,
    ):
        # Force the molecule path: drop both chemical potentials
        for ads in minimal_config.adsorbates:
            ads.chemical_potential = None
        # Slab energy already set, so only the molecule path runs
        result = calibrate(minimal_config, run_dir=tmp_path)
        # mu(O) = -7.8, mu(OH) = mu(O) + mu(H) = -7.8 + -3.35 = -11.15
        assert result["mu_O"] == pytest.approx(-7.8)
        assert result["mu_OH"] == pytest.approx(-11.15)
        # Adsorbate config was mutated in place
        ads_by_symbol = {a.symbol: a for a in minimal_config.adsorbates}
        assert ads_by_symbol["O"].chemical_potential == pytest.approx(-7.8)
        assert ads_by_symbol["OH"].chemical_potential == pytest.approx(-11.15)
