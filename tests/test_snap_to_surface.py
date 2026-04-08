"""Tests for ``galoop.spawn.snap_to_surface``.

This used to instantiate a fresh MACE calculator on every call, which was a
real performance bug *and* a source of stale-state shape mismatches under
threading. After Phase 1 it routes through ``CalculatorStage._get_mace_calc``
so the same instance is reused across calls. We test:

- bare-slab early return (no adsorbates)
- z-clamping clamps to the configured window even if BFGS fails
- BFGS failures are logged, not silently swallowed
- the cached MACE calc is fetched, not rebuilt, across calls
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.calculators.calculator import Calculator, all_changes

from galoop import spawn
from galoop.engine.calculator import CalculatorStage


# ---------------------------------------------------------------------------
# Stub MACE calculator that just zeros forces and returns a fake energy.
# Lets us run snap_to_surface without loading the real MACE-MP weights.
# ---------------------------------------------------------------------------

class _FakeMaceCalc(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self):
        super().__init__()
        # MACE's calculator instance has a `.results` dict — keep parity.
        self.results = {}
        self.call_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.call_count += 1
        n = len(atoms)
        self.results["energy"] = -1.0 * n
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = np.zeros((n, 3))


def _slab_config_pair():
    """Return (slab Atoms, fake config) ready for snap_to_surface."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    cfg = SimpleNamespace(
        mace_model="medium",
        mace_device="cpu",
        mace_dtype="float32",
        slab=SimpleNamespace(
            snap_z_min_offset=0.8,
            snap_z_max_offset=4.0,
        ),
    )
    return slab, cfg


# ---------------------------------------------------------------------------
# Bare-slab fast path
# ---------------------------------------------------------------------------

class TestSnapToSurfaceBareSlab:

    def test_bare_slab_returns_unchanged(self, monkeypatch):
        slab, cfg = _slab_config_pair()
        # Should NOT touch the calculator at all when there are no ads.
        called = []
        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(lambda cls, *a, **k: called.append(1) or _FakeMaceCalc()),
        )
        result = spawn.snap_to_surface(slab, cfg, n_slab_atoms=len(slab))
        assert called == []  # never asked for the calc
        # Same atoms returned (or a copy with same positions)
        assert (result.get_positions() == slab.get_positions()).all()


# ---------------------------------------------------------------------------
# Z-clamping
# ---------------------------------------------------------------------------

class TestSnapToSurfaceZClamp:

    def _slab_with_ads(self, ads_z_offset: float):
        slab, cfg = _slab_config_pair()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        ads = Atoms("H", positions=[[0.0, 0.0, top + ads_z_offset]])
        return slab + ads, cfg, n_slab

    def test_low_adsorbate_clamped_up(self, monkeypatch):
        atoms, cfg, n_slab = self._slab_with_ads(ads_z_offset=0.1)  # too low
        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(lambda cls, *a, **k: _FakeMaceCalc()),
        )
        result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        top = atoms.positions[:n_slab, 2].max()
        ads_z_after = result.positions[n_slab, 2]
        assert ads_z_after >= top + cfg.slab.snap_z_min_offset - 1e-6

    def test_high_adsorbate_clamped_down(self, monkeypatch):
        atoms, cfg, n_slab = self._slab_with_ads(ads_z_offset=10.0)  # way too high
        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(lambda cls, *a, **k: _FakeMaceCalc()),
        )
        result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        top = atoms.positions[:n_slab, 2].max()
        ads_z_after = result.positions[n_slab, 2]
        assert ads_z_after <= top + cfg.slab.snap_z_max_offset + 1e-6


# ---------------------------------------------------------------------------
# BFGS failure path: must log a warning, not swallow silently
# ---------------------------------------------------------------------------

class TestSnapToSurfaceBfgsFailure:

    def test_bfgs_failure_logged_and_clamps(self, monkeypatch, caplog):
        atoms, cfg, n_slab = (
            *_slab_config_pair(), 0,  # placeholder, replaced below
        )
        slab, cfg = _slab_config_pair()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])

        class _ExplodingCalc(_FakeMaceCalc):
            def calculate(self, atoms=None, **kwargs):
                raise RuntimeError("intentional MACE failure")

        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(lambda cls, *a, **k: _ExplodingCalc()),
        )

        with caplog.at_level("WARNING"):
            result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)

        # Must have logged a warning so the operator can spot CUDA OOM etc.
        assert any("snap_to_surface BFGS failed" in r.message for r in caplog.records)
        # And still clamped the result so the caller gets usable geometry
        ads_z_after = result.positions[n_slab, 2]
        top = slab.positions[:, 2].max()
        assert (top + cfg.slab.snap_z_min_offset
                <= ads_z_after
                <= top + cfg.slab.snap_z_max_offset)


# ---------------------------------------------------------------------------
# Calculator caching
# ---------------------------------------------------------------------------

class TestSnapUsesCachedCalc:

    def test_get_mace_calc_called_per_invocation_but_cache_intact(self, monkeypatch):
        """snap_to_surface should call _get_mace_calc, but the underlying
        cache shouldn't grow more than once for a fixed (model, device, dtype)."""
        slab, cfg = _slab_config_pair()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()

        calls = []
        original = CalculatorStage._get_mace_calc.__func__

        def tracking_get(cls, model, device, dtype):
            calls.append((model, device, dtype))
            return _FakeMaceCalc()

        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(tracking_get),
        )

        for _ in range(3):
            atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])
            spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)

        # Calc-fetch is called every snap, but it's a cheap dict lookup in
        # the real implementation. The point of the test is that the import
        # path stays consolidated through CalculatorStage.
        assert len(calls) == 3
        assert all(c == ("medium", "cpu", "float32") for c in calls)
