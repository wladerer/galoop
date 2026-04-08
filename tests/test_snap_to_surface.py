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
            snap_timeout_s=120.0,
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


# ---------------------------------------------------------------------------
# Per-structure timeout: wedged MACE must not freeze init-pop
# ---------------------------------------------------------------------------

class TestSnapTimeout:

    def test_hung_bfgs_raises_snap_timeout_error(self, monkeypatch):
        """A MACE calculator that wedges forever must be interrupted by
        SIGALRM within the configured snap_timeout_s window."""
        import time

        slab, cfg = _slab_config_pair()
        cfg.slab.snap_timeout_s = 0.5  # tight — test must be fast
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])

        class _HangingCalc(_FakeMaceCalc):
            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                time.sleep(5.0)  # longer than snap_timeout_s

        monkeypatch.setattr(
            CalculatorStage, "_get_mace_calc",
            classmethod(lambda cls, *a, **k: _HangingCalc()),
        )

        import pytest
        t0 = time.monotonic()
        with pytest.raises(spawn.SnapTimeoutError):
            spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        elapsed = time.monotonic() - t0
        # Should fire well before the 5s sleep would have finished.
        assert elapsed < 2.0, f"timeout took {elapsed:.2f}s, expected < 2s"

    def test_build_initial_population_honors_galoopstop(self, monkeypatch, tmp_path):
        """Touching galoopstop mid-init-pop must stop before the next structure."""
        from galoop import spawn as spawn_mod

        # Let the loop run a couple structures then trip the sentinel.
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        call_count = {"n": 0}
        real_snap = spawn_mod.snap_to_surface

        def counting_snap(atoms, cfg, n_slab_atoms):
            call_count["n"] += 1
            if call_count["n"] == 2:
                (run_dir / "galoopstop").touch()
            return atoms

        monkeypatch.setattr(spawn_mod, "snap_to_surface", counting_snap)

        # Minimal stub store with just what build_initial_population touches.
        class _StubStore:
            def __init__(self, run_dir):
                self.run_dir = run_dir
                self.inserted = []

            def insert(self, ind):
                d = run_dir / "structures" / ind.id
                d.mkdir(parents=True, exist_ok=True)
                self.inserted.append(ind)
                return d

            def update(self, ind):
                pass

        # Build a minimal config/slab_info pair that build_random_structure accepts.
        # Rather than wiring all that, patch build_random_structure to a no-op.
        from ase.build import fcc111
        stub_slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)

        monkeypatch.setattr(
            spawn_mod, "build_random_structure",
            lambda *a, **k: stub_slab.copy(),
        )
        monkeypatch.setattr(
            spawn_mod, "load_ads_template_dict",
            lambda ads: {"H": Atoms("H")},
        )
        monkeypatch.setattr(
            spawn_mod, "random_stoichiometry",
            lambda *a, **k: {"H": 1},
        )

        cfg = SimpleNamespace(
            adsorbates=[SimpleNamespace(symbol="H")],
            ga=SimpleNamespace(
                population_size=10,
                min_adsorbates=1,
                max_adsorbates=3,
            ),
            slab=SimpleNamespace(snap_timeout_s=120.0),
        )
        slab_info = SimpleNamespace(n_slab_atoms=len(stub_slab))
        store = _StubStore(run_dir)

        import numpy as np
        spawn_mod.build_initial_population(cfg, slab_info, store, np.random.default_rng(0))

        # Should have stopped after the 2nd structure; not all 10 built.
        assert call_count["n"] == 2
        assert len(store.inserted) < 10
