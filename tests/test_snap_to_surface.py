"""Tests for ``galoop.spawn.snap_to_surface``.

snap_to_surface routes through :mod:`galoop.engine.backends` — whatever the
resolved snap stage's ``type`` is, that backend's factory is called to
build the ASE calculator used for the constrained pre-relax. Tests here
monkeypatch the built-in ``mace`` registry entry with a fake factory so we
don't need to load real MACE weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.calculators.calculator import Calculator, all_changes

from galoop import spawn
from galoop.engine import backends


# ---------------------------------------------------------------------------
# Stub calculator that just zeros forces and returns a fake energy.
# ---------------------------------------------------------------------------

class _FakeMaceCalc(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self):
        super().__init__()
        self.results = {}
        self.call_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.call_count += 1
        n = len(atoms)
        self.results["energy"] = -1.0 * n
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = np.zeros((n, 3))


def _make_fake_stage() -> SimpleNamespace:
    """Pydantic-free shim that mimics StageConfig.model_dump()."""
    return SimpleNamespace(
        model_dump=lambda: {
            "name": "preopt",
            "type": "mace",
            "fmax": 0.05,
            "max_steps": 300,
            "fix_slab_first": False,
            "prescan_fmax": None,
            "params": {"model": "medium", "device": "cpu", "dtype": "float32"},
        },
    )


def _slab_config_pair():
    """Return (slab Atoms, fake config) ready for snap_to_surface."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    cfg = SimpleNamespace(
        calculator_stages=[_make_fake_stage()],
        snap_stage=None,
        slab=SimpleNamespace(
            snap_z_min_offset=0.8,
            snap_z_max_offset=4.0,
            snap_timeout_s=120.0,
        ),
    )
    return slab, cfg


def _install_fake_backend(monkeypatch, calc_factory):
    """Swap the built-in ``mace`` backend with a user-supplied factory."""
    monkeypatch.setitem(
        backends._BUILTIN, "mace",
        (lambda params: calc_factory(), False),
    )


# ---------------------------------------------------------------------------
# Bare-slab fast path
# ---------------------------------------------------------------------------

class TestSnapToSurfaceBareSlab:

    def test_bare_slab_returns_unchanged(self, monkeypatch):
        slab, cfg = _slab_config_pair()
        called = []
        _install_fake_backend(
            monkeypatch,
            lambda: (called.append(1), _FakeMaceCalc())[1],
        )
        result = spawn.snap_to_surface(slab, cfg, n_slab_atoms=len(slab))
        assert called == []  # never asked for the calc
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
        atoms, cfg, n_slab = self._slab_with_ads(ads_z_offset=0.1)
        _install_fake_backend(monkeypatch, _FakeMaceCalc)
        result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        top = atoms.positions[:n_slab, 2].max()
        ads_z_after = result.positions[n_slab, 2]
        assert ads_z_after >= top + cfg.slab.snap_z_min_offset - 1e-6

    def test_high_adsorbate_clamped_down(self, monkeypatch):
        atoms, cfg, n_slab = self._slab_with_ads(ads_z_offset=10.0)
        _install_fake_backend(monkeypatch, _FakeMaceCalc)
        result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        top = atoms.positions[:n_slab, 2].max()
        ads_z_after = result.positions[n_slab, 2]
        assert ads_z_after <= top + cfg.slab.snap_z_max_offset + 1e-6


# ---------------------------------------------------------------------------
# BFGS failure path: must log a warning, not swallow silently
# ---------------------------------------------------------------------------

class TestSnapToSurfaceBfgsFailure:

    def test_bfgs_failure_logged_and_clamps(self, monkeypatch, caplog):
        slab, cfg = _slab_config_pair()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])

        class _ExplodingCalc(_FakeMaceCalc):
            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                raise RuntimeError("intentional MACE failure")

        _install_fake_backend(monkeypatch, _ExplodingCalc)

        with caplog.at_level("WARNING"):
            result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)

        assert any("snap_to_surface BFGS failed" in r.message for r in caplog.records)
        ads_z_after = result.positions[n_slab, 2]
        top = slab.positions[:, 2].max()
        assert (top + cfg.slab.snap_z_min_offset
                <= ads_z_after
                <= top + cfg.slab.snap_z_max_offset)


# ---------------------------------------------------------------------------
# Backend dispatch: built-in vs import-path
# ---------------------------------------------------------------------------

# Module-level factory so it's importable via "tests.test_snap_to_surface:..."
def _importable_fake_factory(params):
    return _FakeMaceCalc()


class TestSnapBackendDispatch:

    def test_snap_respects_import_path_backend(self, monkeypatch):
        """snap_to_surface with a 'pkg.mod:func' stage type should route
        through importlib and call the user-supplied factory directly,
        without touching the built-in mace registry entry."""
        slab, _ = _slab_config_pair()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])

        # Poison the built-in 'mace' entry — if resolve() touches it, the
        # test blows up loudly, proving dispatch went to the import path.
        def _poisoned(_params):
            raise AssertionError("built-in mace backend should not be called")
        monkeypatch.setitem(backends._BUILTIN, "mace", (_poisoned, False))

        import_path_stage = SimpleNamespace(model_dump=lambda: {
            "name": "snap",
            "type": "tests.test_snap_to_surface:_importable_fake_factory",
            "fmax": 0.2,
            "max_steps": 30,
            "fix_slab_first": False,
            "prescan_fmax": None,
            "params": {},
        })
        cfg = SimpleNamespace(
            calculator_stages=[_make_fake_stage()],
            snap_stage=import_path_stage,
            slab=SimpleNamespace(
                snap_z_min_offset=0.8,
                snap_z_max_offset=4.0,
                snap_timeout_s=120.0,
            ),
        )

        result = spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        top = slab.positions[:, 2].max()
        ads_z_after = result.positions[n_slab, 2]
        assert (top + cfg.slab.snap_z_min_offset
                <= ads_z_after
                <= top + cfg.slab.snap_z_max_offset)


# ---------------------------------------------------------------------------
# Per-structure timeout: wedged MACE must not freeze init-pop
# ---------------------------------------------------------------------------

class TestSnapTimeout:

    def test_hung_bfgs_raises_snap_timeout_error(self, monkeypatch):
        """A calculator that wedges forever must be interrupted by SIGALRM
        within the configured snap_timeout_s window."""
        import time

        slab, cfg = _slab_config_pair()
        cfg.slab.snap_timeout_s = 0.5
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        atoms = slab + Atoms("H", positions=[[0.0, 0.0, top + 2.0]])

        class _HangingCalc(_FakeMaceCalc):
            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                time.sleep(5.0)

        _install_fake_backend(monkeypatch, _HangingCalc)

        import pytest
        t0 = time.monotonic()
        with pytest.raises(spawn.SnapTimeoutError):
            spawn.snap_to_surface(atoms, cfg, n_slab_atoms=n_slab)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"timeout took {elapsed:.2f}s, expected < 2s"

    def test_build_initial_population_honors_galoopstop(self, monkeypatch, tmp_path):
        """Touching galoopstop mid-init-pop must stop before the next structure."""
        from galoop import spawn as spawn_mod

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        call_count = {"n": 0}

        def counting_snap(atoms, cfg, n_slab_atoms, snap_stage=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                (run_dir / "galoopstop").touch()
            return atoms

        monkeypatch.setattr(spawn_mod, "snap_to_surface", counting_snap)

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
            calculator_stages=[_make_fake_stage()],
            snap_stage=None,
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

        assert call_count["n"] == 2
        assert len(store.inserted) < 10
