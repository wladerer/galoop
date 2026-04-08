"""Tests for ``validate_surface_binding`` with multi-atom adsorbates.

The post-relax binding check used to be exercised only with single-atom O on
Cu in test_population.py. After Phase 1 it's load-bearing for the new
``galoop sample`` command and the GA harvest pipeline, and we want explicit
coverage of multi-atom (CO, OH, NH3) cases plus the unbound failure path.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111

from galoop.science.surface import validate_surface_binding


def _cu_slab(size=(3, 3, 3)):
    slab = fcc111("Cu", size=size, vacuum=10.0, periodic=True)
    return slab


def _top_layer_indices(slab: Atoms) -> list[int]:
    """Return indices of top-layer slab atoms (highest z)."""
    z = slab.positions[:, 2]
    top_z = z.max()
    return [i for i, zi in enumerate(z) if abs(zi - top_z) < 0.1]


def _add_atom(slab: Atoms, symbol: str, top_idx: int, z_offset: float) -> Atoms:
    """Place a single atom *z_offset* above slab's *top_idx*-th top-layer atom."""
    out = slab.copy()
    top_indices = _top_layer_indices(slab)
    anchor = slab.positions[top_indices[top_idx]].copy()
    anchor[2] = slab.positions[:, 2].max() + z_offset
    out += Atoms(symbol, positions=[anchor])
    return out


# ---------------------------------------------------------------------------
# Single-atom adsorbates (CO removed: see CO test below)
# ---------------------------------------------------------------------------

class TestSingleAtomBinding:

    def test_bare_slab_returns_bound(self):
        slab = _cu_slab()
        ok, unbound = validate_surface_binding(slab, n_slab=len(slab))
        assert ok is True
        assert unbound == []

    def test_single_h_close_to_surface_is_bound(self):
        slab = _cu_slab()
        with_h = _add_atom(slab, "H", top_idx=0, z_offset=1.6)
        ok, unbound = validate_surface_binding(with_h, n_slab=len(slab))
        assert ok is True
        assert unbound == []

    def test_single_h_floating_far_above_is_unbound(self):
        slab = _cu_slab()
        with_h = _add_atom(slab, "H", top_idx=0, z_offset=5.0)
        ok, unbound = validate_surface_binding(with_h, n_slab=len(slab))
        assert ok is False
        assert len(unbound) == 1
        # The single H atom is the only unbound molecule.
        n_slab = len(slab)
        assert unbound[0] == [n_slab]


# ---------------------------------------------------------------------------
# Multi-atom adsorbates: CO upright on top
# ---------------------------------------------------------------------------

class TestMultiAtomBinding:

    def test_co_upright_with_c_down_is_bound(self):
        slab = _cu_slab()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        top_indices = _top_layer_indices(slab)
        # C above a real top-layer atom, O above C
        c_pos = slab.positions[top_indices[0]].copy()
        c_pos[2] = top + 1.85
        o_pos = c_pos.copy()
        o_pos[2] += 1.15
        co = Atoms("CO", positions=[c_pos, o_pos])
        with_co = slab + co
        ok, unbound = validate_surface_binding(with_co, n_slab=n_slab)
        assert ok is True
        assert unbound == []

    def test_co_floating_high_is_unbound_as_one_molecule(self):
        """Both atoms of the same CO must report as one unbound molecule,
        not two."""
        slab = _cu_slab()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        top_indices = _top_layer_indices(slab)
        c_pos = slab.positions[top_indices[0]].copy()
        c_pos[2] = top + 5.0
        o_pos = c_pos.copy()
        o_pos[2] += 1.15
        co = Atoms("CO", positions=[c_pos, o_pos])
        with_co = slab + co
        ok, unbound = validate_surface_binding(with_co, n_slab=n_slab)
        assert ok is False
        assert len(unbound) == 1  # exactly ONE molecule
        # Both CO atoms reported (indices n_slab and n_slab+1)
        assert set(unbound[0]) == {n_slab, n_slab + 1}


# ---------------------------------------------------------------------------
# Multiple molecules: one bound, one unbound
# ---------------------------------------------------------------------------

class TestMixedBinding:

    def test_one_bound_one_unbound(self):
        slab = _cu_slab()
        n_slab = len(slab)
        top = slab.positions[:, 2].max()
        top_indices = _top_layer_indices(slab)

        # Bound H above the first top-layer atom
        bound_h = slab.positions[top_indices[0]].copy()
        bound_h[2] = top + 1.6
        # Unbound H far above the last top-layer atom
        floating_h = slab.positions[top_indices[-1]].copy()
        floating_h[2] = top + 6.0

        with_two = slab + Atoms("HH", positions=[bound_h, floating_h])
        ok, unbound = validate_surface_binding(with_two, n_slab=n_slab)
        assert ok is False
        assert len(unbound) == 1
        # The unbound atom is index n_slab + 1 (the floating one).
        assert unbound[0] == [n_slab + 1]


# ---------------------------------------------------------------------------
# Subsurface atom should still count as bound (chemistry is real here:
# H below Cu surfaces happens, see CLAUDE.md)
# ---------------------------------------------------------------------------

class TestSubsurfaceCountsAsBound:

    def test_subsurface_h_is_bound(self):
        slab = _cu_slab()
        n_slab = len(slab)
        # Place H well *below* the slab top, between top and second layer.
        positions = slab.positions
        top = positions[:, 2].max()
        # Find an atom in the second layer for reference
        second_layer_z = sorted(np.unique(np.round(positions[:, 2], 1)))[-2]
        sub_pos = positions[0].copy()
        sub_pos[2] = (top + second_layer_z) / 2
        with_sub = slab + Atoms("H", positions=[sub_pos])
        ok, unbound = validate_surface_binding(with_sub, n_slab=n_slab)
        assert ok is True
        assert unbound == []
