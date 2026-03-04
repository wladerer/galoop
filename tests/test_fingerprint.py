"""
tests/test_fingerprint.py

Unit tests for the multi-stage duplicate detection cascade in fingerprint.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import fcc111, molecule
from ase import Atoms

from galoop.fingerprint import (
    StructRecord,
    _composition,
    _dist_histogram,
    _cosine,
    _energy_gate_passes,
    classify_postrelax,
    tanimoto_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cu_slab():
    """Small Cu(111) slab for testing."""
    return fcc111("Cu", size=(2, 2, 2), vacuum=8.0, periodic=True)


@pytest.fixture
def cu_slab_with_o(cu_slab):
    """Cu slab + one O atom placed above the surface."""
    slab = cu_slab.copy()
    top_z = slab.get_positions()[:, 2].max()
    slab.append("O")
    pos = slab.get_positions()
    pos[-1] = [slab.cell[0, 0] / 2, slab.cell[1, 1] / 2, top_z + 1.5]
    slab.set_positions(pos)
    return slab


# ---------------------------------------------------------------------------
# _composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_pure_metal(self, cu_slab):
        comp = _composition(cu_slab)
        assert "Cu" in comp

    def test_same_atoms_equal(self, cu_slab):
        assert _composition(cu_slab) == _composition(cu_slab.copy())

    def test_different_composition(self, cu_slab, cu_slab_with_o):
        assert _composition(cu_slab) != _composition(cu_slab_with_o)


# ---------------------------------------------------------------------------
# _dist_histogram
# ---------------------------------------------------------------------------

class TestDistHistogram:
    def test_shape(self, cu_slab):
        hist = _dist_histogram(cu_slab, n_bins=30)
        assert hist.shape == (30,)

    def test_normalised(self, cu_slab):
        hist = _dist_histogram(cu_slab)
        assert abs(hist.sum() - 1.0) < 1e-10

    def test_identical_structures_same_hist(self, cu_slab):
        h1 = _dist_histogram(cu_slab)
        h2 = _dist_histogram(cu_slab.copy())
        np.testing.assert_allclose(h1, h2)

    def test_single_atom_returns_zeros(self):
        single = Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        hist = _dist_histogram(single, n_bins=20)
        assert hist.shape == (20,)
        assert hist.sum() == 0.0

    def test_cosine_identical_near_one(self, cu_slab):
        h1 = _dist_histogram(cu_slab)
        h2 = _dist_histogram(cu_slab.copy())
        assert _cosine(h1, h2) > 0.9999


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine(v, v) - 1.0) < 1e-12

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine(a, b)) < 1e-12

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine(a, b) == 0.0
        assert _cosine(b, a) == 0.0


# ---------------------------------------------------------------------------
# _energy_gate_passes
# ---------------------------------------------------------------------------

class TestEnergyGate:
    def test_same_energy_passes(self):
        assert _energy_gate_passes(-100.0, -100.0, 5.0) is True

    def test_within_tolerance_passes(self):
        # 4% difference — within 5% tolerance
        assert _energy_gate_passes(-104.0, -100.0, 5.0) is True

    def test_outside_tolerance_fails(self):
        # 10% difference — outside 5% tolerance
        assert _energy_gate_passes(-110.0, -100.0, 5.0) is False

    def test_none_energy_passes(self):
        assert _energy_gate_passes(None, -100.0, 5.0) is True
        assert _energy_gate_passes(-100.0, None, 5.0) is True
        assert _energy_gate_passes(None, None, 5.0) is True

    def test_zero_tol_passes(self):
        # tol_pct=0 disables gate → always True
        assert _energy_gate_passes(-110.0, -100.0, 0.0) is True

    def test_zero_existing_energy_passes(self):
        # avoid divide-by-zero
        assert _energy_gate_passes(-1.0, 0.0, 5.0) is True

    def test_nan_energy_passes(self):
        import math
        assert _energy_gate_passes(float("nan"), -100.0, 5.0) is True


# ---------------------------------------------------------------------------
# classify_postrelax (full cascade)
# ---------------------------------------------------------------------------

class TestClassifyPostrelax:
    """Integration tests for the cascade classifier."""

    def _make_record(self, atoms, energy, id_="ref"):
        """Build a StructRecord without SOAP (use zero vector for speed)."""
        return StructRecord(
            id=id_,
            soap_vector=np.zeros(10),
            energy=energy,
            composition=_composition(atoms),
            dist_hist=_dist_histogram(atoms),
        )

    def test_empty_cache_returns_unique(self, cu_slab):
        label, dup_id, _ = classify_postrelax(
            cu_slab, energy=None, struct_cache={},
        )
        assert label == "unique"
        assert dup_id is None

    def test_different_composition_returns_unique(self, cu_slab, cu_slab_with_o):
        """Composition gate: Cu vs Cu+O → unique."""
        rec = self._make_record(cu_slab, energy=-500.0, id_="cu")
        # Override soap_vector to 1s so Tanimoto would be 1.0 if gate fails
        rec.soap_vector = np.ones(10)

        label, _, _ = classify_postrelax(
            cu_slab_with_o,
            energy=-500.0,
            struct_cache={"cu": rec},
            # Use zero SOAP so Tanimoto would be 0 anyway
            r_cut=3.0, n_max=3, l_max=2,
        )
        assert label == "unique"

    def test_large_energy_difference_returns_unique(self, cu_slab):
        """Energy gate: 20% difference → not a duplicate."""
        rec = StructRecord(
            id="ref",
            soap_vector=np.ones(10),
            energy=-500.0,
            composition=_composition(cu_slab),
            dist_hist=_dist_histogram(cu_slab),
        )
        label, _, _ = classify_postrelax(
            cu_slab,
            energy=-600.0,   # 20% difference
            struct_cache={"ref": rec},
            energy_tol_pct=5.0,
            r_cut=3.0, n_max=3, l_max=2,
        )
        assert label == "unique"

    def test_identical_structure_is_duplicate(self, cu_slab):
        """Full cascade: identical structure must be flagged as duplicate."""
        try:
            from dscribe.descriptors import SOAP  # noqa: F401
        except ImportError:
            pytest.skip("dscribe not installed")

        # First call — empty cache → unique
        label1, _, soap1 = classify_postrelax(
            cu_slab, energy=-500.0, struct_cache={},
            r_cut=4.0, n_max=4, l_max=3,
        )
        assert label1 == "unique"

        # Build a proper StructRecord with real SOAP
        rec = StructRecord(
            id="original",
            soap_vector=soap1,
            energy=-500.0,
            composition=_composition(cu_slab),
            dist_hist=_dist_histogram(cu_slab),
        )

        # Second call — same structure → duplicate
        label2, dup_id, _ = classify_postrelax(
            cu_slab.copy(),
            energy=-500.0,
            struct_cache={"original": rec},
            duplicate_threshold=0.90,
            r_cut=4.0, n_max=4, l_max=3,
        )
        assert label2 == "duplicate"
        assert dup_id == "original"

    def test_distinct_structures_are_unique(self, cu_slab, cu_slab_with_o):
        """Two genuinely different structures (different composition) → both unique."""
        try:
            from dscribe.descriptors import SOAP  # noqa: F401
        except ImportError:
            pytest.skip("dscribe not installed")

        # Add first (cu_slab) to cache
        label1, _, soap1 = classify_postrelax(
            cu_slab, energy=-500.0, struct_cache={},
            r_cut=4.0, n_max=4, l_max=3,
        )
        assert label1 == "unique"

        rec = StructRecord(
            id="cu",
            soap_vector=soap1,
            energy=-500.0,
            composition=_composition(cu_slab),
            dist_hist=_dist_histogram(cu_slab),
        )

        # Second (cu + O) → different composition, must be unique
        label2, _, _ = classify_postrelax(
            cu_slab_with_o,
            energy=-502.0,
            struct_cache={"cu": rec},
            r_cut=4.0, n_max=4, l_max=3,
        )
        assert label2 == "unique"
