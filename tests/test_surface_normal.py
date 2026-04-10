"""Tests for ``_surface_normal`` (galoop/science/surface.py).

The function used to compute the surface normal as ``c / |c|``, which is
the wrong answer for non-orthogonal cells where the c-vector is tilted.
Commit ``d5e5687`` switched to ``(a × b) / |a × b|``, which is
perpendicular to the surface plane by construction. None of the slabs
built by ``ase.build.{fcc111, fcc100, fcc211, hcp0001}`` produce a
tilted-c cell — they all have ``c ∥ z`` so old and new code happen to
agree — so the fix was never exercised in production until this test.

These tests construct hand-crafted cells where the two definitions
disagree, so the regression is real if anyone reverts the fix.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111

from galoop.science.surface import _surface_normal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_cell(a, b, c) -> Atoms:
    """Build a one-atom Atoms object with the given lattice vectors.

    The single atom keeps ASE's machinery happy without affecting any
    cell-vector-only computation.
    """
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[a, b, c], pbc=True)


def _legacy_normal(atoms: Atoms) -> np.ndarray:
    """The pre-fix definition: c-vector normalised. Used to assert that
    the new code disagrees with the old one on tilted cells (so reverting
    the fix would actually break this test rather than silently passing).
    """
    c = np.asarray(atoms.cell[2], dtype=float)
    return c / np.linalg.norm(c)


# ---------------------------------------------------------------------------
# Orthogonal cells: old and new agree
# ---------------------------------------------------------------------------

class TestOrthogonalCells:
    """For orthogonal cells with c ∥ z, the (a×b) and c/|c| definitions
    are identical. We pin both."""

    def test_simple_cubic(self):
        n = _surface_normal(_make_cell([3, 0, 0], [0, 3, 0], [0, 0, 12]))
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

    def test_orthorhombic(self):
        n = _surface_normal(_make_cell([4.1, 0, 0], [0, 5.7, 0], [0, 0, 18]))
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

    def test_fcc111_from_ase_builder(self):
        """ASE's fcc111 returns a hexagonal-in-ab cell with c ∥ z. The
        normal must be exactly [0, 0, 1] regardless of the 60° angle in
        the ab plane."""
        slab = fcc111("Cu", size=(2, 2, 3), a=3.615, vacuum=8.0,
                      orthogonal=False)
        n = _surface_normal(slab)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

    def test_orthogonal_old_and_new_agree(self):
        """Sanity check that the legacy and current definitions match
        for orthogonal cells (so the new code doesn't accidentally break
        the production case)."""
        atoms = fcc111("Pt", size=(2, 2, 3), a=3.92, vacuum=8.0,
                       orthogonal=False)
        np.testing.assert_allclose(
            _surface_normal(atoms), _legacy_normal(atoms), atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Tilted-c cells: old and new must DISAGREE; new must be correct
# ---------------------------------------------------------------------------

class TestTiltedCCells:
    """The bug 1 regression cases: cells where ``c`` is not perpendicular
    to the ab plane. Old code returned ``c/|c|`` (wrong), new code returns
    the true a×b normal."""

    def test_30_degree_tilt(self):
        """ab in the xy plane, c tilted 30° from z toward +x. The true
        normal is +z; the legacy c-vector definition would give a tilted
        unit vector with a non-zero x component."""
        a = [3.0, 0.0, 0.0]
        b = [0.0, 3.0, 0.0]
        # c length = 12, tilted 30° toward +x
        theta = np.deg2rad(30.0)
        c = [12 * np.sin(theta), 0.0, 12 * np.cos(theta)]
        atoms = _make_cell(a, b, c)

        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

        # Legacy definition: c-vector normalised. Should be markedly
        # different (sin 30° = 0.5 along x).
        legacy = _legacy_normal(atoms)
        np.testing.assert_allclose(legacy, [0.5, 0.0, np.cos(theta)], atol=1e-12)
        assert not np.allclose(n, legacy, atol=1e-3), (
            "regression: tilted-c case must give a different answer than "
            "the legacy c/|c| definition"
        )

    def test_45_degree_tilt_diagonal(self):
        """c tilted 45° toward +x+y diagonal — both lateral components
        non-zero, larger angular departure from z."""
        a = [4.0, 0.0, 0.0]
        b = [0.0, 4.0, 0.0]
        c = [5.0, 5.0, 10.0]
        atoms = _make_cell(a, b, c)

        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)
        assert not np.allclose(n, _legacy_normal(atoms), atol=1e-3)

    def test_hexagonal_ab_with_tilted_c(self):
        """A hexagonal in-plane lattice (60° between a and b) plus a
        tilted c. The true normal is still ±z, but |a×b| is now the
        rhombic-cell area, not |a||b|."""
        a = [3.0, 0.0, 0.0]
        b = [3.0 * np.cos(np.deg2rad(60)), 3.0 * np.sin(np.deg2rad(60)), 0.0]
        c = [1.0, 0.5, 11.0]   # arbitrary tilt
        atoms = _make_cell(a, b, c)

        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)
        assert not np.allclose(n, _legacy_normal(atoms), atol=1e-3)


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    """The normal must point into the same half-space as c (i.e. into
    the vacuum side, by ASE convention). This matters when the user
    flips the c-vector or builds a cell upside down."""

    def test_flipped_c_orthogonal(self):
        """A slab with c pointing in -z must yield a normal in -z, not
        +z. Otherwise downstream height projections would put the
        adsorbate sampling window inside the slab."""
        atoms = _make_cell([3, 0, 0], [0, 3, 0], [0, 0, -10])
        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, -1], atol=1e-12)
        assert np.dot(n, atoms.cell[2]) > 0, (
            "normal must point into the same half-space as c"
        )

    def test_flipped_c_tilted(self):
        atoms = _make_cell([3, 0, 0], [0, 3, 0], [3.0, 0.0, -9.0])
        n = _surface_normal(atoms)
        # cross([3,0,0], [0,3,0]) = [0,0,9]; flipping to match -z c gives [0,0,-1]
        np.testing.assert_allclose(n, [0, 0, -1], atol=1e-12)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

class TestDegenerate:
    """Cells where (a × b) collapses fall back to [0, 0, 1] without
    raising. These cells are nonsensical for a surface but the function
    should not blow up."""

    def test_collinear_ab_falls_back(self):
        """a and b parallel → cross product = 0 → fallback."""
        atoms = _make_cell([3, 0, 0], [6, 0, 0], [0, 0, 10])
        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

    def test_zero_a_falls_back(self):
        atoms = _make_cell([0, 0, 0], [0, 3, 0], [0, 0, 10])
        n = _surface_normal(atoms)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# Output properties (always-true invariants)
# ---------------------------------------------------------------------------

class TestNormalInvariants:
    """Properties the returned vector must always satisfy, regardless
    of the input cell shape."""

    @staticmethod
    def _random_well_formed_cell(seed: int) -> Atoms:
        rng = np.random.default_rng(seed)
        a = rng.uniform(2.5, 5.0, size=3)
        b = rng.uniform(2.5, 5.0, size=3)
        # Ensure a, b not collinear
        while abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) > 0.99:
            b = rng.uniform(2.5, 5.0, size=3)
        c = rng.uniform(-2.0, 2.0, size=3)
        c[2] = rng.uniform(8.0, 15.0)   # c always has a +z component
        return _make_cell(a.tolist(), b.tolist(), c.tolist())

    def test_returns_unit_vector(self):
        for seed in range(10):
            atoms = self._random_well_formed_cell(seed)
            n = _surface_normal(atoms)
            np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-12)

    def test_perpendicular_to_a_and_b(self):
        for seed in range(10):
            atoms = self._random_well_formed_cell(seed)
            n = _surface_normal(atoms)
            a = np.asarray(atoms.cell[0])
            b = np.asarray(atoms.cell[1])
            assert abs(np.dot(n, a)) < 1e-10
            assert abs(np.dot(n, b)) < 1e-10

    def test_points_into_c_halfspace(self):
        for seed in range(10):
            atoms = self._random_well_formed_cell(seed)
            n = _surface_normal(atoms)
            c = np.asarray(atoms.cell[2])
            assert np.dot(n, c) > 0
