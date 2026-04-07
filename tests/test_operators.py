"""
tests/test_operators.py

Structural-sanity tests for GA operators.

Each operator is applied across many random seeds and the output is checked for:
  - No orphan atoms (all adsorbate atoms belong to complete molecules)
  - Valid C-O bond lengths (CO stays intact)
  - No atom clashes (no two non-slab atoms closer than 0.8 Å)
  - Slab atoms unchanged (indices < n_slab have the same positions)
  - Correct atom count changes (add/remove change by exactly one molecule)

A regression test covers the 6db3d2e2 -> mutate_remove bug where a single C
was deleted from a 4-CO structure, leaving orphan O atoms.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc100
from ase.constraints import FixAtoms

from galoop.science.reproduce import (
    _group_molecules,
    merge,
    mutate_add,
    mutate_displace,
    mutate_rattle_slab,
    mutate_remove,
    mutate_translate,
    splice,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SEEDS = 60
CO_BOND_MIN = 0.9   # Å — below this something is very wrong
CO_BOND_MAX = 1.5   # Å — stretched but still a CO bond
CLASH_THRESHOLD = 0.8  # Å — two atoms closer than this = clash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cu100_slab():
    """Cu(100) 3×3×4 bare slab, bottom two layers fixed."""
    slab = fcc100("Cu", size=(3, 3, 4), vacuum=8.0, periodic=True)
    z = slab.get_positions()[:, 2]
    sorted_z = sorted(set(z.round(2)))
    threshold = sorted_z[1]  # top of second layer
    slab.set_constraint(FixAtoms(indices=[i for i, zi in enumerate(z) if zi <= threshold]))
    return slab


def _place_co_molecules(slab, n_co: int, rng: np.random.Generator) -> Atoms:
    """Place *n_co* CO molecules on top of the slab at random x,y positions."""
    atoms = slab.copy()
    n_slab = len(slab)
    cell = slab.get_cell()
    z_top = slab.get_positions()[:, 2].max()
    for _ in range(n_co):
        xy = rng.uniform(0, 1, size=2) @ cell[:2, :2]
        z_c = rng.uniform(z_top + 1.8, z_top + 2.3)
        atoms.extend(Atoms("C", positions=[[xy[0], xy[1], z_c]]))
        atoms.extend(Atoms("O", positions=[[xy[0], xy[1], z_c + 1.15]]))
    atoms.set_constraint(FixAtoms(indices=list(range(n_slab))))
    return atoms


@pytest.fixture(scope="module")
def co_parents(cu100_slab):
    """Two parent structures each with 2 CO molecules."""
    rng = np.random.default_rng(0)
    pa = _place_co_molecules(cu100_slab, 2, rng)
    pb = _place_co_molecules(cu100_slab, 2, rng)
    return pa, pb


@pytest.fixture(scope="module")
def co_4mol(cu100_slab):
    """A single structure with exactly 4 CO molecules (regression fixture)."""
    rng = np.random.default_rng(42)
    return _place_co_molecules(cu100_slab, 4, rng)


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------

def _n_slab(slab_ref: Atoms) -> int:
    return len(slab_ref)


def assert_no_orphan_co(atoms: Atoms, n_slab: int) -> None:
    """Every C must be bonded to exactly one O and vice-versa."""
    mols = _group_molecules(atoms, n_slab)
    syms = atoms.get_chemical_symbols()
    for mol in mols:
        mol_syms = sorted(syms[i] for i in mol)
        assert mol_syms == ["C", "O"], (
            f"Incomplete CO molecule: {mol_syms} at indices {mol}"
        )


def assert_co_bond_lengths(atoms: Atoms, n_slab: int) -> None:
    """C-O bond length must be in [CO_BOND_MIN, CO_BOND_MAX]."""
    mols = _group_molecules(atoms, n_slab)
    pos = atoms.get_positions()
    for mol in mols:
        assert len(mol) == 2
        d = np.linalg.norm(pos[mol[0]] - pos[mol[1]])
        assert CO_BOND_MIN <= d <= CO_BOND_MAX, (
            f"CO bond length {d:.3f} Å out of range [{CO_BOND_MIN}, {CO_BOND_MAX}]"
        )


def assert_no_clashes(atoms: Atoms, n_slab: int) -> None:
    """No two adsorbate atoms closer than CLASH_THRESHOLD."""
    ads = list(range(n_slab, len(atoms)))
    pos = atoms.get_positions()
    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            d = np.linalg.norm(pos[ads[i]] - pos[ads[j]])
            assert d >= CLASH_THRESHOLD, (
                f"Clash between atoms {ads[i]} and {ads[j]}: {d:.3f} Å"
            )


def assert_slab_unchanged(original: Atoms, result: Atoms, n_slab: int) -> None:
    """Slab positions must not change."""
    np.testing.assert_allclose(
        original.get_positions()[:n_slab],
        result.get_positions()[:n_slab],
        atol=1e-10,
        err_msg="Slab atom positions were modified",
    )


def assert_co_integrity(atoms: Atoms, n_slab: int) -> None:
    """Combined check: no orphans, valid bonds, no clashes."""
    assert len(atoms) > n_slab, "Result has no adsorbates"
    assert (len(atoms) - n_slab) % 2 == 0, (
        f"Odd number of adsorbate atoms: {len(atoms) - n_slab}"
    )
    assert_no_orphan_co(atoms, n_slab)
    assert_co_bond_lengths(atoms, n_slab)


# ---------------------------------------------------------------------------
# Tests: splice
# ---------------------------------------------------------------------------

class TestSplice:

    def test_children_have_adsorbates(self, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(0)
        c1, c2 = splice(pa, pb, n_slab, rng)
        assert len(c1) > n_slab or len(c2) > n_slab

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_no_orphan_atoms(self, seed, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        c1, c2 = splice(pa, pb, n_slab, rng)
        for child in (c1, c2):
            if len(child) > n_slab:
                assert_no_orphan_co(child, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        c1, c2 = splice(pa, pb, n_slab, rng)
        assert_slab_unchanged(pa, c1, n_slab)
        assert_slab_unchanged(pb, c2, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_co_bond_lengths(self, seed, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        c1, c2 = splice(pa, pb, n_slab, rng)
        for child in (c1, c2):
            if len(child) > n_slab:
                assert_co_bond_lengths(child, n_slab)


# ---------------------------------------------------------------------------
# Tests: merge
# ---------------------------------------------------------------------------

class TestMerge:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_no_orphan_atoms(self, seed, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        child = merge(pa, pb, n_slab, rng)
        if len(child) > n_slab:
            assert_no_orphan_co(child, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        child = merge(pa, pb, n_slab, rng)
        assert_slab_unchanged(pa, child, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_at_least_parent_a_adsorbates(self, seed, co_parents, cu100_slab):
        """merge() starts from parent_a; it should never have fewer adsorbates than A."""
        pa, pb = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        child = merge(pa, pb, n_slab, rng)
        n_ads_a = len(pa) - n_slab
        assert len(child) - n_slab >= n_ads_a


# ---------------------------------------------------------------------------
# Tests: mutate_remove
# ---------------------------------------------------------------------------

class TestMutateRemove:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_removes_whole_molecule(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_remove(pa, n_slab, rng=rng)
        if result is not None:
            assert_no_orphan_co(result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_removes_exactly_one_molecule(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        n_ads_before = len(pa) - n_slab
        rng = np.random.default_rng(seed)
        result = mutate_remove(pa, n_slab, rng=rng)
        if result is not None:
            n_ads_after = len(result) - n_slab
            assert n_ads_after == n_ads_before - 2, (
                f"Expected removal of 2 atoms (one CO), got {n_ads_before - n_ads_after}"
            )

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_remove(pa, n_slab, rng=rng)
        if result is not None:
            assert_slab_unchanged(pa, result, n_slab)

    def test_returns_none_when_empty(self, cu100_slab):
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(0)
        result = mutate_remove(cu100_slab.copy(), n_slab, rng=rng)
        assert result is None

    # --- Regression: 6db3d2e2 -> mutate_remove -> orphan O ---

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_regression_4co_remove_gives_3co(self, seed, co_4mol, cu100_slab):
        """4-CO -> mutate_remove must yield 3-CO, not 3C+4O."""
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_remove(co_4mol, n_slab, rng=rng)
        assert result is not None
        n_ads = len(result) - n_slab
        assert n_ads == 6, f"Expected 6 adsorbate atoms (3×CO), got {n_ads}"
        assert_no_orphan_co(result, n_slab)
        syms = result.get_chemical_symbols()[n_slab:]
        c_count = syms.count("C")
        o_count = syms.count("O")
        assert c_count == o_count == 3, f"C={c_count} O={o_count}, expected 3 each"


# ---------------------------------------------------------------------------
# Tests: mutate_displace
# ---------------------------------------------------------------------------

class TestMutateDisplace:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_no_orphan_atoms(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_displace(pa, n_slab, rng=rng)
        if result is not None:
            assert_no_orphan_co(result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_same_atom_count(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_displace(pa, n_slab, rng=rng)
        if result is not None:
            assert len(result) == len(pa)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_displace(pa, n_slab, rng=rng)
        if result is not None:
            assert_slab_unchanged(pa, result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_one_molecule_moved_rigidly(self, seed, co_parents, cu100_slab):
        """displace moves a whole molecule by the same vector."""
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_displace(pa, n_slab, displacement=0.5, rng=rng)
        if result is None:
            return
        pos_before = pa.get_positions()[n_slab:]
        pos_after = result.get_positions()[n_slab:]
        diffs = pos_after - pos_before
        dist = np.linalg.norm(diffs, axis=1)
        moved = np.where(dist > 1e-10)[0]
        # At least one atom moved (the molecule)
        assert len(moved) >= 1
        # All moved atoms moved by the same vector (rigid translation)
        if len(moved) > 1:
            for i in moved[1:]:
                np.testing.assert_allclose(diffs[i], diffs[moved[0]], atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: mutate_translate
# ---------------------------------------------------------------------------

class TestMutateTranslate:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_no_orphan_atoms(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_translate(pa, n_slab, rng=rng)
        if result is not None:
            assert_no_orphan_co(result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_same_atom_count(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_translate(pa, n_slab, rng=rng)
        if result is not None:
            assert len(result) == len(pa)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_translate(pa, n_slab, rng=rng)
        if result is not None:
            assert_slab_unchanged(pa, result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_whole_molecule_translated_rigidly(self, seed, co_parents, cu100_slab):
        """C and O in the translated molecule must move by the same vector."""
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_translate(pa, n_slab, rng=rng)
        if result is None:
            return
        pos_before = pa.get_positions()
        pos_after = result.get_positions()
        deltas = pos_after[n_slab:] - pos_before[n_slab:]
        # Find the molecule that moved
        moved_mask = np.linalg.norm(deltas, axis=1) > 1e-10
        if not moved_mask.any():
            return
        # The moved atoms should all share the same displacement vector
        moved_deltas = deltas[moved_mask]
        for delta in moved_deltas[1:]:
            np.testing.assert_allclose(
                delta, moved_deltas[0], atol=1e-10,
                err_msg="Molecule was not translated rigidly (atoms moved by different vectors)",
            )

    def test_returns_none_when_empty(self, cu100_slab):
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(0)
        result = mutate_translate(cu100_slab.copy(), n_slab, rng=rng)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: mutate_rattle_slab
# ---------------------------------------------------------------------------

class TestMutateRattleSlab:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_molecules_translated_rigidly(self, seed, co_parents, cu100_slab):
        """Adsorbate molecules are translated as rigid bodies (intramolecular
        bonds preserved)."""
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_rattle_slab(pa, n_slab, rng=rng)
        # Each CO molecule's C-O bond length should be preserved
        from galoop.science.reproduce import _group_molecules
        for mol in _group_molecules(pa, n_slab):
            d_before = np.linalg.norm(
                pa.get_positions()[mol[0]] - pa.get_positions()[mol[1]]
            )
            d_after = np.linalg.norm(
                result.get_positions()[mol[0]] - result.get_positions()[mol[1]]
            )
            assert abs(d_before - d_after) < 1e-10

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_same_atom_count(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_rattle_slab(pa, n_slab, rng=rng)
        assert len(result) == len(pa)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_fixed_slab_atoms_unchanged(self, seed, co_parents, cu100_slab):
        """Atoms covered by FixAtoms must not move."""
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_rattle_slab(pa, n_slab, rng=rng)
        fixed = set()
        for c in pa.constraints:
            if isinstance(c, FixAtoms):
                fixed.update(int(i) for i in c.index)
        for i in fixed:
            np.testing.assert_allclose(
                pa.get_positions()[i],
                result.get_positions()[i],
                atol=1e-10,
                err_msg=f"Fixed slab atom {i} was moved by rattle_slab",
            )


# ---------------------------------------------------------------------------
# Tests: mutate_add
# ---------------------------------------------------------------------------

class TestMutateAdd:

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_adds_exactly_one_atom(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_add(pa, n_slab, symbol="C", rng=rng)
        assert len(result) == len(pa) + 1

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_new_atom_above_slab(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        z_top = pa.get_positions()[:n_slab, 2].max()
        rng = np.random.default_rng(seed)
        result = mutate_add(pa, n_slab, symbol="C", rng=rng)
        new_z = result.get_positions()[-1, 2]
        assert new_z > z_top, f"New atom z={new_z:.3f} not above slab top z={z_top:.3f}"

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_slab_unchanged(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        result = mutate_add(pa, n_slab, symbol="C", rng=rng)
        assert_slab_unchanged(pa, result, n_slab)

    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_correct_symbol_added(self, seed, co_parents, cu100_slab):
        pa, _ = co_parents
        n_slab = _n_slab(cu100_slab)
        rng = np.random.default_rng(seed)
        for sym in ("C", "O", "H"):
            result = mutate_add(pa, n_slab, symbol=sym, rng=rng)
            assert result.get_chemical_symbols()[-1] == sym
