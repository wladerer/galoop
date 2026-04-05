"""Shared pytest fixtures."""

import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary GaloopStore."""
    from galoop.store import GaloopStore
    return GaloopStore(tmp_path)


@pytest.fixture
def cu_slab_path(tmp_path):
    """Write a simple Cu(111) 2x2x3 slab POSCAR to tmp_path."""
    from ase.build import fcc111
    from ase.io import write

    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    path = tmp_path / "slab.vasp"
    write(str(path), slab, format="vasp")
    return path


@pytest.fixture
def minimal_config(cu_slab_path):
    """GaloopConfig with a small population and two adsorbate species."""
    from ase.io import read
    from galoop.config import GaloopConfig

    slab = read(str(cu_slab_path), format="vasp")
    top_z = float(slab.get_positions()[:, 2].max())

    return GaloopConfig.model_validate({
        "slab": {
            "geometry": str(cu_slab_path),
            "energy": -1000.0,
            "sampling_zmin": top_z + 0.5,
            "sampling_zmax": top_z + 3.5,
        },
        "adsorbates": [
            {"symbol": "O",  "chemical_potential": -4.0, "min_count": 0, "max_count": 3},
            {"symbol": "OH", "chemical_potential": -3.5, "min_count": 0, "max_count": 2,
             "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]},
        ],
        "calculator_stages": [{"name": "preopt", "type": "mace"}],
        "ga": {
            "population_size": 8,
            "min_adsorbates": 1,
            "max_adsorbates": 4,
            "min_structures": 10,
            "max_structures": 100,
            "max_stall": 5,
        },
    })


@pytest.fixture
def slab_info(minimal_config):
    """SlabInfo loaded from the test slab."""
    from galoop.science.surface import load_slab

    return load_slab(
        minimal_config.slab.geometry,
        zmin=minimal_config.slab.sampling_zmin,
        zmax=minimal_config.slab.sampling_zmax,
    )


@pytest.fixture
def converged_population(tmp_path, minimal_config, slab_info, temp_db):
    """
    Build the initial population and fake-converge every structure so they
    become selectable parents for spawn tests.

    Returns the list of converged Individuals.
    """
    from galoop.galoop import _build_initial_population
    from galoop.individual import STATUS

    rng = np.random.default_rng(0)
    _build_initial_population(minimal_config, slab_info, temp_db, rng)

    converged = []
    for i, ind in enumerate(temp_db.get_by_status(STATUS.PENDING)):
        struct_dir = temp_db.individual_dir(ind.id)
        poscar = struct_dir / "POSCAR"
        contcar = struct_dir / "CONTCAR"
        shutil.copy(poscar, contcar)
        updated = ind.with_energy(raw=-500.0 - i, grand_canonical=-1.0 - i * 0.1)
        temp_db.update(updated)
        converged.append(updated)

    return converged
