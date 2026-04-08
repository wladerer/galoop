"""Tests for the ``galoop sample`` CLI command.

We exercise the parallel, placement-only default path: it builds candidates
on a fresh ProcessPoolExecutor, validates them, dedupes via SOAP, and writes
an extended-XYZ file. No MACE calls happen, so the test runs in seconds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from ase.io import read as ase_read
from click.testing import CliRunner

from galoop.cli import cli, sample


@pytest.fixture
def sample_config_path(tmp_path):
    """Write a minimal galoop.yaml + slab POSCAR for sampling tests.

    Uses the same Cu(111) builder as the rest of the tests, with a single
    monoatomic O adsorbate so we don't need to ship coordinates.
    """
    from ase.build import fcc111
    from ase.io import write as ase_write

    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    slab_path = tmp_path / "slab.vasp"
    ase_write(str(slab_path), slab, format="vasp")

    top_z = float(slab.get_positions()[:, 2].max())

    cfg = {
        "slab": {
            "geometry": str(slab_path),
            "energy": -1000.0,
            "sampling_zmin": top_z + 1.5,
            "sampling_zmax": top_z + 3.5,
        },
        "adsorbates": [
            {"symbol": "O", "chemical_potential": -4.0,
             "min_count": 0, "max_count": 3},
        ],
        "calculator_stages": [{"name": "preopt", "type": "mace"}],
        "ga": {
            "population_size": 4,
            "min_adsorbates": 1,
            "max_adsorbates": 3,
            "min_structures": 5,
            "max_structures": 50,
            "max_stall": 5,
        },
    }
    cfg_path = tmp_path / "galoop.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


# ---------------------------------------------------------------------------
# Smoke tests on the parallel placement-only path
# ---------------------------------------------------------------------------

class TestSampleCommand:

    def test_writes_xyz_and_manifest(self, sample_config_path, tmp_path):
        out_xyz = tmp_path / "samples.xyz"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sample",
                "-c", str(sample_config_path),
                "-n", "5",
                "-o", str(out_xyz),
                "--workers", "1",
                "--soap-threshold", "0.99",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_xyz.exists()
        manifest = out_xyz.with_suffix(".manifest.csv")
        assert manifest.exists()

    def test_xyz_round_trips_through_ase_read(self, sample_config_path, tmp_path):
        out_xyz = tmp_path / "samples.xyz"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sample",
                "-c", str(sample_config_path),
                "-n", "5",
                "-o", str(out_xyz),
                "--workers", "1",
                "--soap-threshold", "0.99",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0, result.output

        atoms_list = ase_read(str(out_xyz), index=":")
        assert len(atoms_list) >= 1
        for ats in atoms_list:
            assert "sample_id" in ats.info
            assert "adsorbate_counts" in ats.info
            assert "n_slab_atoms" in ats.info
            # adsorbate_counts is JSON-serialised; verify round-trip
            counts = json.loads(ats.info["adsorbate_counts"])
            assert isinstance(counts, dict)
            assert sum(counts.values()) >= 1

    def test_manifest_csv_is_well_formed(self, sample_config_path, tmp_path):
        out_xyz = tmp_path / "samples.xyz"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sample",
                "-c", str(sample_config_path),
                "-n", "5",
                "-o", str(out_xyz),
                "--workers", "1",
                "--soap-threshold", "0.99",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0, result.output

        manifest = out_xyz.with_suffix(".manifest.csv")
        rows = list(csv.DictReader(manifest.open()))
        assert len(rows) >= 1
        for row in rows:
            assert {"id", "n_atoms", "energy_eV", "adsorbate_counts"} <= set(row.keys())
            # placement-only has no energy
            assert row["energy_eV"] == ""
            counts = json.loads(row["adsorbate_counts"])
            assert sum(counts.values()) >= 1

    def test_seed_makes_run_reproducible(self, sample_config_path, tmp_path):
        runner = CliRunner()

        def _run(out_path):
            return runner.invoke(
                cli,
                [
                    "sample",
                    "-c", str(sample_config_path),
                    "-n", "4",
                    "-o", str(out_path),
                    "--workers", "1",
                    "--soap-threshold", "0.99",
                    "--seed", "7",
                ],
            )

        out_a = tmp_path / "a.xyz"
        out_b = tmp_path / "b.xyz"
        result_a = _run(out_a)
        result_b = _run(out_b)
        assert result_a.exit_code == 0
        assert result_b.exit_code == 0

        atoms_a = ase_read(str(out_a), index=":")
        atoms_b = ase_read(str(out_b), index=":")
        assert len(atoms_a) == len(atoms_b)
        for a, b in zip(atoms_a, atoms_b, strict=True):
            # Same composition, same atom positions
            assert a.get_chemical_formula() == b.get_chemical_formula()
            assert (a.get_positions() == b.get_positions()).all()

    def test_truncates_pre_existing_output(self, sample_config_path, tmp_path):
        out_xyz = tmp_path / "samples.xyz"
        out_xyz.write_text("STALE CONTENT\n")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sample",
                "-c", str(sample_config_path),
                "-n", "3",
                "-o", str(out_xyz),
                "--workers", "1",
                "--soap-threshold", "0.99",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0
        # The file should no longer contain the stale content; instead it
        # should be a valid extxyz that ASE can parse.
        atoms = ase_read(str(out_xyz), index=":")
        assert len(atoms) >= 1


# ---------------------------------------------------------------------------
# Composition-bucketed dedup
# ---------------------------------------------------------------------------

class TestSampleDedup:
    """Verify the parent-side composition bucket actually filters duplicates.

    With seed=42 and soap_threshold=0.5 (very aggressive), the dedup should
    accept far fewer structures than max_attempts.
    """

    def test_aggressive_dedup_rejects_most(self, sample_config_path, tmp_path):
        out_xyz = tmp_path / "samples.xyz"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sample",
                "-c", str(sample_config_path),
                "-n", "100",  # ask for many; expect to fall short
                "-o", str(out_xyz),
                "--workers", "1",
                "--soap-threshold", "0.50",  # very low → most things flagged duplicate
                "--max-attempts", "200",
                "--seed", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should not generate the full 100 — saturation is the point of this test
        atoms = ase_read(str(out_xyz), index=":") if out_xyz.exists() else []
        assert len(atoms) < 100
