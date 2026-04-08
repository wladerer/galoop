"""Tests for galoop.gpr.CompositionGPR.

These cover the GP fit/predict/suggest path with synthetic data so the
tests stay fast and don't depend on MACE or any GA state.
"""

from __future__ import annotations

import numpy as np
import pytest

from galoop.gpr import CompositionGPR


class _StubAdsorbateConfig:
    """Tiny duck-typed AdsorbateConfig for unit tests.

    Pydantic AdsorbateConfig requires geometry/coordinates for polyatomics
    and chemical-potential fiddling we don't care about here. The GPR code
    only touches `.symbol`, `.min_count`, `.max_count`, so a stub is enough.
    """

    def __init__(self, symbol: str, min_count: int = 0, max_count: int = 4):
        self.symbol = symbol
        self.min_count = min_count
        self.max_count = max_count


@pytest.fixture
def gpr_two_species():
    """Two-species GPR (e.g. CO/H on Pt) with min_total=1, max_total=4."""
    species = ["CO", "H"]
    cfgs = [_StubAdsorbateConfig("CO"), _StubAdsorbateConfig("H")]
    return CompositionGPR(species=species, ads_configs=cfgs,
                          min_total=1, max_total=4)


# ---------------------------------------------------------------------------
# is_ready / counts_to_vec / random_composition
# ---------------------------------------------------------------------------

class TestCompositionGPRBasics:

    def test_not_ready_before_fit(self, gpr_two_species):
        assert gpr_two_species.is_ready is False

    def test_counts_to_vec_orders_by_species(self, gpr_two_species):
        v = gpr_two_species._counts_to_vec({"H": 2, "CO": 3})
        # species ordering is ["CO", "H"]
        assert list(v) == [3.0, 2.0]

    def test_counts_to_vec_missing_species_zero(self, gpr_two_species):
        v = gpr_two_species._counts_to_vec({"CO": 1})
        assert list(v) == [1.0, 0.0]

    def test_random_composition_respects_bounds(self, gpr_two_species):
        rng = np.random.default_rng(0)
        for _ in range(50):
            comp = gpr_two_species._random_composition(rng)
            total = sum(comp.values())
            assert 1 <= total <= 4
            assert all(0 <= comp[s] <= 4 for s in ("CO", "H"))

    def test_predict_returns_zero_when_untrained(self, gpr_two_species):
        mean, std = gpr_two_species.predict({"CO": 1, "H": 0})
        assert mean == 0.0
        assert std == float("inf")


# ---------------------------------------------------------------------------
# fit / predict / suggest
# ---------------------------------------------------------------------------

class TestCompositionGPRFit:

    def test_fit_with_too_few_samples_is_noop(self, gpr_two_species):
        gpr_two_species.fit([{"CO": 1, "H": 0}], [-1.0])
        assert gpr_two_species.is_ready is False

    def test_fit_with_synthetic_data_trains(self, gpr_two_species):
        # Synthetic linear-ish landscape: GCE drops with more CO, rises with H.
        rng = np.random.default_rng(0)
        compositions = []
        energies = []
        for n_co in range(0, 5):
            for n_h in range(0, 5):
                if 1 <= n_co + n_h <= 4:
                    compositions.append({"CO": n_co, "H": n_h})
                    energies.append(-1.0 * n_co + 0.5 * n_h
                                    + rng.normal(0, 0.05))
        gpr_two_species.fit(compositions, energies)
        assert gpr_two_species.is_ready is True
        assert gpr_two_species._n_train > 0

    def test_fit_dedupes_keeping_best_energy(self, gpr_two_species):
        # Three observations of the same composition; only the lowest survives.
        compositions = [
            {"CO": 1, "H": 0},
            {"CO": 1, "H": 0},
            {"CO": 1, "H": 0},
            {"CO": 2, "H": 1},  # second unique point so fit() proceeds
        ]
        energies = [-0.5, -1.0, -0.8, -2.0]
        gpr_two_species.fit(compositions, energies)
        assert gpr_two_species.is_ready
        # Two unique compositions, not four
        assert gpr_two_species._n_train == 2

    def test_predict_after_fit_returns_finite(self, gpr_two_species):
        compositions = [
            {"CO": 1, "H": 0},
            {"CO": 2, "H": 0},
            {"CO": 0, "H": 1},
        ]
        energies = [-1.0, -1.5, 0.5]
        gpr_two_species.fit(compositions, energies)
        mean, std = gpr_two_species.predict({"CO": 3, "H": 0})
        assert np.isfinite(mean)
        assert std >= 0.0

    def test_suggest_falls_back_when_untrained(self, gpr_two_species):
        rng = np.random.default_rng(0)
        comp = gpr_two_species.suggest(rng)
        assert isinstance(comp, dict)
        assert sum(comp.values()) >= 1

    def test_suggest_returns_valid_composition_after_fit(self, gpr_two_species):
        compositions = [{"CO": i, "H": 0} for i in range(1, 5)]
        energies = [-float(i) for i in range(1, 5)]
        gpr_two_species.fit(compositions, energies)

        rng = np.random.default_rng(0)
        comp = gpr_two_species.suggest(rng, n_candidates=20, kappa=1.0)
        assert isinstance(comp, dict)
        total = sum(comp.values())
        assert 1 <= total <= 4
