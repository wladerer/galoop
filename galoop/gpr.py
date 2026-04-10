"""
galoop/gpr.py

Gaussian Process Regression surrogate for composition-guided spawning.

Trains a GP on converged structures to predict grand canonical energy
as a function of adsorbate composition, then suggests promising
compositions via Upper Confidence Bound acquisition.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UCB exploration schedule
# ---------------------------------------------------------------------------

def effective_kappa(config, total_evals: int, stall_count: int) -> float:
    """Return the UCB exploration parameter for the current loop state.

    Two effects are combined:

    - **Exponential decay** of ``ga.gpr_kappa`` toward ``ga.gpr_kappa_min``
      with half-life ``ga.gpr_kappa_half_life`` (in converged evaluations).
      Set ``half_life`` to 0 to disable decay entirely (legacy behavior).
    - **Stall-driven boost** that adds ``ga.gpr_kappa_stall_boost``
      multiplied by ``stall_count / max_stall`` so a run that stops
      improving forces re-exploration without operator intervention.

    The returned value is the kappa to pass to :func:`CompositionGPR.suggest`
    on this poll cycle.
    """
    k0 = float(config.ga.gpr_kappa)
    k_min = float(config.ga.gpr_kappa_min)
    half_life = int(getattr(config.ga, "gpr_kappa_half_life", 0))
    boost = float(getattr(config.ga, "gpr_kappa_stall_boost", 0.0))
    max_stall = max(1, int(config.ga.max_stall))

    if half_life > 0:
        decayed = k0 * (0.5 ** (total_evals / half_life))
        base = max(k_min, decayed)
    else:
        base = k0

    stall_frac = min(1.0, max(0, stall_count) / max_stall)
    return base + boost * stall_frac


class CompositionGPR:
    """GPR surrogate over adsorbate composition space.

    Features are integer adsorbate count vectors (e.g., [n_O, n_OH, n_H]).
    Target is grand canonical energy (lower = better).

    Parameters
    ----------
    species : ordered list of adsorbate symbols (from config)
    ads_configs : list of AdsorbateConfig objects (for per-species bounds)
    min_total : minimum total adsorbate count
    max_total : maximum total adsorbate count
    """

    def __init__(
        self,
        species: list[str],
        ads_configs: list,
        min_total: int = 1,
        max_total: int = 8,
    ) -> None:
        self.species = species
        self.ads_configs = ads_configs
        self.min_total = min_total
        self.max_total = max_total
        self._gp = None
        self._X_train = None
        self._y_train = None
        self._n_train = 0

    def _counts_to_vec(self, counts: dict[str, int]) -> np.ndarray:
        """Convert adsorbate count dict to feature vector."""
        return np.array([counts.get(s, 0) for s in self.species], dtype=float)

    def fit(self, compositions: list[dict], energies: list[float]) -> None:
        """Train the GP on observed composition → GCE data.

        Deduplicates compositions by taking the minimum energy at each
        unique composition (best structure per stoichiometry).
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        if len(compositions) < 2:
            return

        # Deduplicate: best energy per composition
        comp_best: dict[tuple, float] = {}
        for counts, energy in zip(compositions, energies, strict=True):
            key = tuple(counts.get(s, 0) for s in self.species)
            if key not in comp_best or energy < comp_best[key]:
                comp_best[key] = energy

        X = np.array(list(comp_best.keys()), dtype=float)
        y = np.array(list(comp_best.values()), dtype=float)

        if len(X) < 2:
            return

        # Normalize y for numerical stability
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 1e-8 else 1.0
        y_norm = (y - self._y_mean) / self._y_std

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=False,
            alpha=1e-6,
        )
        self._gp.fit(X, y_norm)
        self._X_train = X
        self._y_train = y
        self._n_train = len(X)

        log.debug("GPR trained on %d unique compositions", len(X))

    def predict(self, counts: dict[str, int]) -> tuple[float, float]:
        """Predict GCE mean and std for a composition.

        Returns (mean, std) in original energy units.
        """
        if self._gp is None:
            return 0.0, float("inf")
        x = self._counts_to_vec(counts).reshape(1, -1)
        mean_norm, std_norm = self._gp.predict(x, return_std=True)
        mean = float(mean_norm[0]) * self._y_std + self._y_mean
        std = float(std_norm[0]) * self._y_std
        return mean, std

    def suggest(
        self,
        rng: np.random.Generator,
        n_candidates: int = 200,
        kappa: float = 1.5,
    ) -> dict[str, int]:
        """Suggest a composition via UCB acquisition.

        Samples random candidate compositions, evaluates GP, and returns
        the one with the best (lowest) UCB score:

            UCB(x) = mean(x) - kappa * std(x)

        Lower UCB means either predicted-low-energy or high-uncertainty
        (exploration).

        Parameters
        ----------
        rng : numpy random generator
        n_candidates : number of random compositions to evaluate
        kappa : exploration parameter (higher = more exploration)

        Returns
        -------
        dict[str, int] — suggested adsorbate counts
        """
        if self._gp is None:
            return self._random_composition(rng)

        # Sample candidate compositions
        candidates = []
        for _ in range(n_candidates):
            comp = self._random_composition(rng)
            candidates.append(comp)

        # Evaluate GP on all candidates
        X_cand = np.array([self._counts_to_vec(c) for c in candidates])
        means_norm, stds_norm = self._gp.predict(X_cand, return_std=True)
        means = means_norm * self._y_std + self._y_mean
        stds = stds_norm * self._y_std

        # UCB acquisition (minimize)
        ucb = means - kappa * stds
        best_idx = np.argmin(ucb)

        chosen = candidates[best_idx]
        mean_chosen, std_chosen = means[best_idx], stds[best_idx]
        log.debug(
            "GPR suggest (kappa=%.3f): %s  predicted=%.3f±%.3f eV  UCB=%.3f",
            kappa, chosen, mean_chosen, std_chosen, ucb[best_idx],
        )
        return chosen

    def _random_composition(self, rng: np.random.Generator) -> dict[str, int]:
        """Sample a random valid composition within bounds."""
        counts = {}
        for cfg in self.ads_configs:
            counts[cfg.symbol] = int(rng.integers(cfg.min_count, cfg.max_count + 1))

        total = sum(counts.values())
        # Shrink if over max
        while total > self.max_total:
            shrinkable = [
                s for s in counts
                if counts[s] > next(c.min_count for c in self.ads_configs if c.symbol == s)
            ]
            if not shrinkable:
                break
            counts[str(rng.choice(shrinkable))] -= 1
            total -= 1
        # Grow if under min
        while total < self.min_total:
            growable = [
                s for s in counts
                if counts[s] < next(c.max_count for c in self.ads_configs if c.symbol == s)
            ]
            if not growable:
                break
            counts[str(rng.choice(growable))] += 1
            total += 1

        return counts

    @property
    def is_ready(self) -> bool:
        """True if the GP has been trained with enough data."""
        return self._gp is not None and self._n_train >= 2
