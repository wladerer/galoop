"""
galoop/galoop.py

Backwards-compat shim. The GA loop was split into three focused modules
during the Phase 2 refactor:

  - galoop.loop      — main `run()` event loop, signal handling, stop checks
  - galoop.harvest   — post-relax classification, SOAP cache, validity gates
  - galoop.spawn     — initial population, operators, GPR / random fallback

Existing imports (`from galoop.galoop import run`, `_build_initial_population`,
`_spawn_one`, etc.) keep working through this module's re-exports so test
fixtures and CLI subcommands don't need to be touched.
"""

from __future__ import annotations

from galoop.harvest import (
    best_similarity as _best_similarity,
)

# Internals re-exported under their pre-split names for back-compat with
# tests/cli.py. New code should import from galoop.harvest / galoop.spawn
# directly.
from galoop.harvest import (
    handle_converged as _handle_converged,
)
from galoop.harvest import (
    has_atom_overlap as _has_atom_overlap,
)
from galoop.harvest import (
    is_prerelax_duplicate as _is_prerelax_duplicate,
)
from galoop.harvest import (
    read_final_energy as _read_energy,
)
from galoop.harvest import (
    rebuild_struct_cache as _rebuild_struct_cache,
)

# Public API
from galoop.loop import _should_stop, _stop_requested, run
from galoop.spawn import (
    build_initial_population as _build_initial_population,
)
from galoop.spawn import (
    fill_workers as _fill_workers,
)
from galoop.spawn import (
    infer_adsorbate_counts as _infer_adsorbate_counts,
)
from galoop.spawn import (
    random_stoichiometry as _random_stoichiometry,
)
from galoop.spawn import (
    retrain_gpr as _retrain_gpr,
)
from galoop.spawn import (
    sample_operator as _sample_operator,
)
from galoop.spawn import (
    snap_to_surface as _snap_to_surface,
)
from galoop.spawn import (
    spawn_one as _spawn_one,
)
from galoop.spawn import (
    spawn_random as _place_random,
)
from galoop.spawn import (
    spawn_via_gpr as _spawn_gpr,
)

__all__ = [
    "_best_similarity",
    "_build_initial_population",
    "_fill_workers",
    "_handle_converged",
    "_has_atom_overlap",
    "_infer_adsorbate_counts",
    "_is_prerelax_duplicate",
    "_place_random",
    "_random_stoichiometry",
    "_read_energy",
    "_rebuild_struct_cache",
    "_retrain_gpr",
    "_sample_operator",
    "_should_stop",
    "_snap_to_surface",
    "_spawn_gpr",
    "_spawn_one",
    "_stop_requested",
    "run",
]
