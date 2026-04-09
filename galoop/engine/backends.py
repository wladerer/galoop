"""
galoop/engine/backends.py

Pluggable calculator backend registry.

A **backend** is a factory callable that returns a configured ASE
``Calculator``. Factories are looked up by the ``type`` field of a stage
config, using two resolution strategies:

1. **Built-in name** — ``"mace"`` or ``"vasp"`` (or anything registered via
   :func:`register`) resolves through the module-level ``_BUILTIN`` dict.
2. **Import path** — any string containing ``":"`` (``"pkg.mod:func"``) is
   resolved via ``importlib.import_module`` + ``getattr``. This lets users
   bring their own MLIP with nothing more than a .py file on PYTHONPATH.

Backends that relax *inside* the calculator itself (e.g. VASP, where
``atoms.get_potential_energy()`` triggers a full DFT relaxation) set
``drives_own_relaxation=True``. Backends whose calculator only provides
energies + forces (MACE, most MLIPs) leave it False, and the
:class:`CalculatorStage` runs BFGS on the Python side.

An import-path factory may return either:
- a bare callable ``(params: dict) -> Calculator``, in which case
  ``drives_own_relaxation`` defaults to False, or
- a 2-tuple ``(callable, drives_own_relaxation: bool)``.

This module has no dependency on the rest of galoop — it imports only ASE
and the stdlib — so it's safe to import from Parsl workers.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable

from ase.calculators.calculator import Calculator as ASECalculator

log = logging.getLogger(__name__)

BackendFactory = Callable[[dict], ASECalculator]

# Registry entry: (factory, drives_own_relaxation)
_BUILTIN: dict[str, tuple[BackendFactory, bool]] = {}


def register(
    name: str,
    factory: BackendFactory,
    *,
    drives_own_relaxation: bool = False,
) -> None:
    """Register a built-in backend by name.

    Tests and user plugins can monkeypatch ``_BUILTIN`` directly; this is
    just the convenience entry point used at module import time.
    """
    _BUILTIN[name.lower()] = (factory, drives_own_relaxation)


def resolve(type_str: str) -> tuple[BackendFactory, bool]:
    """Resolve a yaml ``type:`` value to ``(factory, drives_own_relaxation)``.

    - ``"mace"`` / ``"vasp"`` → built-in registry lookup.
    - ``"pkg.mod:func"``      → import-path resolution via importlib.

    Raises
    ------
    ValueError
        if the name is unknown or the import path is malformed.
    ImportError
        if the import-path module cannot be imported.
    """
    if not isinstance(type_str, str) or not type_str:
        raise ValueError(f"calculator type must be a non-empty string, got {type_str!r}")

    if ":" in type_str:
        module_name, _, attr = type_str.partition(":")
        if not module_name or not attr:
            raise ValueError(
                f"import-path backend {type_str!r} must be 'pkg.mod:callable'"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import backend module {module_name!r}: {exc}"
            ) from exc
        if not hasattr(module, attr):
            raise ValueError(
                f"Module {module_name!r} has no attribute {attr!r}"
            )
        obj = getattr(module, attr)
        return _coerce_resolved(obj, origin=type_str)

    name = type_str.lower()
    if name not in _BUILTIN:
        raise ValueError(
            f"Unknown calculator type {type_str!r}. "
            f"Built-ins: {sorted(_BUILTIN)}. "
            "For a custom backend use a 'pkg.mod:callable' import path."
        )
    return _BUILTIN[name]


def _coerce_resolved(obj: Any, origin: str) -> tuple[BackendFactory, bool]:
    """Normalise an import-path target into a ``(factory, bool)`` pair."""
    if callable(obj):
        return (obj, False)
    if isinstance(obj, tuple) and len(obj) == 2:
        fn, drives = obj
        if callable(fn) and isinstance(drives, bool):
            return (fn, drives)
    raise ValueError(
        f"Backend {origin!r} must resolve to a callable or a "
        "(callable, drives_own_relaxation: bool) tuple; "
        f"got {type(obj).__name__}."
    )


# ---------------------------------------------------------------------------
# Built-in: MACE
# ---------------------------------------------------------------------------

# Module-level cache + lock (used to live on CalculatorStage). Keyed on the
# full params dict turned into a frozen key so any distinct model/device/dtype
# triple gets its own calculator instance.
_mace_cache: dict[tuple, Any] = {}
_mace_lock = threading.Lock()


def _mace_cache_key(params: dict) -> tuple:
    return (
        params.get("model", "medium"),
        params.get("device", "cpu"),
        params.get("dtype", "float32"),
    )


def _mace_factory(params: dict) -> ASECalculator:
    """Return a MACE calculator, memoized on (model, device, dtype).

    ``params`` keys:
        model  — model name ('small', 'medium', …) or path to a .pt file
        device — 'cpu' / 'cuda' / 'auto'
        dtype  — 'float32' / 'float64'
    """
    from pathlib import Path as _Path

    key = _mace_cache_key(params)
    if key in _mace_cache:
        return _mace_cache[key]

    with _mace_lock:
        if key in _mace_cache:
            return _mace_cache[key]

        model, device, dtype = key
        model_path = _Path(model)
        if model_path.exists():
            from mace.calculators import MACECalculator
            calc = MACECalculator(
                model_paths=str(model_path),
                device=device,
                default_dtype=dtype,
            )
        else:
            from mace.calculators import mace_mp
            calc = mace_mp(model=model, device=device, default_dtype=dtype)

        _mace_cache[key] = calc
        return calc


# ---------------------------------------------------------------------------
# Built-in: VASP
# ---------------------------------------------------------------------------

def _vasp_factory(params: dict) -> ASECalculator:
    """Return an ASE ``Vasp`` calculator configured from ``params['incar']``.

    The calculator itself drives the relaxation (via ``ibrion``/``nsw``), so
    this backend is registered with ``drives_own_relaxation=True``.
    ``params`` keys:
        incar      — dict of INCAR-style options (keys lowercased per ASE)
        directory  — optional work dir (usually set by CalculatorStage)
    """
    from ase.calculators.vasp import Vasp

    incar = dict(params.get("incar", {}))
    vasp_kwargs: dict[str, Any] = {
        "ismear": 0,
        "sigma": 0.05,
        "algo": "Fast",
        "lreal": "Auto",
        "lwave": False,
        "lcharg": False,
        "ibrion": 2,
        "ediff": 1e-5,
    }
    for key, val in incar.items():
        vasp_kwargs[key.lower()] = val

    directory = params.get("directory")
    if directory is not None:
        vasp_kwargs["directory"] = str(directory)

    return Vasp(**vasp_kwargs)


register("mace", _mace_factory, drives_own_relaxation=False)
register("vasp", _vasp_factory, drives_own_relaxation=True)


# ---------------------------------------------------------------------------
# Built-in: CHGNet
# ---------------------------------------------------------------------------

_chgnet_cache: dict[tuple, Any] = {}
_chgnet_lock = threading.Lock()


def _chgnet_factory(params: dict) -> ASECalculator:
    """Return a CHGNet ASE calculator, memoized on (model_path, device).

    ``params`` keys:
        model   — path to a .pt.gz checkpoint, or omit for the bundled default
        device  — 'cpu' / 'cuda' / 'auto'
    """
    model_path = params.get("model")
    device = params.get("device", "cpu")
    if device == "auto":
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    key = (model_path, device)
    if key in _chgnet_cache:
        return _chgnet_cache[key]

    with _chgnet_lock:
        if key in _chgnet_cache:
            return _chgnet_cache[key]

        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        model = CHGNet.from_file(model_path) if model_path else CHGNet.load()
        calc = CHGNetCalculator(model=model, use_device=device)
        _chgnet_cache[key] = calc
        return calc


register("chgnet", _chgnet_factory, drives_own_relaxation=False)
