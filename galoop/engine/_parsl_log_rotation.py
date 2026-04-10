"""Install rotating-file log handlers on parsl's logger.

Bug 5 (validation sweep, 2026-04-09): every parsl HTEX worker writes
``manager.log`` unbounded. Workers that survive their parent — after a
``pkill``, segfault, pytest interruption, or any other un-cleaned exit —
keep writing forever, leaking multi-GB log files into ``/tmp`` and
``runs/`` until the disk fills up. The leak observed in the validation
sweep was 30+ GB across 18 orphaned workers, several of which were
holding 6+ GB ``manager.log`` files open on disk.

Fix: monkey-patch :func:`parsl.log_utils.set_file_logger` to use a
:class:`logging.handlers.RotatingFileHandler` that caps each log file at
``GALOOP_PARSL_LOG_MAX_BYTES`` with ``GALOOP_PARSL_LOG_BACKUPS`` rotated
files kept. The worst-case per-worker on-disk footprint is therefore
``MAX_BYTES * (1 + BACKUP_COUNT)`` instead of unbounded.

Defaults: 50 MiB per file, 3 backups → 200 MiB worst case per worker
(vs the multi-GB unbounded growth that motivated this fix).

Patching strategy:

- Replace :func:`parsl.log_utils.set_file_logger` so any future caller
  picks up the rotating variant.
- Walk :data:`sys.modules` and rebind any already-imported parsl
  module that grabbed a reference via ``from parsl.log_utils import
  set_file_logger`` (e.g. ``process_worker_pool``, ``interchange``,
  ``db_manager``, the monitoring radios). Without this step, modules
  that imported parsl before the patch would still hold the original
  unrotated function.
- Idempotent — repeated calls are no-ops.

This module is imported at the top of :mod:`galoop.engine.scheduler`
so the main galoop process gets the patch as soon as anything from
``engine`` is touched. Worker subprocesses get the patch via
:mod:`galoop.engine._parsl_worker_wrapper`, which they invoke as
``python -m galoop.engine._parsl_worker_wrapper`` instead of the
default ``process_worker_pool.py`` script.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Callable

# Per-file cap before rotation kicks in. 50 MiB is enough to capture a
# couple of hours of DEBUG-level worker chatter under normal load.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024

# Number of rotated backup files kept on disk.
DEFAULT_BACKUP_COUNT = 3

_PATCHED = False


def install_rotating_handler() -> None:
    """Patch parsl's file-logger to use rotation. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    max_bytes = int(os.environ.get("GALOOP_PARSL_LOG_MAX_BYTES", DEFAULT_MAX_BYTES))
    backup_count = int(os.environ.get("GALOOP_PARSL_LOG_BACKUPS", DEFAULT_BACKUP_COUNT))

    # Lazy import so this module can be imported even if parsl isn't
    # available yet (e.g. by the worker wrapper before parsl is on
    # PYTHONPATH).
    import parsl.log_utils as log_utils

    _orig = log_utils.set_file_logger
    _DEFAULT_FORMAT = log_utils.DEFAULT_FORMAT

    def set_file_logger_rotating(
        filename: str,
        name: str = "parsl",
        level: int = logging.DEBUG,
        format_string: str | None = None,
    ) -> Callable[[], None]:
        """Drop-in replacement for parsl.log_utils.set_file_logger.

        Same signature, same return value (a callback that removes the
        installed handler), but uses :class:`RotatingFileHandler` so
        long-running workers can't grow the log file without bound.
        """
        if format_string is None:
            format_string = _DEFAULT_FORMAT

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        handler = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setLevel(level)
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # parsl also routes concurrent.futures' logger through the same
        # file handler — preserve that behavior so we don't accidentally
        # silence those messages.
        futures_logger = logging.getLogger("concurrent.futures")
        futures_logger.addHandler(handler)

        def unregister_callback():
            logger.removeHandler(handler)
            futures_logger.removeHandler(handler)

        return unregister_callback

    # Patch the canonical home in log_utils.
    log_utils.set_file_logger = set_file_logger_rotating

    # Walk already-imported parsl modules and rebind any that grabbed
    # set_file_logger by name. Without this, anything that did
    # `from parsl.log_utils import set_file_logger` before the patch
    # would still hold the original.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("parsl"):
            continue
        if getattr(mod, "set_file_logger", None) is _orig:
            try:
                setattr(mod, "set_file_logger", set_file_logger_rotating)
            except (AttributeError, TypeError):
                # Some modules expose read-only attributes; skip them.
                pass

    _PATCHED = True
