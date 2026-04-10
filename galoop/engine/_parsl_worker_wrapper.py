"""Worker entry point that installs rotating log handlers, then runs
parsl's ``process_worker_pool`` script as if it were ``__main__``.

The default Parsl HTEX launch_cmd invokes ``process_worker_pool.py``
directly, which means each worker subprocess starts a fresh Python
interpreter that has not seen any of galoop's runtime patches. The
:func:`galoop.engine._parsl_log_rotation.install_rotating_handler`
monkey-patch installed in the main galoop process therefore does NOT
apply to workers — they need their own copy of the patch installed
before parsl's worker code runs.

This module is that wrapper. We override Parsl HTEX's ``launch_cmd``
in :mod:`galoop.engine.scheduler` to invoke
``python -m galoop.engine._parsl_worker_wrapper ...`` instead of
``process_worker_pool.py ...``. The wrapper:

1. Installs the rotating-log patch (which rebinds
   ``parsl.log_utils.set_file_logger`` and walks ``sys.modules``).
2. Hands off to parsl's actual worker entry point via :mod:`runpy`,
   so all the original ``__main__`` argument parsing and lifecycle
   logic runs unchanged.

The result is that ``manager.log`` in every worker is a
:class:`RotatingFileHandler` instead of a plain
:class:`FileHandler`, capping the per-worker on-disk footprint.
"""
from __future__ import annotations


def main() -> None:
    # Install BEFORE importing parsl modules that grab set_file_logger
    # by name. The wrapper module imports nothing from parsl at the
    # top level, and we install before runpy.run_module so the
    # process_worker_pool's `from parsl.log_utils import set_file_logger`
    # picks up the patched version.
    from galoop.engine._parsl_log_rotation import install_rotating_handler

    install_rotating_handler()

    import runpy

    # Use runpy with run_name="__main__" so the script's
    # `if __name__ == "__main__":` block runs as if invoked via
    # `python -m parsl.executors.high_throughput.process_worker_pool`.
    # sys.argv is inherited from our own invocation, so all the
    # --logdir / --port / --block_id flags reach the worker unchanged.
    runpy.run_module(
        "parsl.executors.high_throughput.process_worker_pool",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
