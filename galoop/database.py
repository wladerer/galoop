"""
galoop/database.py  — DEPRECATED

GaloopDB (SQLite) has been replaced by GaloopProject (signac).
This module re-exports the config-diff utilities that some tests may still
import directly; everything else has moved to galoop.project.
"""

from galoop.project import diff_configs, _flatten_config  # noqa: F401

__all__ = ["diff_configs"]
