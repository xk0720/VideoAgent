"""Console-script entry points (declared in pyproject.toml [project.scripts]).

These are thin wrappers around the matching modules in ``scripts/`` so
``pip install -e .`` gives the user lva-preprocess / lva-run / etc.
"""
from __future__ import annotations


def preprocess_main():
    from .scripts_impl import preprocess_main as _impl
    return _impl()


def build_memory_main():
    from .scripts_impl import build_memory_main as _impl
    return _impl()


def run_pipeline_main():
    from .scripts_impl import run_pipeline_main as _impl
    return _impl()


def eval_main():
    from .scripts_impl import eval_main as _impl
    return _impl()


def viz_trajectory_main():
    from .scripts_impl import viz_trajectory_main as _impl
    return _impl()
