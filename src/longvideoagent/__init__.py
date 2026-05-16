"""LongVideoEditAgent — instruction-driven long-video editing.

Top-level package. Public re-exports are kept minimal; everything below the
package boundary is importable as ``longvideoagent.<subpackage>.<module>``.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("longvideoagent")
except PackageNotFoundError:        # editable install before metadata is built
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
