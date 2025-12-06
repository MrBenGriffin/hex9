"""
Part of the H9 project
e.g. Domain-specific, but widely reusable put it in domains/
Core infrastructure (used across points, formats, projections, etc.) → put it in base/
H9 Subpackage Initialization.
Order matters here to avoid circular imports.
"""

from .constants import H9Const, H9K
from .classifier import H9Classifier, H9CL, classify_cell, in_scope, in_down, in_up
from .lattice import H9C, H9Cell
from .region import H9R, H9Region
from .addressing import H9_RA, HEX_LUTS
from .polygon import H9P, H9Polygon

__all__ = [
    "H9K", "H9Const",
    "H9CL", "H9Classifier",
    "H9C", "H9Cell",
    "H9R", "H9Region",
    "H9P", "H9Polygon",
    "H9_RA", "HEX_LUTS",
    "classify_cell",
    "in_scope", "in_down", "in_up"
]
