"""
This package provides utility functions for computing various metrics,
 used in landmark removal algorithms.
"""

from .utils import (
    compute_degree,
    compute_uncertainty,
    k_cover_algorithm,
    compute_mutual_information,
    compute_reprojection_error
)
from .evals import (
    compute_ate,
    compute_are,
    compute_ud
)

__all__ = [
    'compute_degree',
    'compute_uncertainty',
    'k_cover_algorithm',
    'compute_mutual_information',
    'compute_reprojection_error',
    'compute_ate',
    'compute_are',
    'compute_ud'
]
