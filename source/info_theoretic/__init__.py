"""
This package provides utility functions for computing various metrics,
 used in landmark removal algorithms.
"""

from source.info_theoretic.utils import (
    log_det,
    compute_information_gain
)

from source.info_theoretic.evals import (
    compute_ate,
    compute_are,
    compute_ud
)

__all__ = [
    'compute_information_gain',
    'log_det',
    'compute_ate',
    'compute_are',
    'compute_ud'
]
