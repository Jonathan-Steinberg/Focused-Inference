"""
This module provides functions to evaluate SLAM performance using different metrics:
- Average Trajectory Error (ATE)
- Average Rotation Error (ARE)
- Difference in Determinant of Uncertainty (UD)
"""

import numpy as np

def compute_ate(estimated_poses, ground_truth_poses):
    """
    Compute the Average Trajectory Error (ATE).

    Args:
        estimated_poses (numpy.ndarray): Array of estimated poses (N, 3).
        ground_truth_poses (numpy.ndarray): Array of ground truth poses (N, 3).

    Returns:
        float: The root mean squared error (RMSE) between estimated and ground truth poses.
    """
    if estimated_poses.shape != ground_truth_poses.shape:
        raise ValueError("Estimated and ground truth poses must have the same shape.")

    errors = np.linalg.norm(estimated_poses - ground_truth_poses, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    return ate


def compute_are(estimated_rotations, ground_truth_rotations):
    """
    Compute the Average Rotation Error (ARE).

    Args:
        estimated_rotations (numpy.ndarray): Array of estimated rotations (N, 3).
        ground_truth_rotations (numpy.ndarray): Array of ground truth rotations (N, 3).

    Returns:
        float: The average angular difference in degrees between estimated and ground truth rotations.
    """
    if estimated_rotations.shape != ground_truth_rotations.shape:
        raise ValueError("Estimated and ground truth rotations must have the same shape.")

    dot_products = np.einsum('ij,ij->i',
                             estimated_rotations.reshape(-1, 3),
                             ground_truth_rotations.reshape(-1, 3))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * (180.0 / np.pi)
    are = np.mean(angular_errors)
    return are


def compute_ud(estimated_covariance, ground_truth_covariance):
    """
    Compute the Difference in Determinant of Uncertainty (UD).

    Args:
        estimated_covariance (numpy.ndarray): Covariance matrix of the estimated pose (3, 3).
        ground_truth_covariance (numpy.ndarray): Covariance matrix of the ground truth pose (3, 3).

    Returns:
        float: The difference in the determinant of the covariance matrices.
    """
    if estimated_covariance.shape != (3, 3) or ground_truth_covariance.shape != (3, 3):
        raise ValueError("Covariance matrices must be of shape (3, 3).")

    det_estimated = np.linalg.det(estimated_covariance)
    det_ground_truth = np.linalg.det(ground_truth_covariance)
    ud = det_estimated - det_ground_truth
    return ud
