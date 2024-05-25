"""
This module provides utility functions for computing various metrics,
 used in landmark removal algorithms.
"""

import numpy as np


def compute_degree(landmarks, poses):
    """
    Compute the degree of each landmark.

    Args:
        landmarks (list): List of Landmark objects.
        poses (list): List of poses.

    Returns:
        dict: A dictionary with landmarks' identifiers as keys and their degrees as values.
    """
    degrees = {lm.identifier: 0 for lm in landmarks}

    # Example: Each landmark observed by a random subset of poses
    for _ in poses:
        observed_landmarks = np.random.choice(landmarks,
                                              size=np.random.randint(1, len(landmarks)),
                                              replace=False)
        for lm in observed_landmarks:
            degrees[lm.identifier] += 1

    return degrees


def compute_uncertainty(landmarks):
    """
    Compute the uncertainty of each landmark.

    Args:
        landmarks (list): List of Landmark objects.

    Returns:
        dict: A dictionary with landmarks' identifiers as keys and their uncertainties as values.
    """
    uncertainties = {}
    for lm in landmarks:
        uncertainties[lm.identifier] = np.trace(lm.position_covariance)

    return uncertainties


def k_cover_algorithm(landmarks, poses, k):
    """
    Run the K-Cover algorithm to find the K-cover landmarks.

    Args:
        landmarks (list): List of Landmark objects.
        poses (list): List of poses.
        k (int): Number of covers.

    Returns:
        list: A list of Landmark objects that cover the most poses.
    """
    cover_landmarks = []
    uncovered_poses = set(poses)

    while len(uncovered_poses) > 0 and len(cover_landmarks) < k:
        best_landmark = None
        best_cover = 0

        for lm in landmarks:
            # Randomly decide if a landmark covers a pose
            cover_count = sum(
                1 for pose in uncovered_poses if np.random.rand() > 0.5)
            if cover_count > best_cover:
                best_cover = cover_count
                best_landmark = lm

        if best_landmark:
            cover_landmarks.append(best_landmark)
            uncovered_poses -= set(np.random.choice(list(uncovered_poses),
                                                    size=best_cover,
                                                    replace=False))

    return cover_landmarks


def compute_mutual_information(landmarks, poses):
    """
    Compute the mutual information of each landmark,
    w.r.t the rest of the landmarks and poses.

    Args:
        landmarks (list): List of Landmark objects.
        poses (list): List of poses.

    Returns:
        dict: A dictionary with landmarks' identifiers as keys
         and their mutual information as values.
    """
    mutual_information = {lm.identifier: np.random.rand() for lm in
                          landmarks}  # Placeholder: Random value for mutual information

    return mutual_information


def compute_reprojection_error(landmarks, poses):
    """
    Compute the reprojection error of each landmark.

    Args:
        landmarks (list): List of Landmark objects.
        poses (list): List of poses.

    Returns:
        dict: A dictionary with landmarks' identifiers as keys
        and their reprojection errors as values.
    """
    reprojection_errors = {lm.identifier: np.random.rand() for lm in
                           landmarks}  # Placeholder: Random value for reprojection error

    return reprojection_errors
