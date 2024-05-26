"""
This module defines the LandmarkRemoval class,
 which contains various algorithms for removing landmarks.
"""

import numpy as np

from source.info_theoretic import (
    compute_degree,
    compute_uncertainty,
    k_cover_algorithm,
    compute_mutual_information,
    compute_reprojection_error
)
from source.landmarks.landmark import Landmark

class LandmarkRemoval:
    """
    LandmarkRemoval class contains various algorithms for removing landmarks.
    """

    def __init__(self, landmarks, poses):
        """
        Initialize the LandmarkRemoval with landmarks and poses.

        Args:
            landmarks (list): List of Landmark objects.
            poses (list): List of poses.
        """
        self._landmarks = {lm.identifier: lm for lm in landmarks}
        self._poses = poses

    @property
    def landmarks(self):
        """Get the list of landmarks."""
        return list(self._landmarks.values())

    @landmarks.setter
    def landmarks(self, new_landmarks):
        """Set a new list of landmarks."""
        if not isinstance(new_landmarks, list):
            raise ValueError("Landmarks must be a list.")
        self._landmarks = {lm.identifier: lm for lm in new_landmarks}

    @property
    def poses(self):
        """Get the list of poses."""
        return self._poses

    @poses.setter
    def poses(self, new_poses):
        """Set a new list of poses."""
        if not isinstance(new_poses, list):
            raise ValueError("Poses must be a list.")
        self._poses = new_poses

    def least_degree_removal(self):
        """
        Remove landmarks based on the least degree.
        """
        degrees = compute_degree(self.landmarks, self.poses)
        sorted_landmarks = sorted(self.landmarks, key=lambda lm: degrees[lm.identifier])
        self._landmarks = {lm.identifier: lm for lm in sorted_landmarks}
        return self.landmarks

    def max_uncertainty_removal(self):
        """
        Remove landmarks based on maximum uncertainty.
        """
        uncertainties = compute_uncertainty(self.landmarks)
        sorted_landmarks = sorted(self.landmarks,
                                  key=lambda lm: uncertainties[lm.identifier],
                                  reverse=True)
        self._landmarks = {lm.identifier: lm for lm in sorted_landmarks}
        return self.landmarks

    def k_cover_removal(self, k=1):
        """
        Remove landmarks based on the K-Cover algorithm.

        Args:
            k (int): Number of covers.
        """
        k_cover_landmarks = k_cover_algorithm(self.landmarks, self.poses, k)
        sorted_landmarks = list(reversed(k_cover_landmarks))
        self._landmarks = {lm.identifier: lm for lm in sorted_landmarks}
        return self.landmarks

    def least_informative_removal(self):
        """
        Remove landmarks based on the least informative criterion.
        """
        while len(self.landmarks) > 0:
            infos = compute_mutual_information(self.landmarks, self.poses)
            least_informative_landmark = min(infos, key=infos.get)
            del self._landmarks[least_informative_landmark]
        return self.landmarks

    def least_reprojection_error_removal(self):
        """
        Remove landmarks based on the least reprojection error.
        """
        reprojection_errors = compute_reprojection_error(self.landmarks, self.poses)
        sorted_landmarks = sorted(self.landmarks,
                                  key=lambda lm: reprojection_errors[lm.identifier])
        self._landmarks = {lm.identifier: lm for lm in sorted_landmarks}
        return self.landmarks

    def _remove_landmarks(self, sorted_landmarks):
        """
        Helper function to remove landmarks in the given order.

        Args:
            sorted_landmarks (list): List of sorted landmarks to remove.
        """
        for landmark in sorted_landmarks:
            del self._landmarks[landmark.identifier]
        return self.landmarks

    @staticmethod
    def get_algorithm_names():
        """Get the names of the algorithms."""
        return [
            "least_degree_removal",
            "max_uncertainty_removal",
            "k_cover_removal",
            "least_informative_removal",
            "least_reprojection_error_removal"
        ]


# Example usage:
if __name__ == "__main__":
    landmarks = [
        Landmark(identifier=i,
                 initial_position_mean=np.random.rand(3),
                 initial_position_covariance=np.eye(3))
        for i in range(10)
    ]
    poses = [...]  # List of poses
    remover = LandmarkRemoval(landmarks, poses)
    remaining_landmarks = remover.least_degree_removal()
    print("Remaining Landmarks:", remaining_landmarks)
