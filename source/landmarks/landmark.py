"""
This module provides the Landmark class for representing landmarks with position mean and covariance.
"""

# source/landmarks/landmark.py

import numpy as np
from gtsam import Pose3, Rot3

class Landmark:
    def __init__(self, position, covariance, identifier: int):
        if not isinstance(position, Pose3):
            self._position = Pose3(Rot3(), position.reshape((3, 1)))
        else:
            self._position = position
        self._covariance = covariance
        self.identifier = identifier

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, Pose3):
            self._position = value
        else:
            self._position = Pose3(Rot3(), value.reshape((3, 1)))

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if value.shape == (3, 3):
            self._covariance = value
        else:
            raise ValueError("Covariance must be a 3x3 NumPy array.")

    def update_position(self, new_position, new_covariance):
        self.position = new_position
        self.covariance = new_covariance
