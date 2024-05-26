"""
This module defines the Agent class for representing an agent with position mean and covariance.
"""

# source/agents/agent.py

from gtsam import Pose3, Rot3
import numpy as np

class Agent:
    def __init__(self, position: Pose3):
        self._position = position
        self._position_covariance = np.eye(3)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, Pose3):
            self._position = value
        elif isinstance(value, np.ndarray) and value.shape == (3,):
            self._position = Pose3(Rot3(), value.reshape((3, 1)))
        else:
            raise ValueError("Position must be a gtsam.Pose3 or a numpy array of shape (3,)")

    @property
    def position_covariance(self):
        return self._position_covariance

    @position_covariance.setter
    def position_covariance(self, value):
        # print(f"Setting position covariance: {value}")
        if isinstance(value, np.ndarray) and value.shape == (3, 3):
            self._position_covariance = value
        else:
            raise ValueError("Position covariance must be a 3x3 NumPy array.")
