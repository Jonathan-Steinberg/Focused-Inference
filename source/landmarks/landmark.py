"""
This module defines the Landmark class for representing
 landmarks with position mean and covariance.
"""

import numpy as np
from .utils import *

class Landmark:
    """
    Represents a landmark with position mean and covariance.
    """

    def __init__(self, identifier, initial_position_mean, initial_position_covariance):
        """
        Initialize the Landmark with its position distribution and an identifier.

        Args:
            identifier (any): An identifier for the landmark.
            initial_position_mean (numpy.ndarray): A 3-element array for
             the initial mean position (x, y, z).
            initial_position_covariance (numpy.ndarray): A 3x3 array for
             the initial position covariance.
        """
        self._identifier = identifier
        self._position_mean = initial_position_mean
        self._position_covariance = initial_position_covariance
        self.observations = []
        self._position_history = [(initial_position_mean, initial_position_covariance)]

    @property
    def identifier(self):
        """Get the identifier of the landmark."""
        return self._identifier

    @property
    def position_mean(self):
        """Get the mean position of the landmark."""
        return self._position_mean

    @position_mean.setter
    def position_mean(self, new_position_mean):
        """Set a new mean position for the landmark."""
        if (not isinstance(new_position_mean, np.ndarray) or
                new_position_mean.shape != (3,)):
            raise ValueError("Position mean must be a 3-element NumPy array.")
        self._position_mean = new_position_mean

    @property
    def position_covariance(self):
        """Get the position covariance of the landmark."""
        return self._position_covariance

    @position_covariance.setter
    def position_covariance(self, new_position_covariance):
        """Set a new position covariance for the landmark."""
        if (not isinstance(new_position_covariance, np.ndarray) or
                new_position_covariance.shape != (3, 3)):
            raise ValueError("Position covariance must be a 3x3 NumPy array.")
        self._position_covariance = new_position_covariance

    def update_position(self, new_position_mean, new_position_covariance):
        """
        Update the position of the landmark.

        Args:
            new_position_mean (numpy.ndarray): The new mean position of the landmark.
            new_position_covariance (numpy.ndarray): The new covariance matrix
            of the landmark's position.
        """
        self.position_mean = new_position_mean
        self.position_covariance = new_position_covariance

    def get_position(self):
        """
        Get the current position of the landmark.

        Returns:
            tuple: A tuple containing the position mean and covariance.
        """
        return self.position_mean, self.position_covariance

    def add_observation(self, observation):
        """
        Add an observation related to the landmark.

        Args:
            observation (any): The observation to add.
        """
        self.observations.append(observation)

    def get_observations(self):
        """
        Get all observations related to the landmark.

        Returns:
            list: A list of observations.
        """
        return self.observations

    def get_position_history(self):
        """
        Get the history of the landmark's positions and associated uncertainties.

        Returns:
            list: A list of tuples, each containing a position mean
             and its corresponding covariance.
        """
        return self._position_history

    def __repr__(self):
        """Provide a detailed string representation of the landmark."""
        return (f"Landmark(id={self.identifier},"
                f" position_mean={self.position_mean},"
                f" position_covariance={self.position_covariance})")
