"""
This module defines the Agent class for representing an agent with position mean and covariance.
"""

import numpy as np


class Agent:
    """
    Represents an agent with a position mean and covariance.
    """

    def __init__(self, initial_position_mean, initial_position_covariance):
        """
        Initialize the Agent with its initial position distribution.

        Args:
            initial_position_mean (numpy.ndarray): A 3-element NumPy array
             representing the initial mean of the agent's position (x, y, z).

            initial_position_covariance (numpy.ndarray): A 3x3 NumPy array
             representing the initial covariance matrix of the agent's position.
        """
        self._position_mean = initial_position_mean
        self._position_covariance = initial_position_covariance
        self._position = [(initial_position_mean, initial_position_covariance)]

    @property
    def position_mean(self):
        """Get the mean position of the agent."""
        return self._position_mean

    @position_mean.setter
    def position_mean(self, new_position_mean):
        """Set a new mean position for the agent."""
        if not isinstance(new_position_mean, np.ndarray) or new_position_mean.shape != (3,):
            raise ValueError("Position mean must be a 3-element NumPy array.")
        self._position_mean = new_position_mean
        self._position.append((new_position_mean.copy(), self._position_covariance.copy()))

    @property
    def position_covariance(self):
        """Get the position covariance of the agent."""
        return self._position_covariance

    @position_covariance.setter
    def position_covariance(self, new_position_covariance):
        """Set a new position covariance for the agent."""
        if (not isinstance(new_position_covariance, np.ndarray) or
                new_position_covariance.shape != (3, 3)):
            raise ValueError("Position covariance must be a 3x3 NumPy array.")
        self._position_covariance = new_position_covariance
        self._position[-1] = (self._position_mean.copy(), new_position_covariance.copy())

    def get_position_uncertainty(self):
        """
        Get the uncertainty (covariance) associated with the agent's position.

        Returns:
            float: The trace of the position covariance matrix,
             representing the overall uncertainty.
        """
        return np.trace(self._position_covariance)

    def get_position(self):
        """
        Get the history of the agent's positions and associated uncertainties.

        Returns:
            list: A list of tuples, where each tuple contains a 3-element NumPy array
            representing the agent's position mean and a 3x3 NumPy array representing
             the position covariance at a specific timestep.
        """
        return self._position

    def move_agent(self, delta_position, new_position_covariance):
        """
        Update the agent's position based on a given position change and the new covariance.

        Args:
            delta_position (numpy.ndarray): A 3-element NumPy array
            representing the change in position (x, y, z).

            new_position_covariance (numpy.ndarray): A 3x3 NumPy array
             representing the new covariance matrix of the agent's position.
        """
        new_position_mean = self.position_mean + delta_position
        self.position_mean = new_position_mean
        self.position_covariance = new_position_covariance
        self._position.append((new_position_mean, new_position_covariance))
