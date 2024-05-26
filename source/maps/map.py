"""
This module provides the Map class for managing the spatial relationships
 between an agent and multiple landmarks.
"""

from source.landmarks.landmark import Landmark
import numpy as np

class Map:
    """
    Map manages the spatial relationships between an agent and multiple landmarks.
    """
    def __init__(self, agent):
        """
        Initialize the map with a single agent.

        Args:
            agent (Agent): The single agent to be managed by the map.
        """
        self.agent = agent
        self.landmarks = {}

    def add_landmark(self, landmark_id, landmark):
        """
        Add a landmark to the map.

        Args:
            landmark_id (str): Unique identifier for the landmark.
            landmark (Landmark): The landmark object to be added.
        """
        if not isinstance(landmark, Landmark):
            raise ValueError("Only Landmark instances can be added.")
        self.landmarks[landmark_id] = landmark

    def remove_landmark(self, landmark_id):
        """
        Remove a landmark from the map by its identifier.

        Args:
            landmark_id (str): The identifier of the landmark to remove.
        """
        if landmark_id in self.landmarks:
            del self.landmarks[landmark_id]

    def update_landmark(self, landmark_id, position_mean, position_covariance):
        """
        Update the position of a landmark.

        Args:
            landmark_id (str): The identifier of the landmark.
            position_mean (numpy.ndarray):
             The new mean position of the landmark.

            position_covariance (numpy.ndarray):
             The new covariance matrix of the landmark's position.
        """
        if landmark_id in self.landmarks:
            landmark = self.landmarks[landmark_id]
            landmark.update_position(position_mean, position_covariance)
        else:
            self.add_landmark(landmark_id,
                              Landmark(landmark_id,
                                       position_mean,
                                       position_covariance))

    def update_agent_position(self, delta_position, position_covariance):
        """
        Update the position of the agent based on a change and new covariance.

        Args:
            delta_position (numpy.ndarray): The change in position (x, y, z).
            position_covariance (numpy.ndarray): The new covariance matrix of the agent's position.
        """
        self.agent.move_agent(delta_position, position_covariance)

    def get_landmark(self, landmark_id):
        """
        Retrieve a landmark by its identifier.

        Returns:
            Landmark: The landmark object if found, otherwise None.
        """
        return self.landmarks.get(landmark_id, None)


