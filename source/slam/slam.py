"""
This module provides the SLAM class for managing the simultaneous localization and mapping process using GTSAM.
"""

import gtsam

# source/slam/slam.py

import numpy as np
from gtsam import (Values, NonlinearFactorGraph, GaussNewtonOptimizer,
                   Pose3, PriorFactorPose3, BetweenFactorPose3, Marginals, Rot3)
from source.agents.agent import Agent
from source.landmarks.landmark import Landmark
from source.info_theoretic.utils import compute_information_gain

class SLAM:
    """
    SLAM class handles the simultaneous localization and mapping process incrementally using GTSAM.
    """

    def __init__(self, initial_pose, landmarks, minimization_interval=10):
        """
        Initialize the SLAM class.

        Args:
            initial_pose (numpy.ndarray): Initial pose of the agent.
            landmarks (list of Landmark): List of landmarks.
            minimization_interval (int): Interval at which to perform landmark minimization.
        """
        self.agent = Agent(position=Pose3(Rot3(), initial_pose))
        self._landmarks = {lm.identifier: lm for lm in landmarks}
        self._poses = [Pose3(Rot3(), initial_pose)]
        self.minimization_interval = minimization_interval
        self.step_count = 0

        # Initialize GTSAM structures
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()

        # Add prior on the first pose
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        initial_pose_gtsam = Pose3(Rot3(), initial_pose)
        self.graph.add(PriorFactorPose3(0, initial_pose_gtsam, prior_noise))
        self.initial_estimate.insert(0, initial_pose_gtsam)

    @property
    def landmarks(self):
        """Get the landmarks."""
        return self._landmarks

    @property
    def poses(self):
        """Get the poses."""
        return self._poses

    def perform_slam_step(self, control_input, measurements):
        """
        Perform a single SLAM step.

        Args:
            control_input (numpy.ndarray): Control input for the motion model.
            measurements (list of dict): List of measurements containing landmark id and observations.
        """
        if control_input.shape != (3,):
            raise ValueError("control_input must be a 1D array of shape (3,)")

        # Predict the next pose using the motion model
        new_pose_index = len(self.poses)
        previous_pose = self.poses[-1]
        delta_pose = Pose3(Rot3(), control_input.reshape((3, 1)))
        new_pose = previous_pose.compose(delta_pose)
        self.agent.position = new_pose
        self._poses.append(new_pose)

        # Add the new pose to the graph
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        self.graph.add(BetweenFactorPose3(new_pose_index - 1, new_pose_index, delta_pose, odometry_noise))
        self.initial_estimate.insert(new_pose_index, new_pose)

        # Update landmarks based on measurements
        for measurement in measurements:
            lm_id = measurement['id']
            observation = measurement['observation']
            if lm_id in self._landmarks:
                lm = self._landmarks[lm_id]
                lm.update_position(Pose3(Rot3(), observation['mean'].reshape((3, 1))), observation['covariance'])

        # Perform graph optimization periodically
        self.step_count += 1
        # if self.step_count % self.minimization_interval == 0:
        #    self.perform_information_theoretic_minimization()

        # Calculate marginals for the current pose
        marginals = Marginals(self.graph, self.initial_estimate)
        try:
            full_covariance = marginals.marginalCovariance(new_pose_index)
            # print(f"Full covariance matrix: {full_covariance}")
            position_covariance = full_covariance[:3, :3]  # Extract the top-left 3x3 submatrix
            self.agent.position_covariance = position_covariance
        except Exception as e:
            print(f"Error computing marginal covariance: {e}")

    def perform_information_theoretic_minimization(self):
        """
        Perform information-theoretic minimization to reduce the number of landmarks.
        """
        information_gains = {lm_id: compute_information_gain(self.agent.position, lm, self.poses) for lm_id, lm in self.landmarks.items()}
        sorted_landmarks = sorted(information_gains, key=information_gains.get, reverse=True)
        num_landmarks_to_keep = int(len(self.landmarks) * 0.5)
        self._landmarks = {lm_id: self.landmarks[lm_id] for lm_id in sorted_landmarks[:num_landmarks_to_keep]}
