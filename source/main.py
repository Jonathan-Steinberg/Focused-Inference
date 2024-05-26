'''
This script demonstrates how to use the SLAM system to estimate the trajectory of a robot and the positions of landmarks
'''

# source/main.py

import os
import numpy as np
from gtsam import Pose3, Rot3

from source.slam.slam import SLAM
from source.landmarks.landmark import Landmark
from source.run import run_slam
from source.maps.visualization import MapVisualizer

# Define paths to save results
poses_dir = 'results/poses'
metrics_dir = 'results/metrics'
os.makedirs(poses_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)


def create_environment(num_landmarks, num_steps):
    """
    Create the environment with given number of landmarks and steps.

    Args:
        num_landmarks (int): Number of landmarks.
        num_steps (int): Number of steps.

    Returns:
        initial_pose (Pose3): Initial pose of the agent.
        landmarks (list of Landmark): List of landmarks.
        control_inputs (numpy.ndarray): Control inputs for each step.
        measurements (list): Measurements for each step.
        ground_truth_poses (list of Pose3): Ground truth poses.
    """
    initial_pose = Pose3(Rot3(), np.array([0, 0, 0]).reshape((3, 1)))

    # Generate random landmarks
    landmarks = [
        {
            'position': np.random.rand(3) * 10,
            'covariance': np.eye(3) * 0.1,
            'id': i
        }
        for i in range(num_landmarks)
    ]

    # Convert landmarks to Landmark objects
    landmark_objs = [Landmark(Pose3(Rot3(), lm['position'].reshape((3, 1))), lm['covariance'], lm['id']) for lm in
                     landmarks]

    # Generate random control inputs and ground truth poses
    control_inputs = np.random.rand(num_steps, 3) - 0.5
    ground_truth_poses = [initial_pose]
    for i in range(num_steps):
        new_pose = ground_truth_poses[-1].compose(Pose3(Rot3(), control_inputs[i].reshape((3, 1))))
        ground_truth_poses.append(new_pose)

    # Generate measurements for each step
    measurements = []
    for step in range(num_steps):
        measurement = []
        for lm in landmark_objs:
            observation = {
                'mean': lm.position.translation().flatten(),
                'covariance': lm.covariance,
                'id': lm.identifier
            }
            measurement.append(observation)
        measurements.append(measurement)

    return initial_pose, landmark_objs, control_inputs, measurements, ground_truth_poses


def plot_results(visualizer, landmarks, supposed_trajectory, slam_trajectory):
    """
    Plot the landmarks, supposed trajectory, and SLAM trajectory using the MapVisualizer.

    Args:
        visualizer (MapVisualizer): Visualizer for plotting.
        landmarks (list of Landmark): List of landmarks.
        supposed_trajectory (numpy.ndarray): Supposed trajectory from control inputs.
        slam_trajectory (numpy.ndarray): SLAM estimated trajectory.
    """
    visualizer.update_landmarks(np.array([lm.position.translation().flatten() for lm in landmarks]))
    visualizer.update_agent_position(supposed_trajectory[-1].flatten())
    visualizer.ax.plot(supposed_trajectory[:, 0], supposed_trajectory[:, 1], supposed_trajectory[:, 2], c='blue',
                       label='Supposed Trajectory')
    visualizer.ax.plot(slam_trajectory[:, 0], slam_trajectory[:, 1], slam_trajectory[:, 2], c='green',
                       label='SLAM Trajectory')
    visualizer.run()


def main():
    # Create environment
    (initial_pose,
     landmarks,
     control_inputs,
     measurements,
     ground_truth_poses) = create_environment(num_landmarks=10, num_steps=100)

    # Initialize SLAM system
    slam_system = SLAM(initial_pose, landmarks)

    # Run SLAM
    results = run_slam(slam_system, control_inputs, measurements, ground_truth_poses, len(control_inputs), poses_dir,
                       metrics_dir)

    # Extract supposed trajectory and SLAM trajectory
    supposed_trajectory = np.cumsum(control_inputs, axis=0)
    supposed_trajectory = np.vstack((np.array([0, 0, 0]), supposed_trajectory))  # Include the initial pose

    slam_trajectory = np.array([pose.translation().flatten() for pose in slam_system.poses])

    # Initialize visualizer
    visualizer = MapVisualizer(plot_3d=False)

    # Plot results
    plot_results(visualizer, landmarks, supposed_trajectory, slam_trajectory)


if __name__ == '__main__':
    main()
