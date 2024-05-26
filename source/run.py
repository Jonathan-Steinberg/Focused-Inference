'''
This file contains the main function that runs the SLAM system for a given number of steps.
'''

from gtsam import Pose3, Rot3

# source/run.py

import numpy as np
from source.info_theoretic.evals import compute_ate, compute_are, compute_ud

def run_slam(slam_system, visualizer, control_inputs, measurements, ground_truth_poses, num_steps):
    results = {
        'landmarks_removed': [],
        'ate_values': [],
        'are_values': [],
        'ud_values': []
    }

    for step in range(num_steps):
        control_input = control_inputs[step]
        measurement = measurements[step]

        try:
            slam_system.perform_slam_step(control_input, measurement)
            visualizer.update_agent_position(slam_system.agent.position.translation().flatten())
            visualizer.update_landmarks(np.array([lm.position.translation().flatten() for lm in slam_system.landmarks.values()]))
            visualizer.update()

            estimated_poses = np.array([pose.translation().flatten() for pose in slam_system.poses])
            ground_truth_poses_array = np.array([pose.flatten() for pose in ground_truth_poses[:len(slam_system.poses)]])

            # print(f"Step {step}: Estimated poses shape: {estimated_poses.shape}, Ground truth poses shape: {ground_truth_poses_array.shape}")

            if estimated_poses.shape != ground_truth_poses_array.shape:
                raise ValueError("Estimated and ground truth poses must have the same shape.")

            ate = compute_ate(estimated_poses, ground_truth_poses_array)
            estimated_rotations = np.array([pose.rotation().matrix() for pose in slam_system.poses])
            ground_truth_rotations = np.array([Pose3(Rot3(), gt_pose.reshape((3, 1))).rotation().matrix() for gt_pose in ground_truth_poses[:len(slam_system.poses)]])
            are = compute_are(estimated_rotations, ground_truth_rotations)
            ud = compute_ud(slam_system.agent.position_covariance, np.eye(3))

            results['landmarks_removed'].append(len(slam_system.landmarks))
            results['ate_values'].append(ate)
            results['are_values'].append(are)
            results['ud_values'].append(ud)

        except Exception as e:
            print(f"Error during SLAM step: {e}")

    return results
