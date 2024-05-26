'''
This file contains the main function that runs the SLAM system for a given number of steps.
'''

from gtsam import Pose3, Rot3

# source/run.py

import os
import numpy as np
from source.info_theoretic.evals import compute_ate, compute_are, compute_ud

def run_slam(slam_system, control_inputs, measurements, ground_truth_poses, num_steps, poses_dir, metrics_dir):
    results = {'landmarks_removed': [], 'ate_values': [], 'are_values': [], 'ud_values': []}

    for step in range(num_steps):
        control_input = control_inputs[step]
        measurement = measurements[step]

        try:
            slam_system.perform_slam_step(control_input, measurement)
            estimated_poses = slam_system.poses
            ground_truth_subset = ground_truth_poses[:len(estimated_poses)]

            # print(f"Step {step}: Estimated poses shape: ({len(estimated_poses)},), Ground truth poses shape: ({len(ground_truth_subset)}, 3)")

            # Extract translation components for metric computation
            estimated_positions = np.array([pose.translation() for pose in estimated_poses])
            ground_truth_positions = np.array([pose.translation() for pose in ground_truth_subset])

            # print(f"Step {step}: Estimated positions shape: {estimated_positions.shape}, Ground truth positions shape: {ground_truth_positions.shape}")

            # Ensure the shapes are correct
            if estimated_positions.shape == ground_truth_positions.shape:
                # Compute metrics
                ate = compute_ate(estimated_positions, ground_truth_positions)
                are = compute_are(
                    np.array([pose.rotation().rpy() for pose in estimated_poses]),  # using rpy() instead of xyz()
                    np.array([pose.rotation().rpy() for pose in ground_truth_subset])
                )
                ud = compute_ud(np.eye(3), np.eye(3))  # Dummy covariances for example

                results['landmarks_removed'].append(0)  # Placeholder value
                results['ate_values'].append(ate)
                results['are_values'].append(are)
                results['ud_values'].append(ud)

                # Save poses and metrics to files
                np.save(os.path.join(poses_dir, f'estimated_poses_step_{step}.npy'), estimated_positions)
                np.save(os.path.join(metrics_dir, f'metrics_step_{step}.npy'), [ate, are, ud])
            else:
                print(f"Error computing metrics: Estimated and ground truth positions must have the same shape.")
        except Exception as e:
            print(f"Error during SLAM step: {e}")

    return results
