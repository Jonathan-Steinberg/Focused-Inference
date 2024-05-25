"""
This is the main entry point of the project.
"""

import numpy as np
from source.info_theoretic import (
    # compute_degree,
    # compute_uncertainty,
    # k_cover_algorithm,
    # compute_mutual_information,
    # compute_reprojection_error,
    compute_ate,
    compute_are,
    compute_ud
)

# from source.algorithms import LandmarkRemoval
from source.landmarks.landmark import Landmark
from source.agents.agent import Agent
from source.maps.visualization import MapVisualizer

def main():
    # Example usage
    agent = Agent(initial_position_mean=np.array([5, 5, 5]),
                  initial_position_covariance=np.eye(3))
    landmarks = [
        Landmark(identifier=i,
                 initial_position_mean=np.random.rand(3) * 10,
                 initial_position_covariance=np.eye(3))
        for i in range(10)
    ]
    poses = np.random.rand(10, 3) * 10  # Example poses

    # Initialize the visualizer
    plot_type = (input("Enter '3D' for 3D plotting or '2D' for 2D plotting: ")
                 .strip().upper() == '3D')
    visualizer = MapVisualizer(plot_3d=plot_type)
    visualizer.run()

    # Update the visualization
    landmark_positions = np.array([lm.position_mean for lm in landmarks])
    visualizer.update_landmarks(landmark_positions)
    visualizer.update_agent_position(np.array([agent.position_mean]))

    # Example evaluation metrics
    estimated_poses = poses
    ground_truth_poses = poses + np.random.normal(0,
                                                  0.5,
                                                  poses.shape)  # Simulated ground truth with noise
    estimated_rotations = np.random.rand(10, 3)
    ground_truth_rotations = estimated_rotations + np.random.normal(0,
                                                                    0.1,
                                                                    estimated_rotations.shape)
    estimated_covariance = np.eye(3)
    ground_truth_covariance = np.eye(3) * 1.1

    ate = compute_ate(estimated_poses, ground_truth_poses)
    are = compute_are(estimated_rotations, ground_truth_rotations)
    ud = compute_ud(estimated_covariance, ground_truth_covariance)

    print(f"ATE: {ate:.2f} m")
    print(f"ARE: {are:.2f} deg")
    print(f"UD: {ud:.2f}")

    # Real-time update loop
    while True:
        visualizer.update()

if __name__ == "__main__":
    main()
