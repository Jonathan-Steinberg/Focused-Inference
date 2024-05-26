'''
This script demonstrates how to use the SLAM system to estimate the trajectory of a robot and the positions of landmarks
'''

# source/main.py

import numpy as np
from gtsam import Pose3, Rot3
from source.slam.slam import SLAM
from source.landmarks.landmark import Landmark
from source.maps.visualization import MapVisualizer
from source.utils.graphs import plot_metrics
from source.run import run_slam

def main():
    initial_pose = np.array([0.0, 0.0, 0.0])
    # Generate some landmarks
    n_landmarks = 100
    landmark_data = [{'mean': np.random.rand(3) * 10, 'covariance': np.eye(3) * 0.1, 'id': i} for i in range(n_landmarks)]

    # Create the SLAM system
    landmarks = [Landmark(position=Pose3(Rot3(), lm['mean'].reshape((3, 1))), covariance=np.array(lm['covariance']), identifier=lm['id']) for lm in landmark_data]
    slam_system = SLAM(initial_pose, landmarks)

    # Create the visualizer
    visualizer = MapVisualizer(plot_3d=True)
    num_steps = 10

    # Control inputs and measurements (example data)
    CONTROL_INPUTS = [np.array([0.01, 0.01, 0.0]) for _ in range(num_steps)]
    MEASUREMENTS = [
        [{'id': 1, 'observation': {'mean': np.array([5.0, 5.0, 0.0]), 'covariance': np.eye(3) * 0.1}}],
        [{'id': 2, 'observation': {'mean': np.array([10.0, 0.0, 0.0]), 'covariance': np.eye(3) * 0.1}}],
        # Add more measurements as needed
    ] * num_steps

    GROUND_TRUTH_POSES = np.array([[0.1 * i, 0.1 * i, 0.0] for i in range(num_steps + 1)])

    try:
        results = run_slam(slam_system, visualizer, CONTROL_INPUTS, MEASUREMENTS, GROUND_TRUTH_POSES, num_steps)
        plot_metrics(results)
    except Exception as e:
        print(f"Error during SLAM: {e}")

if __name__ == "__main__":
    main()
