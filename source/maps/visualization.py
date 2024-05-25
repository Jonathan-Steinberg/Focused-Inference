"""
This module provides the MapVisualizer class for visualizing the map with landmarks and agent poses.
Utilizes matplotlib for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

class MapVisualizer:
    """
    MapVisualizer handles the visualization of the map with landmarks and agent positions.
    Utilizes matplotlib for visualization.
    """

    def __init__(self, plot_3d=True):
        """
        Initialize the MapVisualizer.
        Sets up the canvas and markers.

        Args:
            plot_3d (bool): If True, use 3D plotting. If False, use 2D plotting.
        """
        self.plot_3d = plot_3d
        self.fig = plt.figure()
        if self.plot_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        self.landmarks = None
        self.agent = None
        self.initial_scale_done = False
        self.scaling_factor = 2  # Initial scale relative to the furthest landmark
        self.rescale_threshold = 0.2  # Threshold for rescaling (20% of the current plot size)

    def setup_canvas(self):
        """Set up the matplotlib canvas."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if self.plot_3d:
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Map of Landmarks and Agent')
        else:
            self.ax.set_title('2D Map of Landmarks and Agent')

    def setup_markers(self):
        """Set up the markers for landmarks and the agent."""
        if self.plot_3d:
            self.landmarks, = self.ax.plot([], [], [], 'r^', label='Landmarks')
            self.agent, = self.ax.plot([], [], [], 'bo', label='Agent')
        else:
            self.landmarks, = self.ax.plot([], [], 'r^', label='Landmarks')
            self.agent, = self.ax.plot([], [], 'bo', label='Agent')
        self.ax.legend()

    def update_landmarks(self, positions):
        """
        Update the positions of the landmarks.

        Args:
            positions (numpy.ndarray): Array of landmark positions with shape (N, 3).
        """
        if self.plot_3d:
            self.landmarks.set_data(positions[:, 0], positions[:, 1])
            self.landmarks.set_3d_properties(positions[:, 2])
        else:
            self.landmarks.set_data(positions[:, 0], positions[:, 1])
        if not self.initial_scale_done:
            self._initial_auto_scale(positions)

    def update_agent_position(self, position):
        """
        Update the position of the agent.

        Args:
            position (numpy.ndarray): Array of the agent's position with shape (1, 3).
        """
        if self.plot_3d:
            self.agent.set_data(position[:, 0], position[:, 1])
            self.agent.set_3d_properties(position[:, 2])
        else:
            self.agent.set_data(position[:, 0], position[:, 1])
        self._check_and_rescale(position)

    def _initial_auto_scale(self, positions):
        """
        Perform the initial auto-scaling of the plot based on landmark positions.

        Args:
            positions (numpy.ndarray): Array of positions with shape (N, 3).
        """
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        if self.plot_3d:
            z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        else:
            max_range = max(x_max - x_min, y_max - y_min)
        scale = max_range * self.scaling_factor
        self.ax.set_xlim(x_min - scale, x_max + scale)
        self.ax.set_ylim(y_min - scale, y_max + scale)
        if self.plot_3d:
            self.ax.set_zlim(z_min - scale, z_max + scale)
        self.initial_scale_done = True

    def _check_and_rescale(self, position):
        """
        Check the agent's position and rescale the plot if needed.

        Args:
            position (numpy.ndarray): Array of the agent's position with shape (1, 3).
        """
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        if self.plot_3d:
            z_min, z_max = self.ax.get_zlim()
            pos = position[0]

            rescale_needed = (
                pos[0] < x_min + (x_max - x_min) * self.rescale_threshold or
                pos[0] > x_max - (x_max - x_min) * self.rescale_threshold or
                pos[1] < y_min + (y_max - y_min) * self.rescale_threshold or
                pos[1] > y_max - (y_max - y_min) * self.rescale_threshold or
                pos[2] < z_min + (z_max - z_min) * self.rescale_threshold or
                pos[2] > z_max - (z_max - z_min) * self.rescale_threshold
            )

            if rescale_needed:
                scale = max(x_max - x_min, y_max - y_min, z_max - z_min)
                new_scale = scale * self.scaling_factor
                self.ax.set_xlim(x_min - new_scale, x_max + new_scale)
                self.ax.set_ylim(y_min - new_scale, y_max + new_scale)
                self.ax.set_zlim(z_min - new_scale, z_max + new_scale)
        else:
            pos = position[0]

            rescale_needed = (
                pos[0] < x_min + (x_max - x_min) * self.rescale_threshold or
                pos[0] > x_max - (x_max - x_min) * self.rescale_threshold or
                pos[1] < y_min + (y_max - y_min) * self.rescale_threshold or
                pos[1] > y_max - (y_max - y_min) * self.rescale_threshold
            )

            if rescale_needed:
                scale = max(x_max - x_min, y_max - y_min)
                new_scale = scale * self.scaling_factor
                self.ax.set_xlim(x_min - new_scale, x_max + new_scale)
                self.ax.set_ylim(y_min - new_scale, y_max + new_scale)

    def run(self):
        """Run the application to start the visualization."""
        self.setup_canvas()
        self.setup_markers()
        plt.ion()  # Turn on interactive mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.show()

    def update(self):
        """Update the plot in real-time."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# Example usage:
if __name__ == "__main__":
    visualizer = MapVisualizer(plot_3d=True)  # Change to False for 2D plotting
    visualizer.run()
    visualizer.update_landmarks(np.random.rand(100, 3) * 10)  # Random landmarks
    visualizer.update_agent_position(np.array([[5, 5, 5]]))  # Single agent in the center
    while True:
        visualizer.update()
