"""
This module provides the MapVisualizer class for visualizing the map with landmarks and agent poses.
Utilizes matplotlib for visualization.
"""
# source/maps/visualization.py

import matplotlib.pyplot as plt
import numpy as np

class MapVisualizer:
    def __init__(self, plot_3d=True):
        self.plot_3d = plot_3d
        self.fig = plt.figure()
        if self.plot_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        self.landmarks = None
        self.agent = None

    def update_landmarks(self, positions):
        if self.landmarks:
            self.landmarks.remove()
        if self.plot_3d:
            self.landmarks = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', label='Landmarks')
        else:
            self.landmarks = self.ax.scatter(positions[:, 0], positions[:, 1], c='red', label='Landmarks')
        self.ax.legend()

    def update_agent_position(self, position):
        if self.agent:
            self.agent.remove()
        position = position.reshape((3,))
        if self.plot_3d:
            self.agent = self.ax.scatter(position[0], position[1], position[2], c='green', label='Agent')
        else:
            self.agent = self.ax.scatter(position[0], position[1], c='green', label='Agent')
        self.ax.legend()

    def run(self):
        plt.show()

    def update(self):
        plt.draw()
        plt.pause(0.01)
