"""
This module provides utility functions for information-theoretic computations.
"""

import numpy as np
import gtsam

def compute_information_gain(agent, landmark, poses):
    try:
        # Assuming `agent.position` returns a numpy array
        agent_position = agent.position
        agent_covariance = agent.position_covariance

        # Compute mutual information between the agent's position and the landmark
        X = [0]  # Placeholder, replace with the actual variable indices in the factor graph
        g_theta = gtsam.NonlinearFactorGraph()  # Placeholder, replace with the actual factor graph
        theta_star = gtsam.Values()  # Placeholder, replace with the actual values

        log_det_theta = log_det(X + [landmark.identifier], g_theta, theta_star)
        reduced_graph = gtsam.NonlinearFactorGraph(g_theta)
        reduced_graph.remove(landmark.identifier)
        reduced_theta_star = gtsam.Values(theta_star)
        reduced_theta_star.erase(landmark.identifier)
        log_det_theta_reduced = log_det(X, reduced_graph, reduced_theta_star)

        return 0.5 * (log_det_theta_reduced - log_det_theta)
    except Exception as e:
        print(f"Error computing information gain: {e}")
        return 0

def log_det(X, g_theta, theta_star):
    try:
        ordering = gtsam.Ordering()
        for var in X:
            ordering.push_back(var)
        linearized_graph = g_theta.linearize(theta_star)
        bayes_net, bayes_tree = linearized_graph.eliminatePartialSequential(ordering)
        marginals = gtsam.Marginals(g_theta, theta_star)
        r_matrix = marginals.marginalCovariance(ordering.front())
        log_det = -2 * np.sum(np.log(np.diag(r_matrix)))
        return log_det
    except Exception as e:
        print(f"Error computing marginal covariance: {e}")
        return 0
