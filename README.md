# Focused Inference Project

## Overview

The Focused Inference Project is an undergrad final project aimed at improving Simultaneous Localization and Mapping (SLAM) by leveraging information-theoretic approaches to reduce the number of landmarks without compromising the accuracy and efficiency of navigation. This project integrates various landmark removal algorithms and evaluates their performance using metrics such as Average Trajectory Error (ATE), Average Rotation Error (ARE), and Uncertainty Determinant (UD).

## Objectives

1. **Reduce Landmark Density**: Implement and evaluate several algorithms to reduce the number of landmarks while maintaining accurate localization and mapping.
2. **Improve Computational Efficiency**: Reduce the number of landmarks to make SLAM algorithms more computationally efficient.
3. **Maintain Accuracy**: Ensure that the reduced landmark set provides sufficient information for accurate SLAM performance.

## Algorithms

### Landmark Removal Techniques

1. **Least Degree Based Removal**: Removes landmarks with the least connectivity (degree) first.
2. **Maximum Uncertainty Based Removal**: Removes landmarks with the highest uncertainty first.
3. **K-Cover Based Removal**: This method uses a greedy algorithm to cover the most poses with the fewest landmarks and then removes landmarks in reverse order.
4. **Least Informative Landmark Based Removal**: Computes the mutual information of each landmark and removes the least informative ones.
5. **Least Reprojection Error Based Removal**: Removes landmarks based on their reprojection error from low to high.

### Evaluation Metrics

1. **Average Trajectory Error (ATE)**: Measures the RMSE of pose translation differences between standard SLAM and reduced landmark SLAM.
2. **Average Rotation Error (ARE)**: Measures the average angular difference in pose heading directions.
3. **Uncertainty Determinant (UD)**: Evaluate the change in the uncertainty of the latest pose by comparing the determinant of its covariance.

## Project Structure
Focused Inference Project/
├── source/
│ ├── agents/
│ │ ├── agent.py
│ │ ├── init.py
│ ├── algorithms/
│ │ ├── landmark_removal.py
│ │ ├── init.py
│ ├── info_theoretic/
│ │ ├── utils.py
│ │ ├── evals.py
│ │ ├── init.py
│ ├── landmarks/
│ │ ├── landmark.py
│ │ ├── utils.py
│ │ ├── init.py
│ ├── maps/
│ │ ├── map.py
│ │ ├── visualization.py
│ │ ├── init.py
│ ├── slam/
│ ├── utils/
│ ├── config.py
│ ├── main.py
├── tests/
│ ├── test_import.py
├── scripts/
│ ├── opengl_compatability_check.py
├── README.md


## Getting Started

### Prerequisites

- Python 3.6+
- NumPy
- Matplotlib
- PyCUDA (optional, for GPU acceleration)
- Vispy (optional, for advanced visualization)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Jonathan-Steinberg/Focused-Inference.git
    cd Focused-Inference
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

1. **Navigate to the source directory**:
    ```sh
    cd source
    ```

2. **Run the main script**:
    ```sh
    python main.py
    ```

