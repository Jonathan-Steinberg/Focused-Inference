import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(results):
    """
    Plot the evaluation metrics against the number of landmarks removed.

    Args:
        results (dict): Dictionary where keys are algorithm names and values are dictionaries with keys:
                        'landmarks_removed', 'ate_values', 'are_values', and 'ud_values'.
    """
    sns.set(style="whitegrid")

    # Create separate figures for each metric
    fig_ate, ax_ate = plt.subplots(figsize=(10, 6))
    fig_are, ax_are = plt.subplots(figsize=(10, 6))
    fig_ud, ax_ud = plt.subplots(figsize=(10, 6))

    for algorithm, metrics in results.items():
        data = {
            'Landmarks Removed': metrics['landmarks_removed'],
            'ATE (m)': metrics['ate_values'],
            'ARE (deg)': metrics['are_values'],
            'UD': metrics['ud_values']
        }

        df = pd.DataFrame(data)

        sns.lineplot(ax=ax_ate, x='Landmarks Removed', y='ATE (m)', data=df, marker='o', label=algorithm)
        sns.lineplot(ax=ax_are, x='Landmarks Removed', y='ARE (deg)', data=df, marker='o', label=algorithm)
        sns.lineplot(ax=ax_ud, x='Landmarks Removed', y='UD', data=df, marker='o', label=algorithm)

    # Customize ATE plot
    ax_ate.set_title('ATE vs Landmarks Removed')
    ax_ate.set_xlabel('Landmarks Removed')
    ax_ate.set_ylabel('ATE (m)')
    ax_ate.legend()

    # Customize ARE plot
    ax_are.set_title('ARE vs Landmarks Removed')
    ax_are.set_xlabel('Landmarks Removed')
    ax_are.set_ylabel('ARE (deg)')
    ax_are.legend()

    # Customize UD plot
    ax_ud.set_title('UD vs Landmarks Removed')
    ax_ud.set_xlabel('Landmarks Removed')
    ax_ud.set_ylabel('UD')
    ax_ud.legend()

    plt.show()
