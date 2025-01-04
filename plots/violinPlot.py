#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

custom_palette = sns.color_palette("Set2", 8)
DEFAULT_OUTPUT_FILENAME = "violin_plot.png"
DEFAULT_OUTPUT_FILEPATH = "/Users/uw20204/Documents/figures/"

def plot_violin(df, features, negative_dataset, output_filename=DEFAULT_OUTPUT_FILENAME, driver_stat_col="driver_stat", output_path=DEFAULT_OUTPUT_FILEPATH):
    """
    Generate a violin plot for specified features, comparing two datasets based on a driver_stat column.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the data.
    - features (list): A list of column names in df to be plotted.
    - output_filename (str, optional): The name of the output file to save the plot. Defaults to 'violin_plot.png'.
    - driver_stat_col (str, optional): The column indicating the dataset categories (e.g., 1 for COSMIC, 0 for SomaMutDB). Defaults to 'driver_stat'.

    Usage:
    - plot_violin(your_dataframe, ["feature1", "feature2"], output_filename="output_plot.png", driver_stat_col="driver_category")

    Note:
    - The function generates a 2x4 grid of violin plots, comparing the distribution of specified features between positive and negative classes.
    - The color palette used for the plots is derived from Seaborn's 'Set2' palette.
    """

    if not features:
        print("No features provided for plotting.")
        return

    # Create a 2x4 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    sns.set_palette(custom_palette)

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.violinplot(x=driver_stat_col, y=feature, data=df, ax=axes[i], palette='Set2', split=True)
        
        # Set labels and title for each subplot
        axes[i].set_xlabel('Dataset', fontsize=14, weight='bold')
        axes[i].set_ylabel('Values', fontsize=14, weight='bold')
        axes[i].set_title(f'{feature}', fontsize=16, weight='bold')
        
        # Set custom x-axis ticks and labels
        axes[i].set_xticks([1, 0])
        axes[i].set_xticklabels(['COSMIC', negative_dataset], fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with a high DPI
    plt.savefig(output_path + output_filename, dpi=300)

    # Show the plot
    plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

custom_palette = sns.color_palette("Set2", 8)
DEFAULT_OUTPUT_FILENAME = "swarm_plot.png"
DEFAULT_OUTPUT_FILEPATH = "/Users/uw20204/Documents/figures/"

def plot_swarm(df, features, negative_dataset, output_filename=DEFAULT_OUTPUT_FILENAME, driver_stat_col="driver_stat", output_path=DEFAULT_OUTPUT_FILEPATH):
    """
    Generate a swarm plot for specified features, comparing two datasets based on a driver_stat column.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the data.
    - features (list): A list of column names in df to be plotted.
    - output_filename (str, optional): The name of the output file to save the plot. Defaults to 'swarm_plot.png'.
    - driver_stat_col (str, optional): The column indicating the dataset categories (e.g., 1 for COSMIC, 0 for SomaMutDB). Defaults to 'driver_stat'.

    Usage:
    - plot_swarm(your_dataframe, ["feature1", "feature2"], output_filename="output_plot.png", driver_stat_col="driver_category")

    Note:
    - The function generates a 2x4 grid of swarm plots, comparing the distribution of specified features between positive and negative classes.
    - The color palette used for the plots is derived from Seaborn's 'Set2' palette.
    """

    if not features:
        print("No features provided for plotting.")
        return

    # Limit the number of features to the number of subplots available
    num_plots = min(len(features), 8)
    features = features[:num_plots]

    # Create a 2x4 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    sns.set_palette(custom_palette)

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.swarmplot(x=driver_stat_col, y=feature, data=df, ax=axes[i], palette='Set2')
        
        # Set labels and title for each subplot
        axes[i].set_xlabel('Dataset', fontsize=14, weight='bold')
        axes[i].set_ylabel('Values', fontsize=14, weight='bold')
        axes[i].set_title(f'{feature}', fontsize=16, weight='bold')
        
        # Set custom x-axis ticks and labels
        axes[i].set_xticks([1, 0])
        axes[i].set_xticklabels(['COSMIC', negative_dataset], fontsize=12)

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with a high DPI
    plt.savefig(output_path + output_filename, dpi=300)

    # Show the plot
    plt.show()


# %%
