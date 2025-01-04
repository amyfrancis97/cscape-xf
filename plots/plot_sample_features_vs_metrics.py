#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_sample_metrics_smooth_lowess(sample_size_data, outputDir, x_axis_name, x_label, plot_time=False, lower_ylim=False, upper_ylim=False, vline_x=None, vline_label=None):
    """
    Plot sample vs accuracy, F1, precision, and optionally Time with lowess smoothing.

    Parameters:
        sample_size_data (pd.DataFrame): The dataframe from the "get_sample_vs_accuracy.py" script.
        outputDir (str): The directory to save the plot.
        x_axis_name (str): The name of the column to be used as the x-axis.
        x_label (str): The label for the x-axis.
        plot_time (bool): Whether to plot elapsed time on a secondary y-axis. Default is False.
        lower_xlim (float): Lower limit for the x-axis. Default is False (no limit).
        upper_xlim (float): Upper limit for the x-axis. Default is False (no limit).
        vline_x (float): X-coordinate of the vertical line. Default is None (no line).
        vline_label (str): Label for the vertical line. Default is None (no label).
    """

    # Sort dataframe based on sample size and remove outliers
    sample_size_data = sample_size_data.drop_duplicates(subset=[x_axis_name], keep="first")

    if pd.api.types.is_string_dtype(sample_size_data['accuracy']):
        print("yes")
        # Convert metrics to floats
        sample_size_data["accuracy"] = sample_size_data["accuracy"].str.split(" ± ", expand=True)[0].astype("float")
        sample_size_data["f1"] = sample_size_data["f1"].str.split(" ± ", expand=True)[0].astype("float")
        sample_size_data["precision"] = sample_size_data["precision"].str.split(" ± ", expand=True)[0].astype("float")
    if plot_time:
        sample_size_data["elapsed_time (s)"] = sample_size_data["elapsed_time (s)"].astype("float")

    # Extracting the data
    x = sample_size_data[x_axis_name].values
    metrics = ['accuracy', 'f1', 'precision']
    if plot_time:
        metrics.append('elapsed_time (s)')

    # Define the custom palette
    custom_palette = sns.color_palette("Dark2", len(metrics))

    # Set plot style for better readability
    sns.set(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust figure size for clarity
    ax2 = ax1.twinx() if plot_time else None  # Create a second y-axis only if plot_time is True

    line_styles = ['-', '--', '-.', ':']  # Different line styles for each metric

    # Apply plotting within your loop, integrating the suggestions
    for i, metric in enumerate(metrics):
        y = sample_size_data[metric].values
        smoothed = lowess(y, x, frac=0.2)  # Keeping your smoothing logic
        if metric == 'elapsed_time (s)':
            # Adjustments for secondary axis plotting
            ax2.plot(smoothed[:, 0], smoothed[:, 1], label=metric, color=custom_palette[i], linestyle=line_styles[i], linewidth=2)
            ax2.set_ylabel('Time (s)', fontsize=14)
        else:
            ax1.plot(smoothed[:, 0], smoothed[:, 1], label=metric, color=custom_palette[i], linestyle=line_styles[i], linewidth=2)
            ax1.set_xlabel(x_label, fontsize=14)
            ax1.set_ylabel('Metric Value', fontsize=14)
    
    print(lower_ylim)

    # Add a vertical dashed line if specified
    if vline_x is not None:
        ax1.axvline(x=vline_x, color='darkblue', linestyle='--', linewidth=1.5, label=vline_label)

    # Adjust legend for better readability
    if plot_time:
        ax2.legend(loc='lower right')
        ax1.legend(loc='upper left')
    else:
        ax1.legend(loc='lower right')

    # Further enhance the title, labels, and ticks for clarity
    ax1.set_title('Model Performance for Different ' + x_label, fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    if plot_time:
        ax2.tick_params(axis='y', which='major', labelsize=12)
    
    if lower_ylim != False:
        plt.ylim(lower_ylim, upper_ylim)

    # Saving the figure with adjustments
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()  # Adjust layout to make sure nothing is clipped
    plt.savefig(f"{outputDir}/{x_axis_name}_vs_metrics_enhanced.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Read in sample size output file from get_sample_vs_accuracy.py script
    sample_size_data = pd.read_csv("/Volumes/Samsung_T5/data/sample_vs_accuracy.txt", sep = "\t")
    feature_size_data = pd.read_csv("/Volumes/Samsung_T5/data/number_features_vs_accuracy.txt", sep = "\t")
    outputDir = "/Users/uw20204/Documents/figures/"
    plot_sample_metrics_smooth_lowess(sample_size_data, outputDir, "number_samples", "Sample Size")
    plot_sample_metrics_smooth_lowess(feature_size_data, outputDir, "number_features", "Number of Features")

# %%
