import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

custom_palette = sns.color_palette("Set2", 8)
DEFAULT_OUTPUT_FILENAME = "metrics_barplot.png"
DEFAULT_OUTPUT_FILEPATH = "/Users/uw20204/Documents/figures/"
    

def plot_metrics(metrics, output_filename=DEFAULT_OUTPUT_FILENAME, output_path=DEFAULT_OUTPUT_FILEPATH):

    sns.set_palette(custom_palette)

    # Means
    means = metrics.iloc[0, :].tolist()

    # Standard Deviations
    std_devs = metrics.iloc[1, :].tolist()

    # Labels for the metrics
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    # Setting the positions and width for the bars
    positions = np.arange(len(labels))
    width = 0.35

    # Customize error bars
    error_bar_props = {
        'ecolor': custom_palette[2],
        'capsize': 5,  
        'elinewidth': 2,
        'capthick': 2    
    }

    # Creating the bar plot
    bars = plt.bar(positions, means, width, yerr=std_devs, capsize=5, color=custom_palette[0], label='Mean Values', error_kw = error_bar_props)

    # Adding labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Bar Plot of Metrics with Standard Deviations')
    plt.xticks(positions, labels)
    plt.ylim(0.5, 1)

    # Adding the value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval+ 0.02, round(yval, 3), ha='center', va='bottom')


    # Adding a legend
    plt.legend()

    # Save the figure with a high DPI
    plt.savefig(output_path + output_filename, dpi=300)

    # Show the plot
    plt.show()