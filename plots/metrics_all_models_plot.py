#%%

def plot_model_metrics(model_comparison):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    custom_palette = sns.color_palette("Set2", 8)
    res = model_comparison
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_dictionary = Counter()
    for i in metrics:
        metric_dictionary[i] = res[i].str.split(" ± ", expand = True)[0].astype("float64").tolist()
        metric_dictionary[f'std_{i}'] = res[i].str.split(" ± ", expand = True)[1].astype("float64").tolist()


    metric_dictionary["model_time"] = res["time (s)"].tolist()

    models = ['XGB', 'SVM', 'RF', 'DFFN', 'ANN']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']

    # Assuming metric_dictionary has been defined with your data
    # Assuming sns and plt have been imported

    sns.set(style="whitegrid")
    palette = sns.color_palette("colorblind", len(metric_labels) + 1)  # Colorblind-friendly palette

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    x = np.arange(len(models))
    width = 0.12

    # Calculate the offset needed for centering bars
    num_bars_per_group = len(metric_labels) + 1  # Including the Time bar
    offset = width * num_bars_per_group / 2 - width / 2  # Offset for centering

    # Plotting metrics
    for i, metric in enumerate(metric_labels):
        values = metric_dictionary[metric.lower()]
        errors = metric_dictionary[f'std_{metric.lower()}']
        ax.bar(x + i * width - offset, values, width, label=metric, yerr=errors,
            color=palette[i], edgecolor='black', linewidth=0.5,
            error_kw={'elinewidth': 1, 'capsize': 2, 'capthick': 1, 'ecolor': 'black'})

    # Plotting "Time" with adjusted position
    time_values = metric_dictionary['model_time']
    ax2.bar(x + (len(metric_labels) * width) - offset, time_values, width,
            label='Time', color=palette[len(metric_labels)], edgecolor='black', linewidth=0.5)

    ax2.set_yscale('log')
    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Metric Value (%)', fontsize=12)
    ax.set_title('Metrics for Different Models Including Time on Log Scale', fontsize=14)

    # Improve legend readability and aesthetics
    ax.legend(loc="lower left", facecolor='white', framealpha=0.7)
    ax2.legend(loc="upper right", facecolor='white', framealpha=1)
    ax2.grid(None)
    plt.tight_layout()
    plt.savefig("/Users/uw20204/Documents/figures/model_comparison.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import numpy as np
    res = pd.read_csv('/Volumes/Samsung_T5/data/model_comparison_res.txt', sep = "\t")
    plot_model_metrics(res)
# %%
#%%
