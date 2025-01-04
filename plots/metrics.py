#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

dir = "/Users/uw20204/Documents/data/somamut_cosmic"
metrics = pd.read_csv(dir + "/all_metrics.txt", sep = "\t", header = None)
print(metrics)
metrics.columns =['tissue', 'accuracy', 'accuracy_std', 'precision', 'precision_std', 
                  'recall','recall_std', 'f1', 'f1_std']
metrics = metrics.sort_values('accuracy', ascending=False).reset_index(drop = True)

metrics = metrics[0:10]
# Custom color palette
custom_palette = sns.color_palette("Set2", 8)
# Use custom color palette
sns.set_palette(custom_palette)


# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Scores by Tissue', fontsize=16)

# Set the bar width
bar_width = 0.2

# Create a list of positions for each tissue
positions = np.arange(len(metrics['tissue']))

# Plot each metric with different colors and error bars
for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
    ax.barh(positions + i * bar_width, metrics[metric], bar_width, label=metric.capitalize(),
            xerr=metrics[f'{metric}_std'], capsize=2, edgecolor='black', linewidth=1)  # Add error bars

ax.set_yticks(positions + bar_width * 1.5)
ax.set_yticklabels(metrics['tissue'])
ax.set_ylabel('Tissue')
ax.set_xlabel('Score')
ax.legend()
# Save the figure 
plt.savefig('/Users/uw20204/Documents/figures/SomaMut_COSMIC.png', dpi=300, bbox_inches='tight')

plt.show()


# %%
