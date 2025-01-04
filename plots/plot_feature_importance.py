#%%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(feature_importance_df, output_dir):
    # Set up the matplotlib figure
    plt.figure(figsize=(6, 5))

    # Create a bar plot with reversed color palette
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='coolwarm_r')

    # Add labels and title with increased font size
    plt.xlabel('Importance', fontsize=10)
    plt.ylabel('Feature', fontsize=10)
    plt.title('Feature Importances', fontsize=14)

    # Increase the size of tick labels
    plt.xticks(fontsize=7.6)
    plt.yticks(fontsize=7.6)

    # Improve layout
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    # Show plot
    plt.show()
# %%
