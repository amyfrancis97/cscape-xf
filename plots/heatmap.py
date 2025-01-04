#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%

common_cols = list(set(placenta_NS_scores.columns).intersection(lung_non_small_cell_carcinoma_scores.columns))

placenta_NS_scores = placenta_NS_scores[common_cols].transpose()
placenta_NS_scores.columns = ['FeatureImportance']

lung_non_small_cell_carcinoma_scores = lung_non_small_cell_carcinoma_scores[common_cols].transpose()
lung_non_small_cell_carcinoma_scores.columns = ['FeatureImportance']

df_list = [placenta_NS_scores, lung_non_small_cell_carcinoma_scores]

df_scores = pd.concat([placenta_NS_scores, lung_non_small_cell_carcinoma_scores], axis =1)
df_scores.columns = ["placenta_NS_scores", "lung_non_small_cell_carcinoma_scores"]
df_scores = df_scores.sort_values(by = "placenta_NS_scores", ascending=False)[:20]
df_scores['placenta_NS_scores'] = (df_scores['placenta_NS_scores'] - df_scores['placenta_NS_scores'].mean()) / df_scores['placenta_NS_scores'].std() 
df_scores['lung_non_small_cell_carcinoma_scores'] = (df_scores['lung_non_small_cell_carcinoma_scores'] - df_scores['lung_non_small_cell_carcinoma_scores'].mean()) / df_scores['lung_non_small_cell_carcinoma_scores'].std() 
df_scores
#%%
df_scores = df_scores.sample(frac=1)
# %%
# Plot heatmap using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(df_scores, cmap='viridis', annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': 'Feature Importance'})
plt.title('Feature Importance Heatmap')
plt.xlabel('Samples')
plt.ylabel('Features')
plt.show()
# %%
