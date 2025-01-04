#%%
import pandas as pd
all_tissues = pd.read_csv("/Users/uw20204/Documents/data/gnomad_cosmic/combined.csv", sep = "\t")
# %%
std_df = all_tissues[all_tissues["accuracy"] < 0.5]
# %%
mean_df = all_tissues[all_tissues["accuracy"] > 0.5]
# %%
mean_df.sort_values("accuracy", ascending=False)
# %%
tissue = "skin"
try:
    chunks = 100000
    df = []
    for chunk in pd.read_csv(f"/Users/uw20204/Documents/data/gnomad_cosmic/all_tissues_full_datasets/{tissue}_cosmic_all_features_full_dataset.txt.gz", sep = "\t", chunksize=chunks):
        df.append(chunk)
    cosmic= pd.concat(df)
except:
    print("cosmic dataset doesnt exist")
gnomad =  pd.read_csv("/Users/uw20204/Documents/data/gnomad_cosmic/gnomad_all_features_full_dataset.txt", sep = "\t")

# %%
import os
os.chdir("/Users/uw20204/Documents/scripts/CanDrivR/models/")
from selected_features import features

import pandas as pd
sampled = pd.read_csv("/Users/uw20204/Documents/data/gnomad_cosmic/sample_cosmic_gnomad14000.txt", sep = "\t")
sampled
#%%
cosmic_gnomad_sampled = sampled[features[:62]+["driver_stat"]]
cosmic_gnomad_sampled["driver_stat"]

#%%
gold_standard = pd.read_csv("/Users/uw20204/Documents/data/gnomad_cosmic/known_drivers.bed", sep = "\t")
#%%
gold_standard = gold_standard[cosmic_gnomad_sampled.columns.tolist()]
gold_standard["driver_stat"] = 3
#%%

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



def plot_distributions(feature, dataset, clusters):
    # Create a side-by-side violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=clusters, y=feature, data=dataset, palette='Set1')

    # Add labels and title
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.title(f'Distribution of {feature} for Clusters 0 and 1')

    # Show the plot
    plt.show()


# Below basically gives you the percentage of positives found in the cluster

def get_percentages(dataset, clusters):
    print("percentage of cosmic in each cluster: ")
    print("cluster 1:", len(dataset[clusters == 0][dataset[clusters == 0]["driver_stat"] == 1]) / len(dataset[dataset["driver_stat"] == 1])*100)
    print("cluster 2:", len(dataset[clusters == 1][dataset[clusters == 1]["driver_stat"] == 1]) / len(dataset[dataset["driver_stat"] == 1])*100)
    if 2 in set(clusters.tolist()):
        print("cluster 3:", len(dataset[clusters == 2][dataset[clusters == 2]["driver_stat"] == 1]) / len(dataset[dataset["driver_stat"] == 1])*100)

    if 3 in dataset["driver_stat"].tolist():
        print(dataset["driver_stat"])
        print("percentage of gold standard in each cluster: ")
        print("cluster 1:", len(dataset[clusters == 0][dataset[clusters == 0]["driver_stat"] == 3]) / len(dataset[dataset["driver_stat"] == 3])*100)
        print("cluster 2:", len(dataset[clusters == 1][dataset[clusters == 1]["driver_stat"] == 3]) / len(dataset[dataset["driver_stat"] == 3])*100)
        if 2 in set(clusters.tolist()):
            print("cluster 2:", len(dataset[clusters == 2][dataset[clusters == 2]["driver_stat"] == 3]) / len(dataset[dataset["driver_stat"] == 3])*100)
                


def get_cluster_res(dataset, gold_standard_variants, n_clusters, plot_violins=False):
    dataset_no_ds = dataset.drop("driver_stat", axis = 1)
    # Generate some sample data (replace this with your dataset)
    X = np.array(dataset.drop("driver_stat", axis = 1))

    # Create a K-Means clustering model
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit the model to the data
    kmeans.fit(X)

    # Assuming your data is in a variable 'X'
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    scaled_data = scaler.fit_transform(X)

    # Create a PCA object with the number of components you want (e.g., 2 for 2D visualization)
    n_components = 2  # You can adjust this based on your preference
    pca = PCA(n_components=n_components)

    # Fit and transform the scaled data using PCA
    principal_components = pca.fit_transform(scaled_data)

    # Fit the K-Means model to the PCA-reduced data
    clusters = kmeans.fit_predict(principal_components)

    # Calculate the silhouette scores
    silhouette_avg = silhouette_score(principal_components, clusters)

    gold_standard_variants = gold_standard_variants.drop("driver_stat", axis= 1)

    # Step 1: Scale the gold standard data
    scaled_new_data = scaler.transform(gold_standard_variants)

    # Step 2: Apply PCA
    pca_new_data = pca.transform(scaled_new_data)
    # Use the model to predict clusters for the new dataset
    predicted_clusters = kmeans.predict(pca_new_data)

    print("predicted_cluster_gold_standard", predicted_clusters)

    print("The average silhouette score for the clusters is:", silhouette_avg)

    # Visualize clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization with Clusters')
    plt.grid(True)
    plt.show()


    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(principal_components)  # Replace 'data' with your dataset
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()

    if plot_violins == True:
        print(len(dataset_no_ds.columns))
        print(len(clusters))
        [plot_distributions(i, dataset_no_ds, clusters) for i in dataset_no_ds.columns]


    get_percentages(dataset, clusters)
#%%

# %%

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering

#%%
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_cluster_res(dataset, gold_standard_variants, n_clusters, plot_violins=False):
    dataset_no_ds = dataset.drop("driver_stat", axis=1)
    X = np.array(dataset_no_ds)

    # Common preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # K-Means Clustering (for reference)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_clusters = kmeans.fit_predict(principal_components)
    print_silhouette_score(principal_components, kmeans_clusters, 'K-Means')

    # Predict clusters for the PCA-reduced gold standard data
    gold_standard_variants = gold_standard_variants.drop("driver_stat", axis =1)
    pca_new_data = pca.transform(scaler.transform(gold_standard_variants))  # Assuming pca and scaler are already fitted to the dataset
    predicted_clusters_gold_standard = kmeans.predict(pca_new_data)  # Using KMeans as an example
    
    # Calculate and print percentages
    total_gold_standard = len(predicted_clusters_gold_standard)
    unique_clusters = np.unique(predicted_clusters_gold_standard)
    print("Predicted cluster distribution for gold standard variants:")
    for cluster in unique_clusters:
        cluster_count = np.sum(predicted_clusters_gold_standard == cluster)
        percentage = (cluster_count / total_gold_standard) * 100
        print(f"Cluster {cluster}: {percentage:.2f}% of gold standard variants")
    
    # Gaussian Mixture Models
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm_clusters = gmm.fit_predict(principal_components)
    print_silhouette_score(principal_components, gmm_clusters, 'GMM')

    predicted_clusters_gold_standard = gmm.predict(pca_new_data)  # Using KMeans as an example
    
    # Calculate and print percentages
    total_gold_standard = len(predicted_clusters_gold_standard)
    unique_clusters = np.unique(predicted_clusters_gold_standard)
    print("Predicted cluster distribution for gold standard variants:")
    for cluster in unique_clusters:
        cluster_count = np.sum(predicted_clusters_gold_standard == cluster)
        percentage = (cluster_count / total_gold_standard) * 100
        print(f"Cluster {cluster}: {percentage:.2f}% of gold standard variants")

def print_silhouette_score(data, clusters, method_name):
    # Calculate silhouette score only if more than one cluster exists
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(data, clusters)
        print(f"The average silhouette score for {method_name} is: {score}")
    else:
        print(f"{method_name} produced a single cluster.")

get_cluster_res(cosmic_gnomad_sampled, gold_standard, 2, True)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

def get_cluster_res(dataset, gold_standard_variants, n_clusters, plot_violins=False):
    dataset_no_ds = dataset.drop("driver_stat", axis=1)
    X = np.array(dataset_no_ds)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_clusters = kmeans.fit_predict(principal_components)
    dataset_no_ds['Cluster'] = kmeans_clusters  # Append cluster assignments to the dataset

    # Select top 5 features based on variance
    top_features = select_top_features_by_variance(dataset_no_ds, 6)
    print(top_features)
    
    if plot_violins:
        plot_distributions_by_cluster(dataset_no_ds, kmeans_clusters, "K-Means", top_features)

def select_top_features_by_variance(dataset, n_features=5):
    # Calculate variances of each feature
    variances = dataset.var().sort_values(ascending=False)
    # Select the names of the top n features
    top_features = variances.head(n_features).index.tolist()
    return top_features

def plot_distributions_by_cluster(dataset, clusters, title, top_features):
    # Set up the matplotlib figure with a 3x2 grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    
    # Flatten the array of axes to easily iterate over it
    axs = axs.flatten()
    
    # Iterate over the top 6 features and their respective axes to plot
    for i, feature in enumerate(top_features):
        sns.violinplot(x='Cluster', y=feature, data=dataset, ax=axs[i])
        axs[i].set_title(feature)
        axs[i].set_xlabel('Cluster')
        axs[i].set_ylabel('Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

get_cluster_res(cosmic_gnomad_sampled, gold_standard, 2, True)

# %%
