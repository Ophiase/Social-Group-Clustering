import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

def reduce_dimensionality(df, n_components, method):
    """
    Reduce the dimensionality of the data using PCA or t-SNE.
    :param df: The data frame with the dataset.
    :param n_components: The number of components to reduce the data to (2 for 2D plotting).
    :param method: The method for dimensionality reduction ('PCA' or 't-SNE').
    :return: The reduced data (2D).
    """
    if method == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(df)
    elif method == 't-SNE':
        tsne = TSNE(n_components=n_components)
        reduced_data = tsne.fit_transform(df)
    else:
        raise ValueError("Method must be either 'PCA' or 't-SNE'.")
    
    return reduced_data

def spectral_clustering(X, n_clusters, k_neighbors):
    """
    Apply Spectral Clustering using a graph of k-nearest neighbors.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param k_neighbors: The number of neighbors for the graph construction.
    :return: The cluster labels.
    """
    connectivity = kneighbors_graph(X, n_neighbors=k_neighbors, include_self=False)
    
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = spectral_clustering.fit_predict(connectivity.toarray())
    
    return labels



