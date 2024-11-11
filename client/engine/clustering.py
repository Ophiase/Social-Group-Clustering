from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px



def reduce_dimensionality(
    df: pd.DataFrame, 
    n_components: int, 
    method: Literal['PCA', 't-SNE'], 
    random_state: int
) -> np.ndarray:
    """
    Reduce the dimensionality of the data using PCA or t-SNE.

    :param df: The data frame with the dataset.
    :param n_components: The number of components to reduce the data to (2 for 2D plotting).
    :param method: The method for dimensionality reduction ('PCA' or 't-SNE').
    :param random_state: The random state to control the randomness of the transformation.
    :return: The reduced data (2D).
    """
    if method == 'PCA':
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_data = pca.fit_transform(df)
    elif method == 't-SNE':
        tsne = TSNE(n_components=n_components, random_state=random_state)
        reduced_data = tsne.fit_transform(df)
    else:
        raise ValueError("Method must be either 'PCA' or 't-SNE'.")
    
    return reduced_data


def spectral_clustering_neighbors(
    X: np.ndarray, 
    n_clusters: int, 
    k_neighbors: int, 
    random_state: int
) -> np.ndarray:
    """
    Apply Spectral Clustering using a graph of k-nearest neighbors.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param k_neighbors: The number of neighbors for the graph construction.
    :param random_state: The seed for random number generation (optional).
    :return: The cluster labels.
    """
    connectivity = kneighbors_graph(X, n_neighbors=k_neighbors, include_self=False)
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                             assign_labels='kmeans', random_state=random_state)
    labels = spectral_clustering.fit_predict(connectivity.toarray())
    return labels


def spectral_clustering_epsilon(
    X: np.ndarray, 
    n_clusters: int, 
    epsilon: float, 
    random_state: int
) -> np.ndarray:
    """
    Apply Spectral Clustering using a graph of epsilon-neighborhoods.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param epsilon: The radius for constructing the epsilon-neighborhood graph.
    :param random_state: The seed for random number generation (optional).
    :return: The cluster labels.
    """
    connectivity = radius_neighbors_graph(X, radius=epsilon, mode='distance', include_self=False)
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='sigmoid', 
                                             assign_labels='kmeans', random_state=random_state)
    labels = spectral_clustering.fit_predict(connectivity.toarray())
    return labels


