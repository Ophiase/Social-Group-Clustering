from typing import Literal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from client.engine.analysis import silhouette_score
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import random

def clustering(
        df: pd.DataFrame, 
        n_clusters: int, 
        epsilon : float = 15, k_neighbors : int = 30,
        n_components : int = 2,
        method: Literal['PCA', 't-SNE'] = 't-SNE', 
        random_state: int = None
    ) -> np.ndarray:
    
    X = reduce_dimensionality(df, n_components, method, random_state)

    cluster_labels_k = spectral_clustering_neighbors(X, n_clusters=n_clusters, k_neighbors=k_neighbors, random_state=2838)
    cluster_labels_epsilon = spectral_clustering_epsilon(X, n_clusters=n_clusters, epsilon=epsilon, random_state=3982)

    score_k = silhouette_score(X, cluster_labels_k)
    score_epsilon = silhouette_score(X, cluster_labels_epsilon)

    return cluster_labels_k if score_k > score_epsilon else cluster_labels_epsilon

def reduce_dimensionality(
    df: pd.DataFrame, 
    n_components: int, 
    method: Literal['PCA', 't-SNE'], 
    random_state: int = None
) -> np.ndarray:
    """
    Reduce the dimensionality of the data using PCA or t-SNE.

    :param df: The data frame with the dataset.
    :param n_components: The number of components to reduce the data to (2 for 2D plotting).
    :param method: The method for dimensionality reduction ('PCA' or 't-SNE').
    :param random_state: The random state to control the randomness of the transformation.
    :return: The reduced data (2D).
    """

    if random_state is None : random_state = random.randint(1, 299333)

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
    random_state: int = None
) -> np.ndarray:
    """
    Apply Spectral Clustering using a graph of k-nearest neighbors.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param k_neighbors: The number of neighbors for the graph construction.
    :param random_state: The seed for random number generation (optional).
    :return: The cluster labels.
    """
    if random_state is None : random_state = random.randint(1, 299333)

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


def evaluate_silhouette_scores(X: np.ndarray, n_clusters: int, k_neighbors: int, epsilon: float) -> tuple:
    """
    Évalue les scores de silhouette pour les deux méthodes de clustering spectral : k-nearest neighbors et epsilon-neighborhood.
    :param X: Les données après réduction de dimension (PCA ou t-SNE).
    :param n_clusters: Le nombre de clusters à former.
    :param k_neighbors: Le nombre de voisins pour le graph de k-nearest neighbors.
    :param epsilon: Le rayon pour le graph de epsilon-neighborhoods.
    :return: Le score de silhouette pour k-nearest neighbors et epsilon-neighborhoods.
    """
    
    cluster_labels_k = spectral_clustering_neighbors(X, n_clusters=n_clusters, k_neighbors=k_neighbors)
    cluster_labels_epsilon = spectral_clustering_epsilon(X, n_clusters=n_clusters, epsilon=epsilon)

    score_k = silhouette_score(X, cluster_labels_k)
    score_epsilon = silhouette_score(X, cluster_labels_epsilon)
    
    return score_epsilon, score_k