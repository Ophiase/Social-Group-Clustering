import numpy as np
import pandas as pd
from client.engine.clustering import spectral_clustering_neighbors, spectral_clustering_epsilon
from sklearn.metrics import silhouette_score


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


def analyze_cluster_characteristics(df: pd.DataFrame, labels: pd.Series) -> tuple:
    """
    Analyze the characteristics of each cluster by calculating the mean and standard deviation of the features.

    :param df: DataFrame containing the dataset
    :param labels: Cluster labels for each data point
    :return: Tuple containing the mean and standard deviation DataFrames for each cluster
    """
    df['Cluster'] = labels
    cluster_summary = df.groupby('Cluster').mean()
    cluster_summary_std = df.groupby('Cluster').std()
    return cluster_summary, cluster_summary_std
