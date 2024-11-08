from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph


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

def spectral_clustering_neighbors(X, n_clusters, k_neighbors):
    """
    Apply Spectral Clustering using a graph of k-nearest neighbors.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param k_neighbors: The number of neighbors for the graph construction.
    :return: The cluster labels.
    not super sensitive to the number of neighbors.
    """
    connectivity = kneighbors_graph(X, n_neighbors=k_neighbors, include_self=False)
    
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = spectral_clustering.fit_predict(connectivity.toarray())
    
    return labels


def spectral_clustering_epsilon(X, n_clusters, epsilon):
    """
    Apply Spectral Clustering using a graph of epsilon-neighborhoods.
    :param X: The dataset after dimensionality reduction.
    :param n_clusters: The number of clusters to form.
    :param epsilon: The radius for constructing the epsilon-neighborhood graph.
    :return: The cluster labels.
    Good for big epsilon 10,15
    """
    # Construire un graphe basé sur les epsilon-voisins (c'est-à-dire les points dans un rayon de distance epsilon)
    connectivity = radius_neighbors_graph(X, radius=epsilon, mode='distance', include_self=False)
    
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='sigmoid', assign_labels='kmeans', random_state=random_state)
    labels = spectral_clustering.fit_predict(connectivity.toarray())
    
    return labels

