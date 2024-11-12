import numpy as np
import pandas as pd
# from client.engine.clustering import spectral_clustering_neighbors, spectral_clustering_epsilon
from sklearn.metrics import silhouette_score


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
