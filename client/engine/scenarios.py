import random
from typing import Dict
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from client.engine.clustering import reduce_dimensionality, spectral_clustering_neighbors, spectral_clustering_epsilon, evaluate_silhouette_scores
from client.engine.analysis import analyze_cluster_characteristics
from client.engine.visualisations import plot_cluster_heatmap, plot_numerical_distribution, plot_cluster_correlation_heatmap, plot_clustering
from client.engine.preprocessing import Weighting, apply_weighting

def test_and_visualize_scenarios(
    df: pd.DataFrame,
    n_clusters: int,
    method: str,
    cluster_labels: pd.Series,
    cluster_summary: pd.DataFrame,
    cluster_summary_std: pd.DataFrame,
    scenarios: Dict[str, Weighting],
    n_components: int,
    k_neighbors: int,
    epsilon: float
) -> None:    
    """
    Test different weighting scenarios, apply the weighting, reduce dimensionality, and generate all visualizations.

    :param df: DataFrame containing the original data
    :param cluster_labels: Cluster labels for visualization
    :param cluster_summary: Summary of characteristics for each cluster
    :param cluster_summary_std: Standard deviation of characteristics for each cluster
    :param scenarios: Dictionary of weighting scenarios
    :param n_components: Number of components for dimensionality reduction (default 2)
    :param k_neighbors: Number of neighbors for k-nearest neighbors clustering
    :param epsilon: Epsilon value for density-based (epsilon) clustering
    """
    random_state = random.randint(1, 900)

    reduced_data_normal = reduce_dimensionality(df, n_components,method,random_state)
    
    score_epsilon_normal, score_k_normal = evaluate_silhouette_scores(reduced_data_normal, n_clusters, k_neighbors, epsilon)

    print(f"Silhouette score for original data (epsilon method): {score_epsilon_normal:.3f}")
    print(f"Silhouette score for original data (k-nearest neighbors): {score_k_normal:.3f}")

    plot_cluster_heatmap(cluster_summary_std, title_suffix="Standard Deviation (Original)")
    plot_cluster_heatmap(cluster_summary, title_suffix="Mean (Original)")

    selected_features = ['Anxiety', 'Depression', 'Hours per day', 'Music effects', 'While working']
    plot_numerical_distribution(df, 'Anxiety', cluster_labels)
    plot_numerical_distribution(df, 'Depression', cluster_labels)
    plot_cluster_correlation_heatmap(df, cluster_labels)
    # plot_selected_features_histogram(df, selected_features, cluster_labels)

    for scenario_name, weighting_dict in scenarios.items():
        print(f"\nTesting {scenario_name}...")

        df_weighted = apply_weighting(df, weighting_dict)
        reduced_data_weighted = reduce_dimensionality(df_weighted, n_components,method,random_state)

        score_epsilon_weighted, score_k_weighted = evaluate_silhouette_scores(reduced_data_weighted, n_clusters, k_neighbors, epsilon)

        if score_epsilon_weighted > score_k_weighted:
            cluster_labels_weighted = spectral_clustering_neighbors(reduced_data_weighted, n_clusters, k_neighbors, random_state=random_state)
        else:
            cluster_labels_weighted = spectral_clustering_epsilon(reduced_data_weighted, n_clusters, epsilon, random_state=random_state)

        print(f"Silhouette score for weighted data (epsilon method): {score_epsilon_weighted:.3f}")
        print(f"Silhouette score for weighted data (k-neighbors method): {score_k_weighted:.3f}")
        
        plot_cluster_heatmap(cluster_summary_std, title_suffix=f"Standard Deviation ({scenario_name})")
        plot_cluster_heatmap(cluster_summary, title_suffix=f"Mean ({scenario_name})")
        
        plot_numerical_distribution(df_weighted, 'Anxiety', cluster_labels_weighted)
        plot_numerical_distribution(df_weighted, 'Depression', cluster_labels_weighted)
        
        selected_features = ['Anxiety', 'Depression', 'Hours per day', 'Music effects', 'While working']
        # plot_selected_features_histogram(df, selected_features, cluster_labels)

        print(f"Reduced Dimensions Visualization - Before vs After Weighting for {scenario_name}")
        
        # plot_clustering(
        #     scenario_name: str,
        #     reduced_data_normal,
        #     reduced_data_weighted,
        #     cluster_labels_weighted,
        #     cluster_labels
        # )    