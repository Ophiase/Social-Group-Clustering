import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from client.engine.clustering import reduce_dimensionality, spectral_clustering_neighbors, spectral_clustering_epsilon
from client.engine.analysis import evaluate_silhouette_scores, analyze_cluster_characteristics
from client.engine.visualisations import plot_cluster_heatmap, plot_numerical_distribution, plot_cluster_correlation_heatmap


def apply_weighting(
    df: pd.DataFrame, 
    weighting_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply feature weighting to the DataFrame based on the given weighting dictionary.

    :param df: DataFrame containing the dataset
    :param weighting_dict: Dictionary containing the features and their corresponding weights
    :return: DataFrame with weighted features
    """
    weighted_df = df.copy()
    for feature, weight in weighting_dict.items():
        if feature in weighted_df.columns:
            weighted_df[feature] *= weight
    return weighted_df


def define_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Define different weighting scenarios for the dataset.

    :return: Dictionary of scenarios with feature weights
    """
    scenarios = {
        "Scenario 1": {
            'Anxiety': 10,
            'Depression': 7,
            'Music_Hours_Per_Week': 10,
            'OCD': 4,
            'Age': 2
        },
        "Scenario 2": {
            'Anxiety': 5,
            'Depression': 5,
            'Music_Hours_Per_Week': 8,
            'OCD': 3,
            'Age': 1
        },
        "Scenario 3": {
            'Anxiety': 15,
            'Depression': 10,
            'Music_Hours_Per_Week': 12,
            'OCD': 5,
            'Age': 3
        }
    }
    return scenarios


def test_and_visualize_scenarios(
    df: pd.DataFrame,
    n_clusters: int,
    method: str,
    cluster_labels: pd.Series,
    cluster_summary: pd.DataFrame,
    cluster_summary_std: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
    n_components: int,
    k_neighbors: int,
    epsilon: float
) -> None:    """
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
    plot_selected_features_histogram(df, selected_features, cluster_labels)
    
    for scenario_name, weighting_dict in scenarios.items():
        print(f"\nTesting {scenario_name}...")

        df_weighted = apply_weighting(df, weighting_dict)reduced_data_weighted = reduce_dimensionality(df_weighted, n_components,method,random_state)

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
        plot_selected_features_histogram(df, selected_features, cluster_labels)

        print(f"Reduced Dimensions Visualization - Before vs After Weighting for {scenario_name}")
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{scenario_name} - Before Weighting", f"{scenario_name} - After Weighting"])

        
        fig.add_trace(go.Scatter(
            x=reduced_data_normal[:, 0], 
            y=reduced_data_normal[:, 1], 
            mode='markers', 
            marker=dict(color=cluster_labels, colorscale='Viridis'), 
            name="Original Data"
        ), row=1, col=1)
       fig.add_trace(go.Scatter(
            x=reduced_data_weighted[:, 0], 
            y=reduced_data_weighted[:, 1], 
            mode='markers', 
            marker=dict(color=cluster_labels_weighted, colorscale='Viridis'), 
            name="Weighted Data"
        ), row=1, col=2)

        
        fig.update_layout(
            title=f"Reduced Dimensions Visualization - Before vs After Weighting for {scenario_name}",
            xaxis=dict(range=[-60, 60]),   
            yaxis=dict(range=[-20, 20]),   
            xaxis2=dict(range=[-60, 60]),  
            yaxis2=dict(range=[-20, 20]),  
            width=900,
            height=600, 
            autosize=False
        )

        fig.show()
