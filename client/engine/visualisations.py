from typing import Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import plotly.express as px
from plotly.subplots import make_subplots

from client.engine.analysis import analyze_cluster_characteristics, top_k_attributes_by_variance

def generate_graphics(
    df: pd.DataFrame,
    clusters : np.ndarray,
    suffix : str = "",
    max_important_attributes: int = 5
) -> Dict[str, Figure]:
    result = {}

    cluster_summary, cluster_summary_std, cluster_variance = analyze_cluster_characteristics(df, pd.Series(clusters))
    best_attributes = top_k_attributes_by_variance(cluster_variance, k=max_important_attributes)

    result["heat_map"] = figure_cluster_heatmap(cluster_summary, f"Heatmap | Mean : {suffix}")
    result["heat_map_std"] = figure_cluster_heatmap(cluster_summary_std, f"Heatmap | Std : {suffix}")
    result["cluster_correlation_heat_map"] = figure_cluster_correlation_heatmap(df, clusters)

    for attribute in best_attributes :
       result[f"numerical_distribution_of_{attribute}"] = figure_numerical_distribution(df, attribute, clusters)
       result[f"features_histogram_of_{attribute}"] = plot_selected_features_histogram(df, attribute, clusters)

    return result

def figure_cluster_heatmap(
    cluster_summary: pd.DataFrame, 
    title_suffix: str = ""
) -> Figure:
    """
    Plot a heatmap of mean characteristics for each cluster.

    :param cluster_summary: DataFrame containing the summary statistics for each cluster
    :param title_suffix: Suffix to append to the plot title (optional)
    """
    cluster_summary_transposed = cluster_summary.T

    fig = go.Figure(data=go.Heatmap(
        z=cluster_summary_transposed.values,
        x=cluster_summary_transposed.columns.astype(str),
        y=cluster_summary_transposed.index,
        colorscale="YlOrRd", 
        colorbar=dict(title="Mean Value"),
        text=cluster_summary_transposed.values,
        hoverinfo='text',
    ))

    fig.update_layout(
        title=f"Heatmap of Mean Characteristics by Cluster {title_suffix}",
        xaxis=dict(
            title="Clusters",
            tickmode="array",
            tickvals=cluster_summary_transposed.columns,
            ticktext=[str(int(val)) for val in cluster_summary_transposed.columns]
        ),
        yaxis_title="Features",
        width=1000,
        height=1000,
    )

    return fig

def figure_numerical_distribution(
    df: pd.DataFrame, 
    numerical_column: str, 
    clusters: np.ndarray
) -> Figure:

    """
    Plot a box plot for the distribution of a numerical column across clusters.

    :param df: DataFrame containing the dataset
    :param numerical_column: The name of the numerical column to plot
    :param labels: Cluster labels for each data point
    """
    df['Cluster'] = clusters
    fig = px.box(df, x='Cluster', y=numerical_column, title=f"Distribution of {numerical_column} by Cluster")
    fig.update_layout(xaxis_title="Clusters", yaxis_title=numerical_column)

    return fig

def figure_cluster_correlation_heatmap(
    df: pd.DataFrame, 
    clusters: np.ndarray
) -> Figure:

    """
    Plot correlation heatmaps for each cluster.

    :param df: DataFrame containing the dataset
    :param labels: Cluster labels for each data point
    """
    df['Cluster'] = clusters
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster].drop('Cluster', axis=1)
        corr = cluster_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            colorbar=dict(title="Correlation Coefficient"),
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title=f"Correlation Heatmap of Characteristics for Cluster {cluster}",
            xaxis_title="Characteristics",
            yaxis_title="Characteristics",
            xaxis=dict(tickangle=45),
            width=1000,
            height=800
        )
        
        return fig

def figure_clustering(
    scenario_name: str,
    reduced_data_normal,
    reduced_data_weighted,
    cluster_labels_weighted,
    cluster_labels
    ) -> Figure : 
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

    return fig

def plot_selected_features_histogram(
        df : pd.DataFrame, 
        feature : str, 
        clusters : np.array
        ) -> Figure:
    """
    Plot histograms for selected features, colored by cluster.

    :param df: DataFrame containing the dataset
    :param feature: List of feature name to plot
    :param cluster_labels: Cluster labels for each data point
    """
    df['Cluster'] = clusters

    fig = go.Figure()
    fig = px.histogram(df, x=feature, color="Cluster", barmode="overlay", nbins=20,
                           title=f"Distribution of {feature} by Cluster")
    fig.update_layout(xaxis_title=feature, yaxis_title="Frequency")

    return fig
