import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_cluster_heatmap(cluster_summary, title_suffix=""):
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
        height=600,
    )
    
    fig.show()



def plot_numerical_distribution(df, numerical_column, labels):
    """
    Plot a box plot for the distribution of a numerical column across clusters.

    :param df: DataFrame containing the dataset
    :param numerical_column: The name of the numerical column to plot
    :param labels: Cluster labels for each data point
    """
    df['Cluster'] = labels
    fig = px.box(df, x='Cluster', y=numerical_column, title=f"Distribution of {numerical_column} by Cluster")
    fig.update_layout(xaxis_title="Clusters", yaxis_title=numerical_column)
    fig.show()




def plot_cluster_correlation_heatmap(df, labels):
    """
    Plot correlation heatmaps for each cluster.

    :param df: DataFrame containing the dataset
    :param labels: Cluster labels for each data point
    """
    df['Cluster'] = labels
    
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
        
        fig.show()



        
