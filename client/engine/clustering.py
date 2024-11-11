from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


####CLUSTERING#####

def reduce_dimensionality(df, n_components, method, random_state):
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


def spectral_clustering_neighbors(X, n_clusters, k_neighbors, random_state):
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


def spectral_clustering_epsilon(X, n_clusters, epsilon, random_state):
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

####CLUSTERING ANALYSIS######

def evaluate_silhouette_scores(X, n_clusters, k_neighbors, epsilon):
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


def analyze_cluster_characteristics(df, labels):
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


#####VISUALISATION#####

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

###CLUSTERING ON SCENARIOS#####


def apply_weighting(df, weighting_dict):
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


def define_scenarios():
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


def test_and_visualize_scenarios(df, n_clusters,method, cluster_labels, cluster_summary, cluster_summary_std, scenarios, n_components, k_neighbors, epsilon):
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


    n_clusters = 2
    k_neighbors = 30
    epsilon = 15
    n_components = 2
    random_state = 42
    method='t-SNE'


