from gui.application import Application
from engine.clustering import reduce_dimensionality, spectral_clustering_neighbors, spectral_clustering_epsilon
from engine.analysis import evaluate_silhouette_scores, analyze_cluster_characteristics
from engine.visualisations import plot_cluster_heatmap, plot_numerical_distribution, plot_cluster_correlation_heatmap
from engine.scenarios import apply_weighting, define_scenarios, test_and_visualize_scenarios
from .load_data import transform, extract, load

n_clusters = 2
k_neighbors = 30
epsilon = 15
n_components = 2
random_state = 42
method = 't-SNE'


def main():
    """
    Main function to load, clean, and prepare the dataset.
    """
    pd.set_option('display.max_columns', 10)  
    path = extract()  
    df = load(path)    
    transformed_df = transform(df)  
    display_data(transformed_df)   


def display_data(df):
    """
    Function to process the data, apply clustering, evaluate clustering, and visualize results.

    :param df: DataFrame containing the processed dataset
    """
    reduced_data = reduce_dimensionality(df, n_components, method, random_state)
    score_epsilon, score_k = evaluate_silhouette_scores(reduced_data, n_clusters, k_neighbors, epsilon)
    
    if score_k > score_epsilon:
        cluster_labels = spectral_clustering_neighbors(reduced_data, n_clusters, k_neighbors, random_state=random_state)
    else:
        cluster_labels = spectral_clustering_epsilon(reduced_data, n_clusters, epsilon, random_state=random_state)

    cluster_summary, cluster_summary_std = analyze_cluster_characteristics(df, cluster_labels)
    
    scenarios = define_scenarios()
    test_and_visualize_scenarios(df, n_clusters, method, cluster_labels, cluster_summary, cluster_summary_std, scenarios, n_components, k_neighbors, epsilon)


if __name__ == '__main__':
    main()
