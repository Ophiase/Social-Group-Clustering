
from typing import Dict
import pandas as pd
from client.engine.clustering import evaluate_silhouette_scores, reduce_dimensionality, spectral_clustering_neighbors, spectral_clustering_epsilon, clustering
from client.engine.analysis import analyze_cluster_characteristics
from client.engine.save import save
from client.engine.visualisations import plot_cluster_heatmap, plot_numerical_distribution, plot_cluster_correlation_heatmap
from client.engine.preprocessing import apply_weighting
from client.engine.scenarios import test_and_visualize_scenarios
from .load_data import transform, extract, load

N_CLUSTERS = 5
METHOD = "t-SNE"
SCENARIOS = {
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
        'Anxiety': 1,
        'Depression': 1,
        'Music_Hours_Per_Week': 10,
        'While working' : 10,
        'Frequency [classical]' : 10
    }
}


def main():
    """
    Main function to load, clean, and prepare the dataset.
    """

    print("ELT\n---\n")

    pd.set_option('display.max_columns', 3)  
    df = transform(load(extract()))
    print(df)

    #########################################

    print("\nEvaluate scenarios\n---\n")
    for scenario in SCENARIOS:
        print(f"Evaluate the scenario\n---\n : {scenario}")

        graphics = {} # generate_graphics()
        clusters = clustering(df, N_CLUSTERS, method=METHOD)
        save(df, graphics, clusters, scenario)

if __name__ == '__main__':
    main()


# def display_data(
#         df : pd.DataFrame,
#         n_clusters : int = 2, 
#         k_neighbors : int = 30, 
#         epsilon : float = 15, 
#         n_components : int = 2, 
#         random_state : int = 83, 
#         method : str = 't-SNE'
#         ):
#     """
#     Function to process the data, apply clustering, evaluate clustering, and visualize results.

#     :param df: DataFrame containing the processed dataset
#     """
#     reduced_data = reduce_dimensionality(df, n_components, method, random_state)
#     score_epsilon, score_k = evaluate_silhouette_scores(reduced_data, n_clusters, k_neighbors, epsilon)
    
#     if score_k > score_epsilon:
#         cluster_labels = spectral_clustering_neighbors(reduced_data, n_clusters, k_neighbors, random_state=random_state)
#     else:
#         cluster_labels = spectral_clustering_epsilon(reduced_data, n_clusters, epsilon, random_state=random_state)

#     cluster_summary, cluster_summary_std = analyze_cluster_characteristics(df, cluster_labels)
    
#     test_and_visualize_scenarios(df, n_clusters, method, cluster_labels, cluster_summary, cluster_summary_std, SCENARIOS, n_components, k_neighbors, epsilon)


