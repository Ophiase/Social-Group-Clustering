
from typing import Dict
import pandas as pd
from client.engine.clustering import evaluate_silhouette_scores, reduce_dimensionality, spectral_clustering_neighbors, spectral_clustering_epsilon, clustering
from client.engine.analysis import analyze_cluster_characteristics
from client.engine.save import save
from client.engine.visualisations import figure_cluster_heatmap, figure_numerical_distribution, figure_cluster_correlation_heatmap, generate_graphics
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

        clusters = clustering(df, N_CLUSTERS, method=METHOD)
        graphics = generate_graphics(df, clusters, None, None, None)
        save(df, graphics, clusters, scenario)

if __name__ == '__main__':
    main()