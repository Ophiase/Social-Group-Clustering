from typing import Dict
import pandas as pd

from client.engine.clustering import clustering
from client.engine.dimensionality_reduction_method import DimensionalityReductionMethod
from client.engine.preprocessing import apply_weights
from client.engine.save import save
from client.engine.visualisations import generate_graphics

class Application:

    DEFAULT_N_CLUSTERS = 5
    DEFAULT_METHOD = DimensionalityReductionMethod.TSNE

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS, method: DimensionalityReductionMethod = DEFAULT_METHOD, suffix: str = "default") -> None:
        self.n_clusters = n_clusters
        self.method = method
        self.suffix = suffix

    def process(self, df: pd.DataFrame, weightings: Dict = None):
        if weightings is not None:
            df = apply_weights(df, weightings)
        
        clustering_result = clustering(df, self.n_clusters, method=self.method)
        graphics = generate_graphics(df, clustering_result["clusters"])
        save(df, graphics, clustering_result, self.suffix)
