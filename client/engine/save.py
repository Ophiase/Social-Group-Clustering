
from typing import Dict
import numpy as np
import os
from plotly.graph_objects import Figure
import pandas as pd

def save(
        df: pd.DataFrame, 
        graphics: Dict[str, Figure], 
        clustering_result: Dict, 
        where: str
        ) -> None:
    
    path = os.path.join("tmp", where)
    os.makedirs("tmp", exist_ok=True)
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    else:
        os.makedirs(path, exist_ok=True)

    clusters_df = pd.DataFrame({"index": df.index, "cluster": clustering_result["clusters"]}).sort_values(by="cluster")    
    clusters_df.to_csv(os.path.join(path, "clusters.csv"), index=False)

    metadata_df = pd.DataFrame({
        'Which method is used ?': [ 
            "k-neighbors spectral clustering" 
            if clustering_result["used_neighbors_instead_of_k"] 
            else "epsilon spectral clustering"
            ],
        'Score using k-neighbors spectral clustering': [ clustering_result["score_k"] ],
        'Score using epsilon spectral clustering': [ clustering_result["score_epsilon"] ]
    })
    metadata_df.to_csv(os.path.join(path,'metadata.csv'), index=False)

    for name, fig in graphics.items():
        fig.write_image(f"{path}/{name}.png")