
from typing import Dict
import numpy as np
import os
from plotly.graph_objects import Figure
import pandas as pd

def save(df: pd.DataFrame, graphics: Dict[str, Figure], clusters: np.ndarray, where: str):
    path = f"tmp/{where}"
    os.makedirs("tmp", exist_ok=True)
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    else:
        os.makedirs(path, exist_ok=True)

    clusters_df = pd.DataFrame({"index": df.index, "cluster": clusters}).sort_values(by="cluster")
    clusters_df.to_csv(f"{path}/clusters.csv", index=False)

    for name, fig in graphics.items():
        fig.write_image(f"{path}/{name}.png")