import os
import warnings
import pandas as pd
import argparse
from gui.application import Application
from gui.application import DimensionalityReductionMethod

DF_DEFAULT_PATH : str = os.path.join("resources", "default_df.csv")
WEIGHTS_DEFAULT_PATH : str = os.path.join("resources", "default_weights.csv")

def read_weights(file_path: str) -> dict:
    weights_df = pd.read_csv(file_path)
    if weights_df.shape[1] != 2 or not {"key", "value"}.issubset(weights_df.columns):
        raise ValueError("Weight CSV must have two columns: 'key' and 'value'.")
    return dict(zip(weights_df["key"], weights_df["value"]))

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option('display.max_columns', 3)

    parser = argparse.ArgumentParser(description="Run clustering application")
    parser.add_argument('--n_clusters', type=int, default=Application.DEFAULT_N_CLUSTERS)
    parser.add_argument('--method', type=str, default=Application.DEFAULT_METHOD.value)
    parser.add_argument('--suffix', type=str, default="default")
    parser.add_argument('--file', type=str, required=False, default=DF_DEFAULT_PATH, help="Path to input CSV file")
    parser.add_argument('--weights', type=str, required=False, default=WEIGHTS_DEFAULT_PATH, help="Path to weight CSV file")

    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    weights = read_weights(args.weights)

    method = DimensionalityReductionMethod.from_string(args.method)

    app = Application(n_clusters=args.n_clusters, method=method, suffix=args.suffix)
    app.process(df, weights)
