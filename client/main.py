import warnings
import pandas as pd
import argparse
from gui.application import Application

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option('display.max_columns', 3)

    parser = argparse.ArgumentParser(description="Run clustering application")
    parser.add_argument('--n_clusters', type=int, default=Application.DEFAULT_N_CLUSTERS)
    parser.add_argument('--method', type=str, default=Application.DEFAULT_METHOD)
    parser.add_argument('--suffix', type=str, default="default")
    parser.add_argument('--file', type=str, required=True, help="Path to input CSV file")

    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    app = Application(n_clusters=args.n_clusters, method=args.method, suffix=args.suffix)
    app.process(df)

