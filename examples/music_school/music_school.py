# from client.engine.clustering import cluster
from .load_data import extract, load, transform
import pandas as pd

def main():
    
    pd.set_option('display.max_columns', 10)
    path = extract()
    df = load(path)
    transformed_df = transform(df)
    display_data(transformed_df)

    #########################################
    
    # TODO

def display_data(df):
    print(df.head())

if __name__ == '__main__':
    main()