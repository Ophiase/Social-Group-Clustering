
from typing import Dict
import pandas as pd
from .load_data import transform, extract, load
import warnings
from client.gui.application import Application

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
    
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option('display.max_columns', 3)  

    print("ELT\n---\n")
    df = transform(load(extract()))
    print(df)

    #########################################

    print("\nEvaluate scenarios : \n---\n")
    for scenario in SCENARIOS:
        print(f"\nCurrent scenario : {scenario}\n---\n")
        print(SCENARIOS[scenario])
        Application(N_CLUSTERS, METHOD).process(df, SCENARIOS[scenario])
        print("========================================================")

if __name__ == '__main__':
    main()