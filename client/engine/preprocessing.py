
from typing import Dict

import pandas as pd


Weighting = Dict[str, float]

def apply_weights(
    df: pd.DataFrame, 
    weighting_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply feature weighting to the DataFrame based on the given weighting dictionary.

    :param df: DataFrame containing the dataset
    :param weighting_dict: Dictionary containing the features and their corresponding weights
    :return: DataFrame with weighted features
    """
    weighted_df = df.copy()
    for feature, weight in weighting_dict.items():
        if feature in weighted_df.columns:
            weighted_df[feature] *= weight
    return weighted_df

