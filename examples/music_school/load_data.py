import kagglehub
import pandas as pd
import os
import numpy as np

def extract() -> str:
    folder = kagglehub.dataset_download("catherinerasgaitis/mxmh-survey-results")
    return os.path.join(folder, "mxmh_survey_results.csv")

def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = scale_frequency_columns(df)
    df = encode_categorical(df)
    df = scale_numeric_columns(df)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Timestamp", "Permissions"])

    boolean_columns = ["While working", "Instrumentalist", "Composer", "Exploratory", "Foreign languages"]
    # boolean_columns = df.columns[df.apply(lambda col: col.astype(str).isin(['True', 'False', 'yes', 'no']).any())]
    for col in boolean_columns:
        df[col] = df[col].astype(str).map({'True': 10, 'False': 0, 'Yes': 10, 'No': 0}).fillna(0).astype(int)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    df = df.fillna(0)

    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ["Primary streaming service", "While working", "Instrumentalist", "Composer", "Fav genre", "Exploratory", "Foreign languages", "Music effects"]
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

def scale_frequency_columns(df: pd.DataFrame) -> pd.DataFrame:
    frequency_mapping = {"Never": 0, "Rarely": 2, "Sometimes": 5, "Very frequently": 8, "Always": 10}
    frequency_columns = [col for col in df.columns if col.startswith("Frequency")]
    for col in frequency_columns:
        df[col] = df[col].map(frequency_mapping).fillna(0)
    return df

def scale_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["Age", "BPM", "Hours per day"]
    
    # print(df["BPM"].min(), df["BPM"].max)
    # df["BPM"] = np.log(df["BPM"])

    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val) * 10
        else:
            df[col] = 0 
    return df