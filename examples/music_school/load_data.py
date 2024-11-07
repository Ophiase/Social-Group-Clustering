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
    df = encode_frequency_columns(df)
    df = encode_yes_no_columns(df)
    df = encode_music_effects(df)
    df = clean_data(df)
    df = apply_one_hot_encoding(df)
    df = transform_true_false_to_int(df)
    df = scale_numeric_columns(df)
    df = integer(df)
    return df

def load_data(file_path):
    """
    Loads the data from the specified CSV file.
    """
    df = pd.read_csv(file_path, sep=',')
    return df

def encode_frequency_columns(df):
    """
    Encodes the frequency columns with numeric values.
    """
    frequency_mapping = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Very frequently': 3
    }
    
    frequency_columns = [
        'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]',
        'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Latin]', 'Frequency [Lofi]', 
        'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 
        'Frequency [Video game music]'
    ]
    
    for col in frequency_columns:
        df[col] = df[col].map(frequency_mapping)
    
    return df

def encode_yes_no_columns(df):
    """
    Encodes columns with 'Yes'/'No' responses to 1/0.
    """
    yes_no_columns = ['While working','Instrumentalist','Composer','Exploratory','Foreign languages']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

def encode_music_effects(df):
    """
    Encodes the 'Music effects' column with values -1, 0, 1.
    """
    music_effects_mapping = {
        'Worsen': -1,
        'No effect': 0,
        'Improve': 1
    }
    df['Music effects'] = df['Music effects'].map(music_effects_mapping)
    return df

def clean_data(df):
    """
    Cleans the data by removing unnecessary columns, handling NaN values, and replacing specific values.
    """
    df = df.drop(['Timestamp', 'Permissions'], axis=1)
    
    df = df.fillna(0)
    
    df['BPM'] = df['BPM'].replace(0, 105)
    
    df['Primary streaming service'] = df['Primary streaming service'].fillna('Other')
    df['Primary streaming service'] = df['Primary streaming service'].replace('Other streaming service', 'Other')
    df['Primary streaming service'] = df['Primary streaming service'].replace('I do not use a streaming service.', 'No service')
    df['Primary streaming service'] = df['Primary streaming service'].replace(0, 'No service')
    
    return df



def apply_one_hot_encoding(df):
    """
    Applies One-Hot Encoding to the categorical columns 'Fav genre' and 'Primary streaming service'.
    """
    df = pd.get_dummies(df, columns=['Fav genre'])
    df = pd.get_dummies(df, columns=['Primary streaming service'])
    
    return df

def transform_true_false_to_int(df):
    """
    Transforms columns containing True/False values to 1/0.
    """
    true_false_columns = [
        'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
        'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop',
        'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 
        'Fav genre_R&B', 'Fav genre_Rap', 'Fav genre_Rock', 'Fav genre_Video game music',
        'Primary streaming service_Apple Music', 'Primary streaming service_No service', 
        'Primary streaming service_Other', 'Primary streaming service_Pandora', 
        'Primary streaming service_Spotify', 'Primary streaming service_YouTube Music'
    ]
    
    for col in true_false_columns:
        df[col] = df[col].astype(int)
    
    return df
    
def integer(df):
    """
    Converts all columns in the dataframe to integer type.
    """
    df = df.astype(int)
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
