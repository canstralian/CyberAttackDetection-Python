import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()
    return df

def split_data(df, test_size=0.3, random_state=42):
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
