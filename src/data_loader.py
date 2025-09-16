import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import logger
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Load IMDB data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path, sep=',', header=0, names=['Review', 'Sentiment'])
    logger.info("Data loaded successfully.")
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split data, stratify by sentiment."""
    X = df['Review']
    y = df['Sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    logger.info(f"Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test