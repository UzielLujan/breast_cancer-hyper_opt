# data_utils.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """Carga el dataset de cáncer de mama como DataFrame."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def data_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """Divide el conjunto para visualizaciones o predicción ilustrativa."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
