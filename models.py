# models.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def RL_pipeline(random_state=42):
    """Pipeline: StandardScaler + LogisticRegression."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state))
    ])
    return pipe

def MLP_pipeline(random_state=42):
    """Pipeline: StandardScaler + MLPClassifier."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=2000, random_state=random_state))
    ])
    return pipe
