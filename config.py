# config.py

# Semilla global para reproducibilidad
SEED = 40

# Parámetros para GridSearch de Regresión Logística
GRID_PARAMS_RL = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2']
}

# Parámetros para RandomSearch de Regresión Logística 
from scipy.stats import loguniform
RANDOM_PARAMS_RL = {
    'clf__C': loguniform(1e-4, 1e2),
    'clf__penalty': ['l1', 'l2']
}

# Parámetros para GridSearch de MLP
GRID_PARAMS_MLP = {
    'mlp__hidden_layer_sizes': [(50,), (60,), (120, 60), (240, 120, 60)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.00001, 0.0001, 0.001],
    'mlp__learning_rate_init': [0.001, 0.01]
}

# Parámetros para RandomSearch de MLP
RANDOM_PARAMS_MLP = {
    'mlp__hidden_layer_sizes': [(50,), (60,), (120, 60), (240, 120, 60)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': loguniform(1e-5, 1e-2),
    'mlp__learning_rate_init': loguniform(1e-3, 1e-1)
}
# Número de folds para validación cruzada y Nested CV
N_FOLDS = 5

# Métricas por defecto
SCORING = 'accuracy'
