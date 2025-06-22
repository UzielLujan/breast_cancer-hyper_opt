# main.py

from data_utils import load_data, data_split
from config import SEED, GRID_PARAMS_RL, RANDOM_PARAMS_RL, GRID_PARAMS_MLP, RANDOM_PARAMS_MLP
from models import RL_pipeline, MLP_pipeline 
from tuning import run_grid_search, run_random_search
from evaluation import print_classification_report, nested_cv_score
from vis import plot_3d_pca, plot_f1_comparison

# 1. ______________________ Carga y preparación del dataset ____________________
X , y = load_data()
# División train/test para visualizaciones o predicción ilustrativa
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=SEED, stratify=True)

# 2. ________________________ Modelos y búsquedas ________________________

# ============== Logistic Regression =================
pipe_rl = RL_pipeline()

grid_rl = run_grid_search(pipe_rl, GRID_PARAMS_RL, X, y, cv=5, scoring='accuracy', n_jobs=-1)

random_rl = run_random_search(pipe_rl, RANDOM_PARAMS_RL, X, y, cv=5, scoring='accuracy', n_iter=50, random_state=SEED, n_jobs=-1)

# ============== MLP Classifier =================
pipe_mlp = MLP_pipeline()

grid_mlp = run_grid_search(pipe_mlp, GRID_PARAMS_MLP, X, y, cv=5, scoring='f1_macro', n_jobs=-1)

random_mlp = run_random_search(pipe_mlp, RANDOM_PARAMS_MLP, X, y, cv=5, scoring='f1_macro', n_iter=50, random_state=SEED, n_jobs=-1)

# 3._______________________ Evaluación en test set _________________________
print_classification_report(grid_rl, X_test, y_test, "Logistic Regression (Grid Search)")
print_classification_report(random_rl, X_test, y_test, "Logistic Regression (Random Search)")
print_classification_report(grid_mlp, X_test, y_test, "MLP Classifier (Grid Search)")
print_classification_report(random_mlp, X_test, y_test, "MLP Classifier (Random Search)")

# 4._________________________ Nested CV _________________________
nested_cv_score(grid_rl.best_estimator_, X, y, cv=5, scoring='accuracy', name="Regresion Logistica")
nested_cv_score(grid_mlp.best_estimator_, X, y, cv=5, scoring='accuracy', name="Multilayer Perceptron")

# 5. ________________________ Visualizaciones PCA 3D ________________________

plot_3d_pca(X, y)
