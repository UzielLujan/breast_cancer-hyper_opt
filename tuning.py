# tuning.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def run_grid_search(model, param_grid, X, y, cv=5, scoring='accuracy', n_jobs=-1):
    """Ejecuta una búsqueda por malla (GridSearchCV)."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X, y)
    return grid

def run_random_search(model, param_dist, X, y, cv=5, scoring='accuracy',
                      n_iter=50, random_state=42, n_jobs=-1):
    """Ejecuta una búsqueda aleatoria (RandomizedSearchCV)."""
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )
    random_search.fit(X, y)
    return random_search
