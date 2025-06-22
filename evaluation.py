# evaluation.py

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def print_classification_report(model, X_test, y_test, title="Reporte de Clasificación"):
    """Imprime el reporte de clasificación con encabezado claro."""
    print(f"\n{title}")
    print("========================================")
    print(f"Mejores parámetros: {model.best_params_ if hasattr(model, 'best_params_') else 'N/A'}")
    print("========================================")
    print(classification_report(y_test, model.predict(X_test), digits=3))
    # Guardamos una imagen del reporte de clasificación y la matriz de confusión
    plt.title(title)
    plt.text(0.01, 0.05, classification_report(y_test, model.predict(X_test), digits=3), 
             {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f"BreastCancer-hyperparam-opt/Visualizations/{title.replace(' ', '_').lower()}_report.png")
    plt.close()
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.title(title)
    plt.savefig(f"BreastCancer-hyperparam-opt/Visualizations/{title.replace(' ', '_').lower()}.png")
    plt.close()

def nested_cv_score(estimator, X, y, cv=5, scoring='accuracy', name="Modelo"):
    """Evalúa un modelo con validación cruzada anidada."""
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\nNested CV {scoring.capitalize()} para {name}: {scores.mean():.3f} ± {scores.std():.3f}")
    # Guardamos los scores en un archivo
    with open(f"BreastCancer-hyperparam-opt/Visualizations/{name.replace(' ', '_').lower()}_nested_cv_scores.txt", 'w') as f:
        f.write(f"Nested CV {scoring.capitalize()} para {name}: {scores.mean():.3f} ± {scores.std():.3f}\n")
        f.write("Scores: " + ", ".join([f"{score:.3f}" for score in scores]) + "\n")
    return scores


