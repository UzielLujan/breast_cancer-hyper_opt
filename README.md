# breast_cancer-hyper_opt

Optimización de hiperparámetros para la clasificación binaria del cáncer de mama utilizando `Logistic Regression` y `Multilayer Perceptron`, con evaluación robusta vía `Nested Cross Validation` y visualizaciones interactivas en línea.

---

## Descripción del proyecto

Este proyecto aplica técnicas de optimización de hiperparámetros (`Grid Search` y `Random Search`) sobre dos modelos clásicos de clasificación:

- **Regresión Logística (RL)**
- **Multilayer Perceptron (MLP)**

Se evalúa el rendimiento en distintas métricas y se emplea una validación cruzada anidada para estimar de forma robusta la capacidad de generalización de cada modelo.  
Además, se generan visualizaciones detalladas para análisis exploratorio y presentación de resultados.

---

## Dataset

El conjunto de datos utilizado es el **Breast Cancer Wisconsin Diagnostic Dataset**, disponible en la [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

- 569 instancias  
- 30 características numéricas  
- Objetivo binario: diagnóstico `Maligno` o `Benigno`

---

## Visualizaciones

Puedes ver todos los gráficos y reportes del proyecto directamente en:

🔗 **Sitio del proyecto con visualizaciones**:  
➡️ [https://uziellujan.github.io/breast_cancer-hyper_opt/](https://uziellujan.github.io/breast_cancer-hyper_opt/)

Incluye:

- Proyección interactiva de **PCA en 3D**
- Reportes visuales por modelo (F1, matrices, etc.)
- Resultados de validación cruzada anidada

---

## Estructura del proyecto

```bash
breast_cancer-hyper_opt/
├── BreastCancer_HyperOpt.ipynb   # Notebook principal
├── main.py                       # Script orquestador
├── config.py                     # Parámetros y grids de búsqueda
├── data_utils.py                 # Carga y división del dataset
├── models.py                     # Pipelines de RL y MLP
├── tuning.py                     # GridSearch y RandomizedSearch
├── evaluation.py                 # Métricas y validación cruzada
├── vis.py                        # Visualización PCA y gráficas de F1
├── docs/                         # Visualizaciones publicadas en GitHub Pages
│   ├── index.html                # Página principal del proyecto
│   ├── pca_3d.html               # Visualización interactiva
│   ├── *.png                     # Reportes visuales por modelo
│   ├── *.txt                     # Resultados de Nested CV
└── requirements.txt              # Dependencias del proyecto





