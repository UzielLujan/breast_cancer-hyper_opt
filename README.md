#  breast_cancer-hyper_opt

Optimización de hiperparámetros para la clasificación binaria del cáncer de mama utilizando `Logistic Regression` y `Multilayer Perceptron`, con evaluación robusta vía `Nested Cross Validation` y visualizaciones interactivas.

---

## Descripción del proyecto

Este proyecto aplica técnicas de optimización de hiperparámetros (Grid Search y Random Search) sobre dos modelos clásicos de clasificación:
- **Regresión Logística (RL)**
- **Multilayer Perceptron (MLP)**

Se evalúa el rendimiento en distintas métricas, y se realiza una validación cruzada anidada para estimar de forma robusta la capacidad de generalización de cada modelo. Finalmente, se generan visualizaciones gráficas para análisis exploratorio y presentación de resultados.

---

## Dataset

El conjunto de datos utilizado es el **Breast Cancer Wisconsin Diagnostic Dataset**, disponible en la [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

- 569 instancias
- 30 características numéricas
- Objetivo binario: diagnóstico `Maligno` o `Benigno`

---

##  Visualizaciones

Las visualizaciones generadas (gráficos y reportes) están en la carpeta [`Visualizations/`](./Visualizations).

Puedes descargar y abrir el archivo de Proyeccion PCA 3D en [`pca_3d.html`](./Visualizations/pca_3d.html) 

Y para ver la proyección interactiva en tu navegador directamente: 

- [PCA en 3D](https://uziellujan.github.io/breast_cancer-hyper_opt/pca_3d.html)


---

## Estructura del proyecto

```bash
breast_cancer-hyper_opt/
├── BreastCancer_HyperOpt.ipynb # Notebook que incluye todo el flujo de trabajo
├── main.py                  # Script principal: orquesta todo el flujo
├── config.py                # Parámetros generales, grids y random search
├── data_utils.py            # Carga y partición del dataset
├── models.py                # Pipelines de RL y MLP
├── tuning.py                # GridSearch y RandomizedSearch
├── evaluation.py            # Reports y Nested CV
├── vis.py                   # Visualización 3D PCA y gráficas de F1
├── Visualizations/          # Resultados y visualizaciones generadas
│   ├── pca_3d.html          # Gráfico interactivo de PCA
│   ├── *.png                # Reportes visuales por modelo
│   ├── *.txt                # Scores de Nested CV
└── requirements.txt         # Dependencias del proyecto




