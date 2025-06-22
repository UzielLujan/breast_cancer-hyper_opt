# viz.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

def plot_3d_pca(X, y, title="Visualización PCA en 3D del dataset Cancer Breast", color_sequence=px.colors.qualitative.Set1):
    """
    Genera un scatter plot 3D con PCA a partir de X (features) y y (etiquetas).
    """
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['target'] = y.astype(str) if hasattr(y, 'astype') else [str(val) for val in y]

    fig = px.scatter_3d(
        df_pca,
        x='PC1', y='PC2', z='PC3',
        color='target',
        title=title,
        labels={'target': 'Clase'},
        color_discrete_sequence=color_sequence
    )
    fig.update_traces(marker=dict(
        size=5,           # Tamaño de los puntos
        opacity=0.7,      # Transparencia (0=transparente, 1=opaco)
        line=dict(
            width=1,      # Grosor del borde
            color='black' # Color del borde
        )
    ))
    # Calcula los rangos de cada eje
    x_range = [df_pca['PC1'].min(), df_pca['PC1'].max()]
    y_range = [df_pca['PC2'].min(), df_pca['PC2'].max()]
    z_range = [df_pca['PC3'].min(), df_pca['PC3'].max()]

    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range)
        ),
        legend_title_text='Clase',
        legend=dict(
            font=dict(size=18),
            title_font=dict(size=20),
            itemsizing='constant'
        )
    )

    pio.renderers.default = "browser"
    fig.write_html("breast_cancer-hyper_op/Visualizations/pca_3d.html")
    fig.show()

# Graficar F1-score de múltiples modelos
def plot_f1_comparison(y_test, preds_dict):
    """
    Compara F1-score de múltiples modelos.
    
    Parameters:
        y_test: Etiquetas verdaderas.
        preds_dict: Diccionario {nombre_modelo: y_pred}.
    """
    from sklearn.metrics import classification_report

    f1_scores = {}
    for name, preds in preds_dict.items():
        report = classification_report(y_test, preds, output_dict=True)
        f1_scores[name] = [report['0']['f1-score'], report['1']['f1-score']]

    labels = ['Clase 0', 'Clase 1']
    x = np.arange(len(labels))
    bar_width = 0.15
    offset = 0

    plt.figure(figsize=(8, 5))
    for i, (name, scores) in enumerate(f1_scores.items()):
        plt.bar(x + offset, scores, width=bar_width, label=name)
        offset += bar_width

    plt.xticks(x + bar_width * (len(f1_scores) - 1) / 2, labels)
    plt.ylabel('F1-score')
    plt.title('Comparación de F1-score por clase')
    plt.ylim(0.9, 1.0)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()