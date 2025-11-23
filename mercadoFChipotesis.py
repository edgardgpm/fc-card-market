# Importación de librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy.stats import shapiro
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import probplot


# Configuración para gráficos
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)


# Enrutado del archivo .csv
file_location = os.path.dirname(os.path.abspath(__file__))

# Creación del dataframe
df = pd.read_csv(file_location + "/fc26.csv")


# Función para agregar labels a las barras
def add_bar_labels(ax, decimal=2):
    for p in ax.patches:
        value = round(p.get_height(), decimal)
        ax.text(
            p.get_x() + p.get_width()/2,
            p.get_height(),
            value,
            ha='center', va='bottom',
            fontsize=10
        )


# ==== Resolución de Hipótesis ==== #


# Hipótesis 1 - Cartas con Media >= 86, Mayor probabilidad de fluctuar en sus precios (durante semanas de evento)

print("\nHipótesis #1\n")

df["Volatilidad"] = df[["Precio S1","Precio S2","Precio S3","Precio S4","Precio S5","Precio S6","Precio S7","Precio S8"]].std(axis=1)

vol_alta = df[df["Media"] >= 86]["Volatilidad"].mean()
vol_baja = df[df["Media"] < 86]["Volatilidad"].mean()

print("Volatilidad media de cartas 86+: ", vol_alta)
print("Volatilidad media de cartas <86: ", vol_baja)


# Hipótesis 1 - Gráfico
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x=df["Media"] >= 86,
    y="Volatilidad",
    palette="muted"
)
plt.xticks([0,1], ["Media < 86", "Media ≥ 86"])
plt.title("Comparación de Volatilidad según Media")
plt.xlabel("Categoría de Media")
plt.ylabel("Volatilidad (STD de precios por carta)")
plt.show()


# Hipótesis 2 - Tendencia hacia una distribución normal de precios en periodos normales

print("\nHipótesis #2\n")

precios_sin_evento = df[["Precio S1","Precio S2","Precio S3","Precio S4",
                         "Precio S5","Precio S6","Precio S7","Precio S8"]].stack()

stat, p = shapiro(precios_sin_evento)
alpha = 0.05
print("p-value:", p)

if p > alpha:
    print("Conclusión: No se rechaza la normalidad (los precios parecen normales).")
else:
    print("Conclusión: Se rechaza la normalidad (los precios NO siguen una normal).")


# Hipótesis 2 - Gráfico
plt.figure(figsize=(8,5))
sns.histplot(precios_sin_evento, kde=True, bins=20, color="steelblue")
plt.title("Distribución de Precios (Períodos sin evento)")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.show()


plt.figure(figsize=(6,6))
probplot(precios_sin_evento, dist="norm", plot=plt)
plt.title("QQ-Plot de precios sin evento")
plt.show()


# Hipótesis 3 - Probabilidad de obtener cartas especiales (Media >= 86) en sobres estándar es < 10%

print("\nHipótesis #3\n")

freq_total = df["Frecuencia en sobres"].sum()

p_86plus = df[df["Media"] >= 86]["Frecuencia en sobres"].sum() / freq_total
print("Probabilidad total de 86+:", p_86plus)

print("¿Es menor al 10%?:", p_86plus < 0.10)


# Hipótesis 3 - Gráfico

# asegurar columna binaria
df["Es_86plus"] = (df["Media"] >= 86).astype(int)

# contar y renombrar índice correctamente
vc = df["Es_86plus"].value_counts().sort_index()  # índice: 0,1

# renombrar índice (robusto ante True/False o 0/1)
vc = vc.rename(index={0: "< 86", 1: "≥ 86", False: "< 86", True: "≥ 86"})

# Pie chart bonito con seaborn/matplotlib
plt.figure(figsize=(6,6))
vc.plot(kind="pie",
        autopct="%1.2f%%",
        colors=sns.color_palette("pastel"),
        startangle=90,
        wedgeprops={"edgecolor": "k", "linewidth": 0.5})
plt.title("Proporción de cartas Media ≥ 86")
plt.ylabel("")  # quitar label vertical
plt.gca().set_aspect("equal")  # círculo perfecto
plt.show()

# Alternativa: gráfico de barras (más claro para comparar)
plt.figure(figsize=(6,4))
sns.barplot(x=vc.index, y=vc.values, palette="pastel")
plt.ylabel("Número de cartas")
plt.title("Conteo de cartas por rango de Media")
for i, v in enumerate(vc.values):
    plt.text(i, v + max(vc.values)*0.01, str(v), ha="center")
plt.show()


# Hipótesis 4 - Mayor probabilidad de que una carta disminuya su precio al incrementar la apertura de sobres

print("\nHipótesis #4\n")

correlacion = np.corrcoef(df["Frecuencia en sobres"], df["Variación (%)"])[0,1]

print("Correlación:", correlacion)


# Hipótesis 4 - Gráfico
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Frecuencia en sobres",
    y="Variación (%)",
    hue="Media",
    palette="viridis",
    alpha=0.7
)
plt.title("Frecuencia en sobres vs Variación (%)")
plt.xlabel("Frecuencia en sobres")
plt.ylabel("Variación del precio (%)")
plt.show()


# Hipótesis 5 - Las cartas con rareza especial presentan una volatilidad mayor que las cartas oro estándar

print("\nHipótesis #5\n")

df.groupby("Rareza")["Volatilidad"].mean()

vol_por_rareza = df.groupby("Rareza")["Volatilidad"].mean()
print(vol_por_rareza)


# Hipótesis 5 - Gráfico
plt.figure(figsize=(8,5))
ax = sns.barplot(
    data=df,
    x="Rareza",
    y="Volatilidad",
    palette="muted",
    estimator=np.mean
)
plt.xticks(rotation=45)
plt.title("Volatilidad promedio según rareza")
plt.xlabel("Rareza")
plt.ylabel("Volatilidad (STD)")

add_bar_labels(ax)

plt.show()

