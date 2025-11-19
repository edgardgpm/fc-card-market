# Importación de librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import binom
from scipy.stats import poisson


# Configuración para gráficos
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)


# Enrutado del archivo .csv
file_location = os.path.dirname(os.path.abspath(__file__))

# Creación del dataframe
df = pd.read_csv(file_location + "/fc26.csv")


# Descripción Estadística Básica
print(df.head())
print(df.info())
print(df.describe())


# ==== Resolución de Hipótesis ==== #


# Hipótesis 1 - Cartas con Media >= 86, Mayor probabilidad de fluctuar en sus precios (durante semanas de evento)

print("\nHipótesis #1\n")

df["Volatilidad"] = df[["Precio S1","Precio S2","Precio S3","Precio S4","Precio S5","Precio S6","Precio S7","Precio S8"]].std(axis=1)

vol_alta = df[df["Media"] >= 86]["Volatilidad"].mean()
vol_baja = df[df["Media"] < 86]["Volatilidad"].mean()

print("Volatilidad media de cartas 86+: ", vol_alta)
print("Volatilidad media de cartas <86: ", vol_baja)


# Hipótesis 2 - Tendencia hacia una distribución normal de precios en periodos normales

print("\nHipótesis #2\n")

precios_sin_evento = df[["Precio S1","Precio S2","Precio S3","Precio S4",
                         "Precio S5","Precio S6","Precio S7","Precio S8"]].stack()

stat, p = shapiro(precios_sin_evento)
print("p-value:", p)


# Hipótesis 3 - Probabilidad de obtener cartas especiales (Media > 86) en sobres estándar es < 10%

print("\nHipótesis #3\n")

p_86plus = df[df["Media"] >= 86]["Frecuencia en sobres"].sum()
print("Probabilidad total de 86+:", p_86plus)

print("¿Es menor al 10%?:", p_86plus < 0.10)


# Hipótesis 4 - Mayor probabilidad de que una carta disminuya su precio al incrementar la apertura de sobres

print("\nHipótesis #4\n")

correlacion = np.corrcoef(df["Frecuencia en sobres"], df["Variación (%)"])[0,1]

print("Correlación:", correlacion)


# Hipótesis 5 - Las cartas con rareza especial presentan una volatilidad mayor que las cartas oro estándar

print("\nHipótesis #5\n")

df.groupby("Rareza")["Volatilidad"].mean()

vol_por_rareza = df.groupby("Rareza")["Volatilidad"].mean()
print(vol_por_rareza)


# ==== 4. Análisis Probabilístico ==== #

# 4.1 Identificación de Distribuciones


# 4.1.1 Distribución Normal
# Histograma

print("\nDist. Normal\n")

plt.figure()
sns.histplot(df["Variación (%)"], bins=8, kde=True)
plt.title("Distribución de Variación (%) con KDE")
plt.xlabel("Variación (%)")
plt.ylabel("Frecuencia")
plt.show()

stat, p = shapiro(df["Variación (%)"])
print("p-value Shapiro-Wilk:", p)

alpha = 0.05
print("No se rechaza normalidad." if p > alpha else "Se rechaza normalidad.")


# 4.1.2 Distribución Binomial

print("\nDist. Binomial\n")

# Mapeo de Datos (Subida de Precio)
df["¿Subió de precio?"] = df["¿Subió de precio?"].astype(str).str.strip()
df["Subida_bin"] = df["¿Subió de precio?"].map({"Sí":1, "No":0})

# Probabilidad de que una carta haya subido de precio (Subida_bin = 1)
p_subida = df["Subida_bin"].mean()

plt.figure()
df["¿Subió de precio?"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Proporción de cartas que subieron de precio")
plt.ylabel("")
plt.show()

print("P(subida):", p_subida)
print("P(3 subidas en 5 cartas):", binom.pmf(3, 5, p_subida))


# Probabilidad de que una carta sea especial (Media >= 86)
p_especial = df[df["Media"] >= 86]["Frecuencia en sobres"].sum()

# Tamaño del sobre
n = 7

# Valores posibles: 0 a 7 cartas especiales
k_vals = np.arange(0, n+1)

# PMF de la binomial
pmf_vals = binom.pmf(k_vals, n, p_especial)

# Gráfico
plt.figure(figsize=(10,6))
sns.barplot(x=k_vals, y=pmf_vals, color="steelblue", edgecolor="black")

plt.title(f"Distribución Binomial – Cartas 86+ en un sobre (n={n}, p={p_especial:.4f})")
plt.xlabel("Número de cartas 86+ en el sobre")
plt.ylabel("Probabilidad (PMF)")
plt.ylim(0, max(pmf_vals)*1.15)

for x, y in zip(k_vals, pmf_vals):
    plt.text(x, y + 0.001, f"{y:.3f}", ha='center')

plt.show()


# Probabilidad específica de 1 carta especial
p_1 = binom.pmf(1, n, p_especial)
print("Probabilidad de obtener EXACTAMENTE 1 carta especial:", p_1)


# 4.1.3 Distribución Poisson

print("\nDist. Poisson\n")

eventos_subida = (df["Variación (%)"] > 10).sum()
lambda_ = eventos_subida


# Visualización de la PMF
k_vals = range(0, lambda_ + 5)

plt.figure()
plt.bar(k_vals, poisson.pmf(k_vals, lambda_))
plt.title(f"PMF de Poisson (lambda={lambda_})")
plt.xlabel("Número de eventos fuertes")
plt.ylabel("Probabilidad")
plt.show()

print("Lambda (eventos fuertes):", lambda_)
print("P(k=2 eventos fuertes):", poisson.pmf(2, lambda_))


# Distribución Uniforme

print("\nDist. Uniforme\n")

plt.figure()
sns.histplot(df["Media"], bins=10)
plt.title("Distribución Uniforme Aproximada – Media de Cartas")
plt.xlabel("Media")
plt.ylabel("Frecuencia")
plt.show()


# 4.2 Cálculo de Probabilidades


