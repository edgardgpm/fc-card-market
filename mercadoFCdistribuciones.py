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


# ==== 4. Análisis Probabilístico ==== #

# 4.1 Identificación de Distribuciones


# 4.1.1 Distribución Normal
# Histograma

print("\nDist. Normal\n")

plt.figure()
sns.histplot(df["Variación (%)"], bins=8, kde=True) # Uso de Kernel Density Estimation (KDE)
plt.title("Distribución de Variación (%) con KDE")
plt.xlabel("Variación (%)")
plt.ylabel("Frecuencia")
plt.show()

stat, p = shapiro(df["Variación (%)"])
print("p-value Shapiro-Wilk:", p)       # Comparación del valor-p de una distribución normal

alpha = 0.05                            # Nivel de significancia
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


# Probabilidad de que una carta sea común (Media < 86)
freq_total = df["Frecuencia en sobres"].sum()
p_menor_86 = df[df["Media"] < 86]["Frecuencia en sobres"].sum() / freq_total

# Tamaño del sobre
n = 7

# Valores posibles: 0 a 7 cartas especiales
k_vals = np.arange(0, n+1)

# PMF de la binomial
pmf_vals = binom.pmf(k_vals, n, p_menor_86)

# Gráfico
plt.figure(figsize=(10,6))
sns.barplot(x=k_vals, y=pmf_vals, color="steelblue", edgecolor="black")

plt.title(f"Distribución Binomial – Cartas con Media < 86 en un sobre (n={n}, p={p_menor_86:.4f})")
plt.xlabel("Número de cartas con Media < 86 en el sobre")
plt.ylabel("Probabilidad (PMF)")
plt.ylim(0, max(pmf_vals)*1.15)

for x, y in zip(k_vals, pmf_vals):
    plt.text(x, y + 0.001, f"{y:.3f}", ha='center')

plt.show()


# Probabilidad de que una carta sea >= 86
freq_total = df["Frecuencia en sobres"].sum()
p_mayor_86 = df[df["Media"] >= 86]["Frecuencia en sobres"].sum() / freq_total

# Tamaño del sobre (7 cartas)
n = 7

# Valores posibles de 0 a 7 cartas 86+
k_vals = np.arange(0, n + 1)

# Distribución Binomial (PMF)
pmf_vals = binom.pmf(k_vals, n, p_mayor_86)

# Gráfico
plt.figure(figsize=(10,6))
sns.barplot(x=k_vals, y=pmf_vals, color="steelblue", edgecolor="black")
plt.title(f"Distribución Binomial – Cartas ≥ 86 en un sobre (n={n}, p={p_mayor_86:.4f})")
plt.xlabel("Número de cartas ≥ 86 en el sobre")
plt.ylabel("Probabilidad (PMF)")
plt.ylim(0, max(pmf_vals)*1.15)

# Etiquetas arriba de cada barra
for x, y in zip(k_vals, pmf_vals):
    plt.text(x, y + 0.001, f"{y:.3f}", ha='center')

plt.show()


# Distribución Poisson

print("\nDist. Poisson\n")


# --------- 1. Calcular subidas y bajadas fuertes por semana ---------
subidas_fuertes = []
bajadas_fuertes = []
labels = []

for i in range(1, 8):  # S1→S2 ... S7→S8
    col_a = f"Precio S{i}"
    col_b = f"Precio S{i+1}"

    variacion = (df[col_b] - df[col_a]) / df[col_a] * 100

    subidas = (variacion > 5).sum()
    bajadas = (variacion < -5).sum()

    subidas_fuertes.append(subidas)
    bajadas_fuertes.append(bajadas)

    labels.append(f"S{i}→S{i+1}")

# --------- 2. Gráfico de subidas y bajadas fuertes ---------
plt.figure(figsize=(10,5))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, subidas_fuertes, width, label="Subidas fuertes (>5%)")
plt.bar(x + width/2, bajadas_fuertes, width, label="Bajadas fuertes (<-5%)")

plt.xticks(x, labels)
plt.ylabel("Número de eventos")
plt.title("Eventos Fuertes por Semana")
plt.legend()
plt.show()

# --------- 3. Calcular lambda para Poisson ---------

lambda_subidas = np.mean(subidas_fuertes)
lambda_bajadas = np.mean(bajadas_fuertes)

print("Lambda (subidas fuertes):", lambda_subidas)
print("Lambda (bajadas fuertes):", lambda_bajadas)

# --------- 4. Gráfica Poisson para subidas y bajadas ---------
plt.figure(figsize=(10,5))

k_vals = np.arange(0, max(max(subidas_fuertes), max(bajadas_fuertes)) + 4)

plt.bar(k_vals - 0.15, poisson.pmf(k_vals, lambda_subidas), width=0.3, label="Poisson - Subidas fuertes")
plt.bar(k_vals + 0.15, poisson.pmf(k_vals, lambda_bajadas), width=0.3, label="Poisson - Bajadas fuertes")

plt.xticks(k_vals)
plt.xlabel("Número de eventos fuertes en una semana")
plt.ylabel("Probabilidad")
plt.title("Distribución de Poisson - Eventos Fuertes")
plt.legend()
plt.show()


# Distribución Uniforme

print("\nDist. Uniforme\n")

plt.figure()
sns.histplot(df["Media"], bins=10)
plt.title("Media de Cartas en base a la Frecuencia")
plt.xlabel("Media")
plt.ylabel("Frecuencia")
plt.show()


# Agregar nuevas columnas de subida y bajada de precios (S1 v S2, S2 v S3, etc.)

for i in range(1, 8):
    col_actual = f"Precio S{i}"
    col_siguiente = f"Precio S{i+1}"
    nueva_col = f"Subida_{i}a{i+1}"

    df[nueva_col] = (df[col_siguiente] > df[col_actual]).astype(int)

# Mostrar las nuevas columnas creadas
df[[col for col in df.columns if "Subida_" in col]].head()


# Tendencia semanal por carta
semanas = ["Precio S1","Precio S2","Precio S3","Precio S4",
           "Precio S5","Precio S6","Precio S7","Precio S8"]

# Calcular tendencia semanal (+1 sube, -1 baja, 0 igual)
for i in range(len(semanas)-1):
    df[f"Trend_{i+1}"] = np.sign(df[semanas[i+1]] - df[semanas[i]])



print("\n--- Distribución Binomial por Semana ---\n")

resultados = {}

for i in range(1, 8):
    col = f"Subida_{i}a{i+1}"

    p_subida = df[col].mean()   # prob de éxito
    prob_3_en_5 = binom.pmf(3, 5, p_subida)

    resultados[col] = {
        "P(subida)": p_subida,
        "P(3 subidas en 5 cartas)": prob_3_en_5
    }

    print(f"Semana {i} → Semana {i+1}")
    print("   P(subida):", round(p_subida, 4))
    print("   P(3 subidas en 5 cartas):", round(prob_3_en_5, 6))
    print()


# Crear lista de probabilidades
p_list = [df[f"Subida_{i}a{i+1}"].mean() for i in range(1, 8)]
labels = [f"S{i}→S{i+1}" for i in range(1, 8)]

plt.figure(figsize=(8,5))
ax = sns.barplot(x=labels, y=p_list, palette="pastel")

plt.title("Probabilidad de Subida de Precio por Semana")
plt.xlabel("Intervalo Semanal")
plt.ylabel("P(Subida)")
plt.ylim(0, 1)

# Agregar labels
for i, v in enumerate(p_list):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()



plt.figure(figsize=(8,5))
sns.lineplot(x=labels, y=p_list, marker="o")

plt.title("Tendencia de P(Subida) por Semana")
plt.xlabel("Intervalo Semanal")
plt.ylabel("P(Subida)")
plt.ylim(0, 1)

for i, v in enumerate(p_list):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()


intervalos = [f"Subida_{i}a{i+1}" for i in range(1, 8)]
rareza_unicas = df["Rareza"].unique()

# Diccionario para guardar probabilidades por rareza
prob_rareza = {r: [] for r in rareza_unicas}

for r in rareza_unicas:
    df_r = df[df["Rareza"] == r]

    for col in intervalos:
        p = df_r[col].mean()
        prob_rareza[r].append(p)



plt.figure(figsize=(10,6))

for r in rareza_unicas:
    plt.plot(intervalos, prob_rareza[r], marker="o", label=r)

plt.title("Tendencia de P(Subida) por Semana según Rareza")
plt.xlabel("Intervalo Semanal")
plt.ylabel("P(Subida)")
plt.ylim(0, 1)
plt.legend(title="Rareza")
plt.grid(alpha=0.3)

plt.show()