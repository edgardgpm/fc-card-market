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


# Descripción Estadística Básica
print(df.head())
print(df.info())
print(df.describe())


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


# Tendencia inconsistente
df["Tendencia_inconsistente"] = df[[f"Trend_{i}" for i in range(1,8)]].apply(
    lambda row: (1 in row.values and -1 in row.values), axis=1
)


# Tendencia consistente
def cuatro_seguidas(trends, objetivo):
    return any(all(x == objetivo for x in trends[i:i+4]) for i in range(len(trends)-3))

df["Consistente_subida"] = df[[f"Trend_{i}" for i in range(1,8)]].apply(
    lambda row: cuatro_seguidas(list(row.values), 1), axis=1
)

df["Consistente_bajada"] = df[[f"Trend_{i}" for i in range(1,8)]].apply(
    lambda row: cuatro_seguidas(list(row.values), -1), axis=1
)


# Ranking por volatilidad y tendencia por tipo de carta
ranking_volatilidad = df.groupby("Rareza")["Volatilidad"].mean().sort_values(ascending=False)
print(ranking_volatilidad)

ranking_inconsistencia = df.groupby("Rareza")["Tendencia_inconsistente"].mean().sort_values(ascending=False)
print(ranking_inconsistencia)


# Cantidad de cartas por tipo de tendencia (consistent vs inconsistent)
plt.figure(figsize=(7,4))
ax = sns.barplot(
    x=["Consistentes (↑)", "Consistentes (↓)", "Inconsistentes"],
    y=[
        df["Consistente_subida"].sum(),
        df["Consistente_bajada"].sum(),
        df["Tendencia_inconsistente"].sum()
    ],
    palette="muted"
)
plt.title("Distribución de Tendencias de Precio")
plt.ylabel("Número de Cartas")

add_bar_labels(ax)

plt.show()


# Ranking de volatilidad por rareza
plt.figure(figsize=(8,4))
ax = sns.barplot(
    x=ranking_volatilidad.index,
    y=ranking_volatilidad.values,
    palette="viridis"
)
plt.title("Volatilidad Promedio por Rareza")
plt.ylabel("Volatilidad (STD)")
plt.xlabel("Rareza")

add_bar_labels(ax)

plt.show()


# Evolución promedio de precios por semana
precios_prom = df[semanas].mean()

plt.figure(figsize=(8,4))
ac = sns.lineplot(x=range(1,9), y=precios_prom.values, marker="o")
plt.title("Precio Promedio por Semana (Todas las Cartas)")
plt.xlabel("Semana")
plt.ylabel("Precio promedio")

for x, y in zip(range(1,9), precios_prom.values):
    ax.text(x, y, f"{round(y,2)}", ha='center', va='bottom')

plt.show()



prom_rareza = df.groupby("Rareza")[semanas].mean()

plt.figure(figsize=(10,6))
for rareza in prom_rareza.index:
    y_vals = prom_rareza.loc[rareza].values
    ax = sns.lineplot(x=range(1,9), y=y_vals, marker="o", label=rareza)
    for x, y in zip(range(1,9), y_vals):
        plt.text(x, y, f"{round(y,1)}", ha="center", va="bottom", fontsize=8)

plt.title("Evolución Promedio del Precio por Semana – por Rareza")
plt.xlabel("Semana")
plt.ylabel("Precio promedio")
plt.legend(title="Rareza")
plt.show()


prom_liga = df.groupby("Liga")[semanas].mean()

plt.figure(figsize=(12,6))
for liga in prom_liga.index:
    y_vals = prom_liga.loc[liga].values
    ax = sns.lineplot(x=range(1,9), y=y_vals, marker="o", label=liga)
    for x, y in zip(range(1,9), y_vals):
        plt.text(x, y, f"{round(y,1)}", ha="center", va="bottom", fontsize=8)

plt.title("Evolución Promedio de Precio por Semana – por Liga")
plt.xlabel("Semana")
plt.ylabel("Precio promedio")
plt.legend(title="Liga")
plt.show()


clubs_validos = df["Club"].value_counts()
clubs_validos = clubs_validos[clubs_validos >= 3].index

prom_club = df[df["Club"].isin(clubs_validos)].groupby("Club")[semanas].mean()

plt.figure(figsize=(14,7))
for club in prom_club.index:
    y_vals = prom_club.loc[club].values
    ax = sns.lineplot(x=range(1,9), y=y_vals, marker="o", label=club)
    for x, y in zip(range(1,9), y_vals):
        plt.text(x, y, f"{round(y,1)}", ha="center", va="bottom", fontsize=7)

plt.title("Evolución Promedio de Precio por Semana – por Club")
plt.xlabel("Semana")
plt.ylabel("Precio promedio")
plt.legend(title="Club", bbox_to_anchor=(1.05,1))
plt.show()



vol_liga = df.groupby("Liga")["Volatilidad"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=vol_liga.index, y=vol_liga.values, palette="viridis")
plt.title("Volatilidad Promedio por Liga")
plt.ylabel("Volatilidad (STD)")

for i, v in enumerate(vol_liga.values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.xticks(rotation=45)
plt.show()


vol_club = df.groupby("Club")["Volatilidad"].mean().sort_values(ascending=False)

plt.figure(figsize=(12,5))
sns.barplot(x=vol_club.index, y=vol_club.values, palette="magma")
plt.title("Volatilidad Promedio por Club")
plt.ylabel("Volatilidad (STD)")

for i, v in enumerate(vol_club.values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.xticks(rotation=90)
plt.show()

