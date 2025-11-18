# Importación de librerías
import os
import pandas as pd
import matplotlib.pyplot as plt


# Enrutado del archivo .csv
file_location = os.path.dirname(os.path.abspath(__file__))

# Creación del dataframe
df = pd.read_csv(file_location + "/fc26.csv")

# Descripción Estadística Básica
print(df.head())
print(df.info())
print(df.describe())


# ==== 4. Análisis Probabilístico ==== #

# 4.1 Identificación de Distribuciones


# 4.2 Cálculo de Probabilidades



# Otros

p_subida = df["¿Subió de precio?"].mean()
print("P(Subió el precio) =", p_subida)

plt.bar(df["Jugador"], df["Variación (%)"])
plt.ylabel("Variación (%)")
plt.show()

print(df[["Frecuencia en sobres", "Variación (%)"]].corr())


