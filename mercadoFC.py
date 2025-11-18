from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt

import os
import pandas as pd


file_location = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(file_location + "/fc26.csv")

print(df.head())
print(df.info())
print(df.describe())


plt.bar(df["Jugador"], df["Variación (%)"])
plt.ylabel("Variación (%)")
plt.show()

print(df[["Frecuencia en sobres", "Variación (%)"]].corr())



X = df[["Media", "Frecuencia en sobres", "Precio S1","Precio S8"]]
y = df["¿Subió de precio?"].map({"Sí":1,"No":0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))