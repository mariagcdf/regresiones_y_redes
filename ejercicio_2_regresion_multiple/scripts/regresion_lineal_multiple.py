import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Cargar el archivo CSV
archivo = r'C:\Users\Mario\Desktop\Data Science\regresiones_y_redes\ejercicio_2_regresion_multiple\data\regresion_multiple_normal.csv'
df = pd.read_csv(archivo)

# Seleccionamos las columnas que vamos a usar
X = df[["SNO", "SO", "Qe", "TSS"]]  # Estas son las variables que usamos para predecir
y = df["SNH"]  # Esta es la variable que queremos predecir

# Función para hacer regresión lineal desde cero
def regresion_lineal_simple(X, y, lr=0.01, epochs=1000, verbose=False, normalizar=True):
    if normalizar:
        X = (X - X.mean()) / X.std()  # Si queremos, normalizamos los datos

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    n, m = X.shape

    w = np.zeros((m, 1))  # Pesos iniciales
    b = 0  # Intercepto
    losses = []

    for i in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        # Gradientes para actualizar
        dw = (2/n) * X.T @ error
        db = (2/n) * np.sum(error)

        # Actualizar pesos
        w -= lr * dw
        b -= lr * db

        loss = np.mean(error**2)
        losses.append(loss)

        if verbose and i % 100 == 0:
            print(f"Iteración {i}: Error medio = {loss:.4f}")

    return w.flatten(), b, losses

# Ver cómo cambia el error con distintos learning rates
tasas = [0.001, 0.01, 0.1]
plt.figure(figsize=(8, 5))

for tasa in tasas:
    _, _, perdidas = regresion_lineal_simple(X, y, lr=tasa, epochs=1000)
    plt.plot(perdidas, label=f"lr={tasa}")

plt.title("Cómo cambia el error con el learning rate")
plt.xlabel("Épocas")
plt.ylabel("Error medio cuadrado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Entrenar el modelo con tasa 0.01
w, b, _ = regresion_lineal_simple(X, y, lr=0.01, epochs=1000, verbose=True)

print("\n Coeficientes del modelo propio:")
for nombre, valor in zip(X.columns, w):
    print(f"{nombre}: {valor:.4f}")
print(f"Intercepto: {b:.4f}")

# Comparar con el modelo de sklearn
modelo = LinearRegression().fit(X, y)
print("\n📊 Coeficientes del modelo de sklearn:")
for nombre, valor in zip(X.columns, modelo.coef_):
    print(f"{nombre}: {valor:.4f}")
print(f"Intercepto: {modelo.intercept_:.4f}")

# Ahora hacemos una gráfica 3D para ver los resultados con 2 variables
var1 = "SNO"
var2 = "SO"
X2 = df[[var1, var2]]
y2 = df["SNH"]

w2, b2, _ = regresion_lineal_simple(X2, y2, lr=0.01, epochs=1000)

# Submuestreo del 1% de los puntos reales
muestra = df[[var1, var2, "SNH"]].sample(frac=0.01, random_state=42)

# Crear el plano
x1 = np.linspace(muestra[var1].min(), muestra[var1].max(), 30)
x2 = np.linspace(muestra[var2].min(), muestra[var2].max(), 30)
x1_grid, x2_grid = np.meshgrid(x1, x2)
y_grid = w2[0]*x1_grid + w2[1]*x2_grid + b2

# Plano de sklearn (solo con las dos variables seleccionadas)
modelo2 = LinearRegression().fit(X2, y2)
y_grid_sklearn = modelo2.coef_[0]*x1_grid + modelo2.coef_[1]*x2_grid + modelo2.intercept_

# Gráfico
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Usar solo el 1% de los puntos reales
ax.scatter(muestra[var1], muestra[var2], muestra["SNH"], color='blue', alpha=0.4, label='Datos')
ax.plot_surface(x1_grid, x2_grid, y_grid, color='orange', alpha=0.6)
ax.plot_surface(x1_grid, x2_grid, y_grid_sklearn, color='green', alpha=0.3)

ax.set_xlabel(var1)
ax.set_ylabel(var2)
ax.set_zlabel("SNH")
plt.title("Plano de regresión múltiple (modelo propio vs sklearn)")
plt.tight_layout()

from matplotlib.patches import Patch
custom_lines = [Patch(facecolor='orange', label='Modelo propio'),
                Patch(facecolor='green', label='Modelo sklearn')]
ax.legend(handles=custom_lines, loc='upper left')

plt.show()
