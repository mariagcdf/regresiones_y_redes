import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Ruta base a la carpeta data
ruta_data = r'C:\Users\Mario\Desktop\Data Science\regresiones_y_redes\ejercicio_1_regresion_simple\data'

# Cargar CSVs
df_normal = pd.read_csv(f"{ruta_data}\\qe_cod_normal.csv")
df_toxico = pd.read_csv(f"{ruta_data}\\qe_cod_toxico.csv")

# Extraer variables
X_normal = df_normal["Qe"].values.reshape(-1, 1)
y_normal = df_normal["CODe"].values
X_toxico = df_toxico["Qe"].values.reshape(-1, 1)
y_toxico = df_toxico["CODe"].values

# Ajustar regresión
modelo_normal = LinearRegression().fit(X_normal, y_normal)
modelo_toxico = LinearRegression().fit(X_toxico, y_toxico)

# Predicciones
y_pred_normal = modelo_normal.predict(X_normal)
y_pred_toxico = modelo_toxico.predict(X_toxico)

# Métricas
r2_norm = r2_score(y_normal, y_pred_normal)
r2_tox = r2_score(y_toxico, y_pred_toxico)

# Muestreo para visualizar (5%)
df_normal_sample = df_normal.sample(frac=0.001, random_state=42)
df_toxico_sample = df_toxico.sample(frac=0.001, random_state=42)

# Mostrar en consola
print("🔵 REGRESIÓN SIN FALLO:")
print(f"  Pendiente    : {modelo_normal.coef_[0]:.4f}")
print(f"  Intercepto   : {modelo_normal.intercept_:.4f}")
print(f"  R²           : {r2_norm:.4f}")

print("\n🔴 REGRESIÓN CON TOXICIDAD:")
print(f"  Pendiente    : {modelo_toxico.coef_[0]:.4f}")
print(f"  Intercepto   : {modelo_toxico.intercept_:.4f}")
print(f"  R²           : {r2_tox:.4f}")

# Gráfica
plt.figure(figsize=(8, 5))
plt.scatter(df_normal_sample["Qe"], df_normal_sample["CODe"], color='blue', alpha=0.4, label='Normal')
plt.plot(X_normal, y_pred_normal, color='blue', linewidth=2)

plt.scatter(df_toxico_sample["Qe"], df_toxico_sample["CODe"], color='red', alpha=0.4, label='Toxicidad')
plt.plot(X_toxico, y_pred_toxico, color='red', linewidth=2)

plt.text(0.95, 0.02,
         f'Normal: R²={r2_norm:.2f}, pendiente={modelo_normal.coef_[0]:.2f}\n'
         f'Tóxico: R²={r2_tox:.2f}, pendiente={modelo_toxico.coef_[0]:.2f}',
         transform=plt.gca().transAxes,
         fontsize=9, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

plt.title("Comparación de regresión CODe vs Qe")
plt.xlabel("Qe (mg/L)")
plt.ylabel("CODe (mg/L)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
