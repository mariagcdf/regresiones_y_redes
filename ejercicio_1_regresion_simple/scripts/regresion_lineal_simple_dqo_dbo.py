import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Ruta base
ruta_data = r'C:\Users\Mario\Desktop\Data Science\regresiones_y_redes\ejercicio_1_regresion_simple\data'

# Cargar CSVs
df_normal = pd.read_csv(f"{ruta_data}\\dqo_dbo_normal.csv")
df_toxico = pd.read_csv(f"{ruta_data}\\dqo_dbo_toxico.csv")

# Filtro opcional (por si quieres limitar valores extremos)
# df_normal = df_normal[df_normal["DQO"] <= 1000]
# df_toxico = df_toxico[df_toxico["DQO"] <= 1000]

# Variables completas para regresión
X_normal = df_normal["DQO"].values.reshape(-1, 1)
y_normal = df_normal["DBO"].values
X_toxico = df_toxico["DQO"].values.reshape(-1, 1)
y_toxico = df_toxico["DBO"].values

# Modelos
modelo_normal = LinearRegression().fit(X_normal, y_normal)
modelo_toxico = LinearRegression().fit(X_toxico, y_toxico)

# Predicciones
y_pred_normal = modelo_normal.predict(X_normal)
y_pred_toxico = modelo_toxico.predict(X_toxico)

# R²
r2_norm = r2_score(y_normal, y_pred_normal)
r2_tox = r2_score(y_toxico, y_pred_toxico)

# Muestreo aleatorio para visualización (5%)
frac = 0.001
df_normal_sample = df_normal.sample(frac=frac, random_state=42)
df_toxico_sample = df_toxico.sample(frac=frac, random_state=42)

# Orden para graficar regresión
idx_n = np.argsort(X_normal.flatten())
idx_t = np.argsort(X_toxico.flatten())

# Mostrar resultados
print("🔵 REGRESIÓN DQO → DBO SIN FALLO:")
print(f"  Pendiente    : {modelo_normal.coef_[0]:.4f}")
print(f"  Intercepto   : {modelo_normal.intercept_:.4f}")
print(f"  R²           : {r2_norm:.4f}")

print("\n🔴 REGRESIÓN DQO → DBO CON TOXICIDAD:")
print(f"  Pendiente    : {modelo_toxico.coef_[0]:.4f}")
print(f"  Intercepto   : {modelo_toxico.intercept_:.4f}")
print(f"  R²           : {r2_tox:.4f}")

# Gráfica
plt.figure(figsize=(8, 5))
plt.scatter(df_normal_sample["DQO"], df_normal_sample["DBO"], color='blue', alpha=0.3, label='Normal')
plt.plot(X_normal[idx_n], y_pred_normal[idx_n], color='blue', linewidth=2)

plt.scatter(df_toxico_sample["DQO"], df_toxico_sample["DBO"], color='red', alpha=0.3, label='Toxicidad')
plt.plot(X_toxico[idx_t], y_pred_toxico[idx_t], color='red', linewidth=2)

# Texto resumen
plt.text(0.95, 0.05,
         f'Normal: R²={r2_norm:.2f}, pendiente={modelo_normal.coef_[0]:.2f}\n'
         f'Tóxico: R²={r2_tox:.2f}, pendiente={modelo_toxico.coef_[0]:.2f}',
         transform=plt.gca().transAxes,
         fontsize=9, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

plt.title("Regresión DQO → DBO\nCondición normal vs Toxicidad")
plt.xlabel("DQO (mg/L)")
plt.ylabel("DBO (mg/L)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
