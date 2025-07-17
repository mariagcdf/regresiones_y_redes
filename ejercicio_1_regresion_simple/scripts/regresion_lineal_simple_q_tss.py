import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Ruta base
ruta_data = r'C:\Users\Mario\Desktop\Data Science\regresiones_y_redes\ejercicio_1_regresion_simple\data'

# Leer CSV
df_norm = pd.read_csv(f"{ruta_data}\\qe_tss_normal.csv")
df_tox = pd.read_csv(f"{ruta_data}\\qe_tss_toxico.csv")

# Filtrar Qe <= 60000
df_norm = df_norm[df_norm["Qe"] <= 60000]
df_tox = df_tox[df_tox["Qe"] <= 60000]

# Variables
Xn = df_norm["Qe"].values.reshape(-1, 1)
yn = df_norm["TSS"].values
Xt = df_tox["Qe"].values.reshape(-1, 1)
yt = df_tox["TSS"].values

# Modelos
model_n = LinearRegression().fit(Xn, yn)
model_t = LinearRegression().fit(Xt, yt)

# Predicciones
ypn = model_n.predict(Xn)
ypt = model_t.predict(Xt)

# R²
r2n = r2_score(yn, ypn)
r2t = r2_score(yt, ypt)

# Mostrar resultados
print("🔵 REGRESIÓN TSS vs Q (NORMAL):")
print(f"Pendiente  : {model_n.coef_[0]:.4f}")
print(f"Intercepto : {model_n.intercept_:.4f}")
print(f"R²         : {r2n:.4f}")

print("\n🔴 REGRESIÓN TSS vs Q (TOXICIDAD):")
print(f"Pendiente  : {model_t.coef_[0]:.4f}")
print(f"Intercepto : {model_t.intercept_:.4f}")
print(f"R²         : {r2t:.4f}")

# Muestreo aleatorio del 1% para visualización
frac = 0.01
df_norm_sample = df_norm.sample(frac=frac, random_state=42)
df_tox_sample = df_tox.sample(frac=frac, random_state=42)

# Gráfica
plt.figure(figsize=(8, 5))
plt.scatter(df_norm_sample["Qe"], df_norm_sample["TSS"], color='blue', alpha=0.4, label='Normal')
plt.plot(Xn, ypn, color='blue', linewidth=2)

plt.scatter(df_tox_sample["Qe"], df_tox_sample["TSS"], color='red', alpha=0.4, label='Toxicidad')
plt.plot(Xt, ypt, color='red', linewidth=2)

plt.text(0.95, 0.02,
         f'Normal: R²={r2n:.2f}, pendiente={model_n.coef_[0]:.5f}\n'
         f'Tóxico: R²={r2t:.2f}, pendiente={model_t.coef_[0]:.5f}',
         transform=plt.gca().transAxes,
         fontsize=9, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

plt.title("Comparación de regresión TSS vs Qe")
plt.xlabel("Qe (mg/L)")
plt.ylabel("TSS (mg/L)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
