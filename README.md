# Ejercicio 1: Regresión Lineal Simple en el BSM2

Este ejercicio explora la linealidad entre variables clave del modelo BSM2 bajo dos condiciones:
- 🟦 Simulación sin fallo (condición normal)
- 🔴 Simulación con fallo por toxicidad

---

## 🔍 Variables estudiadas

Se analiza la posible relación lineal entre las siguientes parejas de variables:

| Variable X (predictora) | Variable Y (respuesta) | Fuente de datos |
|--------------------------|-------------------------|------------------|
| SNO (nitrato)            | SNH (amonio)            | efluente         |
| DQO (TCOD)               | DBO (BOD5)              | efluente         |
| Qe (caudal efluente)     | TSS (sólidos totales)   | efluente         |
| Qe (caudal efluente)     | COD (DQO)               | efluente         |

---

## ⚙️ Tecnología usada

- **MATLAB**: generación de CSV desde simulaciones del BSM2
- **Python**: regresiones con `scikit-learn`, `matplotlib`, `pandas`

---

## 📌 Observaciones

- Se filtran outliers en algunas gráficas (por ejemplo, Qe > 60.000 m³/d)
- Se controla el número de puntos graficados para mejorar la visualización, aplicando un muestreo aleatorio del 1% (`frac=0.01`) o del 0,1% (`frac=0.001`)

---

## 📊 Resultados y conclusiones detalladas

### 🔹 1. DQO → DBO (efluente)

![alt text](image.png)

- **R² sin fallo**: 0.6754  
- **R² con toxicidad**: 0.9835

**Conclusión técnica:**

- Cuando el sistema opera con normalidad, los microorganismos degradan parte de la DBO, lo que introduce variabilidad en su relación con la DQO. Esto provoca una **correlación moderada** y no completamente lineal.
- Bajo condiciones de toxicidad, esta actividad biológica se reduce drásticamente. Como consecuencia, la **DBO permanece prácticamente proporcional a la DQO**, manteniendo una relación más estable y lineal similar a la del influente.
- La pendiente de la regresión en este caso se aproxima a la fracción biodegradable teórica del modelo (≈ 0.65), lo que indica que el sistema deja de alterar esa proporción en el tratamiento.

**Aplicación práctica:**

- Este cambio hacia una relación más lineal puede ser un **indicador temprano de fallo por toxicidad o inhibición**.
- Un modelo predictivo entrenado bajo condiciones normales (como una red neuronal o un autoencoder) podría **detectar fácilmente estos cambios estructurales** en las correlaciones entre variables.
- Por tanto, **monitorizar la linealidad entre DQO y DBO podría ayudar a diagnosticar automáticamente fallos en la EDAR** antes de que se manifiesten en los valores absolutos.

### 🔹 2. SNH ↔ SNO (efluente)

![alt text](image-1.png)


- **R² sin fallo**: 0.0007  
- **R² con toxicidad**: 0.8174

**Conclusión técnica:**

- En condiciones normales, el sistema funciona correctamente y el proceso de nitrificación convierte de forma eficiente el amonio (SNH) en nitrato (SNO). Esto da lugar a **valores de SNH muy bajos y estables**, con ligeras fluctuaciones. Como SNO también se mantiene en un rango acotado, **no hay una relación lineal aparente entre ambas variables**.  
  → Resultado: el modelo de regresión no encuentra una pendiente significativa y **R² es prácticamente cero**.

- En cambio, cuando hay toxicidad, los microorganismos nitrificantes se ven afectados y **la conversión de SNH a SNO se interrumpe**. Como consecuencia, el amonio comienza a **acumularse**, mientras que el nitrato disminuye o se estabiliza. Esto genera una **clara relación inversa entre SNH y SNO**, con pendiente negativa pronunciada y un **R² muy alto (0.82)**.  
  → En otras palabras, cuanto menos nitrato, más amonio, lo cual es coherente con una parada de la nitrificación.


### 🔹 3. Qe ↔ TSS (efluente)

![alt text](image-3.png)

### 🔹 3. Qe ↔ TSS (efluente)

![alt text](image-4.png)

- **R² sin fallo**: 0.9762  
- **R² con toxicidad**: 0.9752  
- **Pendiente**: 0.0004 en ambos casos

**Conclusión técnica:**

- Se observa una **relación fuertemente lineal** entre el caudal de efluente (Qe) y la concentración de sólidos totales en suspensión (TSS) tanto en condiciones normales como en presencia de toxicidad.
- La pendiente es baja (≈ 0.0004) debido a que Qe está expresado en valores muy altos (hasta 60.000 m³/día), pero su efecto acumulado es relevante: pequeños aumentos en TSS se explican por grandes cambios en Qe.
- La correlación se mantiene prácticamente idéntica en ambas condiciones, lo cual sugiere que **esta relación no está afectada por el fallo biológico**, sino que refleja fenómenos hidráulicos (como la dilución, arrastre de sólidos o carga hidráulica en el clarificador).


### 🔹 4. Qe ↔ COD (efluente)

![alt text](image-2.png)

- **Tendencia**: leve crecimiento en ambos casos

**Conclusión**:
- Podría haber cierta proporcionalidad entre Qe y la carga orgánica saliente.
- Aun así, la dispersión es elevada y los modelos lineales simples no capturan toda la variabilidad.

---


# Ejercicio 2: Regresión Múltiple en el BSM2
