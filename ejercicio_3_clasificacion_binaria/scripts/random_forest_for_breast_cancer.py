import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Cargar el dataset de cáncer de mama
datos = load_breast_cancer()
X = datos.data
y = datos.target

# Divido entre entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Crear el clasificador Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=1)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Predecir con los datos de test
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidades clase 1

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Imprimir métricas
print("Resultados del modelo Random Forest:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Graficar curva ROC
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Etiquetas de clase (0 = benigno, 1 = maligno en el dataset de cáncer de mama)
nombres_clases = ["Benigno", "Maligno"]

# Mostrar la matriz con etiquetas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres_clases)

fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues', values_format='d')

# Personalizar en castellano
ax.set_title("Matriz de Confusión - Random Forest", fontsize=14)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Valor Real", fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

# Evaluar en entrenamiento
y_train_pred = modelo.predict(X_train)
acc_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

print("Evaluación en entrenamiento:")
print(f"Accuracy (train): {acc_train:.4f}")
print(f"F1 Score (train): {f1_train:.4f}")