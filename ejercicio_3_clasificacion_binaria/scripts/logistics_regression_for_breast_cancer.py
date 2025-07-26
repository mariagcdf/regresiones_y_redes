import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)

# Cargar el dataset
datos = load_breast_cancer()
X = datos.data
y = datos.target

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Crear y entrenar el modelo
modelo = LogisticRegression(max_iter=10000)
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas
print("Resultados del modelo Regresión Logística:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Regresión Logística (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC - Regresión Logística")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Matriz de confusión con etiquetas
cm = confusion_matrix(y_test, y_pred)
nombres_clases = ["Benigno", "Maligno"]

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=nombres_clases)

fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title("Matriz de Confusión - Regresión Logística", fontsize=14)
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
