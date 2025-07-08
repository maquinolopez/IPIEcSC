# -*- coding: utf-8 -*-
# Archivo: simulaciones_intro_estadistica.py
# Descripción: Código guía para la primera lección del curso de estadística
# Herramienta: Spyder (o cualquier entorno compatible con Python)

# ===============================================
# IMPORTAR PAQUETES NECESARIOS
# ===============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===============================================
# 1. COMPARACIÓN DE MODELOS
# ===============================================


# Simulación de datos
np.random.seed(123)
datos = np.random.poisson(lam=1/2, size=10)

# Parámetros estimados
media = np.mean(datos)
se = np.std(datos, ddof=1) / (np.sqrt(len(datos))-1)
intervalo_normal = (media - 2 * se, media + 2 * se)

# Parámetros bayesianos: prior Gamma(1, 1)
alpha0, beta0 = 1, 1
alpha_post = alpha0 + np.sum(datos)
beta_post = beta0 + len(datos)

# Valores de lambda para graficar
x_vals = np.linspace(0.001, 3, 500)

# Verosimilitud (producto de Poisson, escalada)
logpmf_matrix = poisson.logpmf(datos[:, np.newaxis], mu=x_vals)
likelihood = np.exp(np.sum(logpmf_matrix, axis=0))
likelihood_scaled = likelihood / np.max(likelihood)

# Prior Gamma
prior = gamma.pdf(x_vals, a=alpha0, scale=1 / beta0)

# Posterior Gamma
posterior = gamma.pdf(x_vals, a=alpha_post, scale=1 / beta_post)

# Intervalos de credibilidad
ci_lower = gamma.ppf(0.025, a=alpha_post, scale=1 / beta_post)
ci_upper = gamma.ppf(0.975, a=alpha_post, scale=1 / beta_post)

# Figura combinada
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma normalizado
# ax.hist(datos, bins=range(0, max(datos) + 2), density=True, align='left',
#         color='lightgray', edgecolor='black', label='Histograma de datos')
# ax.plot(datos, np.zeros_like(datos), 'k|', marksize=10, label='Datos observados',color='red',alpha=.4)
ax.plot(datos, np.zeros_like(datos), 'kx', markersize=10, label='Datos observados',alpha=0.4,color='r')


# Curvas
ax.plot(x_vals, likelihood_scaled, color='orange', linestyle=':', lw=2, label='Verosimilitud (escalada)')
ax.plot(x_vals, prior, color='gray', linestyle='--', lw=2, label='Distribución previa Gamma(1,1)')
ax.plot(x_vals, posterior, color='darkgreen', lw=2, label=f'Posterior Gamma({alpha_post}, {beta_post})')

# Líneas verticales
ax.axvline(intervalo_normal[0], color='red', linestyle='--', label='IC 95% (Normal)')
ax.axvline(intervalo_normal[1], color='red', linestyle='--')
ax.axvline(ci_lower, color='darkred', linestyle=':', label='IC 95% (Bayesiano)')
ax.axvline(ci_upper, color='darkred', linestyle=':')
ax.axvline(media, color='blue', linestyle='-', label='Media muestral')

# Títulos y leyenda
ax.set_title("")
ax.set_xlabel("Valor de $\lambda$ o conteo observado")
ax.set_ylabel("Densidad relativa")
ax.legend()
plt.tight_layout()

# Guardar figura
plt.savefig("./Compasion_de_modelos.pdf", dpi=300)
plt.show()


# ===============================================
# 2. Modelos lineales
# ===============================================


# Datos simulados
np.random.seed(10)
x = np.linspace(0, 10, 20)
y = 2 + 1.5 * x + .6 * (x**2) + np.random.normal(0, 2, size=20)
media_y = np.mean(y)

# Figura
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Datos observados')
plt.axhline(media_y, color='red', linestyle='--', label='Media constante')
plt.xlabel("Horas estudiadas (x)")
plt.ylabel("Calificación (y)")
plt.title("Modelo constante: sin considerar x")
plt.legend()
plt.tight_layout()

# Guardar figura
plt.savefig("./modelo_constante_vs_datos.png", dpi=300)
plt.show()



# Ajustar modelo lineal y generar figura

# Ajuste de regresión lineal
modelo = LinearRegression()
modelo.fit(x.reshape(-1, 1), y)
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_
y_pred = modelo.predict(x.reshape(-1, 1))

# Visualización
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Datos observados')
plt.plot(x, y_pred, color='darkgreen', label='Modelo lineal ajustado')
plt.xlabel("Horas estudiadas (x)")
plt.ylabel("Calificación (y)")
plt.title("Modelo lineal ajustado")
plt.legend()
plt.tight_layout()

# Guardar figura
plt.savefig("./modelo_lineal_vs_datos.png", dpi=300)
plt.show()


# Modelo cuadrático
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x.reshape(-1, 1))
modelo_cuad = LinearRegression()
modelo_cuad.fit(X_poly, y)
y_pred_cuad = modelo_cuad.predict(X_poly)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Datos observados')
plt.plot(x, y_pred_cuad, color='purple', label='Modelo cuadrático')
plt.xlabel("Horas estudiadas (x)")
plt.ylabel("Calificación (y)")
plt.title("Comparación de modelos sobre los mismos datos")
plt.legend()
plt.tight_layout()

# Guardar figura
plt.savefig("./modelo_cuadri_vs_datos.png", dpi=300)
plt.show()


# ===============================================
# 3. Modelos lineales
# ===============================================



# Predicciones para cada modelo
y_pred_const = np.full_like(y, fill_value=media_y)  # Modelo constante
y_pred_lin = modelo.predict(x.reshape(-1, 1))       # Ya calculado
y_pred_cuad = modelo_cuad.predict(X_poly)           # Ya calculado

# Calcular R²
r2_const = r2_score(y, y_pred_const)
r2_lin = r2_score(y, y_pred_lin)
r2_cuad = r2_score(y, y_pred_cuad)

# Imprimir resultados
print("\nComparación de modelos usando R²:")
print(f"Modelo constante   R² = {r2_const:.3f}")
print(f"Modelo lineal      R² = {r2_lin:.3f}")
print(f"Modelo cuadrático  R² = {r2_cuad:.3f}")



# Calcular AIC para cada modelo
def calcular_aic(y, y_pred, k):
    n = len(y)
    residuales = y - y_pred
    rss = np.sum(residuales ** 2)
    aic = n * np.log(rss / n) + 2 * k
    return aic

# AIC para cada modelo
aic_const = calcular_aic(y, y_pred_const, k=1)
aic_lin = calcular_aic(y, y_pred_lin, k=2)        # intercepto + pendiente
aic_cuad = calcular_aic(y, y_pred_cuad, k=3)      # intercepto + x + x^2

# Imprimir resultados
print("\nComparación de modelos usando AIC:")
print(f"Modelo constante   AIC = {aic_const:.2f}")
print(f"Modelo lineal      AIC = {aic_lin:.2f}")
print(f"Modelo cuadrático  AIC = {aic_cuad:.2f}")



