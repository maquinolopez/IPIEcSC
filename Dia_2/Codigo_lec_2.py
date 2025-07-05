# -*- coding: utf-8 -*-
# Archivo: simulaciones_intro_estadistica.py
# Descripción: Código guía para la primera lección del curso de estadística
# Herramienta: Spyder (o cualquier entorno compatible con Python)

# ===============================================
# IMPORTAR PAQUETES NECESARIOS
# ===============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

# Aseguramos estilo para gráficas
sns.set(style="whitegrid")

# ===============================================
# 1. MEDIA ACUMULADA - CONVERGENCIA
# ===============================================
n = 10000
moneda = np.random.binomial(n=1,p=0.3, size=n)
medias = np.cumsum(moneda) / np.arange(1, n+1)

plt.figure(figsize=(8, 4))
plt.plot(medias, label='Media acumulada')
plt.axhline(0.3, color='red', linestyle='--', label='Valor esperado (0.3)')
plt.xlabel("Número de lanzamientos")
plt.ylabel("Media acumulada")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("./media_acumulada_moneda.png")
plt.savefig("./media_acumulada_bernulli.pdf")
# plt.show()


# ===============================================
# 2. MEDIA ACUMULADA - DISTRIBUCIÓN
# ===============================================

# Parámetros
p = 0.3
reps = 100
n_total = 1000
checkpoints = [50, 200, 1000]
colors = ['tab:blue', 'tab:orange', 'tab:green']  # Colores para cada checkpoint
labels = [f'n = {n}' for n in checkpoints]

# Inicializar estructuras
all_means = np.zeros((reps, n_total))
snapshot_means = {n: [] for n in checkpoints}

# Simulaciones
for r in range(reps):
    data = np.random.binomial(1, p, n_total)
    cumulative_means = np.cumsum(data) / np.arange(1, n_total + 1)
    all_means[r] = cumulative_means
    for n in checkpoints:
        snapshot_means[n].append(cumulative_means[n - 1])

# Gráfico combinado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: trayectorias de medias acumuladas
for r in range(reps):
    ax1.plot(all_means[r], alpha=0.3, linewidth=0.8)

ax1.axhline(p, color='red', linestyle='--', label='Valor real $p$')

# Colorear líneas verticales en el mismo color que los histogramas
for n, color in zip(checkpoints, colors):
    ax1.axvline(n, color=color, linestyle=':', linewidth=2, label=f'n = {n}')

ax1.set_title('Convergencia de la media muestral (100 repeticiones)')
ax1.set_xlabel('Número de ensayos')
ax1.set_ylabel('Media acumulada')
ax1.legend()
ax1.grid(True)

# Panel derecho: histogramas superpuestos
for n, color, label in zip(checkpoints, colors, labels):
    ax2.hist(snapshot_means[n], bins=15, alpha=0.5, color=color, label=label, edgecolor='black')

ax2.axvline(p, color='red', linestyle='--', label='Valor real $p$')
ax2.set_title('Distribuciones de medias muestrales')
ax2.set_xlabel('Media muestral')
ax2.set_ylabel('Frecuencia')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("./medias_convergencia_y_distribuciones.pdf")
# plt.show()




# ===============================================
# 2. TLC
# ===============================================

from scipy.stats import norm

# Configuraciones
R = 10000  # Número de repeticiones
sample_sizes = [3, 5, 30, 100]
true_mean = 3.5  # Esperanza del dado
true_var = 35 / 12  # Varianza del dado justo

# Figura
fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4))

for i, n in enumerate(sample_sizes):
    means = []

    for _ in range(R):
        sample = np.random.randint(1, 7, size=n)
        means.append(np.mean(sample))

    # Histograma
    ax = axes[i]
    ax.hist(means, bins=30, density=True, alpha=0.6, edgecolor='black', label='Simulación')

    # Aproximación normal
    mu = true_mean
    sigma = np.sqrt(true_var / n)
    x = np.linspace(min(means), max(means), 500)
    ax.plot(x, norm.pdf(x, mu, sigma), color='red', lw=2, label='Normal')
    
    ax.set_title(f'Medias muestrales\nn = {n}')
    ax.set_xlabel('Media')
    ax.set_ylabel('Densidad')
    ax.legend()

plt.tight_layout()
plt.savefig("./tcl_dado.pdf")
# plt.show()


# ===============================================
# 3. Intervalos
# ===============================================

import pandas as pd

# Parámetros
p_true = 0.3       # valor verdadero del parámetro
n = 50             # tamaño de muestra
num_intervals = 100
alpha = 0.05
z = 1.96           # valor crítico para 95%

# Almacenar resultados
intervals = []
contains_p = []

for _ in range(num_intervals):
    sample = np.random.binomial(1, p_true, n)
    phat = np.mean(sample)
    se = np.sqrt(phat * (1 - phat) / n)
    lower = phat - z * se
    upper = phat + z * se
    intervals.append((lower, upper, phat))
    contains_p.append(lower <= p_true <= upper)

# Crear la visualización
fig, ax = plt.subplots(figsize=(10, 10))

# Dibujar cada intervalo horizontal
for i, ((low, up, phat), covered) in enumerate(zip(intervals, contains_p)):
    color = 'black' if covered else 'red'
    ax.hlines(y=i, xmin=low, xmax=up, color=color, linewidth=1)
    ax.plot(phat, i, 'o', color='blue', markersize=2)

# Línea vertical con el valor verdadero
ax.axvline(p_true, color='green', linestyle='--', label=f'Valor verdadero $p = {p_true}$')

# Etiquetas
ax.set_xlabel('Proporción estimada $\\hat{{p}}$')
ax.set_ylabel('Número de muestra')
ax.set_title(f'Intervalos de confianza del 95\\% para $p$ con $n = {n}$ y {num_intervals} repeticiones')
ax.invert_yaxis()

# Resumen en texto dentro del gráfico
textstr = '\n'.join((
    f'Total de intervalos: {num_intervals}',
    f'Contienen $p$: {contains_p.count(True)}',
    f'No contienen $p$: {contains_p.count(False)}',
    f'Porcentaje de cobertura: {contains_p.count(True) / num_intervals * 100:.1f}%'
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=props)

ax.legend()
plt.tight_layout()
plt.savefig("./intervalo.pdf")

# Crear y mostrar tabla resumen
df_result = pd.DataFrame({
    "Contiene p verdadero": contains_p
})
df_summary = df_result["Contiene p verdadero"].value_counts().rename(index={True: "Sí", False: "No"}).reset_index()
df_summary.columns = ["¿Contiene p verdadero?", "Frecuencia"]
print(df_summary)


# ===============================================
# 3. Intervalos
# ===============================================

from scipy.stats import beta
# Parámetros
n = 50                    # tamaño de muestra
p_true = 0.3              # valor verdadero de la proporción
alpha_prior = 1           # parámetros de la distribución Beta previa
beta_prior = 1

# Simulación de datos
sample = np.random.binomial(1, p_true, n)
s = np.sum(sample)        # número de éxitos

# Parámetros de la posterior
alpha_post = alpha_prior + s
beta_post = beta_prior + n - s

# Dominio para graficar
x = np.linspace(0, 1, 1000)

# Distribuciones
prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(x, alpha_post, beta_post)
likelihood = x**s * (1 - x)**(n - s)
likelihood /= np.max(likelihood)          # escalar para graficar comparativamente
prior_pdf /= np.max(prior_pdf)            # escalar también la prior (opcional)

# Intervalo de credibilidad del 95%
lower, upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, prior_pdf, label=f'Distribución previa Beta({alpha_prior}, {beta_prior})', color='gray')
plt.plot(x, likelihood, label='Verosimilitud (escalada)', color='orange', linestyle='--')
plt.plot(x, posterior_pdf, label=f'Posterior Beta({alpha_post}, {beta_post})', color='blue')
plt.axvline(lower, color='red', linestyle='--', label=f'2.5% = {lower:.3f}')
plt.axvline(upper, color='red', linestyle='--', label=f'97.5% = {upper:.3f}')
plt.axvline(p_true, color='green', linestyle=':', label=f'Valor verdadero $p = {p_true}$')

# Etiquetas
plt.title('Actualización Bayesiana: previa, verosimilitud y posterior')
plt.xlabel('$p$')
plt.ylabel('Densidad (escalada)')
plt.legend()
plt.tight_layout()
plt.savefig("./intervalo_bayesiano.pdf")





