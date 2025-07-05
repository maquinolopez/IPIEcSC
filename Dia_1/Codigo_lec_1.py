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
# 1. SIMULACIÓN DE UNA MONEDA JUSTA
# ===============================================
n = 10
moneda = np.random.choice([0, 1], size=n)

media = np.mean(moneda)
varianza = np.var(moneda)

print("=== Moneda justa ===")
print(f"Media: {media:.3f}")
print(f"Varianza: {varianza:.3f}")

plt.figure()
sns.histplot(moneda, bins=2, discrete=True)
plt.title("Distribución empírica: moneda justa")
plt.xlabel("Resultado (1 = Cara, 0 = Cruz)")
plt.ylabel("Frecuencia")
# plt.savefig("./histograma_moneda_justa.png")  # <- imagen para la presentación
plt.savefig("./histograma_moneda_justa.pdf")  # <- si prefieres formato vectorial
plt.show()


# ===============================================
# 2. SIMULACIÓN DE UN DADO JUSTO
# ===============================================
n = 50
dado = np.random.randint(1, 7, size=n)

media = np.mean(dado)
varianza = np.var(dado)

print("\n=== Dado justo ===")
print(f"Media: {media:.3f}")
print(f"Varianza: {varianza:.3f}")

plt.figure()
sns.histplot(dado, bins=np.arange(1, 8)-0.5, discrete=True)
plt.title("Distribución empírica: dado justo")
plt.xlabel("Cara del dado")
plt.ylabel("Frecuencia")
# plt.savefig("./histograma_dado_justo.png")
plt.savefig("./histograma_dado_justo.pdf")
plt.show()


# ===============================================
# 3. FUNCIÓN DE DISTRIBUCIÓN EMPÍRICA (FDE)
# ===============================================
ecdf = ECDF(moneda)

plt.figure()
plt.step(ecdf.x, ecdf.y, where="post")
plt.title("Función de distribución empírica: moneda justa")
plt.xlabel("Valor observado")
plt.ylabel("Probabilidad acumulada")
plt.grid(True)
# plt.savefig("./fde_moneda_justa.png")
plt.savefig("./fde_moneda_justa.pdf")
plt.show()


# ===============================================
# 4. FUNCIÓN DE DISTRIBUCIÓN EMPÍRICA DEL DADO
# ===============================================
# Simulamos una nueva muestra del dado justo

n = 1000
dado = np.random.randint(1, 7, size=n)

# Construimos la función de distribución empírica
ecdf_dado = ECDF(dado)

# Graficamos
plt.figure()
plt.step(ecdf_dado.x, ecdf_dado.y, where="post")
plt.title("Función de distribución empírica: dado justo")
plt.xlabel("Cara del dado")
plt.ylabel("Probabilidad acumulada")
plt.grid(True)
# plt.savefig("./fde_dado_justo.png")
plt.savefig("./fde_dado_justo.pdf")
plt.show()


# ===============================================
# 5. MEDIA ACUMULADA - CONVERGENCIA
# ===============================================
n = 10000
moneda = np.random.choice([0, 1], size=n)
medias = np.cumsum(moneda) / np.arange(1, n+1)

plt.figure(figsize=(8, 4))
plt.plot(medias, label='Media acumulada')
plt.axhline(0.5, color='red', linestyle='--', label='Valor esperado (0.5)')
plt.xlabel("Número de lanzamientos")
plt.ylabel("Media acumulada")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("./media_acumulada_moneda.png")
plt.savefig("./media_acumulada_moneda.pdf")
plt.show()

