
# Escuela de Verano CIMAT 2025  
## Introducción práctica a la inferencia estadística con simulaciones computacionales

**Instructor**: Dr. Marco Antonio Aquino López  
**Duración**: 3 sesiones de 90 minutos  
**Dirigido a**: Estudiantes de los primeros semestres de licenciatura  
**Lenguaje de programación**: Python 3  

---

### 🧠 Objetivo general

Brindar una introducción intuitiva y práctica a los conceptos fundamentales de la inferencia estadística utilizando simulaciones computacionales en Python. Se enfatiza el desarrollo de intuición, visualización de resultados y comparación de métodos inferenciales.

---

### 📅 Estructura del curso

#### 📘 Día 1 — ¿Qué es la estadística y por qué simular?

- Motivación: la estadística como herramienta para modelar la incertidumbre.
- Simulación de experimentos aleatorios (monedas, dados).
- Media, varianza y distribución empírica.
- Primeros pasos en Python con `numpy`, `matplotlib`, y `seaborn`.

📁 Carpeta: `dia_1/`  
- `presentacion_dia1.pdf`: Diapositivas del día.  
- `codigo_simulacion.py`: Simulación de moneda, dado y dado cargado.  
- `ejercicios.py`: Actividades guiadas para reforzar conceptos.

---

#### 📙 Día 2 — Estimación e incertidumbre

- Introducción a la estimación puntual y por intervalo.
- Intervalos de confianza vs. intervalos de credibilidad bayesiana.
- Simulación de múltiples muestras y visualización de cobertura.
- Distribuciones Beta como priors y posteriors para proporciones.

📁 Carpeta: `dia_2/`  
- `presentacion_dia2.pdf`: Diapositivas con teoremas, ejemplos y visualizaciones.  
- `intervalos.py`: Comparación de intervalos de confianza y credibilidad.  
- `beta_posterior_visual.py`: Código para visualizar priors, likelihoods y posteriors.

---

#### 📕 Día 3 — Comparación de modelos

- ¿Qué modelo explica mejor los datos?
- Ajuste lineal vs. cuadrático en datos simulados.
- Métricas: error cuadrático medio, validación visual.
- Discusión final sobre toma de decisiones con estadística.

📁 Carpeta: `dia_3/`  
- `presentacion_dia3.pdf`: Diapositivas de comparación de modelos.  
- `modelo_lineal_vs_cuadratico.py`: Comparación mediante simulación.  
- `mini_proyecto_template.ipynb`: Guía para el proyecto final.

---

### 💻 Requisitos técnicos

- Python 3.8 o superior
- Instalar bibliotecas con:  
```bash
pip install numpy matplotlib seaborn scipy statsmodels
```

---

### 📚 Bibliografía recomendada

- Downey, A. (2015). *Think Stats*. [Sitio web](https://greenteapress.com/wp/think-stats/)
- Sivia & Skilling (2006). *Data Analysis: A Bayesian Tutorial*.
- McElreath, R. (2020). *Statistical Rethinking*.
- Casella & Berger (2001). *Statistical Inference*.

---

### 🧪 Proyecto final

Cada grupo elaborará una pequeña presentación donde simulará un experimento aleatorio, estimará parámetros relevantes y discutirá las implicaciones de sus hallazgos con visualizaciones. Los códigos y gráficos se compartirán al final del curso.

---

### 📂 Estructura del repositorio

```
.
├── README.md
├── dia_1/
│   ├── presentacion_dia1.pdf
│   ├── codigo_simulacion.py
│   └── ejercicios.py
├── dia_2/
│   ├── presentacion_dia2.pdf
│   ├── intervalos.py
│   └── beta_posterior_visual.py
├── dia_3/
│   ├── presentacion_dia3.pdf
│   ├── modelo_lineal_vs_cuadratico.py
│   └── mini_proyecto_template.ipynb
```

---

¡Esperamos que disfrutes del curso y te motives a seguir explorando el fascinante mundo de la estadística!

