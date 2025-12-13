# Gold Price Prediction

Proyecto de aprendizaje automático para la **predicción del precio del oro**, desarrollado como parte de mi **Trabajo de Fin de Grado (TFG)**.
Incluye análisis exploratorio, ingeniería de características, selección y entrenamiento de modelos, validación y análisis final.

---

## Objetivos

* Reunir y depurar una amplia colección de características (macroeconómicas, financieras, commodities, volatilidad, divisas, etc.) relacionadas con el precio del oro.
* Realizar un análisis exploratorio completo (EDA) y una selección de variables rigurosa.
* Evaluar diferentes modelos de predicción, desde regresores lineales hasta modelos basados en árboles y ensambles.
* Comparar su rendimiento mediante métricas de evaluación robustas.
* Analizar los resultados y discutir las conclusiones del comportamiento del oro.

---

## Dataset y características

El dataset se compone de **aproximadamente 60 variables** recopiladas de fuentes oficiales como **FRED**, **Investing.com**, **CBOE**, **OECD**, y otros organismos financieros.
Cada característica incluye su frecuencia, fuente y justificación económica.

Puedes consultar el listado completo en:
[`docs/features_reference.md`](docs/features_reference.md)

---

## Estructura del repositorio

```
gold-price-prediction/
│
├── notebooks/
│   └── gold-price-prediction.ipynb
│
├── src/
│   ├── data_preparation/
│   │   ├── processing_FRED_folder.py
│   │   ├── processing_Investing_folder.py
│   │   ├── processing_Others_folder.py
│   │   └── merge_features.py
│   │
│   ├── stage_1_exploration.py
│   ├── stage_2_feature_engineering.py
│   ├── stage_3_model_training.py
│   └── utils/
│       └── data_loader.py
│
├── data/
│   ├── raw/
│   │   ├── FRED/
│   │   ├── Investing/
│   │   └── Others/
│   │
│   ├── processed/
│   │   ├── FRED_processed/
│   │   ├── Investing_processed/
│   │   └── Others_processed/
│   │
│   └── dataset.csv
│
├── docs/
│   ├── features_reference.md
│   └── features_details.md
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

```

---

## Instalación y uso

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/<tu-usuario>/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. **Crear y activar un entorno virtual**

   ```bash
   python -m venv .venv        # O también py -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   ```

3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar los notebooks o scripts**

   ```bash
   jupyter notebook
   # o
   python src/stage_1_exploration.py
   ```

---

## Dataset

El dataset principal de este proyecto, ya procesado y consolidado a partir de múltiples fuentes (FRED, Investing, Others), puede descargarse desde la sección **Releases** de este repositorio. 

Una vez descargado, se recomienda colocar los archivos en la carpeta del proyecto de la siguiente manera:

```
gold-price-prediction/
├── data/
│ ├── raw/ # Carpeta opcional con CSV originales de cada fuente
│ ├── processed/ # Carpeta con CSV ya procesados individualmente
│ └── dataset.csv # Dataset final consolidado y listo para análisis
```


> **Nota de uso:**  
> Este dataset sirve como **punto de partida** para el análisis. Contiene todas las features recolectadas y unidas, incluyendo la columna `date` y el target `gold_spot`. No todas las features se usarán automáticamente; la selección final se realizará durante el EDA y modelado.

**Descarga del dataset:** 

[Dataset Consolidado](https://github.com/sebastianmalbaceda/gold-price-prediction/releases)  

---


## Licencia

Este proyecto se distribuye bajo la licencia **MIT**.
Puedes usarlo, modificarlo o citarlo siempre que mantengas la referencia al autor original.

---

## Autor

**Sebastián Malbaceda Leyva**

Trabajo de Fin de Grado — Universidad Autònoma de Barcelona

[sebastian.malbaceda.leyva@gmail.com](mailto:sebastian.malbaceda.leyva@gmail.com)

Año académico 2025
