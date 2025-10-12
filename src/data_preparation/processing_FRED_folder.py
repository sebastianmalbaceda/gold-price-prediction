import os
import pandas as pd

# Carpetas
input_folder = "../../data/raw/FRED"          # CSV originales FRED
output_folder = "../../data/processed/FRED_processed"  # CSV limpios

os.makedirs(output_folder, exist_ok=True)

# Procesar cada archivo
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)

        # Nombre de la columna 'value' basado en el nombre del archivo (sin extensión)
        col_name = os.path.splitext(file)[0]

        # Leer CSV
        df = pd.read_csv(file_path)

        # Renombrar columnas
        df.columns = ["date", col_name]

        # Convertir columna de fecha
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Convertir valores a float (asegura punto decimal)
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # Eliminar filas con valores no válidos (NaN en fecha o valor)
        df = df.dropna(subset=["date", col_name])

        # Ordenar por fecha descendente
        df = df.sort_values(by="date", ascending=False)

        # Guardar CSV limpio
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)

        print(f"✔ Procesado: {file} → {output_path}")
