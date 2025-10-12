import os
import pandas as pd

# Carpetas
input_folder = "../data/raw/Investing"          # CSV originales Investing
output_folder = "../data/processed/Investing_processed"  # CSV limpios

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)
        col_name = os.path.splitext(file)[0]  # nombre del archivo sin extensión

        # Leer CSV
        df = pd.read_csv(file_path)

        # Nos quedamos solo con Date y Price
        df = df[["Date", "Price"]]

        # Renombrar columnas
        df.columns = ["date", col_name]

        # Convertir fecha de mm/dd/YYYY a YYYY-mm-dd
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")

        # Limpiar el precio: eliminar comas y convertir a float
        df[col_name] = df[col_name].astype(str).str.replace(",", "", regex=False)
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # Eliminar filas inválidas
        df = df.dropna(subset=["date", col_name])

        # Ordenar por fecha descendente
        df = df.sort_values(by="date", ascending=False)

        # Guardar CSV limpio
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)

        print(f"✔ Procesado: {file} → {output_path}")
