import os
import pandas as pd

def consolidate_csv_folders(folders, output_file):
    dfs = []

    # Leer todos los CSV de cada carpeta
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)

                # Nos aseguramos de que tenga columnas: date, value
                if "date" in df.columns:
                    dfs.append(df)

    # Merge progresivo por columna "date"
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    # Convertir a datetime y asegurar rango completo de fechas
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])

    start_date = merged["date"].min()
    end_date = merged["date"].max()
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Reindexar para no dejar días vacíos
    merged = merged.set_index("date").reindex(full_range).reset_index()
    merged.rename(columns={"index": "date"}, inplace=True)

    # Ordenar ascendente (más antigua arriba)
    merged = merged.sort_values(by="date", ascending=True).reset_index(drop=True)

    # --- 🔥 Ordenar columnas por densidad (no nulos) ---
    cols = merged.columns.tolist()
    cols.remove("date")

    if "gold_spot" in cols:
        cols.remove("gold_spot")

        # Ordenar las demás por densidad (más datos primero)
        density_order = merged[cols].count().sort_values(ascending=False).index.tolist()

        # Construir nuevo orden final
        merged = merged[["date", "gold_spot"] + density_order]
    else:
        density_order = merged[cols].count().sort_values(ascending=False).index.tolist()
        merged = merged[["date"] + density_order]

    # Guardar CSV consolidado
    merged.to_csv(output_file, index=False)
    print(f"✔ Súper dataset creado: {output_file}")


if __name__ == "__main__":
    input_folders = [
        "../../data/processed/FRED_processed",
        "../../data/processed/Investing_processed",
        "../../data/processed/Others_processed"
    ]
    output_file = "../../data/dataset.csv"          # CSV consolidado final
    consolidate_csv_folders(input_folders, output_file)

