import os
import pandas as pd

def process_comex_inventories(file_path, output_path):
    df = pd.read_csv(file_path)
    df.columns = ["date", "value"]

    # Convertir fecha con zona horaria a YYYY-mm-dd
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values(by="date", ascending=False)
    df.rename(columns={"value": "comex_inventories"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_consumer_confidence(file_path, output_path):
    df = pd.read_csv(file_path, sep=";", skiprows=3, header=None)
    df.columns = ["date", "value"]

    # Fechas en formato con hora → YYYY-mm-dd
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = df["value"].astype(str).str.replace(",", ".", regex=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values(by="date", ascending=False)
    df.rename(columns={"value": "consumer_confidence"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_etf_gold_flows(file_path, output_path):
    df = pd.read_csv(file_path)

    # Nos quedamos con Date y GLD Close
    df = df[["Date", " GLD Close"]]
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values(by="date", ascending=False)
    df.rename(columns={"value": "etf_gold_flows"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_geopolitical_risk(file_path, output_path):
    df = pd.read_csv(file_path, sep=";")

    # La fecha está en columna "date" (dd/mm/YYYY)
    df = df[["date", "GPRD"]]
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = df["value"].astype(str).str.replace(",", ".", regex=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values(by="date", ascending=False)
    df.rename(columns={"value": "geopolitical_risk"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_google_trends_gold_element(file_path, output_path):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ["date", "value"]

    # Fechas mensuales → YYYY-mm
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values(by="date", ascending=False)
    df.rename(columns={"value": "google_trends_gold_element"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_google_trends_gold_word(file_path, output_path):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values(by="date", ascending=False)
    df.rename(columns={"value": "google_trends_gold_word"}, inplace=True)
    df.to_csv(output_path, index=False)


def process_policy_uncertainty(file_path, output_path):
    df = pd.read_csv(file_path)

    # Crear columna date a partir de year, month, day
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[["date", "daily_policy_index"]]

    df["daily_policy_index"] = pd.to_numeric(df["daily_policy_index"], errors="coerce")
    df = df.dropna().sort_values(by="date", ascending=False)

    df.rename(columns={"daily_policy_index": "policy_uncertainty"}, inplace=True)
    df.to_csv(output_path, index=False)

def process_cftc_gold_positions(file_path, output_path):
    df = pd.read_csv(file_path)

    # Normalizamos columnas
    df = df[["report_date", "ManagedMoneyNet"]]
    df.columns = ["date", "value"]

    # Convertir fechas al formato YYYY-mm-dd
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Limpiar y ordenar
    df = df.dropna().sort_values(by="date", ascending=False)

    # Renombramos la columna final
    df.rename(columns={"value": "cftc_gold_positions"}, inplace=True)

    # Guardar
    df.to_csv(output_path, index=False)

def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    processors = {
        "comex_inventories.csv": process_comex_inventories,
        "consumer_confidence.csv": process_consumer_confidence,
        "etf_gold_flows.csv": process_etf_gold_flows,
        "geopolitical_risk.csv": process_geopolitical_risk,
        "google_trends_gold_element.csv": process_google_trends_gold_element,
        "google_trends_gold_word.csv": process_google_trends_gold_word,
        "policy_uncertainty.csv": process_policy_uncertainty,
        "cftc_gold_positions.csv": process_cftc_gold_positions,
    }

    for file, func in processors.items():
        file_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        if os.path.exists(file_path):
            func(file_path, output_path)
            print(f"✔ Procesado: {file}")
        else:
            print(f"⚠ No encontrado: {file}")


if __name__ == "__main__":
    input_folder = "../data/raw/Others"          # CSV originales Others
    output_folder = "../data/processed/Others_processed"  # CSV limpios
    main(input_folder, output_folder)
