
import argparse
import requests
import re
import json
import logging

def scrape_fuel_prices_ru():
    """
    Tente de récupérer les données officielles depuis fuelprices.ru (Highcharts).
    Cette fonction reproduit la logique ayant généré 'data/raw/donnees_fuel_russia.csv'.
    """
    url = "https://fuelprices.ru/"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    try:
        print(f"Scraping {url}...")
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.text
        
        # Recherche du bloc JS Highcharts
        match = re.search(r'series:\s*(\[.*?\]),\s*responsive:', content, re.DOTALL)
        if not match:
            print("WARN: Structure Highcharts non trouvée (site mis à jour ?).")
            return None
            
        # Parse JS pseudo-JSON
        js_data = match.group(1)
        # Nettoyage minimal pour JSON valide
        js_data = re.sub(r'(\s)([a-zA-Z0-9_]+):', r'\1"\2":', js_data) # Quote keys
        js_data = js_data.replace("'", '"')
        
        series = json.loads(js_data)
        
        # Conversion en DataFrame
        dfs = []
        for item in series:
            name = item.get("name", "ukn")
            data = item.get("data", [])
            df_s = pd.DataFrame(data, columns=["timestamp", name])
            df_s["date"] = pd.to_datetime(df_s["timestamp"], unit="ms").dt.date
            df_s = df_s.drop(columns=["timestamp"]).set_index("date")
            dfs.append(df_s)
            
        if dfs:
            df_final = pd.concat(dfs, axis=1).sort_index()
            return df_final
            
    except Exception as e:
        print(f"Erreur scraping fuelprices: {e}")
        return None

def enrich(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Missing {input_path}")
        return

    df = pd.read_parquet(input_path)

    
    # Filter for relevant date range (>= 2021-01-01) as per user request
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        start_date = pd.to_datetime("2021-01-01").date()
        df = df[df["date"] >= start_date].copy()
        print(f"Loaded {len(df)} rows (filtered >= {start_date}).")
    
    # Recover count columns from shares
    share_cols = [c for c in df.columns if c.startswith("share_")]
    for share_col in share_cols:
        group_name = share_col.replace("share_", "")
        count_col = f"count_{group_name}"
        if "total_messages" in df.columns:
            df[count_col] = (df[share_col] * df["total_messages"]).round().astype(int)
            print(f"Recovered {count_col}")

    if "sentiment_mean" not in df.columns:
        df["sentiment_mean"] = 0.0

    # --- Merge Macro Data (USD/RUB) ---
    macro_path = Path("data/raw/macro.csv")
    if macro_path.exists():
        print("Merging Macro Data (USD/RUB)...")
        macro_df = pd.read_csv(macro_path)
        macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.date
        df = pd.merge(df, macro_df, on="date", how="left")
        if "usd_rub" in df.columns:
            df["usd_rub"] = df["usd_rub"].ffill()
    
    # --- Feature Engineering ---
    print("Computing Rolling Features...")
    df = df.sort_values("date").reset_index(drop=True)
    signals = ["fuel_stress_index", "count_logistics_terms", "sentiment_mean", "avg_price", "usd_rub"]
    
    for col in signals:
        if col not in df.columns:
            continue
        rolling_7 = df[col].rolling(window=7, min_periods=1).mean()
        rolling_30 = df[col].rolling(window=30, min_periods=1).mean()
        df[f"{col}_trend_7d_30d"] = (rolling_7 + 1e-6) / (rolling_30 + 1e-6)
        df[f"{col}_shock_7d"] = (df[col] + 1e-6) / (rolling_7 + 1e-6)
        df[f"{col}_volatility_7d"] = df[col].rolling(window=7, min_periods=1).std().fillna(0.0)

    # Make parent dirs if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved enriched dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/daily_features.parquet")
    parser.add_argument("--output", default="data/processed/merged_enriched.parquet")
    args = parser.parse_args()
    
    enrich(args.input, args.output)
