"""Official data ingestion and merge helpers."""

from __future__ import annotations

import pandas as pd


def load_official_csv(path: str) -> pd.DataFrame:
    """Load official CSV expected to contain a 'date' column.
    
    Handles both daily and weekly data (with interpolation).
    """
    # Try detecting delimiter or just use engine='python'/sep=None if needed, 
    # but specific semicolon support is safer if we know the format.
    try:
        # encoding='utf-8-sig' handles the BOM if present (common in Excel CSVs)
        df = pd.read_csv(path, sep=None, engine='python', encoding="utf-8-sig")
    except Exception:
        # Fallback to default
        df = pd.read_csv(path, encoding="utf-8-sig")

    # Standardize Date column
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})

    if "date" not in df.columns:
        raise ValueError("Official CSV must include a 'date' column (or 'Date').")
    
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date")
    
    # Resample to daily and interpolate if it's not already daily
    # Check frequency roughly
    if len(df) > 1:
        min_diff = df["date"].diff().min()
        if min_diff and min_diff > pd.Timedelta(days=1):
            # It's likely weekly or irregular, upsample to daily
            df = df.set_index("date").resample("D").interpolate(method="linear")
            df = df.reset_index()

    df["date"] = df["date"].dt.date
    
    # Compute Variations (Returns) & Volatility for ML Targets
    if "Diesel_RUB" in df.columns:
        # Daily Percentage Change (Return) - Captures "Shocks"
        df["Diesel_RUB_change"] = df["Diesel_RUB"].pct_change().fillna(0.0)
        # 7-day Rolling Volatility (Standard Deviation of returns) - Captures "Instability"
        df["Diesel_RUB_volatility"] = df["Diesel_RUB_change"].rolling(window=7).std().fillna(0.0)
        
        # --- NEW CLASSIFICATION TARGET (STRATEGIC PIVOT) ---
        # Future 7-day return (looking ahead)
        future_return_7d = df["Diesel_RUB"].pct_change(7).shift(-7)
        # Threshold: 0.5% variation (approx top 10% of movements based on quantile analysis)
        # We target ABSOLUTE variation (crisis can be crash or surge), or just surge? 
        # Usually crisis = surge in price. Let's target Surge > 0.5%.
        df["crisis_7d"] = (future_return_7d > 0.005).astype(int)

    if "Regular92_RUB" in df.columns:
        df["Regular92_RUB_change"] = df["Regular92_RUB"].pct_change().fillna(0.0)
        df["Regular92_RUB_volatility"] = df["Regular92_RUB_change"].rolling(window=7).std().fillna(0.0)
        
    return df



def merge_with_official(features_df: pd.DataFrame, official_df: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """Merge engineered features with official dataset on date."""
    merged = features_df.merge(official_df, on="date", how=how, suffixes=("", "_official"))
    return merged
