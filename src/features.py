"""Feature engineering for Telegram fuel monitoring."""

from __future__ import annotations

import math
from typing import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import keywords as kw
from . import normalization
try:
    from . import nlp
except ImportError:
    nlp = None  # Graceful fallback if dependencies missing


def compute_fuel_stress_index(row: pd.Series) -> float:
    """Heuristic Fuel Stress Index combining messaging, price signals, and sentiment."""
    keyword_mentions = float(row.get("keyword_mentions", 0) or 0)
    price_mentions = float(row.get("price_mentions", 0) or 0)
    avg_price = float(row.get("avg_price", 0) or 0)
    if math.isnan(avg_price):
        avg_price = 0.0
    unique_messages = float(row.get("unique_messages", 0) or 0)
    
    # Sentiment usually -1 (neg) to 1 (pos).
    # We want negative sentiment to INCREASE stress.
    sentiment = float(row.get("sentiment_mean", 0) or 0)
    if math.isnan(sentiment):
        sentiment = 0.0
        
    # Multiplier: 
    # If sentiment is -1 (bad), factor = 1 + 1 = 2 (Double stress)
    # If sentiment is +1 (good), factor = 1 - 1 = 0 (No stress)
    # If sentiment is 0 (neutral), factor = 1
    sentiment_factor = max(0.0, 1.0 - sentiment)

    activity_term = keyword_mentions + price_mentions
    price_term = math.log1p(avg_price + 1.0)
    richness_term = 0.05 * unique_messages
    
    base_stress = activity_term * price_term + richness_term
    return base_stress * sentiment_factor


def build_daily_features(messages: pd.DataFrame, keyword_groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Compute daily aggregates and Fuel Stress Index from Telegram messages."""
    if "date" not in messages.columns:
        raise ValueError("messages DataFrame must contain a 'date' column")

    working = messages.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.date
    working["text"] = working["text"].fillna("")
    working = normalization.deduplicate_messages(working, text_col="text")
    
    # --- Optimize: Run keywords FIRST to filter relevant messages ---
    annotated = kw.keyword_stats(working, text_col="normalized_text", keyword_groups=keyword_groups)
    
    # --- Sentiment Analysis (With Persistence Cache) ---
    if nlp is not None:
        analyzer = nlp.SentimentAnalyzer()
        cache_path = Path("data/cache/sentiment_cache.parquet")
        
        # 1. Load existing cache
        if cache_path.exists():
            print(f"Loading sentiment cache from {cache_path}...")
            cache_df = pd.read_parquet(cache_path)
            # Ensure index is normalized_text for fast lookup
            if "normalized_text" in cache_df.columns:
                cache_df = cache_df.set_index("normalized_text")
            # Cache structure: index=normalized_text, columns=[sentiment]
        else:
            print("No existing sentiment cache found. Starting fresh.")
            cache_df = pd.DataFrame(columns=["sentiment"])

        # 2. Identify what needs computation
        # Relevant messages only (keyword count > 0)
        relevant_mask = annotated["keyword_count"] > 0
        relevant_texts = annotated.loc[relevant_mask, "normalized_text"].unique()
        
        # Filter for texts NOT in cache
        missing_texts = [t for t in relevant_texts if t not in cache_df.index]
        
        if missing_texts:
            print(f"Computing sentiment for {len(missing_texts)} new unique messages...")
            # We need the RAW text for the model, not normalized? 
            # The model is usually better with raw text, but we risk mapping issues if raw texts differ for same normalized.
            # Let's map normalized -> raw (pick first) for prediction
            # Create a lookup map from the dataframe
            norm_to_raw = annotated.loc[annotated["normalized_text"].isin(missing_texts)].groupby("normalized_text")["text"].first()
            raw_inputs = [norm_to_raw[t] for t in missing_texts]
            
            # Predict
            new_scores = analyzer.predict_sentiment(raw_inputs)
            
            # Update Cache
            new_entries = pd.DataFrame({"sentiment": new_scores}, index=missing_texts)
            cache_df = pd.concat([cache_df, new_entries])
            
            # Save Cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.reset_index().rename(columns={"index": "normalized_text"}).to_parquet(cache_path)
            print(f"Updated cache saved to {cache_path} (Total: {len(cache_df)} entries)")
        else:
            print("All relevant messages found in cache. No new computation needed.")

        # 3. Apply scores to current dataframe
        # Join annotated with cache on normalized_text
        annotated = annotated.merge(cache_df, left_on="normalized_text", right_index=True, how="left")
        
        # Fill NaN (irrelevant or missed) with 0.0
        annotated["sentiment"] = annotated["sentiment"].fillna(0.0)
    else:
        annotated["sentiment"] = 0.0
    group_flag_cols = [f"has_{group}" for group in keyword_groups.keys()]

    aggregations: dict[str, tuple[str, str]] = dict(
        total_messages=("text", "count"),
        unique_messages=("normalized_text", "nunique"),
        keyword_mentions=("keyword_count", "sum"),
        price_mentions=("price_count", "sum"),
        price_sum=("price_sum", "sum"),
        # New split aggregations
        price_mentions_diesel=("price_count_diesel", "sum"),
        price_sum_diesel=("price_sum_diesel", "sum"),
        price_mentions_gasoline=("price_count_gasoline", "sum"),
        price_sum_gasoline=("price_sum_gasoline", "sum"),
        # Sentiment
        sentiment_mean=("sentiment", "mean"),
        sentiment_min=("sentiment", "min"),
    )
    for flag_col in group_flag_cols:
        share_name = flag_col.replace("has_", "share_", 1)
        aggregations[share_name] = (flag_col, "mean")
        
        # New: Add raw count of messages containing keywords from this group
        count_name = flag_col.replace("has_", "count_", 1)
        aggregations[count_name] = (flag_col, "sum")

    grouped = annotated.groupby("date").agg(**aggregations)

    grouped["avg_price"] = grouped.apply(
        lambda row: row["price_sum"] / row["price_mentions"] if row["price_mentions"] > 0 else np.nan,
        axis=1,
    )
    grouped["avg_price_diesel"] = grouped.apply(
        lambda row: row["price_sum_diesel"] / row["price_mentions_diesel"] if row["price_mentions_diesel"] > 0 else np.nan,
        axis=1,
    )
    grouped["avg_price_gasoline"] = grouped.apply(
        lambda row: row["price_sum_gasoline"] / row["price_mentions_gasoline"] if row["price_mentions_gasoline"] > 0 else np.nan,
        axis=1,
    )
    
    grouped["fuel_stress_index"] = grouped.apply(compute_fuel_stress_index, axis=1)
    grouped = grouped.reset_index().sort_values("date")
    return grouped
