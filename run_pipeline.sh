#!/bin/bash
set -e  # Exit immediately if any command fails

echo "Starting Russian Fuel Crisis Pipeline..."

# 1. Scraping
# Option A: Full Deep Scraping (2021-2025)
echo "[1/5] Deep Scraping Telegram Channels (2021-2025)..."
# python3 -m src.cli scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 0 --since 2021-01-01

# Option B: Light Scraping (Last 100 messages) - Uncomment for testing
# echo "[1/5] Light Scraping (Test Mode)..."
# python3 -m src.cli scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 100

# Default: Using the pre-scraped data if available, otherwise run Option A.
if [ ! -f "data/raw/messages.parquet" ]; then
    echo "No raw data found. Running full scrape..."
    python3 -m src.cli scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 0 --since 2021-01-01
else
    echo "Raw data found. Skipping scrape (delete data/raw/messages.parquet to re-run)."
fi

# 2. Features (NLP)
# Converts raw text into mathematical signals
echo "[2/5] Extracting NLP Features & Stress Index..."
python3 -m src.cli features --raw data/raw/messages.parquet --output data/interim/daily_features.parquet

# 3. Enrichment (Macro)
# Merges with USD/RUB and official stats
echo "[3/5] Enriching with Macro-Economic Data..."
# Corrected path to point to src module
python3 src/enrich_dataset.py --input data/interim/daily_features.parquet --output data/processed/merged_enriched.parquet

# 4. Training (All Models)
echo "[4/5] Training Models..."

# 4a. Random Forest (Baseline)
echo "   > Training Random Forest..."
python3 -m src.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type rf --target crisis_7d

# 4b. LSTM (Deep Learning)
echo "   > Training LSTM (Sequence Model)..."
python3 -m src.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type lstm --target crisis_7d

# 4c. XGBoost (Final Hybrid)
echo "   > Training XGBoost (Final Production Model)..."
python3 -m src.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type xgb --target crisis_7d

# 5. Visualization
# Check if viz script exists before running
if [ -f "scripts/viz_predictions.py" ]; then
    echo "[5/5] Generating Prediction Graphs..."
    python3 scripts/viz_predictions.py
else
    echo "[5/5] Visualization script not found. Skipping."
fi

echo "PIPELINE SUCCESS! Final report updated."
