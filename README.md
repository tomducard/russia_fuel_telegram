# Détection de Crises de Carburant en Russie par Analyse de Données Telegram

**Projet de Machine Learning avec Python - ENSAE Paris (2025-2026)**

Ce projet vise à anticiper les pénuries de carburant en Russie en analysant les signaux faibles (files d'attente, hausse des prix, panique) sur les canaux Telegram de logistique et de transport, croisés avec des données macro-économiques.

---

## 📋 Pré-requis

*   **Python 3.11+**
*   Compte développeur Telegram (pour obtenir `API_ID` et `API_HASH`)
*   Bibliothèques listées dans `pyproject.toml` (installables via `pip install -e .`)

## ⚙️ Installation

1.  **Cloner le dépôt**
    ```bash
    git clone ...
    cd russia-fuel-telegram
    ```

2.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    # OU
    pip install -e .
    ```

3.  **Configuration Telegram**
    Créer un fichier `.env` à la racine :
    ```env
    TG_API_ID=123456
    TG_API_HASH=votre_hash_ici
    TG_SESSION_NAME=rft_session
    ```

---

## 🚀 Utilisation

### 1. Pipeline Complet (Recommandé)
Un script Bash automatise l'ensemble de la chaîne (Collecte -> Feature Engineering -> Entraînement -> Visualisation).
```bash
./run_pipeline.sh
```

### 2. Commandes Individuelles
Le projet expose une CLI `rft` (ou via `python -m src.cli`) :

*   **Collecte (Scraping)**
    ```bash
    python -m src.cli scrape --limit 1000 --since 2023-01-01
    ```

*   **Feature Engineering (NLP)**
    ```bash
    python -m src.cli features --raw data/raw/messages.parquet --output data/interim/daily_features.parquet
    ```

*   **Enrichissement (Macro)**
    ```bash
    python src/enrich_dataset.py --input data/interim/daily_features.parquet --output data/processed/merged_enriched.parquet
    ```

*   **Entraînement des Modèles**
    ```bash
    # Random Forest
    python -m src.cli train --model-type rf --mode classification
    
    # XGBoost (Modèle Final)
    python -m src.cli train --model-type xgb --mode classification
    ```

---

## 📂 Structure du Projet

*   `src/` : Code source du package (Scraping, NLP, Modélisation).
*   `data/` :
    *   `raw/` : Messages bruts (Parquet).
    *   `processed/` : Données enrichies prêtes pour le ML.
*   `channels/` : Liste des canaux suivis (`channels_seed.csv`) et mots-clés (`keywords_ru.yaml`).
*   `notebooks/` :
    *   `analysis.ipynb` : Analyse exploratoire, tuning des modèles et graphes du rapport final.

---

## 👥 Auteurs
*   **Tom DUCARD**
*   *(Ajouter autres membres)*

**École** : ENSAE Paris
