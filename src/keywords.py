"""Keyword and price extraction utilities."""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import pandas as pd
import regex as re
import yaml


# Patterns for unit-aware price extraction
TON_PATTERN = re.compile(
    r"(?<!\d)(\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?|\d+)\s*(?:₽|руб(?:\.|ля|лей)?)\s*(?:за|/)\s*(?:т|тн|тонну|ton)\b",
    flags=re.IGNORECASE,
)

LITER_PATTERN = re.compile(
    r"(?<!\d)(\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?|\d+)\s*(?:₽|руб(?:\.|ля|лей)?)\s*(?:за|/)\s*(?:л|литр|l)\b",
    flags=re.IGNORECASE,
)

PatternCache = Dict[str, re.Pattern]
KeywordGroups = Dict[str, List[str]]

_PATTERN_CACHE: PatternCache = {}


def _get_pattern(term: str) -> re.Pattern:
    """Return or compile a regex pattern for the given term."""
    cached = _PATTERN_CACHE.get(term)
    if cached:
        return cached
    pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    _PATTERN_CACHE[term] = pattern
    return pattern


def load_keyword_groups(path: str | pathlib.Path) -> KeywordGroups:
    """Load keyword groups from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Keyword YAML must define a mapping of group names to keyword lists.")

    groups: KeywordGroups = {}
    for group, values in data.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            continue
        cleaned: List[str] = []
        seen: Set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(text)
        if cleaned:
            key = str(group).strip()
            if key:
                groups[key] = cleaned
    return groups


def flatten_groups(keyword_groups: Mapping[str, Sequence[str]]) -> List[str]:
    """Return a deduplicated list of keywords across all groups."""
    seen: Set[str] = set()
    flat: List[str] = []
    for terms in keyword_groups.values():
        for term in terms:
            text = str(term).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            flat.append(text)
    return flat


def match_keywords(text: str, keywords: Sequence[str]) -> Set[str]:
    """Return a set of matched keywords within the given text."""
    if not text:
        return set()

    found: Set[str] = set()
    lowered = text.lower()
    for kw in keywords:
        term = str(kw).lower()
        if not term:
            continue
        pattern = _get_pattern(term)
        if pattern.search(lowered):
            found.add(term)
    return found


def extract_ruble_prices(text: str) -> List[float]:
    """Extract numeric ruble amounts from text, normalizing per-ton prices to per-liter."""
    if not text:
        return []

    prices: List[float] = []

    # 1. Check Wholesale (Ton) -> Normalize to Liter (approx 1176 L/ton for Diesel 0.85)
    for match in TON_PATTERN.finditer(text):
        raw_value = match.group(1)
        normalized = re.sub(r"[ \u00A0]", "", raw_value)
        normalized = normalized.replace(",", ".")
        try:
            val = float(normalized)
            # Convert Ton price to Liter price (~ / 1176)
            liter_equiv = val / 1176.0
            if 20.0 <= liter_equiv <= 200.0:  # Valid fuel price range (roughly 50-100)
                prices.append(liter_equiv)
        except ValueError:
            continue

    # 2. Check Retail (Liter)
    for match in LITER_PATTERN.finditer(text):
        raw_value = match.group(1)
        normalized = re.sub(r"[ \u00A0]", "", raw_value)
        normalized = normalized.replace(",", ".")
        try:
            val = float(normalized)
            if 20.0 <= val <= 200.0:
                prices.append(val)
        except ValueError:
            continue
            
    # 3. Check Contextual Proximity (Number near Fuel Keyword)
    # Search for 2-3 digit numbers (20-100) near fuel keywords
    # This captures "AI-95: 55.40" even without explicit unit
    
    # Simple regex for finding float/int numbers 20-100
    POTENTIAL_PRICE = re.compile(r"\b(\d{2,3}(?:[.,]\d+)?)\b")
    
    # We need to know if the text talks about fuel
    # We'll re-use match_keywords logic locally or pass matched keywords? 
    # For independent utility, let's just do a quick scan for major fuel terms
    FUEL_TERMS = re.compile(r"(?:бензин|дизель|дт|аи-9\d|аи-100|fuel|gasoline|diesel)", flags=re.IGNORECASE)
    
    if len(prices) == 0 and FUEL_TERMS.search(text):
        for match in POTENTIAL_PRICE.finditer(text):
            try:
                val = float(match.group(1).replace(",", "."))
                if 30.0 <= val <= 80.0:  # Stricter range for context-less extraction (Retail range)
                     # Only keep if strictly within realistic retail bounds (30-80 RUB)
                    prices.append(val)
            except ValueError:
                continue
            
    return prices


def keyword_stats(df: pd.DataFrame, text_col: str, keyword_groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Annotate DataFrame with keyword and price extraction stats."""
    flat_keywords = flatten_groups(keyword_groups)
    group_sets: Dict[str, Set[str]] = {}
    for group, terms in keyword_groups.items():
        normalized_terms: Set[str] = set()
        for term in terms:
            text = str(term).strip()
            if not text:
                continue
            normalized_terms.add(text.lower())
        if normalized_terms:
            group_sets[group] = normalized_terms

    working = df.copy()
    working["matched_keywords"] = working[text_col].fillna("").map(lambda t: match_keywords(t, flat_keywords))
    working["keyword_count"] = working["matched_keywords"].map(len)
    working["prices"] = working[text_col].fillna("").map(extract_ruble_prices)
    working["price_count"] = working["prices"].map(len)
    working["price_sum"] = working["prices"].map(lambda vals: sum(vals) if vals else 0.0)

    for group, terms in group_sets.items():
        col_name = f"has_{group}"
        # Improved performance: check intersection with pre-computed matched set
        working[col_name] = working["matched_keywords"].map(lambda found: bool(found & terms))

    # Attribution logic for prices by fuel type
    diesel_set = group_sets.get("diesel_terms", set())
    gasoline_set = group_sets.get("gasoline_terms", set())

    def get_attributed_prices(row, fuel_set):
        # If message has fuel-specific keywords, return the prices, else empty
        if row["matched_keywords"] & fuel_set:
            return row["prices"]
        return []

    working["prices_diesel"] = working.apply(lambda r: get_attributed_prices(r, diesel_set), axis=1)
    working["prices_gasoline"] = working.apply(lambda r: get_attributed_prices(r, gasoline_set), axis=1)
    
    working["price_sum_diesel"] = working["prices_diesel"].map(lambda vals: sum(vals) if vals else 0.0)
    working["price_count_diesel"] = working["prices_diesel"].map(len)
    
    working["price_sum_gasoline"] = working["prices_gasoline"].map(lambda vals: sum(vals) if vals else 0.0)
    working["price_count_gasoline"] = working["prices_gasoline"].map(len)

    return working
