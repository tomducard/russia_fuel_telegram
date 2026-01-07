"""Text normalization and deduplication utilities."""

from __future__ import annotations

import re
from typing import Iterable, Set

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_TEXT_PATTERN = re.compile(r"[^\w\s.,:;!?₽руб]+", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize Telegram message text by lowercasing, stripping URLs, and collapsing whitespace."""
    if not isinstance(text, str):
        return ""

    lowered = text.lower()
    without_urls = URL_PATTERN.sub(" ", lowered)
    cleaned = NON_TEXT_PATTERN.sub(" ", without_urls)
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def deduplicate_messages(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Deduplicate messages based on normalized text."""
    working = df.copy()
    working["normalized_text"] = working[text_col].fillna("").map(normalize_text)
    deduped = working.drop_duplicates(subset=["normalized_text"]).reset_index(drop=True)
    return deduped


def unique_tokens(texts: Iterable[str]) -> Set[str]:
    """Extract a set of unique normalized tokens from a collection of texts."""
    tokens: Set[str] = set()
    for text in texts:
        normalized = normalize_text(text)
        tokens.update(normalized.split())
    return tokens
