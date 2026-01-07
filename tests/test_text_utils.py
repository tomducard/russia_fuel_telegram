import math

from rft.normalization import normalize_text
from rft.keywords import extract_ruble_prices


def test_normalize_text_strips_urls_and_lowercases():
    raw = "Visit HTTPS://Example.com now! БЕНЗИН 45 РУБ."
    normalized = normalize_text(raw)
    assert "https" not in normalized
    assert normalized == "visit now! бензин 45 руб."


def test_extract_ruble_prices_handles_spaces_and_symbols():
    text = "Цена 1 234 руб., скидка 999₽ и еще 10 000 руб"
    prices = extract_ruble_prices(text)
    assert prices == [1234.0, 999.0, 10000.0]
