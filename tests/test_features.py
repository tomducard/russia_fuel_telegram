import math
import pandas as pd

from rft.features import build_daily_features


def test_build_daily_features_computes_index():
    messages = pd.DataFrame(
        [
            {"date": "2024-01-01", "text": "Бензин 1 000 руб на азс"},
            {"date": "2024-01-01", "text": "Еще бензин 2 000 руб"},
            {"date": "2024-01-02", "text": "Новость без цены"},
        ]
    )
    keyword_groups = {"fuel_terms": ["бензин"]}

    daily = build_daily_features(messages, keyword_groups)
    day1 = daily[daily["date"] == pd.to_datetime("2024-01-01").date()].iloc[0]
    assert day1["keyword_mentions"] == 2
    assert day1["price_mentions"] == 2
    assert math.isclose(day1["avg_price"], 1500.0, rel_tol=1e-3)
    assert day1["fuel_stress_index"] > 0
    assert math.isclose(day1["share_fuel_terms"], 1.0, rel_tol=1e-6)

    day2 = daily[daily["date"] == pd.to_datetime("2024-01-02").date()].iloc[0]
    assert day2["keyword_mentions"] == 0
    assert day2["price_mentions"] == 0
    assert math.isclose(day2["fuel_stress_index"], 0.05, rel_tol=1e-6)
    assert math.isclose(day2["share_fuel_terms"], 0.0, rel_tol=1e-6)
