from __future__ import annotations

from typing import Any


def _pct(first: float, last: float) -> float:
    if first <= 0:
        return 0.0
    return (last - first) / first * 100


def _ema(values: list[float], period: int) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    k = 2 / (period + 1)
    ema = vals[0]
    for value in vals[1:]:
        ema = value * k + ema * (1 - k)
    return ema


def _frame(history: list[dict[str, Any]], seconds: int) -> dict[str, Any]:
    if not history:
        return {
            "samples": 0,
            "changePct": 0.0,
            "emaDistancePct": 0.0,
            "rangePct": 0.0,
            "wickRatio": 0.0,
            "volumeRatioAvg": 0.0,
            "volumeTrend": "unknown",
            "breakoutState": "NO_DATA",
        }
    end_ts = float(history[-1].get("timestamp", 0) or 0)
    start_ts = end_ts - seconds
    frame = [x for x in history if float(x.get("timestamp", 0) or 0) >= start_ts]
    prices = [float(x.get("price", 0) or 0) for x in frame if float(x.get("price", 0) or 0) > 0]
    if len(prices) < 2:
        return {
            "samples": len(prices),
            "changePct": 0.0,
            "emaDistancePct": 0.0,
            "rangePct": 0.0,
            "wickRatio": 0.0,
            "volumeRatioAvg": 0.0,
            "volumeTrend": "unknown",
            "breakoutState": "NO_DATA",
        }
    open_price = prices[0]
    close_price = prices[-1]
    high = max(prices)
    low = min(prices)
    body = abs(close_price - open_price)
    candle_range = max(high - low, 0.0)
    upper_wick = high - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low
    wick_ratio = (upper_wick + lower_wick) / body if body > 0 else (upper_wick + lower_wick)
    ema = _ema(prices, min(20, max(3, len(prices))))
    ema_distance = _pct(ema, close_price) if ema > 0 else 0.0
    volumes = [float(x.get("volume_ratio", 1) or 1) for x in frame]
    midpoint = max(1, len(volumes) // 2)
    early_vol = sum(volumes[:midpoint]) / midpoint
    late_vol = sum(volumes[midpoint:]) / max(1, len(volumes[midpoint:]))
    volume_trend = "rising" if late_vol > early_vol * 1.08 else "falling" if late_vol < early_vol * 0.92 else "flat"
    change_pct = _pct(open_price, close_price)
    range_pct = _pct(low, high) if low > 0 else 0.0
    breakout_state = "RANGE"
    if abs(change_pct) >= 0.25 and late_vol >= 1.35 and wick_ratio <= 1.8:
        breakout_state = "BREAKOUT_UP" if change_pct > 0 else "BREAKOUT_DOWN"
    elif abs(change_pct) >= 0.25 and (late_vol < 1.15 or wick_ratio > 2.5):
        breakout_state = "FAKEOUT"
    return {
        "samples": len(prices),
        "changePct": round(change_pct, 4),
        "emaDistancePct": round(ema_distance, 4),
        "rangePct": round(range_pct, 4),
        "wickRatio": round(wick_ratio, 4),
        "volumeRatioAvg": round(sum(volumes) / len(volumes), 4),
        "volumeTrend": volume_trend,
        "breakoutState": breakout_state,
    }


def build_multiframe_context(history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "1m": _frame(history, 60),
        "5m": _frame(history, 300),
        "15m": _frame(history, 900),
    }
