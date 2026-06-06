from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from core.candle_intelligence import build_multiframe_context


SNIPER_WINDOW_SECONDS = 300


@dataclass
class MovementFeatures:
    symbol: str
    samples: int
    window_seconds: int
    first_price: float
    last_price: float
    high_price: float
    low_price: float
    price_change_pct: float
    price_acceleration: float
    max_favorable_pct: float
    max_adverse_pct: float
    volume_ratio: float
    oi_change_pct: float
    funding_rate: float
    rsi: float
    atr_pct: float
    btc_regime: str
    direction: str
    movement_state: str


def _pct_change(first: float, last: float) -> float:
    if first <= 0:
        return 0.0
    return (last - first) / first * 100


def _avg(values: list[float], fallback: float = 0.0) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else fallback


def _last_float(history: list[dict], key: str, fallback: float = 0.0) -> float:
    if not history:
        return fallback
    value = history[-1].get(key, fallback)
    try:
        return float(value)
    except Exception:
        return fallback


def _movement_state(
    change_pct: float,
    acceleration: float,
    volume_ratio: float,
    oi_change_pct: float,
    rsi: float,
    btc_regime: str,
) -> str:
    abs_change = abs(change_pct)
    if abs_change < 0.12 or volume_ratio < 1.05:
        return "CHOP"
    if rsi >= 76 and change_pct > 0:
        return "EXHAUSTION_UP"
    if rsi <= 24 and change_pct < 0:
        return "EXHAUSTION_DOWN"
    if change_pct > 0.25 and acceleration > 0 and volume_ratio >= 1.4 and oi_change_pct >= 0:
        return "IMPULSE_UP" if btc_regime != "BEAR" else "BTC_CONFLICT"
    if change_pct < -0.25 and acceleration < 0 and volume_ratio >= 1.4 and oi_change_pct >= 0:
        return "IMPULSE_DOWN" if btc_regime != "BULL" else "BTC_CONFLICT"
    if abs_change >= 0.6 and volume_ratio < 1.2:
        return "FAKE_MOVE"
    return "WAIT"


def build_movement_features(symbol: str, history: list[dict], window_seconds: int = SNIPER_WINDOW_SECONDS) -> MovementFeatures:
    if not history:
        return MovementFeatures(
            symbol=symbol,
            samples=0,
            window_seconds=window_seconds,
            first_price=0.0,
            last_price=0.0,
            high_price=0.0,
            low_price=0.0,
            price_change_pct=0.0,
            price_acceleration=0.0,
            max_favorable_pct=0.0,
            max_adverse_pct=0.0,
            volume_ratio=0.0,
            oi_change_pct=0.0,
            funding_rate=0.0,
            rsi=50.0,
            atr_pct=0.0,
            btc_regime="NEUTRAL",
            direction="FLAT",
            movement_state="NO_DATA",
        )

    prices = [float(x.get("price", 0) or 0) for x in history]
    valid_prices = [p for p in prices if p > 0]
    first_price = valid_prices[0] if valid_prices else 0.0
    last_price = valid_prices[-1] if valid_prices else 0.0
    high_price = max(valid_prices) if valid_prices else 0.0
    low_price = min(valid_prices) if valid_prices else 0.0
    change_pct = _pct_change(prices[0], prices[-1]) if len(prices) >= 2 else 0.0
    max_favorable = _pct_change(first_price, high_price) if first_price > 0 else 0.0
    max_adverse = _pct_change(first_price, low_price) if first_price > 0 else 0.0
    midpoint = max(1, len(prices) // 2)
    early_change = _pct_change(prices[0], prices[midpoint - 1]) if len(prices) >= 4 else 0.0
    late_change = _pct_change(prices[midpoint], prices[-1]) if len(prices) >= 4 else change_pct
    acceleration = late_change - early_change
    volume_ratio = _avg([float(x.get("volume_ratio", 1) or 1) for x in history], 1.0)
    oi_change_pct = _last_float(history, "oi_change_pct")
    funding_rate = _last_float(history, "funding_rate")
    rsi = _last_float(history, "rsi", 50.0)
    atr_pct = _last_float(history, "atr_pct")
    btc_regime = str(history[-1].get("btc_regime", "NEUTRAL"))
    direction = "LONG" if change_pct > 0 else "SHORT" if change_pct < 0 else "FLAT"
    state = _movement_state(change_pct, acceleration, volume_ratio, oi_change_pct, rsi, btc_regime)
    return MovementFeatures(
        symbol=symbol,
        samples=len(history),
        window_seconds=window_seconds,
        first_price=round(first_price, 8),
        last_price=round(last_price, 8),
        high_price=round(high_price, 8),
        low_price=round(low_price, 8),
        price_change_pct=round(change_pct, 4),
        price_acceleration=round(acceleration, 4),
        max_favorable_pct=round(max_favorable, 4),
        max_adverse_pct=round(max_adverse, 4),
        volume_ratio=round(volume_ratio, 4),
        oi_change_pct=round(oi_change_pct, 4),
        funding_rate=round(funding_rate, 8),
        rsi=round(rsi, 2),
        atr_pct=round(atr_pct, 4),
        btc_regime=btc_regime,
        direction=direction,
        movement_state=state,
    )


def classify_btc_commander(btc: MovementFeatures) -> dict[str, Any]:
    if btc.samples < 3:
        return {"state": "BTC_NO_DATA", "confidence": 0.0}
    if btc.movement_state == "IMPULSE_UP":
        return {"state": "BTC_IMPULSE_UP", "confidence": min(0.95, 0.55 + abs(btc.price_change_pct) / 2)}
    if btc.movement_state == "IMPULSE_DOWN":
        return {"state": "BTC_IMPULSE_DOWN", "confidence": min(0.95, 0.55 + abs(btc.price_change_pct) / 2)}
    if btc.movement_state in {"CHOP", "WAIT"}:
        return {"state": "BTC_CHOP", "confidence": 0.6}
    if "EXHAUSTION" in btc.movement_state:
        return {"state": "BTC_EXHAUSTION", "confidence": 0.72}
    return {"state": btc.movement_state, "confidence": 0.5}


def _target_probability(alt: MovementFeatures, btc: MovementFeatures, target_usdt: float) -> float:
    if alt.samples < 3 or btc.samples < 3:
        return 0.0
    direction_ok = (
        alt.movement_state == "IMPULSE_UP" and btc.movement_state == "IMPULSE_UP"
    ) or (
        alt.movement_state == "IMPULSE_DOWN" and btc.movement_state == "IMPULSE_DOWN"
    )
    base = 0.38
    base += min(0.20, abs(alt.price_change_pct) / 4)
    base += min(0.14, max(0.0, alt.volume_ratio - 1.0) / 8)
    base += min(0.10, max(0.0, alt.oi_change_pct) / 20)
    base += min(0.08, abs(btc.price_change_pct) / 5)
    if direction_ok:
        base += 0.12
    if alt.rsi >= 76 or alt.rsi <= 24:
        base -= 0.16
    if alt.movement_state in {"CHOP", "FAKE_MOVE", "BTC_CONFLICT", "NO_DATA"}:
        base -= 0.22
    if target_usdt >= 2:
        base -= 0.10
    elif target_usdt >= 1:
        base -= 0.04
    return round(max(0.0, min(0.95, base)), 4)


def evaluate_sniper_window(
    symbol: str,
    alt_history: list[dict],
    btc_history: list[dict],
    targets_usdt: list[float] | None = None,
    window_seconds: int = SNIPER_WINDOW_SECONDS,
) -> dict[str, Any]:
    targets = targets_usdt or [0.5, 1.0, 2.0]
    alt = build_movement_features(symbol, alt_history, window_seconds)
    btc = build_movement_features("BTC-USDT", btc_history, window_seconds)
    alt_frames = build_multiframe_context(alt_history)
    btc_frames = build_multiframe_context(btc_history)
    btc_state = classify_btc_commander(btc)
    probabilities = {str(t): _target_probability(alt, btc, t) for t in targets}
    best_target = max(probabilities, key=probabilities.get) if probabilities else "0.5"
    best_probability = probabilities.get(best_target, 0.0)

    decision = "WAIT"
    reasons: list[str] = []
    if alt.samples < 3 or btc.samples < 3:
        decision = "WAIT"
        reasons.append("insufficient_realtime_samples")
    elif alt.movement_state == "IMPULSE_UP" and btc_state["state"] == "BTC_IMPULSE_UP":
        decision = "ALLOW_LONG" if best_probability >= 0.58 else "WAIT"
        reasons.extend(["btc_impulse_up", "alt_impulse_up"])
    elif alt.movement_state == "IMPULSE_DOWN" and btc_state["state"] == "BTC_IMPULSE_DOWN":
        decision = "ALLOW_SHORT" if best_probability >= 0.58 else "WAIT"
        reasons.extend(["btc_impulse_down", "alt_impulse_down"])
    elif alt.movement_state == "CHOP" or btc_state["state"] == "BTC_CHOP":
        decision = "BLOCK_CHOP"
        reasons.append("chop_environment")
    elif "EXHAUSTION" in alt.movement_state:
        decision = "BLOCK_EXHAUSTION"
        reasons.append("alt_exhaustion_risk")
    elif alt.movement_state == "BTC_CONFLICT":
        decision = "BLOCK_BTC_CONFLICT"
        reasons.append("alt_move_conflicts_with_btc")
    elif alt.movement_state == "FAKE_MOVE":
        decision = "BLOCK_FAKE_VOLUME"
        reasons.append("price_move_without_volume_confirmation")

    if alt.volume_ratio >= 1.4:
        reasons.append("volume_expansion")
    if alt.oi_change_pct >= 0:
        reasons.append("oi_not_against_move")
    if 35 <= alt.rsi <= 70:
        reasons.append("rsi_healthy")
    if alt_frames["5m"]["breakoutState"] in {"BREAKOUT_UP", "BREAKOUT_DOWN"}:
        reasons.append("confirmed_5m_breakout")
    if alt_frames["1m"]["breakoutState"] == "FAKEOUT":
        reasons.append("one_minute_fakeout_risk")

    return {
        "symbol": symbol,
        "windowSeconds": window_seconds,
        "evaluatedAt": time.time(),
        "decision": decision,
        "score": best_probability,
        "target": f"{best_target} USDT",
        "targetHitProbability": best_probability,
        "targetProbabilities": probabilities,
        "expectedTimeToTargetSec": int(max(30, 150 - best_probability * 90)) if best_probability else None,
        "risk": "standard" if best_probability >= 0.68 else "scout" if best_probability >= 0.58 else "wait",
        "reasons": reasons,
        "btcCommander": btc_state,
        "btcFeatures": btc.__dict__,
        "altFeatures": alt.__dict__,
        "btcTimeframes": btc_frames,
        "altTimeframes": alt_frames,
        "learningMode": "movement_first_pnl_auditor",
    }
