from __future__ import annotations

import hashlib
import time
from typing import Any

from core import knowledge_base as kb
from core.movement_sniper import MovementFeatures
from layers.tactical import get_snapshot_history


TARGETS_USDT = (0.5, 1.0, 2.0)
OUTCOME_SECONDS = (30, 60, 120, 300)
MIN_CONTEXT_SAMPLES = 12


def _bucket(value: float, steps: list[tuple[float, str]], fallback: str) -> str:
    for limit, name in steps:
        if value < limit:
            return name
    return fallback


def _side_from_decision(decision: str, fallback: str) -> str:
    upper = decision.upper()
    if "LONG" in upper:
        return "LONG"
    if "SHORT" in upper:
        return "SHORT"
    return fallback.upper() if fallback else "UNKNOWN"


def _target_move_pct(target_usdt: float, margin_usdt: float, leverage: float) -> float:
    notional = margin_usdt * leverage
    if notional <= 0:
        return 0.0
    return target_usdt / notional * 100


def build_context_key(alt: MovementFeatures, btc: MovementFeatures) -> str:
    volume_bucket = _bucket(
        alt.volume_ratio,
        [(1.2, "vol_low"), (1.8, "vol_ok"), (2.8, "vol_hot")],
        "vol_extreme",
    )
    atr_bucket = _bucket(
        alt.atr_pct,
        [(1.0, "atr_low"), (3.0, "atr_ok"), (6.0, "atr_high")],
        "atr_extreme",
    )
    rsi_bucket = _bucket(
        alt.rsi,
        [(35, "rsi_low"), (55, "rsi_mid"), (70, "rsi_strong"), (78, "rsi_hot")],
        "rsi_exhausted",
    )
    oi_bucket = "oi_down" if alt.oi_change_pct < -0.2 else "oi_flat" if alt.oi_change_pct < 1.0 else "oi_up"
    funding_bucket = (
        "funding_negative" if alt.funding_rate < -0.0003
        else "funding_positive" if alt.funding_rate > 0.0003
        else "funding_neutral"
    )
    return "|".join([
        alt.movement_state,
        btc.movement_state,
        volume_bucket,
        atr_bucket,
        rsi_bucket,
        oi_bucket,
        funding_bucket,
    ])


def _signal_id(symbol: str, side: str, decision: str, created_bucket: int, context_key: str) -> str:
    raw = f"{symbol}|{side}|{decision}|{created_bucket}|{context_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


async def record_signal_from_gate(
    symbol: str,
    fallback_side: str,
    sniper: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    alt = MovementFeatures(**sniper["altFeatures"])
    btc = MovementFeatures(**sniper["btcFeatures"])
    side = _side_from_decision(str(sniper.get("decision", "")), fallback_side)
    entry_price = float(alt.last_price or 0)
    margin = float(config.get("marginPerTrade", 1.0) or 1.0)
    leverage = float(config.get("leverage", 1.0) or 1.0)
    context_key = build_context_key(alt, btc)
    target_moves = {str(t): _target_move_pct(t, margin, leverage) for t in TARGETS_USDT}
    created_bucket = int(time.time() // 5)
    signal_id = _signal_id(symbol, side, str(sniper.get("decision", "")), created_bucket, context_key)
    features = {
        "alt": sniper.get("altFeatures", {}),
        "btc": sniper.get("btcFeatures", {}),
        "alt_timeframes": sniper.get("altTimeframes", {}),
        "btc_timeframes": sniper.get("btcTimeframes", {}),
        "target_moves_pct": target_moves,
        "target_probabilities": sniper.get("targetProbabilities", {}),
    }
    recorded = await kb.record_signal_decision(
        signal_id=signal_id,
        symbol=symbol,
        side=side,
        decision=str(sniper.get("decision", "WAIT")),
        context_key=context_key,
        features=features,
        reasons=list(sniper.get("reasons", [])),
        entry_price=entry_price,
        target_moves=target_moves,
    )
    return {
        "recorded": recorded,
        "signalId": signal_id,
        "contextKey": context_key,
        "side": side,
        "entryPrice": entry_price,
        "targetMovesPct": target_moves,
    }


def _nearest_price(history: list[dict], target_ts: float) -> float | None:
    if not history:
        return None
    best = min(history, key=lambda x: abs(float(x.get("timestamp", 0)) - target_ts))
    price = float(best.get("price", 0) or 0)
    return price if price > 0 else None


def _directional_move_pct(entry: float, price: float, side: str) -> float:
    if entry <= 0 or price <= 0:
        return 0.0
    raw = (price - entry) / entry * 100
    return raw if side == "LONG" else -raw


async def finalize_due_signal_outcomes() -> dict[str, Any]:
    pending = await kb.get_pending_signal_outcomes(min_age_seconds=300, limit=250)
    finalized = 0
    for signal in pending:
        symbol = str(signal["symbol"])
        side = str(signal["side"]).upper()
        entry = float(signal["entry_price"] or 0)
        created_at = float(signal["created_at"] or 0)
        history = get_snapshot_history(symbol, 420)
        future = [h for h in history if float(h.get("timestamp", 0)) >= created_at]
        if not future:
            persisted = await kb.get_feature_history(symbol, hours=2)
            future = [h for h in persisted if float(h.get("timestamp", 0)) >= created_at]
        if entry <= 0 or not future:
            continue
        prices = {}
        for sec in OUTCOME_SECONDS:
            price = _nearest_price(future, created_at + sec)
            if price is not None:
                prices[str(sec)] = price
        moves = [_directional_move_pct(entry, float(h.get("price", 0) or 0), side) for h in future]
        max_favorable = max(moves) if moves else 0.0
        max_adverse = min(moves) if moves else 0.0
        hits = {
            "0.5": max_favorable >= float(signal["target_050_move_pct"] or 0),
            "1.0": max_favorable >= float(signal["target_100_move_pct"] or 0),
            "2.0": max_favorable >= float(signal["target_200_move_pct"] or 0),
        }
        stopped = max_adverse <= -max(float(signal["target_100_move_pct"] or 0), 0.15)
        await kb.finalize_signal_outcome(
            signal_id=str(signal["signal_id"]),
            prices=prices,
            hits=hits,
            stopped=stopped,
            max_favorable_pct=round(max_favorable, 4),
            max_adverse_pct=round(max_adverse, 4),
        )
        finalized += 1
    return {"pending": len(pending), "finalized": finalized}


async def score_signal_context(symbol: str, side: str, context_key: str) -> dict[str, Any]:
    context_stats = await kb.get_signal_edge_stats(symbol=symbol, side=side, context_key=context_key)
    symbol_stats = await kb.get_signal_edge_stats(symbol=symbol, side=side, context_key=None)
    effective = context_stats if context_stats["samples"] >= MIN_CONTEXT_SAMPLES else symbol_stats
    samples = effective["samples"]
    if samples == 0:
        score = 0.5
        verdict = "cold_start"
    else:
        hit = effective["hit_050"]
        stop = effective["stop_rate"]
        quality = hit - (stop * 0.65)
        score = max(0.0, min(0.95, 0.35 + quality))
        if samples < MIN_CONTEXT_SAMPLES:
            verdict = "learning"
        elif hit >= 0.62 and stop <= 0.32:
            verdict = "positive_context"
        elif hit < 0.48 or stop > 0.45:
            verdict = "toxic_context"
        else:
            verdict = "neutral_context"
    return {
        "score": round(score, 4),
        "verdict": verdict,
        "minSamples": MIN_CONTEXT_SAMPLES,
        "context": context_stats,
        "symbolSide": symbol_stats,
    }
