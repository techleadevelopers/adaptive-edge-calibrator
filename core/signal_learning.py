from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from core import knowledge_base as kb
from core.movement_sniper import MovementFeatures
from layers.tactical import get_snapshot_history


TARGETS_USDT = (0.5, 1.0, 2.0)
OUTCOME_SECONDS = (30, 60, 120, 300)
MIN_CONTEXT_SAMPLES = 12
STRATEGY_VERSION = "sniper-v2"
OUTCOME_WINDOW_SECONDS = 300
PRICE_TOLERANCE_SECONDS = 35


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


def _target_move_pct(
    target_usdt: float,
    margin_usdt: float,
    leverage: float,
    estimated_cost_pct: float = 0.0,
) -> float:
    notional = margin_usdt * leverage
    if notional <= 0:
        return 0.0
    return target_usdt / notional * 100 + estimated_cost_pct


def _decision_group(decision: str) -> str:
    upper = decision.upper()
    if upper.startswith("ALLOW_"):
        return "ALLOW"
    if upper.startswith("BLOCK_"):
        return "BLOCK"
    return "WAIT"


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _estimated_cost_pct(alt: MovementFeatures, config: dict[str, Any]) -> float:
    entry_fee_bps = float(config.get("entryFeeBps", config.get("takerFeeBps", 5.0)) or 0)
    exit_fee_bps = float(config.get("exitFeeBps", config.get("takerFeeBps", 5.0)) or 0)
    slippage_bps = float(config.get("slippageBpsPerSide", 2.0) or 0) * 2
    funding_cost_pct = max(0.0, float(config.get("estimatedFundingCostPct", 0.0) or 0))
    # Bid/ask labeling already captures spread. Add only costs absent from prices.
    return max(0.0, (entry_fee_bps + exit_fee_bps + slippage_bps) / 100 + funding_cost_pct)


def _entry_price(alt: MovementFeatures, side: str) -> float:
    if side == "LONG":
        return alt.ask or alt.last_price
    if side == "SHORT":
        return alt.bid or alt.last_price
    return alt.last_price


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
    entry_price = float(_entry_price(alt, side) or 0)
    margin = float(config.get("marginPerTrade", 1.0) or 1.0)
    leverage = float(config.get("leverage", 1.0) or 1.0)
    estimated_cost_pct = _estimated_cost_pct(alt, config)
    context_key = build_context_key(alt, btc)
    target_moves = {
        str(t): _target_move_pct(t, margin, leverage, estimated_cost_pct)
        for t in TARGETS_USDT
    }
    target_moves["configured"] = max(0.0, float(config.get("takeProfitPct", 0.15) or 0.15))
    stop_move_pct = max(0.0, float(config.get("stopLossPct", 0.10) or 0.10))
    created_bucket = int(time.time() // int(config.get("signalDedupeSeconds", 30) or 30))
    signal_id = _signal_id(symbol, side, str(sniper.get("decision", "")), created_bucket, context_key)
    decision = str(sniper.get("decision", "WAIT"))
    source_type = str(config.get("signalSourceType", "hypothetical")).lower()
    features = {
        "alt": sniper.get("altFeatures", {}),
        "btc": sniper.get("btcFeatures", {}),
        "alt_timeframes": sniper.get("altTimeframes", {}),
        "btc_timeframes": sniper.get("btcTimeframes", {}),
        "target_moves_pct": target_moves,
        "target_probabilities": sniper.get("targetProbabilities", {}),
        "estimated_cost_pct": estimated_cost_pct,
        "strategy_version": STRATEGY_VERSION,
        "stop_move_pct": stop_move_pct,
    }
    recorded = await kb.record_signal_decision(
        signal_id=signal_id,
        symbol=symbol,
        side=side,
        decision=decision,
        decision_group=_decision_group(decision),
        source_type=source_type,
        strategy_version=STRATEGY_VERSION,
        config_hash=_config_hash(config),
        context_key=context_key,
        features=features,
        reasons=list(sniper.get("reasons", [])),
        entry_price=entry_price,
        estimated_cost_pct=estimated_cost_pct,
        target_moves=target_moves,
    )
    return {
        "recorded": recorded,
        "signalId": signal_id,
        "contextKey": context_key,
        "side": side,
        "entryPrice": entry_price,
        "targetMovesPct": target_moves,
        "estimatedCostPct": round(estimated_cost_pct, 6),
        "decisionGroup": _decision_group(decision),
        "sourceType": source_type,
        "strategyVersion": STRATEGY_VERSION,
    }


def _nearest_price(history: list[dict], target_ts: float) -> float | None:
    if not history:
        return None
    best = min(history, key=lambda x: abs(float(x.get("timestamp", 0)) - target_ts))
    if abs(float(best.get("timestamp", 0)) - target_ts) > PRICE_TOLERANCE_SECONDS:
        return None
    price = float(best.get("price", 0) or 0)
    return price if price > 0 else None


def _executable_price(snapshot: dict, side: str) -> float:
    if side == "LONG":
        return float(snapshot.get("bid", snapshot.get("price", 0)) or 0)
    return float(snapshot.get("ask", snapshot.get("price", 0)) or 0)


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
        window_end = created_at + OUTCOME_WINDOW_SECONDS
        future = [
            h for h in history
            if created_at <= float(h.get("timestamp", 0)) <= window_end
        ]
        if not future:
            persisted = await kb.get_feature_history(symbol, hours=2)
            future = [
                h for h in persisted
                if created_at <= float(h.get("timestamp", 0)) <= window_end
            ]
        if entry <= 0 or not future:
            continue
        prices = {}
        for sec in OUTCOME_SECONDS:
            price = _nearest_price(future, created_at + sec)
            if price is not None:
                prices[str(sec)] = price
        ordered = sorted(future, key=lambda h: float(h.get("timestamp", 0)))
        moves = [_directional_move_pct(entry, _executable_price(h, side), side) for h in ordered]
        max_favorable = max(moves) if moves else 0.0
        max_adverse = min(moves) if moves else 0.0
        stop_move_pct = float(
            signal["features"].get("stop_move_pct")
            or max(float(signal["target_configured_move_pct"] or 0), 0.15)
        )
        target_thresholds = {
            "configured": float(signal["target_configured_move_pct"] or 0),
            "0.5": float(signal["target_050_move_pct"] or 0),
            "1.0": float(signal["target_100_move_pct"] or 0),
            "2.0": float(signal["target_200_move_pct"] or 0),
        }
        stop_time = next(
            (
                float(item.get("timestamp", 0))
                for item, move in zip(ordered, moves)
                if move <= -stop_move_pct
            ),
            None,
        )
        target_times = {
            target: next(
                (
                    float(item.get("timestamp", 0))
                    for item, move in zip(ordered, moves)
                    if move >= threshold
                ),
                None,
            )
            for target, threshold in target_thresholds.items()
        }
        hits = {
            target: hit_time is not None and (stop_time is None or hit_time <= stop_time)
            for target, hit_time in target_times.items()
        }
        configured_target_time = target_times["configured"]
        stopped = stop_time is not None and (
            configured_target_time is None or stop_time < configured_target_time
        )
        if stopped:
            first_event = "STOP"
            first_event_time = stop_time
        elif configured_target_time is not None:
            first_event = "TARGET_CONFIGURED"
            first_event_time = configured_target_time
        else:
            first_event = "TIMEOUT"
            first_event_time = window_end
        await kb.finalize_signal_outcome(
            signal_id=str(signal["signal_id"]),
            prices=prices,
            hits=hits,
            stopped=stopped,
            first_event=first_event,
            first_event_seconds=round(max(0.0, first_event_time - created_at), 3),
            max_favorable_pct=round(max_favorable, 4),
            max_adverse_pct=round(max_adverse, 4),
        )
        finalized += 1
    return {"pending": len(pending), "finalized": finalized}


async def score_signal_context(
    symbol: str,
    side: str,
    context_key: str,
    decision_group: str = "ALLOW",
    source_type: str = "hypothetical",
) -> dict[str, Any]:
    context_stats = await kb.get_signal_edge_stats(
        symbol=symbol,
        side=side,
        context_key=context_key,
        decision_group=decision_group,
        source_type=source_type,
    )
    symbol_stats = await kb.get_signal_edge_stats(
        symbol=symbol,
        side=side,
        context_key=None,
        decision_group=decision_group,
        source_type=source_type,
    )
    effective = context_stats if context_stats["samples"] >= MIN_CONTEXT_SAMPLES else symbol_stats
    samples = effective["samples"]
    if samples == 0:
        score = 0.5
        verdict = "cold_start"
    else:
        hit = effective["hit_configured"]
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
        "decisionGroup": decision_group,
        "sourceType": source_type,
    }
