from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.movement_sniper import evaluate_sniper_window
from core import knowledge_base as kb
from core.recommendation import recommend_entry
from core.signal_learning import (
    finalize_due_signal_outcomes,
    record_signal_from_gate,
    score_signal_context,
)
from core.shadow_model import predict_shadow
from layers.tactical import get_snapshot_history


def _target_moves(config: dict[str, Any]) -> dict[str, float]:
    margin = max(_num(config.get("marginPerTrade"), 1.0), 0.01)
    leverage = max(_num(config.get("leverage"), 1.0), 1.0)
    notional = margin * leverage
    fees_bps = (
        _num(config.get("entryFeeBps", config.get("takerFeeBps")), 5.0)
        + _num(config.get("exitFeeBps", config.get("takerFeeBps")), 5.0)
        + 2 * _num(config.get("slippageBpsPerSide"), 2.0)
    )
    costs_pct = fees_bps / 100 + max(0.0, _num(config.get("estimatedFundingCostPct"), 0.0))
    targets = {
        str(target): target / notional * 100 + costs_pct
        for target in (0.5, 1.0, 2.0)
    }
    targets["configured"] = max(0.0, _num(config.get("takeProfitPct"), 0.15))
    return targets


def _num(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except Exception:
        return fallback


def _bool(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def _position_to_side(position_side: str | None, side: str | None) -> str:
    raw = (position_side or side or "").upper()
    if raw in {"LONG", "BUY"}:
        return "LONG"
    if raw in {"SHORT", "SELL"}:
        return "SHORT"
    return raw or "UNKNOWN"


def _btc_regime_from_change(change_pct: float, threshold: float) -> str:
    if abs(change_pct) < threshold:
        return "NEUTRAL"
    return "BULL" if change_pct > 0 else "BEAR"


async def evaluate_edge_gate(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol", "")).upper()
    if symbol and not symbol.endswith("-USDT"):
        symbol = f"{symbol}-USDT"
    position_side = _position_to_side(payload.get("positionSide"), payload.get("side"))
    side = str(payload.get("side", "")).upper()
    now_hour = datetime.now(timezone.utc).hour
    hour_utc = int(payload.get("hourUtc", payload.get("hour_utc", now_hour)))
    config = payload.get("config") or {}
    gate_rejects: list[str] = []

    allowed_symbols = _list(config.get("allowedSymbols"))
    if allowed_symbols and symbol not in allowed_symbols:
        gate_rejects.append(f"SYMBOL_REJECT: {symbol} not in allowlist")

    hour_blacklist = [int(x) for x in _list(config.get("hourBlacklist"))]
    if hour_utc in hour_blacklist:
        gate_rejects.append(f"HOUR_REJECT: UTC hour {hour_utc} is blacklisted")

    btc_threshold = _num(config.get("btcRegimeThresholdPct"), 0.5)
    btc_change_pct = _num(payload.get("btcChangePct"), 0.0)
    btc_regime = str(payload.get("btcRegime") or _btc_regime_from_change(btc_change_pct, btc_threshold))
    btc_regime_required = _bool(config.get("btcRegimeRequired"), False)
    allow_counter = _bool(config.get("allowCounterRegimeScalp"), True)

    if btc_regime_required:
        if btc_regime == "NEUTRAL":
            gate_rejects.append(
                f"REGIME_REJECT: BTC change {btc_change_pct:.2f}% < threshold +/-{btc_threshold}%"
            )
        elif not allow_counter:
            want_long = position_side == "LONG"
            btc_bull = btc_regime == "BULL"
            if btc_bull != want_long:
                gate_rejects.append(f"REGIME_DIRECTION: BTC {btc_regime} but entry is {position_side}")

    current_ev = payload.get("currentEv")
    ev_threshold = _num(config.get("evMinThreshold"), 0.0)
    if current_ev is not None and ev_threshold > 0 and _num(current_ev) < ev_threshold:
        gate_rejects.append(f"EV_REJECT: EV {_num(current_ev):.4f} < threshold {ev_threshold:.4f}")

    current_wr = payload.get("currentWinRate")
    wr_min = _num(config.get("winRateMin"), 0.0)
    if current_wr is not None and wr_min > 0 and _num(current_wr) < wr_min:
        gate_rejects.append(f"WR_REJECT: WR {_num(current_wr) * 100:.1f}% < min {wr_min * 100:.1f}%")

    current_pf = payload.get("currentProfitFactor")
    pf_min = _num(config.get("profitFactorMin"), 0.0)
    if current_pf is not None and pf_min > 0 and _num(current_pf) < pf_min:
        gate_rejects.append(f"PF_REJECT: PF {_num(current_pf):.2f}x < min {pf_min:.2f}x")

    alt_history = get_snapshot_history(symbol, 900)
    btc_history = get_snapshot_history("BTC-USDT", 900)
    target_moves_pct = _target_moves(config)
    sniper = evaluate_sniper_window(
        symbol,
        alt_history,
        btc_history,
        target_moves_pct=target_moves_pct,
    )
    await finalize_due_signal_outcomes()
    signal_memory = await record_signal_from_gate(symbol, position_side, sniper, config)
    signal_edge = await score_signal_context(symbol, signal_memory["side"], signal_memory["contextKey"])
    news_context = await kb.get_active_news_context(symbol)
    operational_risk = await kb.get_operational_risk_metrics(hours=24)
    data_quality = {
        "alt1m": sniper["altTimeframes"]["1m"],
        "alt5m": sniper["altTimeframes"]["5m"],
        "alt15m": sniper["altTimeframes"]["15m"],
        "btc1m": sniper["btcTimeframes"]["1m"],
        "btc5m": sniper["btcTimeframes"]["5m"],
        "btc15m": sniper["btcTimeframes"]["15m"],
    }
    if any(frame["quality"] == "STALE" for frame in data_quality.values()):
        gate_rejects.append("DATA_STALE_REJECT: market snapshots are stale")
    if any(frame["quality"] == "GAPPED" for frame in data_quality.values()):
        gate_rejects.append("DATA_GAP_REJECT: snapshot continuity is degraded")
    if _bool(config.get("requireFull15mContext"), True):
        if (
            data_quality["alt15m"]["coveragePct"] < 0.8
            or data_quality["btc15m"]["coveragePct"] < 0.8
        ):
            gate_rejects.append("DATA_15M_REJECT: insufficient 15-minute context")

    max_daily_loss_pct = _num(config.get("maxDailyLossPct"), 0.0)
    max_drawdown_pct = _num(config.get("maxDrawdownPct"), 0.0)
    max_consecutive_losses = int(_num(config.get("maxConsecutiveLosses"), 0))
    if max_daily_loss_pct > 0 and operational_risk["netPnlPct"] <= -max_daily_loss_pct:
        gate_rejects.append("DAILY_LOSS_KILL_SWITCH: daily loss limit reached")
    if max_drawdown_pct > 0 and operational_risk["maxDrawdownPct"] >= max_drawdown_pct:
        gate_rejects.append("DRAWDOWN_KILL_SWITCH: drawdown limit reached")
    if (
        max_consecutive_losses > 0
        and operational_risk["consecutiveLosses"] >= max_consecutive_losses
    ):
        gate_rejects.append("LOSS_STREAK_KILL_SWITCH: consecutive loss limit reached")

    if sniper["decision"].startswith("BLOCK_"):
        gate_rejects.append(f"SNIPER_{sniper['decision']}: {','.join(sniper['reasons'])}")
    elif sniper["decision"] == "WAIT":
        gate_rejects.append(f"SNIPER_WAIT: {','.join(sniper['reasons'])}")

    if signal_edge["verdict"] == "toxic_context":
        gate_rejects.append(
            "SIGNAL_EDGE_REJECT: target-hit context degraded "
            f"(score {signal_edge['score']:.4f})"
        )

    margin = max(_num(config.get("marginPerTrade"), 1.0), 0.01)
    leverage = max(_num(config.get("leverage"), 1.0), 1.0)
    cost_pct = float(signal_memory.get("estimatedCostPct", 0.0))
    configured_target_pct = float(signal_memory["targetMovesPct"].get("configured", 0.0))
    net_target_pct = configured_target_pct - cost_pct
    min_edge_over_cost_pct = _num(config.get("minEdgeOverCostPct"), 0.03)
    if net_target_pct <= min_edge_over_cost_pct:
        gate_rejects.append(
            "COST_EDGE_REJECT: target movement does not clear execution costs and noise"
        )
    effective_stats = (
        signal_edge["context"]
        if signal_edge["context"]["samples"] >= signal_edge["minSamples"]
        else signal_edge["symbolSide"]
    )
    hit_probability = float(effective_stats.get("hit_configured", 0.0))
    stop_move_pct = max(
        _num(config.get("stopMovePct", config.get("stopLossPct")), 0.0),
        0.15,
    )
    notional = margin * leverage
    net_target_usdt = max(0.0, net_target_pct) / 100 * notional
    loss_usdt = (stop_move_pct + cost_pct) / 100 * notional
    net_ev_usdt = hit_probability * net_target_usdt - (1 - hit_probability) * loss_usdt
    if effective_stats.get("samples", 0) >= signal_edge["minSamples"] and net_ev_usdt <= 0:
        gate_rejects.append(f"NET_EV_REJECT: expected value {net_ev_usdt:.4f} USDT")
    shadow_ml = predict_shadow({
        "symbol": symbol,
        "side": signal_memory["side"],
        "context_key": signal_memory["contextKey"],
        "target_configured_move_pct": target_moves_pct["configured"],
        "estimated_cost_pct": cost_pct,
        "features": {
            "alt": sniper["altFeatures"],
            "btc": sniper["btcFeatures"],
            "alt_timeframes": sniper["altTimeframes"],
        },
    })

    if news_context["action"] == "block":
        gate_rejects.append("NEWS_RISK_REJECT: active high-impact event blocks entries")
    elif news_context["action"] == "reduce_aggression" and signal_edge["score"] < 0.72:
        gate_rejects.append("NEWS_RISK_REDUCE: news risk requires stronger target-hit edge")

    recommendation = await recommend_entry({
        "symbol": symbol,
        "position_side": position_side,
        "btc_regime": btc_regime,
        "hour_utc": hour_utc,
        "shadow_only": True,
    })

    # Realized-PnL recommendation is authoritative only after samples exist.
    stats = recommendation.get("stats", {}).get("symbolSide", {})
    samples = int(stats.get("samples", 0) or 0)
    if samples >= recommendation.get("minSamplesForLiveGate", 8) and not recommendation.get("shadowRecommendation"):
        gate_rejects.append(f"REALIZED_EDGE_REJECT: score {recommendation.get('score', 0):.4f}")

    allow = len(gate_rejects) == 0
    score = min(
        float(sniper.get("score", 0.0)),
        float(recommendation.get("score", 0.5)),
        float(signal_edge.get("score", 0.5)),
    )
    if news_context["action"] == "reduce_aggression":
        score = min(score, max(0.0, score - 0.08))
    if samples < recommendation.get("minSamplesForLiveGate", 8):
        score = min(float(sniper.get("score", 0.0)), float(signal_edge.get("score", 0.5)))

    return {
        "allow": allow,
        "gateRejects": gate_rejects,
        "score": round(score, 4),
        "authority": "quant-brain",
        "symbol": symbol,
        "side": side,
        "positionSide": position_side,
        "hourUtc": hour_utc,
        "btcRegime": btc_regime,
        "sniper": sniper,
        "signalMemory": signal_memory,
        "signalEdge": signal_edge,
        "economics": {
            "targetMovesPct": target_moves_pct,
            "estimatedCostPct": round(cost_pct, 6),
            "hitProbability": round(hit_probability, 4),
            "estimatedLossUsdt": round(loss_usdt, 4),
            "estimatedNetTargetUsdt": round(net_target_usdt, 4),
            "netEvUsdt": round(net_ev_usdt, 4),
        },
        "newsContext": news_context,
        "dataQuality": data_quality,
        "operationalRisk": operational_risk,
        "shadowMl": shadow_ml,
        "realizedEdge": recommendation,
        "mode": "movement_first_realized_pnl_auditor",
    }
