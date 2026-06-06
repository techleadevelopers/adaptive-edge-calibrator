import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from core.knowledge_base import DB_PATH


MIN_SAMPLES_FOR_BLOCK = 8
MIN_SCORE_TO_ALLOW = 0.58


@dataclass
class TradeRow:
    symbol: str
    side: str
    pnl_pct: float
    win: int
    btc_regime: str
    timestamp: float

    @property
    def hour_utc(self) -> int:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).hour


def _normalize_symbol(symbol: str) -> str:
    sym = symbol.upper().strip()
    if "-" in sym:
        return sym
    if sym.endswith("USDT"):
        return f"{sym[:-4]}-USDT"
    return sym


def _symbol_variants(symbol: str) -> tuple[str, str]:
    normalized = _normalize_symbol(symbol)
    compact = normalized.replace("-", "")
    return normalized, compact


def _position_to_side(position_side: str | None, side: str | None) -> str:
    raw = (position_side or side or "").upper()
    if raw in {"LONG", "BUY"}:
        return "LONG"
    if raw in {"SHORT", "SELL"}:
        return "SHORT"
    return raw or "UNKNOWN"


def _profit_factor(rows: list[TradeRow]) -> float:
    gross_win = sum(r.pnl_pct for r in rows if r.pnl_pct > 0)
    gross_loss = abs(sum(r.pnl_pct for r in rows if r.pnl_pct < 0))
    if gross_loss == 0:
        return 999.0 if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _stats(rows: list[TradeRow]) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    wins = [r.pnl_pct for r in rows if r.pnl_pct > 0]
    losses = [r.pnl_pct for r in rows if r.pnl_pct <= 0]
    total_pnl = sum(r.pnl_pct for r in rows)
    return {
        "samples": len(rows),
        "win_rate": len(wins) / len(rows),
        "avg_pnl": total_pnl / len(rows),
        "total_pnl": total_pnl,
        "profit_factor": _profit_factor(rows),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score(stats: dict[str, Any], recent_stats: dict[str, Any]) -> float:
    samples = stats["samples"]
    if samples == 0:
        return 0.5
    confidence = _clamp(samples / 30)
    wr_score = stats["win_rate"]
    avg_pnl_score = _clamp((stats["avg_pnl"] + 0.4) / 0.8)
    pf_score = _clamp(stats["profit_factor"] / 2.0)
    recent_bonus = _clamp((recent_stats["avg_pnl"] - stats["avg_pnl"] + 0.2) / 0.4) if recent_stats["samples"] else 0.5
    raw = (wr_score * 0.38) + (avg_pnl_score * 0.28) + (pf_score * 0.22) + (recent_bonus * 0.12)
    return round((raw * confidence) + (0.5 * (1 - confidence)), 4)


def _risk_bucket(score: float, samples: int) -> str:
    if samples < MIN_SAMPLES_FOR_BLOCK:
        return "shadow_only"
    if score >= 0.78:
        return "aggressive"
    if score >= 0.66:
        return "standard"
    if score >= MIN_SCORE_TO_ALLOW:
        return "scout"
    return "reject"


def _suggested_margin(score: float, samples: int) -> float:
    if samples < MIN_SAMPLES_FOR_BLOCK or score < MIN_SCORE_TO_ALLOW:
        return 0.0
    if score >= 0.78:
        return 2.0
    if score >= 0.66:
        return 1.0
    return 0.5


async def _load_trades(days: int = 30) -> list[TradeRow]:
    since = time.time() - days * 86400
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT symbol, side, pnl_pct, win, btc_regime, timestamp
               FROM trade_outcomes
               WHERE timestamp >= ?
               ORDER BY timestamp ASC""",
            (since,),
        )).fetchall()
    return [
        TradeRow(
            symbol=str(r["symbol"]),
            side=_position_to_side(None, str(r["side"])),
            pnl_pct=float(r["pnl_pct"] or 0),
            win=int(r["win"] or 0),
            btc_regime=str(r["btc_regime"] or "NEUTRAL"),
            timestamp=float(r["timestamp"] or 0),
        )
        for r in rows
    ]


async def recommend_entry(payload: dict[str, Any], days: int = 30) -> dict[str, Any]:
    symbol = _normalize_symbol(str(payload.get("symbol", "")))
    side = _position_to_side(payload.get("position_side"), payload.get("side"))
    btc_regime = str(payload.get("btc_regime", "NEUTRAL")).upper()
    hour_utc = int(payload.get("hour_utc", datetime.now(timezone.utc).hour))
    shadow_only = bool(payload.get("shadow_only", True))
    variants = set(_symbol_variants(symbol))

    trades = await _load_trades(days)
    symbol_rows = [r for r in trades if r.symbol in variants and r.side == side]
    cluster_rows = [r for r in symbol_rows if r.btc_regime == btc_regime and r.hour_utc == hour_utc]
    regime_rows = [r for r in symbol_rows if r.btc_regime == btc_regime]
    hour_rows = [r for r in symbol_rows if r.hour_utc == hour_utc]
    recent_rows = symbol_rows[-20:]

    base_stats = _stats(symbol_rows)
    cluster_stats = _stats(cluster_rows)
    regime_stats = _stats(regime_rows)
    hour_stats = _stats(hour_rows)
    recent_stats = _stats(recent_rows)

    effective_stats = cluster_stats if cluster_stats["samples"] >= MIN_SAMPLES_FOR_BLOCK else base_stats
    score = _score(effective_stats, recent_stats)
    risk = _risk_bucket(score, effective_stats["samples"])
    allow = score >= MIN_SCORE_TO_ALLOW and effective_stats["samples"] >= MIN_SAMPLES_FOR_BLOCK

    reasons: list[str] = []
    if effective_stats["samples"] < MIN_SAMPLES_FOR_BLOCK:
        reasons.append("insufficient_realized_samples")
    if base_stats["total_pnl"] < 0 and base_stats["samples"] >= MIN_SAMPLES_FOR_BLOCK:
        reasons.append("negative_symbol_side_total_pnl")
    if hour_stats["samples"] >= MIN_SAMPLES_FOR_BLOCK and hour_stats["avg_pnl"] < 0:
        reasons.append("toxic_hour_for_symbol_side")
    if regime_stats["samples"] >= MIN_SAMPLES_FOR_BLOCK and regime_stats["avg_pnl"] < 0:
        reasons.append("toxic_btc_regime_for_symbol_side")
    if recent_stats["samples"] >= 8 and recent_stats["avg_pnl"] < base_stats["avg_pnl"]:
        reasons.append("recent_edge_drift_down")
    if allow:
        reasons.append("positive_realized_edge_gate")

    return {
        "allow": allow if not shadow_only else False,
        "shadowRecommendation": allow,
        "shadowOnly": shadow_only,
        "score": score,
        "risk": risk,
        "suggestedMarginUsdt": _suggested_margin(score, effective_stats["samples"]),
        "minSamplesForLiveGate": MIN_SAMPLES_FOR_BLOCK,
        "reasons": reasons,
        "context": {
            "symbol": symbol,
            "side": side,
            "btcRegime": btc_regime,
            "hourUtc": hour_utc,
            "days": days,
        },
        "stats": {
            "symbolSide": base_stats,
            "cluster": cluster_stats,
            "regime": regime_stats,
            "hour": hour_stats,
            "recent": recent_stats,
        },
    }


def _apply_gate(rows: list[TradeRow], gate: str, threshold: float) -> tuple[list[TradeRow], list[TradeRow]]:
    grouped: dict[Any, list[TradeRow]] = {}
    for r in rows:
        key = r.symbol if gate == "symbol" else r.hour_utc if gate == "hour" else r.btc_regime
        grouped.setdefault(key, []).append(r)
    stats_by_key = {key: _stats(vals) for key, vals in grouped.items()}
    kept: list[TradeRow] = []
    rejected: list[TradeRow] = []
    for r in rows:
        key = r.symbol if gate == "symbol" else r.hour_utc if gate == "hour" else r.btc_regime
        s = stats_by_key[key]
        should_reject = s["samples"] >= MIN_SAMPLES_FOR_BLOCK and s["avg_pnl"] < threshold
        (rejected if should_reject else kept).append(r)
    return kept, rejected


async def simulate_gate_rejections(days: int = 30, min_avg_pnl: float = 0.0) -> dict[str, Any]:
    rows = await _load_trades(days)
    baseline = _stats(rows)
    simulations = []
    for gate in ("symbol", "hour", "regime"):
        kept, rejected = _apply_gate(rows, gate, min_avg_pnl)
        kept_stats = _stats(kept)
        rejected_stats = _stats(rejected)
        simulations.append({
            "gate": gate,
            "threshold": {"minAvgPnl": min_avg_pnl},
            "keptTrades": kept_stats["samples"],
            "rejectedTrades": rejected_stats["samples"],
            "baselineTotalPnl": baseline["total_pnl"],
            "keptTotalPnl": kept_stats["total_pnl"],
            "rejectedTotalPnl": rejected_stats["total_pnl"],
            "pnlImprovementIfRejected": -rejected_stats["total_pnl"],
            "keptWinRate": kept_stats["win_rate"],
            "rejectedWinRate": rejected_stats["win_rate"],
        })
    simulations.sort(key=lambda x: x["pnlImprovementIfRejected"], reverse=True)
    return {
        "days": days,
        "baseline": baseline,
        "simulations": simulations,
        "note": "Positive pnlImprovementIfRejected means the gate would have removed net losing flow.",
    }
