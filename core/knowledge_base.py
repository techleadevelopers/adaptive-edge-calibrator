"""
Knowledge Base — SQLite persistente para padrões, observações e memória do sistema.
Acumula aprendizado 24h/dia sobre os 10 ativos.
"""
from __future__ import annotations

import json
import time
import aiosqlite
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "knowledge.db"
DB_PATH.parent.mkdir(exist_ok=True)


@dataclass
class Pattern:
    id: Optional[int]
    name: str
    symbol: str
    conditions: dict
    occurrences: int
    wins: int
    total_return: float
    avg_return: float
    win_rate: float
    last_seen: float
    created_at: float


@dataclass
class Observation:
    id: Optional[int]
    symbol: str
    category: str
    text: str
    data: dict
    confidence: float
    timestamp: float


@dataclass
class StrategicInsight:
    id: Optional[int]
    period_days: int
    generated_at: float
    analysis_text: str
    edge_changes: dict
    recommendations: list


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    conditions TEXT NOT NULL,
    occurrences INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    total_return REAL DEFAULT 0.0,
    avg_return REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    last_seen REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL,
    exit_price REAL,
    pnl_pct REAL,
    win INTEGER DEFAULT 0,
    oi_at_entry REAL,
    funding_at_entry REAL,
    volume_ratio REAL,
    btc_regime TEXT,
    rsi_at_entry REAL,
    ema_cross TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    category TEXT NOT NULL,
    text TEXT NOT NULL,
    data TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS strategic_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_days INTEGER NOT NULL,
    generated_at REAL NOT NULL,
    analysis_text TEXT NOT NULL,
    edge_changes TEXT NOT NULL,
    recommendations TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp REAL NOT NULL,
    price REAL,
    price_change_pct REAL,
    volume_ratio REAL,
    oi_change_pct REAL,
    funding_rate REAL,
    rsi REAL,
    ema_cross TEXT,
    atr_pct REAL,
    spread_bps REAL,
    btc_regime TEXT
);

CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    decision TEXT NOT NULL,
    decision_group TEXT NOT NULL DEFAULT 'WAIT',
    source_type TEXT NOT NULL DEFAULT 'hypothetical',
    strategy_version TEXT NOT NULL DEFAULT 'legacy',
    config_hash TEXT NOT NULL DEFAULT '',
    context_key TEXT NOT NULL,
    features TEXT NOT NULL,
    reasons TEXT NOT NULL,
    entry_price REAL NOT NULL,
    estimated_cost_pct REAL NOT NULL DEFAULT 0,
    target_configured_move_pct REAL,
    target_050_move_pct REAL NOT NULL,
    target_100_move_pct REAL NOT NULL,
    target_200_move_pct REAL NOT NULL,
    price_30s REAL,
    price_60s REAL,
    price_120s REAL,
    price_300s REAL,
    hit_configured INTEGER,
    hit_050 INTEGER,
    hit_100 INTEGER,
    hit_200 INTEGER,
    stopped INTEGER,
    first_event TEXT,
    first_event_seconds REAL,
    max_favorable_pct REAL,
    max_adverse_pct REAL,
    finalized INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    finalized_at REAL
);

CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    symbols TEXT NOT NULL,
    category TEXT NOT NULL,
    impact_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    action TEXT NOT NULL,
    raw TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol);
CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(name);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trade_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_symbol ON observations(symbol);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts ON feature_snapshots(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_context ON signal_outcomes(context_key, side);
CREATE INDEX IF NOT EXISTS idx_signal_symbol_ts ON signal_outcomes(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_signal_finalized ON signal_outcomes(finalized, created_at);
CREATE INDEX IF NOT EXISTS idx_news_expires ON news_events(expires_at);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)
        columns = {
            row[1] for row in await (await db.execute("PRAGMA table_info(signal_outcomes)")).fetchall()
        }
        migrations = {
            "decision_group": "TEXT NOT NULL DEFAULT 'WAIT'",
            "source_type": "TEXT NOT NULL DEFAULT 'hypothetical'",
            "strategy_version": "TEXT NOT NULL DEFAULT 'legacy'",
            "config_hash": "TEXT NOT NULL DEFAULT ''",
            "estimated_cost_pct": "REAL NOT NULL DEFAULT 0",
            "target_configured_move_pct": "REAL",
            "hit_configured": "INTEGER",
            "first_event": "TEXT",
            "first_event_seconds": "REAL",
        }
        for name, definition in migrations.items():
            if name not in columns:
                await db.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {name} {definition}")
        await db.commit()


async def record_trade_outcome(
    symbol: str, side: str, pnl_pct: float,
    entry_price: float = 0.0, exit_price: float = 0.0,
    oi_change: float = 0.0, funding: float = 0.0,
    volume_ratio: float = 1.0, btc_regime: str = "NEUTRAL",
    rsi: float = 50.0, ema_cross: str = "FLAT"
):
    win = 1 if pnl_pct > 0 else 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO trade_outcomes
               (symbol, side, entry_price, exit_price, pnl_pct, win,
                oi_at_entry, funding_at_entry, volume_ratio, btc_regime,
                rsi_at_entry, ema_cross, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, side, entry_price, exit_price, pnl_pct, win,
             oi_change, funding, volume_ratio, btc_regime,
             rsi, ema_cross, time.time())
        )
        await db.commit()


async def save_feature_snapshot(symbol: str, features: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO feature_snapshots
               (symbol, timestamp, price, price_change_pct, volume_ratio,
                oi_change_pct, funding_rate, rsi, ema_cross, atr_pct,
                spread_bps, btc_regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol, time.time(),
                features.get("price", 0),
                features.get("price_change_pct", 0),
                features.get("volume_ratio", 1),
                features.get("oi_change_pct", 0),
                features.get("funding_rate", 0),
                features.get("rsi", 50),
                features.get("ema_cross", "FLAT"),
                features.get("atr_pct", 0),
                features.get("spread_bps", 0),
                features.get("btc_regime", "NEUTRAL"),
            )
        )
        await db.commit()


async def record_signal_decision(
    signal_id: str,
    symbol: str,
    side: str,
    decision: str,
    decision_group: str,
    source_type: str,
    strategy_version: str,
    config_hash: str,
    context_key: str,
    features: dict,
    reasons: list,
    entry_price: float,
    estimated_cost_pct: float,
    target_moves: dict[str, float],
) -> bool:
    if entry_price <= 0:
        return False
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                """INSERT INTO signal_outcomes
                   (signal_id, symbol, side, decision, decision_group, source_type,
                    strategy_version, config_hash, context_key, features, reasons,
                    entry_price, estimated_cost_pct,
                    target_configured_move_pct, target_050_move_pct,
                    target_100_move_pct, target_200_move_pct,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    signal_id,
                    symbol,
                    side,
                    decision,
                    decision_group,
                    source_type,
                    strategy_version,
                    config_hash,
                    context_key,
                    json.dumps(features),
                    json.dumps(reasons),
                    entry_price,
                    estimated_cost_pct,
                    float(target_moves.get("configured", 0)),
                    float(target_moves.get("0.5", 0)),
                    float(target_moves.get("1.0", 0)),
                    float(target_moves.get("2.0", 0)),
                    time.time(),
                ),
            )
            await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False


async def get_pending_signal_outcomes(min_age_seconds: int = 300, limit: int = 200) -> list[dict]:
    cutoff = time.time() - min_age_seconds
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT * FROM signal_outcomes
               WHERE finalized=0 AND created_at <= ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (cutoff, limit),
        )).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d["features"])
            d["reasons"] = json.loads(d["reasons"])
            result.append(d)
        return result


async def finalize_signal_outcome(
    signal_id: str,
    prices: dict[str, float],
    hits: dict[str, bool],
    stopped: bool,
    first_event: str | None,
    first_event_seconds: float | None,
    max_favorable_pct: float,
    max_adverse_pct: float,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE signal_outcomes
               SET price_30s=?, price_60s=?, price_120s=?, price_300s=?,
                   hit_configured=?, hit_050=?, hit_100=?, hit_200=?, stopped=?,
                   first_event=?, first_event_seconds=?,
                   max_favorable_pct=?, max_adverse_pct=?,
                   finalized=1, finalized_at=?
               WHERE signal_id=?""",
            (
                prices.get("30"),
                prices.get("60"),
                prices.get("120"),
                prices.get("300"),
                1 if hits.get("configured") else 0,
                1 if hits.get("0.5") else 0,
                1 if hits.get("1.0") else 0,
                1 if hits.get("2.0") else 0,
                1 if stopped else 0,
                first_event,
                first_event_seconds,
                max_favorable_pct,
                max_adverse_pct,
                time.time(),
                signal_id,
            ),
        )
        await db.commit()


async def get_signal_edge_stats(
    symbol: str,
    side: str,
    context_key: str | None = None,
    decision_group: str = "ALLOW",
    source_type: str = "hypothetical",
    days: int = 14,
) -> dict:
    since = time.time() - days * 86400
    params: list = [side, since, decision_group, source_type]
    where = (
        "WHERE finalized=1 AND side=? AND created_at >= ? "
        "AND decision_group=? AND source_type=? AND hit_configured IS NOT NULL"
    )
    if context_key:
        where += " AND context_key=?"
        params.append(context_key)
    else:
        where += " AND symbol=?"
        params.append(symbol)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute(
            f"""SELECT COUNT(*) as samples,
                       AVG(hit_configured) as hit_configured,
                       AVG(hit_050) as hit_050,
                       AVG(hit_100) as hit_100,
                       AVG(hit_200) as hit_200,
                       AVG(stopped) as stop_rate,
                       AVG(max_favorable_pct) as avg_favorable_pct,
                       AVG(max_adverse_pct) as avg_adverse_pct
                FROM signal_outcomes
                {where}""",
            params,
        )).fetchone()
    d = dict(row) if row else {}
    samples = int(d.get("samples") or 0)
    return {
        "samples": samples,
        "hit_configured": round(float(d.get("hit_configured") or 0), 4),
        "hit_050": round(float(d.get("hit_050") or 0), 4),
        "hit_100": round(float(d.get("hit_100") or 0), 4),
        "hit_200": round(float(d.get("hit_200") or 0), 4),
        "stop_rate": round(float(d.get("stop_rate") or 0), 4),
        "avg_favorable_pct": round(float(d.get("avg_favorable_pct") or 0), 4),
        "avg_adverse_pct": round(float(d.get("avg_adverse_pct") or 0), 4),
    }


async def get_signal_training_rows(
    decision_group: str = "ALLOW",
    source_type: str = "hypothetical",
    limit: int = 50000,
) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT signal_id, symbol, side, decision, context_key, features,
                      target_configured_move_pct, estimated_cost_pct, hit_configured,
                      stopped, first_event, created_at
               FROM signal_outcomes
               WHERE finalized=1 AND decision_group=? AND source_type=?
                 AND hit_configured IS NOT NULL
               ORDER BY created_at ASC
               LIMIT ?""",
            (decision_group, source_type, limit),
        )).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["features"] = json.loads(item["features"])
        result.append(item)
    return result


async def get_operational_risk_metrics(hours: int = 24) -> dict:
    since = time.time() - hours * 3600
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await (await db.execute(
            """SELECT pnl_pct, timestamp
               FROM trade_outcomes
               WHERE timestamp >= ?
               ORDER BY timestamp ASC""",
            (since,),
        )).fetchall()
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    consecutive_losses = 0
    current_losses = 0
    for pnl_pct, _ in rows:
        pnl = float(pnl_pct or 0)
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if pnl < 0:
            current_losses += 1
            consecutive_losses = max(consecutive_losses, current_losses)
        else:
            current_losses = 0
    return {
        "hours": hours,
        "trades": len(rows),
        "netPnlPct": round(cumulative, 6),
        "maxDrawdownPct": round(max_drawdown, 6),
        "consecutiveLosses": consecutive_losses,
    }


async def record_news_event(
    source: str,
    title: str,
    symbols: list[str],
    category: str,
    impact_score: float,
    risk_level: str,
    action: str,
    url: str = "",
    raw: dict | None = None,
    ttl_seconds: int = 7200,
) -> None:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO news_events
               (source, title, url, symbols, category, impact_score, risk_level,
                action, raw, created_at, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                source,
                title,
                url,
                json.dumps(symbols),
                category,
                impact_score,
                risk_level,
                action,
                json.dumps(raw or {}),
                now,
                now + ttl_seconds,
            ),
        )
        await db.commit()


async def get_active_news_context(symbol: str, now: float | None = None) -> dict:
    ts = now or time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT * FROM news_events
               WHERE expires_at >= ?
               ORDER BY ABS(impact_score) DESC, created_at DESC
               LIMIT 100""",
            (ts,),
        )).fetchall()
    matched = []
    symbol_upper = symbol.upper()
    for r in rows:
        d = dict(r)
        symbols = json.loads(d["symbols"])
        if symbol_upper in symbols or "BTC-USDT" in symbols or "MARKET" in symbols:
            d["symbols"] = symbols
            d["raw"] = json.loads(d["raw"])
            matched.append(d)
    if not matched:
        return {
            "active": False,
            "newsImpactScore": 0.0,
            "riskLevel": "LOW",
            "action": "none",
            "events": [],
        }
    score = sum(float(x["impact_score"]) for x in matched[:5]) / min(5, len(matched))
    high_risk = any(str(x["risk_level"]).upper() == "HIGH" for x in matched)
    reduce = any(str(x["action"]).lower() in {"block", "reduce_aggression"} for x in matched)
    return {
        "active": True,
        "newsImpactScore": round(score, 4),
        "riskLevel": "HIGH" if high_risk else "MEDIUM" if abs(score) >= 0.35 else "LOW",
        "action": "block" if any(str(x["action"]).lower() == "block" for x in matched) else "reduce_aggression" if reduce else "context_only",
        "events": matched[:10],
    }


async def upsert_pattern(
    name: str, symbol: str, conditions: dict,
    won: bool, pnl_pct: float
):
    async with aiosqlite.connect(DB_PATH) as db:
        row = await (await db.execute(
            "SELECT id, occurrences, wins, total_return FROM patterns WHERE name=? AND symbol=?",
            (name, symbol)
        )).fetchone()

        if row:
            pid, occ, wins, total_ret = row
            occ += 1
            wins += 1 if won else 0
            total_ret += pnl_pct
            avg_ret = total_ret / occ
            wr = wins / occ
            await db.execute(
                """UPDATE patterns SET occurrences=?, wins=?, total_return=?,
                   avg_return=?, win_rate=?, last_seen=?, conditions=?
                   WHERE id=?""",
                (occ, wins, total_ret, avg_ret, wr, time.time(),
                 json.dumps(conditions), pid)
            )
        else:
            wr = 1.0 if won else 0.0
            await db.execute(
                """INSERT INTO patterns
                   (name, symbol, conditions, occurrences, wins, total_return,
                    avg_return, win_rate, last_seen, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (name, symbol, json.dumps(conditions), 1,
                 1 if won else 0, pnl_pct, pnl_pct, wr,
                 time.time(), time.time())
            )
        await db.commit()


async def get_top_patterns(min_occurrences: int = 5, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT * FROM patterns
               WHERE occurrences >= ?
               ORDER BY win_rate DESC, avg_return DESC
               LIMIT ?""",
            (min_occurrences, limit)
        )).fetchall()
        return [dict(r) for r in rows]


async def get_symbol_stats(symbol: str, days: int = 30) -> dict:
    since = time.time() - days * 86400
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT side, COUNT(*) as trades,
               SUM(win) as wins,
               AVG(pnl_pct) as avg_pnl,
               SUM(pnl_pct) as total_pnl,
               MIN(pnl_pct) as worst,
               MAX(pnl_pct) as best
               FROM trade_outcomes
               WHERE symbol=? AND timestamp >= ?
               GROUP BY side""",
            (symbol, since)
        )).fetchall()
        result = {"symbol": symbol, "days": days, "sides": {}}
        for r in rows:
            d = dict(r)
            wr = d["wins"] / d["trades"] if d["trades"] else 0
            result["sides"][d["side"]] = {
                "trades": d["trades"],
                "win_rate": round(wr * 100, 1),
                "avg_pnl": round(d["avg_pnl"] or 0, 4),
                "total_pnl": round(d["total_pnl"] or 0, 4),
                "worst": round(d["worst"] or 0, 4),
                "best": round(d["best"] or 0, 4),
            }
        return result


async def get_all_symbols_stats(days: int = 30) -> list[dict]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "VVVUSDT", "TRUMPUSDT",
               "MELANIAUSDT", "BEATUSDT", "NEARUSDT", "HYPEUSDT", "POLUSDT"]
    return [await get_symbol_stats(s, days) for s in symbols]


async def save_observation(symbol: str, category: str, text: str, data: dict, confidence: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO observations (symbol, category, text, data, confidence, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (symbol, category, text, json.dumps(data), confidence, time.time())
        )
        await db.commit()


async def save_strategic_insight(
    period_days: int, analysis_text: str,
    edge_changes: dict, recommendations: list
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO strategic_insights
               (period_days, generated_at, analysis_text, edge_changes, recommendations)
               VALUES (?,?,?,?,?)""",
            (period_days, time.time(), analysis_text,
             json.dumps(edge_changes), json.dumps(recommendations))
        )
        await db.commit()


async def get_recent_insights(limit: int = 5) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            "SELECT * FROM strategic_insights ORDER BY generated_at DESC LIMIT ?",
            (limit,)
        )).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["edge_changes"] = json.loads(d["edge_changes"])
            d["recommendations"] = json.loads(d["recommendations"])
            result.append(d)
        return result


async def get_feature_history(symbol: str, hours: int = 24) -> list[dict]:
    since = time.time() - hours * 3600
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(
            """SELECT * FROM feature_snapshots
               WHERE symbol=? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (symbol, since)
        )).fetchall()
        return [dict(r) for r in rows]


async def get_recent_observations(symbol: str = None, hours: int = 48, limit: int = 50) -> list[dict]:
    since = time.time() - hours * 3600
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            rows = await (await db.execute(
                """SELECT * FROM observations
                   WHERE symbol=? AND timestamp >= ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, since, limit)
            )).fetchall()
        else:
            rows = await (await db.execute(
                """SELECT * FROM observations
                   WHERE timestamp >= ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (since, limit)
            )).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            result.append(d)
        return result
