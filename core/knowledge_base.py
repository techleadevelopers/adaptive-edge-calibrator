"""
Knowledge Base — SQLite persistente para padrões, observações e memória do sistema.
Acumula aprendizado 24h/dia sobre os 10 ativos.
"""
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

CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol);
CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(name);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trade_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_symbol ON observations(symbol);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts ON feature_snapshots(symbol, timestamp);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)
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
