"""
Knowledge Base — SQLite persistente para padrões, observações e memória do sistema.
Acumula aprendizado 24h/dia sobre os 10 ativos.
Nível Máximo de Excelência: índices otimizados, tabelas de performance,
métricas agregadas, janelas temporais, view materializadas, cache de queries.
"""
from __future__ import annotations

import json
import time
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Any
from pathlib import Path
from core.database import IntegrityError, Row, connect, table_columns

DB_PATH = Path(__file__).parent.parent / "data" / "knowledge.db"
DB_PATH.parent.mkdir(exist_ok=True)

# Cache LRU para consultas frequentes
_query_cache: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = 100
_CACHE_TTL_SECONDS = 30


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
-- ========== TABELAS EXISTENTES (OTIMIZADAS) ==========

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
    pnl_usdt REAL,
    win INTEGER DEFAULT 0,
    oi_at_entry REAL,
    funding_at_entry REAL,
    volume_ratio REAL,
    btc_regime TEXT,
    rsi_at_entry REAL,
    ema_cross TEXT,
    slippage_bps REAL DEFAULT 0,
    fee_paid_usdt REAL DEFAULT 0,
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
    btc_regime TEXT,
    bid_depth_5 REAL,
    ask_depth_5 REAL,
    book_imbalance REAL,
    cvd REAL
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

-- ========== NOVAS TABELAS PARA NÍVEL MÁXIMO DE EXCELÊNCIA ==========

-- Tabela de métricas horárias agregadas para queries rápidas
CREATE TABLE IF NOT EXISTS hourly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    hour_utc INTEGER NOT NULL,
    date TEXT NOT NULL,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    total_pnl_pct REAL DEFAULT 0,
    total_pnl_usdt REAL DEFAULT 0,
    avg_pnl_pct REAL DEFAULT 0,
    win_rate REAL DEFAULT 0,
    UNIQUE(symbol, date, hour_utc)
);

-- Tabela de métricas diárias por símbolo/lado
CREATE TABLE IF NOT EXISTS daily_symbol_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    date TEXT NOT NULL,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    total_pnl_pct REAL DEFAULT 0,
    total_pnl_usdt REAL DEFAULT 0,
    avg_pnl_pct REAL DEFAULT 0,
    win_rate REAL DEFAULT 0,
    max_drawdown_pct REAL DEFAULT 0,
    sharpe_ratio REAL DEFAULT 0,
    UNIQUE(symbol, side, date)
);

-- Tabela de análise de correlação entre símbolos
CREATE TABLE IF NOT EXISTS symbol_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    correlation_1h REAL DEFAULT 0,
    correlation_4h REAL DEFAULT 0,
    correlation_24h REAL DEFAULT 0,
    computed_at REAL NOT NULL,
    UNIQUE(symbol_a, symbol_b)
);

-- Tabela de performance por regime de BTC
CREATE TABLE IF NOT EXISTS regime_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    btc_regime TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    total_pnl_pct REAL DEFAULT 0,
    win_rate REAL DEFAULT 0,
    period_start REAL NOT NULL,
    period_end REAL NOT NULL,
    UNIQUE(btc_regime, symbol, side, period_start)
);

-- Tabela de toxicidade por horário
CREATE TABLE IF NOT EXISTS hour_toxicity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    hour_utc INTEGER NOT NULL,
    side TEXT NOT NULL,
    trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0,
    avg_pnl_pct REAL DEFAULT 0,
    toxicity_score REAL DEFAULT 0,
    updated_at REAL NOT NULL,
    UNIQUE(symbol, hour_utc, side)
);

-- Tabela de rolling edge (janela móvel)
CREATE TABLE IF NOT EXISTS rolling_edge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    window_hours INTEGER NOT NULL,
    win_rate REAL DEFAULT 0,
    avg_pnl_pct REAL DEFAULT 0,
    profit_factor REAL DEFAULT 0,
    computed_at REAL NOT NULL,
    UNIQUE(symbol, side, window_hours, computed_at)
);

-- Tabela de execução quality (slippage real)
CREATE TABLE IF NOT EXISTS execution_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    expected_price REAL NOT NULL,
    executed_price REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    latency_ms REAL NOT NULL,
    timestamp REAL NOT NULL
);

-- ========== ÍNDICES OTIMIZADOS ==========

CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol);
CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(name);
CREATE INDEX IF NOT EXISTS idx_patterns_win_rate ON patterns(win_rate DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trade_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_side_ts ON trade_outcomes(symbol, side, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_btc_regime ON trade_outcomes(btc_regime);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trade_outcomes(pnl_pct);

CREATE INDEX IF NOT EXISTS idx_observations_symbol ON observations(symbol);
CREATE INDEX IF NOT EXISTS idx_observations_category_ts ON observations(category, timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_confidence ON observations(confidence DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts ON feature_snapshots(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_btc_regime ON feature_snapshots(btc_regime);

CREATE INDEX IF NOT EXISTS idx_signal_context ON signal_outcomes(context_key, side);
CREATE INDEX IF NOT EXISTS idx_signal_symbol_ts ON signal_outcomes(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_signal_finalized ON signal_outcomes(finalized, created_at);
CREATE INDEX IF NOT EXISTS idx_news_expires ON news_events(expires_at);
CREATE INDEX IF NOT EXISTS idx_news_impact ON news_events(impact_score DESC);

-- Novos índices
CREATE INDEX IF NOT EXISTS idx_hourly_metrics_symbol ON hourly_metrics(symbol, hour_utc);
CREATE INDEX IF NOT EXISTS idx_hourly_metrics_date ON hourly_metrics(date);

CREATE INDEX IF NOT EXISTS idx_daily_metrics_symbol ON daily_symbol_metrics(symbol, side, date);
CREATE INDEX IF NOT EXISTS idx_daily_metrics_win_rate ON daily_symbol_metrics(win_rate DESC);

CREATE INDEX IF NOT EXISTS idx_regime_performance ON regime_performance(btc_regime, symbol);

CREATE INDEX IF NOT EXISTS idx_hour_toxicity ON hour_toxicity(symbol, hour_utc, toxicity_score DESC);

CREATE INDEX IF NOT EXISTS idx_rolling_edge ON rolling_edge(symbol, side, computed_at);

CREATE INDEX IF NOT EXISTS idx_execution_quality ON execution_quality(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_execution_slippage ON execution_quality(slippage_bps DESC);

-- ========== VIEWS MATERIALIZADAS (via queries otimizadas) ==========
"""


def _get_cache_key(query: str, params: tuple) -> str:
    """Gera chave de cache para query."""
    return f"{query}|{params}"


def _cache_get(query: str, params: tuple) -> Optional[Any]:
    """Recupera do cache se não expirou."""
    key = _get_cache_key(query, params)
    if key in _query_cache:
        value, timestamp = _query_cache[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return value
        del _query_cache[key]
    return None


def _cache_set(query: str, params: tuple, value: Any):
    """Armazena no cache."""
    key = _get_cache_key(query, params)
    _query_cache[key] = (value, time.time())
    while len(_query_cache) > _CACHE_MAX_SIZE:
        _query_cache.popitem(last=False)


async def init_db():
    """Inicializa banco com todas as tabelas e migrações."""
    async with connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)

        # Migrações para tabelas existentes
        columns = await table_columns("signal_outcomes", DB_PATH)
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

        # Migrações para trade_outcomes (campos novos)
        trade_columns = await table_columns("trade_outcomes", DB_PATH)
        trade_migrations = {
            "pnl_usdt": "REAL",
            "slippage_bps": "REAL DEFAULT 0",
            "fee_paid_usdt": "REAL DEFAULT 0",
        }
        for name, definition in trade_migrations.items():
            if name not in trade_columns:
                await db.execute(f"ALTER TABLE trade_outcomes ADD COLUMN {name} {definition}")

        # Migrações para feature_snapshots
        feature_columns = await table_columns("feature_snapshots", DB_PATH)
        feature_migrations = {
            "bid_depth_5": "REAL",
            "ask_depth_5": "REAL",
            "book_imbalance": "REAL",
            "cvd": "REAL",
        }
        for name, definition in feature_migrations.items():
            if name not in feature_columns:
                await db.execute(f"ALTER TABLE feature_snapshots ADD COLUMN {name} {definition}")

        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_decision_group "
            "ON signal_outcomes(decision_group, finalized)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_hit_rate "
            "ON signal_outcomes(hit_configured)"
        )

        await db.commit()


async def record_trade_outcome(
    symbol: str, side: str, pnl_pct: float,
    entry_price: float = 0.0, exit_price: float = 0.0,
    oi_change: float = 0.0, funding: float = 0.0,
    volume_ratio: float = 1.0, btc_regime: str = "NEUTRAL",
    rsi: float = 50.0, ema_cross: str = "FLAT",
    pnl_usdt: float = 0.0, slippage_bps: float = 0.0,
    fee_paid_usdt: float = 0.0
):
    """Registra outcome de trade com métricas avançadas."""
    win = 1 if pnl_pct > 0 else 0
    async with connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO trade_outcomes
               (symbol, side, entry_price, exit_price, pnl_pct, pnl_usdt, win,
                oi_at_entry, funding_at_entry, volume_ratio, btc_regime,
                rsi_at_entry, ema_cross, slippage_bps, fee_paid_usdt, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, side, entry_price, exit_price, pnl_pct, pnl_usdt, win,
             oi_change, funding, volume_ratio, btc_regime,
             rsi, ema_cross, slippage_bps, fee_paid_usdt, time.time())
        )
        await _update_hourly_metrics(db, symbol, side, pnl_pct, win)
        await _update_daily_metrics(db, symbol, side, pnl_pct, win)
        await db.commit()


async def _update_hourly_metrics(db: Any, symbol: str, side: str, pnl_pct: float, win: int):
    """Atualiza métricas horárias agregadas."""
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    hour_utc = now_utc.hour
    date = now_utc.strftime("%Y-%m-%d")

    await db.execute(
        """INSERT INTO hourly_metrics (symbol, hour_utc, date, trades, wins, total_pnl_pct, avg_pnl_pct, win_rate)
           VALUES (?, ?, ?, 1, ?, ?, ?, ?)
           ON CONFLICT(symbol, date, hour_utc) DO UPDATE SET
               trades = hourly_metrics.trades + 1,
               wins = hourly_metrics.wins + excluded.wins,
               total_pnl_pct = hourly_metrics.total_pnl_pct + excluded.total_pnl_pct,
               avg_pnl_pct = (
                   hourly_metrics.total_pnl_pct + excluded.total_pnl_pct
               ) / (hourly_metrics.trades + 1),
               win_rate = CAST(
                   hourly_metrics.wins + excluded.wins AS REAL
               ) / (hourly_metrics.trades + 1)""",
        (symbol, hour_utc, date, win, pnl_pct, pnl_pct, pnl_pct if win else 0)
    )


async def _update_daily_metrics(db: Any, symbol: str, side: str, pnl_pct: float, win: int):
    """Atualiza métricas diárias agregadas."""
    from datetime import datetime, timezone
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    await db.execute(
        """INSERT INTO daily_symbol_metrics (symbol, side, date, trades, wins, total_pnl_pct, avg_pnl_pct, win_rate)
           VALUES (?, ?, ?, 1, ?, ?, ?, ?)
           ON CONFLICT(symbol, side, date) DO UPDATE SET
               trades = daily_symbol_metrics.trades + 1,
               wins = daily_symbol_metrics.wins + excluded.wins,
               total_pnl_pct = daily_symbol_metrics.total_pnl_pct + excluded.total_pnl_pct,
               avg_pnl_pct = (
                   daily_symbol_metrics.total_pnl_pct + excluded.total_pnl_pct
               ) / (daily_symbol_metrics.trades + 1),
               win_rate = CAST(
                   daily_symbol_metrics.wins + excluded.wins AS REAL
               ) / (daily_symbol_metrics.trades + 1)""",
        (symbol, side, date, win, pnl_pct, pnl_pct, pnl_pct if win else 0)
    )


async def save_feature_snapshot(symbol: str, features: dict):
    """Salva snapshot de features com campos avançados."""
    async with connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO feature_snapshots
               (symbol, timestamp, price, price_change_pct, volume_ratio,
                oi_change_pct, funding_rate, rsi, ema_cross, atr_pct,
                spread_bps, btc_regime, bid_depth_5, ask_depth_5, book_imbalance, cvd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                features.get("bid_depth_5", 0),
                features.get("ask_depth_5", 0),
                features.get("book_imbalance", 0),
                features.get("cvd", 0),
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
    async with connect(DB_PATH) as db:
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
        except IntegrityError:
            return False


async def get_pending_signal_outcomes(min_age_seconds: int = 300, limit: int = 200) -> list[dict]:
    cutoff = time.time() - min_age_seconds
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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
    async with connect(DB_PATH) as db:
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

    async with connect(DB_PATH) as db:
        db.row_factory = Row
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
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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


async def get_signal_training_summary(
    decision_group: str = "ALLOW",
    source_type: str = "hypothetical",
) -> dict:
    async with connect(DB_PATH) as db:
        row = await (await db.execute(
            """SELECT COUNT(*) AS samples,
                      SUM(CASE WHEN hit_configured=1 THEN 1 ELSE 0 END) AS hits,
                      SUM(CASE WHEN hit_configured=0 THEN 1 ELSE 0 END) AS misses,
                      MAX(created_at) AS latest_created_at
               FROM signal_outcomes
               WHERE finalized=1 AND decision_group=? AND source_type=?
                 AND hit_configured IS NOT NULL""",
            (decision_group, source_type),
        )).fetchone()

    values = row or (0, 0, 0, 0)
    samples = int(values[0] or 0)
    hits = int(values[1] or 0)
    misses = int(values[2] or 0)
    return {
        "samples": samples,
        "hits": hits,
        "misses": misses,
        "hasBothClasses": hits > 0 and misses > 0,
        "latestCreatedAt": float(values[3] or 0),
    }


async def get_operational_risk_metrics(hours: int = 24) -> dict:
    since = time.time() - hours * 3600
    async with connect(DB_PATH) as db:
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
    async with connect(DB_PATH) as db:
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
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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
    async with connect(DB_PATH) as db:
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
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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
    async with connect(DB_PATH) as db:
        db.row_factory = Row
        rows = await (await db.execute(
            """SELECT side, COUNT(*) as trades,
               SUM(win) as wins,
               AVG(pnl_pct) as avg_pnl,
               SUM(pnl_pct) as total_pnl,
               MIN(pnl_pct) as worst,
               MAX(pnl_pct) as best,
               AVG(pnl_usdt) as avg_pnl_usdt,
               SUM(pnl_usdt) as total_pnl_usdt
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
                "avg_pnl_pct": round(d["avg_pnl"] or 0, 4),
                "total_pnl_pct": round(d["total_pnl"] or 0, 4),
                "avg_pnl_usdt": round(d["avg_pnl_usdt"] or 0, 4),
                "total_pnl_usdt": round(d["total_pnl_usdt"] or 0, 4),
                "worst": round(d["worst"] or 0, 4),
                "best": round(d["best"] or 0, 4),
            }
        return result


async def get_all_symbols_stats(days: int = 30) -> list[dict]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "VVVUSDT", "TRUMPUSDT",
               "MELANIAUSDT", "BEATUSDT", "NEARUSDT", "HYPEUSDT", "POLUSDT"]
    return [await get_symbol_stats(s, days) for s in symbols]


async def save_observation(symbol: str, category: str, text: str, data: dict, confidence: float):
    async with connect(DB_PATH) as db:
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
    async with connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO strategic_insights
               (period_days, generated_at, analysis_text, edge_changes, recommendations)
               VALUES (?,?,?,?,?)""",
            (period_days, time.time(), analysis_text,
             json.dumps(edge_changes), json.dumps(recommendations))
        )
        await db.commit()


async def get_recent_insights(limit: int = 5) -> list[dict]:
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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
    async with connect(DB_PATH) as db:
        db.row_factory = Row
        rows = await (await db.execute(
            """SELECT * FROM feature_snapshots
               WHERE symbol=? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (symbol, since)
        )).fetchall()
        return [dict(r) for r in rows]


async def get_recent_observations(symbol: str = None, hours: int = 48, limit: int = 50) -> list[dict]:
    since = time.time() - hours * 3600
    async with connect(DB_PATH) as db:
        db.row_factory = Row
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


# ========== NOVAS FUNÇÕES PARA NÍVEL MÁXIMO DE EXCELÊNCIA ==========

async def get_hour_toxicity(symbol: str, hour_utc: int, side: str) -> dict:
    """
    Retorna toxicidade por horário baseado em histórico real.
    Quanto maior toxicity_score, mais o horário é perigoso para operar.
    """
    async with connect(DB_PATH) as db:
        db.row_factory = Row
        row = await (await db.execute(
            """SELECT trades, win_rate, avg_pnl_pct, toxicity_score
               FROM hour_toxicity
               WHERE symbol=? AND hour_utc=? AND side=?
               ORDER BY updated_at DESC LIMIT 1""",
            (symbol, hour_utc, side)
        )).fetchone()

    if not row:
        return {"toxic": False, "toxicity_score": 0.0, "trades": 0, "win_rate": 0}

    d = dict(row)
    toxic = d["toxicity_score"] > 0.3
    return {
        "toxic": toxic,
        "toxicity_score": round(d["toxicity_score"], 3),
        "trades": d["trades"],
        "win_rate": round(d["win_rate"] * 100, 1) if d["win_rate"] else 0,
        "avg_pnl_pct": round(d["avg_pnl_pct"] or 0, 4),
    }


async def update_hour_toxicity(symbol: str, hour_utc: int, side: str, trades: list[dict]):
    """Atualiza score de toxicidade para um horário específico."""
    if len(trades) < 5:
        return

    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    total_pnl = sum(t.get("pnl_pct", 0) for t in trades)
    win_rate = wins / len(trades)
    avg_pnl = total_pnl / len(trades)

    # Fórmula de toxicidade: win_rate baixo + avg_pnl negativo = tóxico
    toxicity_score = max(0.0, (0.5 - win_rate) + max(0.0, -avg_pnl * 0.5))

    async with connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO hour_toxicity (symbol, hour_utc, side, trades, win_rate, avg_pnl_pct, toxicity_score, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(symbol, hour_utc, side) DO UPDATE SET
                   trades = excluded.trades,
                   win_rate = excluded.win_rate,
                   avg_pnl_pct = excluded.avg_pnl_pct,
                   toxicity_score = excluded.toxicity_score,
                   updated_at = excluded.updated_at""",
            (symbol, hour_utc, side, len(trades), win_rate, avg_pnl, toxicity_score, time.time())
        )
        await db.commit()


async def get_correlation(symbol_a: str, symbol_b: str, hours: int = 24) -> float:
    """Retorna correlação entre dois símbolos baseado em dados históricos."""
    correlation_key = f"{hours}h"

    async with connect(DB_PATH) as db:
        db.row_factory = Row
        row = await (await db.execute(
            """SELECT correlation_1h, correlation_4h, correlation_24h
               FROM symbol_correlations
               WHERE symbol_a=? AND symbol_b=?
               ORDER BY computed_at DESC LIMIT 1""",
            (symbol_a, symbol_b)
        )).fetchone()

    if not row:
        return 0.5  # Correlação padrão

    d = dict(row)
    if hours <= 1:
        return round(d.get("correlation_1h", 0.5), 3)
    elif hours <= 4:
        return round(d.get("correlation_4h", 0.5), 3)
    else:
        return round(d.get("correlation_24h", 0.5), 3)


async def record_execution_quality(
    symbol: str,
    side: str,
    expected_price: float,
    executed_price: float,
    latency_ms: float
):
    """Registra qualidade de execução para análise de slippage real."""
    slippage_bps = abs(executed_price - expected_price) / expected_price * 10000 if expected_price > 0 else 0

    async with connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO execution_quality
               (symbol, side, expected_price, executed_price, slippage_bps, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, expected_price, executed_price, slippage_bps, latency_ms, time.time())
        )
        await db.commit()


async def get_avg_slippage_bps(symbol: str, hours: int = 24) -> float:
    """Retorna slippage médio real para o símbolo nas últimas N horas."""
    since = time.time() - hours * 3600

    async with connect(DB_PATH) as db:
        row = await (await db.execute(
            """SELECT AVG(slippage_bps) as avg_slippage
               FROM execution_quality
               WHERE symbol=? AND timestamp >= ?
               LIMIT 1000""",
            (symbol, since)
        )).fetchone()

    if not row or row[0] is None:
        return 2.0  # Default 2bps

    return round(float(row[0]), 2)


async def get_realized_sharpe(symbol: str, side: str, days: int = 30) -> float:
    """Calcula Sharpe Ratio realizado para o símbolo/lado."""
    since = time.time() - days * 86400

    async with connect(DB_PATH) as db:
        rows = await (await db.execute(
            """SELECT pnl_pct FROM trade_outcomes
               WHERE symbol=? AND side=? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (symbol, side, since)
        )).fetchall()

    returns = [float(r[0] or 0) for r in rows]
    if len(returns) < 10:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_return = variance ** 0.5 if variance > 0 else 0.0001

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * (365 ** 0.5)  # Anualizado
    return round(sharpe, 3)


async def get_rolling_win_rate(symbol: str, side: str, window_hours: int = 24) -> float:
    """Win rate em janela móvel para detecção de deterioração."""
    since = time.time() - window_hours * 3600

    async with connect(DB_PATH) as db:
        rows = await (await db.execute(
            """SELECT win FROM trade_outcomes
               WHERE symbol=? AND side=? AND timestamp >= ?
               ORDER BY timestamp DESC
               LIMIT 50""",
            (symbol, side, since)
        )).fetchall()

    if len(rows) < 10:
        return 0.5

    wins = sum(1 for r in rows if r[0] == 1)
    return round(wins / len(rows), 3)


async def vacuum_db():
    """Otimiza o banco de dados - agendado semanalmente."""
    async with connect(DB_PATH) as db:
        await db.execute("VACUUM")
        await db.execute("ANALYZE")
