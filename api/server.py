"""
API REST — expõe todos os dados do Quant Brain via HTTP.
Compatível com o dashboard existente e com consultas manuais.
"""
from __future__ import annotations

import asyncio
import time
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.feature_engine import FeatureEngine, SYMBOLS
from core import knowledge_base as kb
from core.recommendation import recommend_entry, simulate_gate_rejections
from core.edge_gate import evaluate_edge_gate
from core.movement_sniper import evaluate_sniper_window, build_movement_features, classify_btc_commander
from core.signal_learning import finalize_due_signal_outcomes, score_signal_context
from layers.tactical import run_tactical_loop, get_active_alerts, get_snapshot_history, TacticalAlert
from layers.strategic import build_strategic_report, report_to_dict, compute_edge_evolution
from analyst.ai_analyst import (
    run_weekly_analysis, run_tactical_analysis, run_hypothesis_generation, _has_ai
)

log = logging.getLogger("api")

engine = FeatureEngine()
_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    await kb.init_db()
    log.info("Knowledge Base inicializada")

    t1 = asyncio.create_task(run_tactical_loop(engine, interval_seconds=5))
    _tasks.append(t1)

    from layers.strategic import run_strategic_loop
    t2 = asyncio.create_task(run_strategic_loop(interval_hours=6))
    _tasks.append(t2)

    log.info("Quant Brain online — monitorando 10 ativos 24h")
    yield

    for t in _tasks:
        t.cancel()
    await engine.close()
    log.info("Quant Brain encerrado")


app = FastAPI(
    title="Quant Brain API",
    description="Motor de análise quantitativa e IA para o bot BingX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── MARKET DATA ────────────────────────────────────────────────────────────

@app.get("/market/snapshots")
async def get_snapshots():
    """Estado atual de todos os 10 ativos."""
    snaps = engine.get_all_snapshots()
    return {
        "timestamp": time.time(),
        "count": len(snaps),
        "symbols": {
            sym: engine.to_dict(snap)
            for sym, snap in snaps.items()
        }
    }


@app.get("/market/snapshots/{symbol}")
async def get_snapshot(symbol: str):
    """Estado atual de um ativo específico."""
    sym = symbol.upper()
    if not sym.endswith("-USDT"):
        sym = sym + "-USDT"
    snap = engine.get_snapshot(sym)
    if not snap:
        raise HTTPException(404, f"Símbolo {sym} não encontrado ou ainda sem dados")
    return engine.to_dict(snap)


@app.get("/market/anomalies")
async def get_anomalies():
    """Todos os ativos com anomalias detectadas agora."""
    snaps = engine.get_all_snapshots()
    result = []
    for sym, snap in snaps.items():
        if snap.anomalies:
            result.append({
                "symbol": sym,
                "anomalies": snap.anomalies,
                "price": snap.price,
                "price_change_pct": snap.price_change_pct,
                "oi_change_pct": snap.oi_change_pct,
                "volume_ratio": snap.volume_ratio,
                "funding_rate": snap.funding_rate,
                "rsi": snap.rsi_approx,
                "btc_regime": snap.btc_regime,
                "timestamp": snap.timestamp,
            })
    result.sort(key=lambda x: len(x["anomalies"]), reverse=True)
    return {"timestamp": time.time(), "count": len(result), "anomalies": result}


@app.get("/sniper/btc-commander")
async def get_btc_commander(window_seconds: int = Query(300, ge=60, le=900)):
    """BTC real-time commander for sniper scalp gating."""
    history = get_snapshot_history("BTC-USDT", window_seconds)
    features = build_movement_features("BTC-USDT", history, window_seconds)
    return {
        "timestamp": time.time(),
        "windowSeconds": window_seconds,
        "commander": classify_btc_commander(features),
        "features": features.__dict__,
    }


@app.get("/sniper/evaluate/{symbol}")
async def evaluate_sniper_symbol(symbol: str, window_seconds: int = Query(300, ge=60, le=900)):
    """Evaluate a symbol against BTC movement for short-target sniper entries."""
    sym = symbol.upper()
    if not sym.endswith("-USDT"):
        sym = f"{sym}-USDT"
    alt_history = get_snapshot_history(sym, window_seconds)
    btc_history = get_snapshot_history("BTC-USDT", window_seconds)
    return evaluate_sniper_window(sym, alt_history, btc_history, window_seconds=window_seconds)


# ─── TACTICAL ────────────────────────────────────────────────────────────────

@app.get("/tactical/alerts")
async def get_tactical_alerts(max_age: int = Query(300, description="Segundos")):
    """Alertas táticos recentes (padrões detectados em tempo real)."""
    alerts = get_active_alerts(max_age)
    return {
        "timestamp": time.time(),
        "count": len(alerts),
        "alerts": [
            {
                "symbol": a.symbol,
                "alert_type": a.alert_type,
                "message": a.message,
                "confidence": a.confidence,
                "similar_occurrences": a.similar_occurrences,
                "avg_return_past": a.avg_return_past,
                "win_rate_past": a.win_rate_past,
                "conditions": a.conditions,
                "timestamp": a.timestamp,
            }
            for a in alerts
        ]
    }


@app.post("/tactical/analyze")
async def run_tactical_ai():
    """Dispara análise tática com IA agora (usa Claude se configurado)."""
    alerts = [
        {
            "symbol": a.symbol,
            "alert_type": a.alert_type,
            "confidence": a.confidence,
            "similar_occurrences": a.similar_occurrences,
            "win_rate_past": a.win_rate_past,
            "avg_return_past": a.avg_return_past,
        }
        for a in get_active_alerts(600)
    ]
    snaps = {sym: engine.to_dict(snap) for sym, snap in engine.get_all_snapshots().items()}
    observations = await kb.get_recent_observations(hours=2, limit=20)
    analysis = await run_tactical_analysis(alerts, snaps, observations)
    return {
        "analysis_type": analysis.analysis_type,
        "generated_at": analysis.generated_at,
        "model": analysis.model,
        "ai_enabled": _has_ai(),
        "full_text": analysis.full_text,
        "summary": analysis.summary,
    }


# ─── STRATEGIC ───────────────────────────────────────────────────────────────

@app.get("/strategic/report")
async def get_strategic_report(days: int = Query(30, ge=1, le=365)):
    """Relatório estratégico: evolução de edge, rankings, mudanças estruturais."""
    report = await build_strategic_report(days)
    return report_to_dict(report)


@app.get("/strategic/edge-evolution")
async def get_edge_evolution(days: int = Query(30, ge=7, le=365)):
    """Evolução de edge por símbolo e lado (primeira vs segunda metade do período)."""
    evolutions = await compute_edge_evolution(days)
    return {
        "period_days": days,
        "generated_at": time.time(),
        "count": len(evolutions),
        "evolutions": [
            {
                "symbol": e.symbol,
                "side": e.side,
                "wr_early": e.win_rate_early,
                "wr_late": e.win_rate_late,
                "delta_wr": e.delta_wr,
                "avg_pnl_early": e.avg_pnl_early,
                "avg_pnl_late": e.avg_pnl_late,
                "trend": e.trend,
                "trades_early": e.trades_early,
                "trades_late": e.trades_late,
            }
            for e in evolutions
        ]
    }


@app.post("/strategic/analyze")
async def run_strategic_ai(days: int = Query(30, ge=7, le=365)):
    """Dispara análise estratégica com IA (relatório semanal / Head of Quant)."""
    report = await build_strategic_report(days)
    evolutions_dicts = [
        {
            "symbol": e.symbol, "side": e.side,
            "wr_early": e.win_rate_early, "wr_late": e.win_rate_late,
            "delta_wr": e.delta_wr, "trend": e.trend,
        }
        for e in report.edge_migrations
    ]
    patterns = await kb.get_top_patterns(min_occurrences=3, limit=20)
    all_stats = await kb.get_all_symbols_stats(days)

    analysis = await run_weekly_analysis(all_stats, evolutions_dicts, patterns)
    await kb.save_strategic_insight(
        period_days=days,
        analysis_text=analysis.full_text,
        edge_changes={e.symbol + "_" + e.side: {"trend": e.trend, "delta_wr": e.delta_wr} for e in report.edge_migrations},
        recommendations=report.structural_changes,
    )

    return {
        "analysis_type": analysis.analysis_type,
        "generated_at": analysis.generated_at,
        "model": analysis.model,
        "ai_enabled": _has_ai(),
        "full_text": analysis.full_text,
        "report_summary": {
            "total_trades": report.total_trades,
            "global_win_rate": report.global_win_rate,
            "structural_changes": report.structural_changes,
        }
    }


@app.post("/strategic/hypotheses")
async def generate_hypotheses():
    """Gera hipóteses originais de edge com base nos padrões acumulados."""
    patterns = await kb.get_top_patterns(min_occurrences=2, limit=20)
    observations = await kb.get_recent_observations(hours=48, limit=30)
    stats = await kb.get_all_symbols_stats(30)
    analysis = await run_hypothesis_generation(patterns, observations, stats)
    return {
        "generated_at": analysis.generated_at,
        "model": analysis.model,
        "ai_enabled": _has_ai(),
        "hypotheses": analysis.full_text,
    }


# ─── KNOWLEDGE BASE ──────────────────────────────────────────────────────────

@app.get("/kb/patterns")
async def get_patterns(min_occurrences: int = Query(1, ge=1), limit: int = Query(50, le=200)):
    """Padrões acumulados na Knowledge Base, ordenados por win rate."""
    patterns = await kb.get_top_patterns(min_occurrences, limit)
    return {"count": len(patterns), "patterns": patterns}


@app.get("/kb/observations")
async def get_observations(
    symbol: str = Query(None),
    hours: int = Query(48, ge=1, le=720),
    limit: int = Query(50, le=200),
):
    """Observações táticas e lead-lag recentes."""
    obs = await kb.get_recent_observations(symbol, hours, limit)
    return {"count": len(obs), "observations": obs}


@app.get("/kb/insights")
async def get_insights(limit: int = Query(5, le=20)):
    """Últimos relatórios estratégicos salvos."""
    insights = await kb.get_recent_insights(limit)
    return {"count": len(insights), "insights": insights}


@app.get("/kb/stats/{symbol}")
async def get_symbol_stats(symbol: str, days: int = Query(30, ge=1, le=365)):
    """Estatísticas de trades por símbolo."""
    stats = await kb.get_symbol_stats(symbol.upper(), days)
    return stats


@app.get("/kb/stats")
async def get_all_stats(days: int = Query(30, ge=1, le=365)):
    """Estatísticas de todos os símbolos."""
    stats = await kb.get_all_symbols_stats(days)
    return {"period_days": days, "symbols": stats}


@app.post("/kb/trades")
async def record_trade(body: dict):
    """Registra resultado de um trade na KB (chamado pelo bot Node.js)."""
    required = ["symbol"]
    for r in required:
        if r not in body:
            raise HTTPException(400, f"Campo obrigatório: {r}")
    side = body.get("positionSide") or body.get("position_side") or body.get("side")
    if not side:
        raise HTTPException(400, "Required field: side or positionSide")
    pnl_pct = body.get("pnl_pct")
    if pnl_pct is None:
        realized_pnl = float(body.get("realizedPnl", body.get("realized_pnl", 0)))
        margin_used = float(body.get("marginUsed", body.get("margin_used", 0)))
        pnl_pct = (realized_pnl / margin_used * 100) if margin_used > 0 else realized_pnl
    await kb.record_trade_outcome(
        symbol=body["symbol"],
        side=side,
        pnl_pct=float(pnl_pct),
        entry_price=float(body.get("entry_price", body.get("entryPrice", 0))),
        exit_price=float(body.get("exit_price", body.get("exitPrice", 0))),
        oi_change=float(body.get("oi_change", body.get("oiChange", 0))),
        funding=float(body.get("funding", body.get("fundingRate", 0))),
        volume_ratio=float(body.get("volume_ratio", body.get("volumeRatio", 1))),
        btc_regime=body.get("btc_regime", body.get("btcRegime", "NEUTRAL")),
        rsi=float(body.get("rsi", body.get("rsiAtEntry", 50))),
        ema_cross=body.get("ema_cross", body.get("emaCross", "FLAT")),
    )
    return {"ok": True, "recorded": body["symbol"], "pnl_pct": float(pnl_pct)}


@app.get("/kb/feature-history/{symbol}")
async def get_feature_history(symbol: str, hours: int = Query(24, ge=1, le=168)):
    """Histórico de snapshots de features para análise temporal."""
    sym = symbol.upper()
    if not sym.endswith("-USDT"):
        sym = sym + "-USDT"
    history = await kb.get_feature_history(sym, hours)
    return {"symbol": sym, "hours": hours, "count": len(history), "history": history}


@app.post("/recommend/entry")
async def recommend_entry_endpoint(body: dict, days: int = Query(30, ge=1, le=365)):
    """Entry allow/reject recommendation in shadow mode by default."""
    if "symbol" not in body:
        raise HTTPException(400, "Required field: symbol")
    return await recommend_entry(body, days=days)


@app.post("/edge/evaluate")
async def evaluate_edge_endpoint(body: dict):
    """Authoritative edge gate: backend sends pending entry context, Quant Brain returns allow/reject."""
    if "symbol" not in body:
        raise HTTPException(400, "Required field: symbol")
    return await evaluate_edge_gate(body)


@app.post("/signals/finalize")
async def finalize_signals_endpoint():
    """Finalize pending signal outcomes after the 300s sniper validation window."""
    return await finalize_due_signal_outcomes()


@app.get("/signals/edge/{symbol}")
async def get_signal_edge_endpoint(
    symbol: str,
    side: str = Query("LONG"),
    context_key: str = Query(None),
):
    """Target-hit memory for sniper signals, including blocked/wait decisions."""
    sym = symbol.upper()
    if not sym.endswith("-USDT"):
        sym = f"{sym}-USDT"
    return await score_signal_context(sym, side.upper(), context_key or "")


@app.post("/news/events")
async def record_news_event_endpoint(body: dict):
    """Store a market/news event as risk context for the edge gate."""
    title = str(body.get("title", "")).strip()
    if not title:
        raise HTTPException(400, "Required field: title")
    symbols = body.get("symbols") or body.get("affectedSymbols") or ["MARKET"]
    if isinstance(symbols, str):
        symbols = [symbols]
    normalized = []
    for item in symbols:
        sym = str(item).upper()
        if sym not in {"MARKET", "BTC-USDT"} and sym.endswith("USDT") and "-" not in sym:
            sym = f"{sym[:-4]}-USDT"
        normalized.append(sym)
    await kb.record_news_event(
        source=str(body.get("source", "manual")),
        title=title,
        symbols=normalized,
        category=str(body.get("category", "market")),
        impact_score=float(body.get("impactScore", body.get("impact_score", 0))),
        risk_level=str(body.get("riskLevel", body.get("risk_level", "LOW"))).upper(),
        action=str(body.get("action", "context_only")),
        url=str(body.get("url", "")),
        raw=body,
        ttl_seconds=int(body.get("ttlSeconds", body.get("ttl_seconds", 7200))),
    )
    return {"ok": True, "symbols": normalized}


@app.get("/news/context/{symbol}")
async def get_news_context_endpoint(symbol: str):
    """Active news/sentiment risk context for a symbol."""
    sym = symbol.upper()
    if not sym.endswith("-USDT") and sym != "MARKET":
        sym = f"{sym}-USDT"
    return await kb.get_active_news_context(sym)


@app.get("/simulate/gate-rejections")
async def simulate_gate_rejections_endpoint(
    days: int = Query(30, ge=1, le=365),
    min_avg_pnl: float = Query(0.0),
):
    """Backtest-style simulation for rejecting losing flow by symbol, hour, or regime."""
    return await simulate_gate_rejections(days=days, min_avg_pnl=min_avg_pnl)


# ─── HEALTH ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    snaps = engine.get_all_snapshots()
    return {
        "status": "ok",
        "ai_enabled": _has_ai(),
        "symbols_monitored": len(SYMBOLS),
        "snapshots_cached": len(snaps),
        "uptime": time.time(),
    }


@app.get("/")
async def root():
    return {
        "name": "Quant Brain",
        "version": "1.0.0",
        "description": "Motor de análise quantitativa 24h para 10 ativos BingX",
        "endpoints": {
            "market": ["/market/snapshots", "/market/anomalies"],
            "sniper": ["/sniper/btc-commander", "/sniper/evaluate/{symbol}"],
            "tactical": ["/tactical/alerts", "/tactical/analyze"],
            "strategic": ["/strategic/report", "/strategic/analyze", "/strategic/hypotheses"],
            "knowledge_base": ["/kb/patterns", "/kb/observations", "/kb/insights", "/kb/stats", "/kb/trades"],
            "recommendation": ["/recommend/entry", "/simulate/gate-rejections"],
            "edge_gate": ["/edge/evaluate"],
            "signals": ["/signals/finalize", "/signals/edge/{symbol}"],
            "news": ["/news/events", "/news/context/{symbol}"],
        }
    }
