"""
Camada Estratégica — analisa 1.000/5.000/10.000 trades e semanas/meses de histórico.
Detecta mudanças estruturais de edge, evolução de win rate por símbolo e lado.
"""
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional

from core import knowledge_base as kb

log = logging.getLogger("strategic")

SYMBOLS_SHORT = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "VVVUSDT", "TRUMPUSDT",
    "MELANIAUSDT", "BEATUSDT", "NEARUSDT", "HYPEUSDT", "POLUSDT",
]


@dataclass
class EdgeEvolution:
    symbol: str
    side: str
    period_label: str
    win_rate_early: float
    win_rate_late: float
    delta_wr: float
    avg_pnl_early: float
    avg_pnl_late: float
    delta_pnl: float
    trades_early: int
    trades_late: int
    trend: str   # IMPROVING / DETERIORATING / STABLE


@dataclass
class StrategicReport:
    period_days: int
    generated_at: float
    total_trades: int
    global_win_rate: float
    top_performers: list[dict]
    worst_performers: list[dict]
    edge_migrations: list[EdgeEvolution]
    structural_changes: list[str]
    raw_stats: dict


async def compute_edge_evolution(days: int = 30) -> list[EdgeEvolution]:
    """
    Divide o período em duas metades e compara win rate / PnL médio.
    Detecta se o edge está melhorando ou deteriorando.
    """
    import aiosqlite
    from core.knowledge_base import DB_PATH

    since = time.time() - days * 86400
    mid = since + (days * 86400) / 2
    evolutions = []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        for sym in SYMBOLS_SHORT:
            for side in ["LONG", "SHORT"]:
                early = await (await db.execute(
                    """SELECT COUNT(*) as n, SUM(win) as w, AVG(pnl_pct) as ap
                       FROM trade_outcomes
                       WHERE symbol=? AND side=? AND timestamp BETWEEN ? AND ?""",
                    (sym, side, since, mid)
                )).fetchone()
                late = await (await db.execute(
                    """SELECT COUNT(*) as n, SUM(win) as w, AVG(pnl_pct) as ap
                       FROM trade_outcomes
                       WHERE symbol=? AND side=? AND timestamp > ?""",
                    (sym, side, mid)
                )).fetchone()

                n_early = early["n"] or 0
                n_late = late["n"] or 0
                if n_early < 3 and n_late < 3:
                    continue

                wr_early = (early["w"] or 0) / n_early if n_early > 0 else 0
                wr_late = (late["w"] or 0) / n_late if n_late > 0 else 0
                ap_early = early["ap"] or 0
                ap_late = late["ap"] or 0
                delta_wr = wr_late - wr_early

                if delta_wr >= 0.05:
                    trend = "IMPROVING"
                elif delta_wr <= -0.05:
                    trend = "DETERIORATING"
                else:
                    trend = "STABLE"

                evolutions.append(EdgeEvolution(
                    symbol=sym,
                    side=side,
                    period_label=f"{days}d",
                    win_rate_early=round(wr_early * 100, 1),
                    win_rate_late=round(wr_late * 100, 1),
                    delta_wr=round(delta_wr * 100, 1),
                    avg_pnl_early=round(ap_early, 4),
                    avg_pnl_late=round(ap_late, 4),
                    delta_pnl=round(ap_late - ap_early, 4),
                    trades_early=n_early,
                    trades_late=n_late,
                    trend=trend,
                ))

    return evolutions


async def build_strategic_report(days: int = 30) -> StrategicReport:
    """Gera relatório completo: stats globais + evolução de edge por símbolo."""
    import aiosqlite
    from core.knowledge_base import DB_PATH

    since = time.time() - days * 86400
    all_stats = await kb.get_all_symbols_stats(days)
    evolutions = await compute_edge_evolution(days)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        totals = await (await db.execute(
            """SELECT COUNT(*) as n, SUM(win) as w, AVG(pnl_pct) as ap
               FROM trade_outcomes WHERE timestamp >= ?""",
            (since,)
        )).fetchone()

    total_trades = totals["n"] or 0
    global_wr = (totals["w"] or 0) / total_trades * 100 if total_trades > 0 else 0

    # Rank símbolos por win rate
    ranked = []
    for stat in all_stats:
        for side, data in stat.get("sides", {}).items():
            if data["trades"] >= 3:
                ranked.append({
                    "symbol": stat["symbol"],
                    "side": side,
                    "win_rate": data["win_rate"],
                    "avg_pnl": data["avg_pnl"],
                    "trades": data["trades"],
                })
    ranked.sort(key=lambda x: x["win_rate"], reverse=True)

    top = ranked[:5]
    worst = ranked[-5:] if len(ranked) >= 5 else []

    # Mudanças estruturais detectadas
    structural = []
    deteriorating = [e for e in evolutions if e.trend == "DETERIORATING" and abs(e.delta_wr) >= 8]
    improving = [e for e in evolutions if e.trend == "IMPROVING" and e.delta_wr >= 8]

    for e in deteriorating:
        structural.append(
            f"⚠️ Edge deteriorando: {e.symbol} {e.side} — WR caiu {abs(e.delta_wr):.0f}pp "
            f"({e.win_rate_early:.0f}% → {e.win_rate_late:.0f}%) nos últimos {days}d"
        )
    for e in improving:
        structural.append(
            f"✅ Edge melhorando: {e.symbol} {e.side} — WR subiu {e.delta_wr:.0f}pp "
            f"({e.win_rate_early:.0f}% → {e.win_rate_late:.0f}%) nos últimos {days}d"
        )

    # Edge migration
    if deteriorating and improving:
        from_syms = ", ".join(f"{e.symbol} {e.side}" for e in deteriorating[:2])
        to_syms = ", ".join(f"{e.symbol} {e.side}" for e in improving[:2])
        structural.append(f"🔄 Migração de edge detectada: [{from_syms}] → [{to_syms}]")

    edge_changes = {
        e.symbol + "_" + e.side: {
            "wr_early": e.win_rate_early,
            "wr_late": e.win_rate_late,
            "delta": e.delta_wr,
            "trend": e.trend,
        }
        for e in evolutions
    }

    return StrategicReport(
        period_days=days,
        generated_at=time.time(),
        total_trades=total_trades,
        global_win_rate=round(global_wr, 1),
        top_performers=top,
        worst_performers=worst,
        edge_migrations=evolutions,
        structural_changes=structural,
        raw_stats={s["symbol"]: s for s in all_stats},
    )


def report_to_dict(report: StrategicReport) -> dict:
    return {
        "period_days": report.period_days,
        "generated_at": report.generated_at,
        "total_trades": report.total_trades,
        "global_win_rate": report.global_win_rate,
        "top_performers": report.top_performers,
        "worst_performers": report.worst_performers,
        "structural_changes": report.structural_changes,
        "edge_migrations": [
            {
                "symbol": e.symbol,
                "side": e.side,
                "wr_early": e.win_rate_early,
                "wr_late": e.win_rate_late,
                "delta_wr": e.delta_wr,
                "trend": e.trend,
                "trades_early": e.trades_early,
                "trades_late": e.trades_late,
            }
            for e in report.edge_migrations
        ],
        "raw_stats": report.raw_stats,
    }


async def run_strategic_loop(interval_hours: int = 6):
    """Roda análise estratégica a cada N horas e salva na KB."""
    await kb.init_db()
    log.info(f"Strategic loop iniciado (interval={interval_hours}h)")
    while True:
        try:
            for days in [7, 30]:
                report = await build_strategic_report(days)
                if report.total_trades > 0:
                    await kb.save_strategic_insight(
                        period_days=days,
                        analysis_text="\n".join(report.structural_changes) or "Sem mudanças estruturais detectadas.",
                        edge_changes={
                            e.symbol + "_" + e.side: {
                                "trend": e.trend,
                                "delta_wr": e.delta_wr,
                            }
                            for e in report.edge_migrations
                        },
                        recommendations=[
                            f"Reduzir exposição em {e.symbol} {e.side} (WR caindo {abs(e.delta_wr):.0f}pp)"
                            for e in report.edge_migrations
                            if e.trend == "DETERIORATING" and abs(e.delta_wr) >= 8
                        ],
                    )
                    log.info(f"Strategic report ({days}d) salvo — {report.total_trades} trades analisados")
        except Exception as e:
            log.error(f"Strategic loop error: {e}")

        await asyncio.sleep(interval_hours * 3600)
