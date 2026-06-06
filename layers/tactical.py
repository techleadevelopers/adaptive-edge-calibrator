"""
Camada Tática — monitora anomalias em tempo real (1s/5s/15s/30s/1min).
Detecta padrões, gera alertas e salva observações na Knowledge Base.
"""
import asyncio
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from core.feature_engine import FeatureEngine, MarketSnapshot, SYMBOLS
from core import knowledge_base as kb

log = logging.getLogger("tactical")


@dataclass
class TacticalAlert:
    symbol: str
    alert_type: str
    message: str
    confidence: float
    similar_occurrences: int
    avg_return_past: float
    win_rate_past: float
    conditions: dict
    timestamp: float = field(default_factory=time.time)


# Buffer de snapshots por símbolo para análise temporal
_snap_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=120))
_active_alerts: list[TacticalAlert] = []
_alert_callbacks: list = []


def on_alert(fn):
    _alert_callbacks.append(fn)


def get_active_alerts(max_age_seconds: int = 300) -> list[TacticalAlert]:
    cutoff = time.time() - max_age_seconds
    return [a for a in _active_alerts if a.timestamp >= cutoff]


def _classify_pattern(snap: MarketSnapshot, history: list[dict]) -> list[str]:
    """Identifica padrões nomeados a partir das condições atuais."""
    patterns = []

    if snap.oi_change_pct >= 5 and snap.price_change_pct >= 0.5 and snap.volume_ratio >= 2:
        patterns.append("OI_PRICE_VOL_TRIPLE_UP")

    if snap.oi_change_pct >= 5 and snap.price_change_pct <= -0.5:
        patterns.append("OI_UP_PRICE_DOWN")  # posições short acumulando

    if snap.funding_rate <= -0.0003 and snap.rsi_approx <= 35:
        patterns.append("NEGATIVE_FUNDING_OVERSOLD")

    if snap.funding_rate >= 0.0003 and snap.rsi_approx >= 65:
        patterns.append("POSITIVE_FUNDING_OVERBOUGHT")

    if snap.volume_ratio >= 3 and abs(snap.price_change_pct) <= 0.3:
        patterns.append("HIGH_VOL_LOW_MOVE")  # absorção

    if snap.btc_regime == "NEUTRAL" and snap.oi_change_pct >= 4 and snap.funding_rate <= 0.0001:
        patterns.append("BTC_LATERAL_OI_BUILD")  # setup long clássico

    if len(history) >= 10:
        prices = [h.get("price", 0) for h in history[-10:]]
        if prices[-1] > 0 and min(prices) > 0:
            move = (prices[-1] - prices[0]) / prices[0] * 100
            if move >= 1.5 and snap.volume_ratio >= 2:
                patterns.append("BREAKOUT_UP_CONFIRMED")
            elif move <= -1.5 and snap.volume_ratio >= 2:
                patterns.append("BREAKOUT_DOWN_CONFIRMED")

    return patterns


async def _process_snapshot(snap: MarketSnapshot):
    """Processa um snapshot: detecta padrões, cria alertas, persiste observações."""
    sym = snap.symbol
    _snap_buffer[sym].append({
        "price": snap.price,
        "price_change_pct": snap.price_change_pct,
        "oi_change_pct": snap.oi_change_pct,
        "funding_rate": snap.funding_rate,
        "volume_ratio": snap.volume_ratio,
        "rsi": snap.rsi_approx,
        "btc_regime": snap.btc_regime,
        "timestamp": snap.timestamp,
    })

    history = list(_snap_buffer[sym])

    # Persiste snapshot na KB a cada ~30s (1 em 6 chamadas a 5s)
    if int(snap.timestamp) % 30 < 5:
        await kb.save_feature_snapshot(sym, {
            "price": snap.price,
            "price_change_pct": snap.price_change_pct,
            "volume_ratio": snap.volume_ratio,
            "oi_change_pct": snap.oi_change_pct,
            "funding_rate": snap.funding_rate,
            "rsi": snap.rsi_approx,
            "ema_cross": "FLAT",
            "atr_pct": snap.atr_pct,
            "spread_bps": snap.spread_bps,
            "btc_regime": snap.btc_regime,
        })

    if not snap.anomalies:
        return

    pattern_names = _classify_pattern(snap, history)

    for pat_name in pattern_names:
        patterns_db = await kb.get_top_patterns(min_occurrences=1, limit=100)
        match = next(
            (p for p in patterns_db if p["name"] == pat_name and p["symbol"] == sym),
            None
        )
        similar_occ = match["occurrences"] if match else 0
        avg_ret = match["avg_return"] if match else 0.0
        win_rate = match["win_rate"] if match else 0.0

        confidence = min(0.95, 0.4 + (similar_occ / 100) * 0.55) if similar_occ else 0.4

        msg = _build_alert_message(sym, pat_name, snap, similar_occ, avg_ret, win_rate)

        alert = TacticalAlert(
            symbol=sym,
            alert_type=pat_name,
            message=msg,
            confidence=round(confidence, 2),
            similar_occurrences=similar_occ,
            avg_return_past=round(avg_ret, 4),
            win_rate_past=round(win_rate, 4),
            conditions={
                "oi_change_pct": snap.oi_change_pct,
                "price_change_pct": snap.price_change_pct,
                "volume_ratio": snap.volume_ratio,
                "funding_rate": snap.funding_rate,
                "rsi": snap.rsi_approx,
                "btc_regime": snap.btc_regime,
            },
        )

        _active_alerts.append(alert)
        if len(_active_alerts) > 500:
            _active_alerts.pop(0)

        await kb.save_observation(
            symbol=sym,
            category="TACTICAL_ALERT",
            text=msg,
            data={
                "pattern": pat_name,
                "anomalies": snap.anomalies,
                "conditions": alert.conditions,
                "confidence": confidence,
                "similar_occ": similar_occ,
                "avg_return": avg_ret,
                "win_rate": win_rate,
            },
            confidence=confidence,
        )

        for cb in _alert_callbacks:
            try:
                await cb(alert)
            except Exception:
                pass

        log.info(f"ALERT [{sym}] {pat_name} | conf={confidence:.0%} | occ={similar_occ} | wr={win_rate:.0%}")


def _build_alert_message(
    symbol: str, pattern: str, snap: MarketSnapshot,
    similar_occ: int, avg_ret: float, win_rate: float
) -> str:
    short = symbol.replace("-USDT", "")
    lines = [f"⚡ ALERTA — {short}", f"Padrão: {pattern}"]

    if snap.oi_change_pct != 0:
        lines.append(f"Open Interest: {snap.oi_change_pct:+.1f}%")
    if snap.price_change_pct != 0:
        lines.append(f"Preço: {snap.price_change_pct:+.2f}%")
    if snap.volume_ratio > 1.2:
        lines.append(f"Volume: {snap.volume_ratio:.1f}×")
    if abs(snap.funding_rate) > 0:
        lines.append(f"Funding: {snap.funding_rate:+.4f}")

    if similar_occ >= 5:
        lines.append(f"\nOcorrências similares: {similar_occ}")
        lines.append(f"Retorno médio histórico: {avg_ret:+.2f}%")
        lines.append(f"Win rate histórico: {win_rate:.0%}")
    else:
        lines.append("\nPadrão novo — sem histórico suficiente")

    return "\n".join(lines)


async def _detect_lead_lag(snaps: dict[str, MarketSnapshot]):
    """Detecta se movimentos de um ativo precedem outros (ex: SOL → ETH)."""
    buf = {}
    for sym, snap in snaps.items():
        h = list(_snap_buffer[sym])
        if len(h) >= 4:
            prices = [x["price"] for x in h[-4:]]
            if prices[0] > 0:
                buf[sym] = (prices[-1] - prices[0]) / prices[0] * 100

    if len(buf) < 2:
        return

    movers = [(sym, mv) for sym, mv in buf.items() if abs(mv) >= 0.5]
    for sym, mv in movers:
        for other_sym, other_mv in buf.items():
            if other_sym == sym:
                continue
            if abs(mv) >= 1.0 and abs(other_mv) <= 0.2:
                direction = "LONG" if mv > 0 else "SHORT"
                msg = (
                    f"Lead-lag detectado: {sym.replace('-USDT','')} moveu {mv:+.2f}% "
                    f"enquanto {other_sym.replace('-USDT','')} ainda está flat ({other_mv:+.2f}%). "
                    f"Potencial setup {direction} em {other_sym.replace('-USDT','')}."
                )
                await kb.save_observation(
                    symbol=other_sym,
                    category="LEAD_LAG",
                    text=msg,
                    data={"leader": sym, "leader_move": mv, "lagger": other_sym, "lagger_move": other_mv},
                    confidence=0.55,
                )


async def run_tactical_loop(engine: FeatureEngine, interval_seconds: int = 5):
    """Loop principal tático — roda a cada N segundos."""
    await kb.init_db()
    log.info(f"Tactical loop iniciado (interval={interval_seconds}s)")
    engine.on_snapshot(_process_snapshot)

    while True:
        try:
            snaps = await engine.snapshot_all()
            await _detect_lead_lag(snaps)
        except Exception as e:
            log.error(f"Tactical loop error: {e}")
        await asyncio.sleep(interval_seconds)
