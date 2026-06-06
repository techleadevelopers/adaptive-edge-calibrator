"""
Feature Engine — coleta dados de mercado da BingX a cada 1s/5s/15s/30s/1min
para os 10 ativos. Calcula: preço, OI, funding, volume, CVD, spread, volatilidade.
"""
import asyncio
import time
import os
import hmac
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional
import httpx

BINGX_BASE = "https://open-api.bingx.com"
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "VVV-USDT", "TRUMP-USDT",
    "MELANIA-USDT", "BEAT-USDT", "NEAR-USDT", "HYPE-USDT", "POL-USDT",
]
SYMBOL_SHORT = {s: s.replace("-USDT", "").replace("-USD", "") for s in SYMBOLS}


@dataclass
class MarketSnapshot:
    symbol: str
    timestamp: float
    price: float
    price_change_pct: float
    volume_24h: float
    volume_ratio: float        # vol atual vs média 24h
    oi: float                  # open interest
    oi_change_pct: float       # variação % OI vs snapshot anterior
    funding_rate: float
    bid: float
    ask: float
    spread_bps: float
    high_24h: float
    low_24h: float
    atr_pct: float             # proxy: (high-low)/price
    btc_regime: str            # BULL / BEAR / NEUTRAL
    rsi_approx: float          # RSI simplificado dos últimos ticks
    anomalies: list[str] = field(default_factory=list)


class FeatureEngine:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._prev_oi: dict[str, float] = {}
        self._prev_prices: dict[str, list[float]] = {s: [] for s in SYMBOLS}
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._btc_change = 0.0
        self._callbacks: list = []

    def on_snapshot(self, fn):
        self._callbacks.append(fn)

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=8.0)
        return self._client

    def _sign(self, params: dict) -> str:
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        secret = os.environ.get("BINGX_SECRET_KEY", "")
        return hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

    async def _get(self, path: str, params: dict = {}, signed: bool = False) -> dict:
        p = dict(params)
        if signed:
            p["timestamp"] = int(time.time() * 1000)
            p["signature"] = self._sign(p)
        headers = {}
        api_key = os.environ.get("BINGX_API_KEY", "")
        if api_key:
            headers["X-BX-APIKEY"] = api_key
        try:
            r = await self.client.get(f"{BINGX_BASE}{path}", params=p, headers=headers)
            return r.json()
        except Exception as e:
            return {"code": -1, "error": str(e)}

    async def fetch_ticker(self, symbol: str) -> dict:
        r = await self._get("/openApi/swap/v2/quote/ticker", {"symbol": symbol})
        if r.get("code") == 0:
            return r.get("data", {})
        return {}

    async def fetch_funding(self, symbol: str) -> float:
        r = await self._get("/openApi/swap/v2/quote/premiumIndex", {"symbol": symbol})
        if r.get("code") == 0:
            data = r.get("data", {})
            try:
                return float(data.get("lastFundingRate", 0))
            except Exception:
                return 0.0
        return 0.0

    async def fetch_oi(self, symbol: str) -> float:
        r = await self._get("/openApi/swap/v2/quote/openInterest", {"symbol": symbol})
        if r.get("code") == 0:
            data = r.get("data", {})
            try:
                return float(data.get("openInterest", 0))
            except Exception:
                return 0.0
        return 0.0

    async def fetch_orderbook(self, symbol: str) -> tuple[float, float]:
        r = await self._get("/openApi/swap/v2/quote/depth", {"symbol": symbol, "limit": "5"})
        if r.get("code") == 0:
            data = r.get("data", {})
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            bid = float(bids[0][0]) if bids else 0.0
            ask = float(asks[0][0]) if asks else 0.0
            return bid, ask
        return 0.0, 0.0

    def _calc_rsi_approx(self, prices: list[float], n: int = 14) -> float:
        if len(prices) < 2:
            return 50.0
        diffs = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d for d in diffs if d > 0]
        losses = [-d for d in diffs if d < 0]
        avg_gain = sum(gains[-n:]) / n if gains else 0.0
        avg_loss = sum(losses[-n:]) / n if losses else 0.0001
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def _detect_anomalies(self, snap: MarketSnapshot) -> list[str]:
        anomalies = []
        if snap.oi_change_pct >= 5.0:
            anomalies.append(f"OI_EXPLOSION:+{snap.oi_change_pct:.1f}%")
        if snap.oi_change_pct <= -5.0:
            anomalies.append(f"OI_COLLAPSE:{snap.oi_change_pct:.1f}%")
        if snap.volume_ratio >= 3.0:
            anomalies.append(f"VOL_SURGE:{snap.volume_ratio:.1f}x")
        if abs(snap.funding_rate) >= 0.0005:
            direction = "HIGH" if snap.funding_rate > 0 else "LOW"
            anomalies.append(f"FUNDING_{direction}:{snap.funding_rate:.4f}")
        if snap.spread_bps >= 10:
            anomalies.append(f"WIDE_SPREAD:{snap.spread_bps:.1f}bps")
        if snap.rsi_approx <= 25:
            anomalies.append(f"RSI_OVERSOLD:{snap.rsi_approx:.0f}")
        if snap.rsi_approx >= 75:
            anomalies.append(f"RSI_OVERBOUGHT:{snap.rsi_approx:.0f}")
        if abs(snap.price_change_pct) >= 1.5 and snap.volume_ratio >= 2.0:
            direction = "UP" if snap.price_change_pct > 0 else "DOWN"
            anomalies.append(f"MOMENTUM_{direction}:{snap.price_change_pct:+.2f}%xVOL{snap.volume_ratio:.1f}x")
        return anomalies

    async def _snapshot_symbol(self, symbol: str) -> Optional[MarketSnapshot]:
        ticker, funding, oi, (bid, ask) = await asyncio.gather(
            self.fetch_ticker(symbol),
            self.fetch_funding(symbol),
            self.fetch_oi(symbol),
            self.fetch_orderbook(symbol),
        )
        if not ticker:
            return None

        try:
            price = float(ticker.get("lastPrice", 0))
            price_change_pct = float(ticker.get("priceChangePercent", 0))
            volume_24h = float(ticker.get("volume", 0))
            high_24h = float(ticker.get("highPrice", price))
            low_24h = float(ticker.get("lowPrice", price))
            avg_vol = float(ticker.get("quoteVolume", volume_24h)) / 24 if volume_24h > 0 else 1
            volume_ratio = volume_24h / avg_vol if avg_vol > 0 else 1.0
        except Exception:
            return None

        spread_bps = ((ask - bid) / price * 10000) if price > 0 and ask > bid else 0.0
        atr_pct = ((high_24h - low_24h) / price * 100) if price > 0 else 0.0

        prev_oi = self._prev_oi.get(symbol, oi)
        oi_change_pct = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0.0
        self._prev_oi[symbol] = oi

        price_history = self._prev_prices[symbol]
        price_history.append(price)
        if len(price_history) > 50:
            price_history.pop(0)
        rsi = self._calc_rsi_approx(price_history)

        if symbol == "BTC-USDT":
            self._btc_change = price_change_pct

        btc_regime = (
            "BULL" if self._btc_change >= 0.5 else
            "BEAR" if self._btc_change <= -0.5 else
            "NEUTRAL"
        )

        snap = MarketSnapshot(
            symbol=symbol,
            timestamp=time.time(),
            price=price,
            price_change_pct=price_change_pct,
            volume_24h=volume_24h,
            volume_ratio=volume_ratio,
            oi=oi,
            oi_change_pct=oi_change_pct,
            funding_rate=funding,
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            high_24h=high_24h,
            low_24h=low_24h,
            atr_pct=atr_pct,
            btc_regime=btc_regime,
            rsi_approx=rsi,
            anomalies=[],
        )
        snap.anomalies = self._detect_anomalies(snap)
        self._snapshots[symbol] = snap
        return snap

    async def snapshot_all(self) -> dict[str, MarketSnapshot]:
        results = await asyncio.gather(
            *[self._snapshot_symbol(s) for s in SYMBOLS],
            return_exceptions=True
        )
        snaps = {}
        for sym, res in zip(SYMBOLS, results):
            if isinstance(res, MarketSnapshot):
                snaps[sym] = res
                for cb in self._callbacks:
                    try:
                        await cb(res)
                    except Exception:
                        pass
        return snaps

    def get_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        return self._snapshots.get(symbol)

    def get_all_snapshots(self) -> dict[str, MarketSnapshot]:
        return dict(self._snapshots)

    def to_dict(self, snap: MarketSnapshot) -> dict:
        return {
            "symbol": snap.symbol,
            "timestamp": snap.timestamp,
            "price": snap.price,
            "price_change_pct": snap.price_change_pct,
            "volume_ratio": snap.volume_ratio,
            "oi": snap.oi,
            "oi_change_pct": snap.oi_change_pct,
            "funding_rate": snap.funding_rate,
            "spread_bps": snap.spread_bps,
            "atr_pct": snap.atr_pct,
            "rsi": snap.rsi_approx,
            "btc_regime": snap.btc_regime,
            "anomalies": snap.anomalies,
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
