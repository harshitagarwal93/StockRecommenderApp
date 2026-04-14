"""Compute technical indicators from OHLCV price data (pure pandas/numpy)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import TechnicalIndicators


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if np.isfinite(val) else 50.0


def _macd(close: pd.Series) -> tuple[float, float, float]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return (
        round(float(macd_line.iloc[-1]), 2),
        round(float(signal.iloc[-1]), 2),
        round(float(hist.iloc[-1]), 2),
    )


def _sma(close: pd.Series, period: int) -> float:
    val = close.rolling(window=period).mean().iloc[-1]
    return round(float(val), 2) if np.isfinite(val) else 0.0


def _ema(close: pd.Series, period: int) -> float:
    val = close.ewm(span=period, adjust=False).mean().iloc[-1]
    return round(float(val), 2) if np.isfinite(val) else 0.0


def _bollinger(close: pd.Series, period: int = 20) -> tuple[float, float, float]:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return (
        round(float(upper.iloc[-1]), 2),
        round(float(sma.iloc[-1]), 2),
        round(float(lower.iloc[-1]), 2),
    )


def _pct_from_high_low(close: pd.Series, window: int = 252) -> tuple[float, float]:
    recent = close.tail(window)
    high = recent.max()
    low = recent.min()
    current = close.iloc[-1]
    pct_high = round(((current - high) / high) * 100, 2) if high else 0.0
    pct_low = round(((current - low) / low) * 100, 2) if low else 0.0
    return pct_high, pct_low


def _price_change_pct(close: pd.Series, days: int) -> float:
    if len(close) < days:
        return 0.0
    old = close.iloc[-days]
    current = close.iloc[-1]
    return round(((current - old) / old) * 100, 2) if old else 0.0


def compute_indicators(ticker: str, df: pd.DataFrame) -> TechnicalIndicators:
    """Compute all technical indicators from an OHLCV DataFrame."""
    close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    volume = df["Volume"].squeeze() if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]

    rsi_val = _rsi(close)
    macd_val, macd_sig, macd_hist = _macd(close)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    pct_high, pct_low = _pct_from_high_low(close)

    vol_avg_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
    current_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0.0

    return TechnicalIndicators(
        ticker=ticker,
        current_price=round(float(close.iloc[-1]), 2),
        rsi_14=rsi_val,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        sma_50=sma50,
        sma_200=sma200,
        ema_20=ema20,
        bollinger_upper=bb_upper,
        bollinger_middle=bb_mid,
        bollinger_lower=bb_lower,
        volume_avg_20=round(vol_avg_20),
        current_volume=round(current_vol),
        pct_from_52w_high=pct_high,
        pct_from_52w_low=pct_low,
        price_change_1m=_price_change_pct(close, 22),
        price_change_3m=_price_change_pct(close, 66),
        price_change_6m=_price_change_pct(close, 132),
    )


def composite_score(tech: TechnicalIndicators) -> float:
    """Heuristic score (0-100) combining multiple technical signals."""
    score = 50.0

    if tech.rsi_14 < 35:
        score += 12
    elif tech.rsi_14 < 45:
        score += 6
    elif tech.rsi_14 > 75:
        score -= 10
    elif tech.rsi_14 > 65:
        score -= 4

    if tech.macd_histogram > 0:
        score += 8
    else:
        score -= 4

    if tech.current_price > tech.sma_200 > 0:
        score += 6
    if tech.current_price > tech.sma_50 > 0:
        score += 4

    if tech.bollinger_lower > 0:
        bb_range = tech.bollinger_upper - tech.bollinger_lower
        if bb_range > 0:
            position = (tech.current_price - tech.bollinger_lower) / bb_range
            if position < 0.25:
                score += 8
            elif position > 0.80:
                score -= 6

    if tech.price_change_3m > 5:
        score += 5
    elif tech.price_change_3m < -10:
        score -= 5

    if tech.volume_avg_20 > 0 and tech.current_volume > tech.volume_avg_20 * 1.2:
        score += 3

    return max(0, min(100, score))
