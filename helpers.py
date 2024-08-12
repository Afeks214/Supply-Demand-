# helpers.py

import numpy as np
import pandas as pd
from typing import Union, List
from datetime import datetime, time
import MetaTrader5 as mt5

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate the Exponential Moving Average (EMA)."""
    return close.ewm(span=period, adjust=False).mean()

def is_within_trading_hours(current_time: datetime, trading_hours: Dict[str, List[str]]) -> bool:
    """Check if the current time is within the specified trading hours."""
    day_of_week = current_time.strftime('%A')
    if day_of_week not in trading_hours:
        return False
    
    current_time = current_time.time()
    for time_range in trading_hours[day_of_week]:
        start, end = map(lambda x: datetime.strptime(x, '%H:%M').time(), time_range.split('-'))
        if start <= current_time <= end:
            return True
    return False

def format_number(number: float, decimals: int = 2) -> str:
    """Format a number with the specified number of decimal places."""
    return f"{number:.{decimals}f}"

def get_mt5_timeframe(timeframe: str) -> int:
    """Convert string timeframe to MT5 timeframe constant."""
    timeframe_map = {
        '1m': mt5.TIMEFRAME_M1,
        '5m': mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15,
        '30m': mt5.TIMEFRAME_M30,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4,
        '1d': mt5.TIMEFRAME_D1,
        '1w': mt5.TIMEFRAME_W1,
        '1mn': mt5.TIMEFRAME_MN1
    }
    return timeframe_map.get(timeframe.lower(), mt5.TIMEFRAME_M1)  # Default to 1 minute if not found

def round_to_tick_size(price: float, tick_size: float) -> float:
    """Round the given price to the nearest valid price based on tick size."""
    return round(price / tick_size) * tick_size

def calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """Calculate pivot points (PP, S1, S2, S3, R1, R2, R3)."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    
    return {
        'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
        'S1': s1, 'S2': s2, 'S3': s3
    }

def normalize_signal(signal: float, min_value: float = -1, max_value: float = 1) -> float:
    """Normalize a signal to be between min_value and max_value."""
    return (signal - min_value) / (max_value - min_value) * 2 - 1

if __name__ == "__main__":
    # Example usage and testing of functions
    print(format_number(3.14159, 3))  # Output: 3.142
    print(get_mt5_timeframe('1h'))  # Output: 16385 (MT5 constant for 1 hour timeframe)
    print(round_to_tick_size(1.23456, 0.00001))  # Output: 1.23456
    print(calculate_pivot_points(100, 90, 95))  # Output: Dictionary of pivot points
    print(normalize_signal(0.5, 0, 1))  # Output: 0.0
