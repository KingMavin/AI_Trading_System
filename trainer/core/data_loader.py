"""
Utility for loading and slicing historical OHLCV data.
Used by both Trainer layers and eventually Engine startup.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

DATA_PATH = Path(__file__).parent.parent / 'trainer_data' / 'historical'

def load_ohlcv(symbol: str,
               timeframe: str,
               start: datetime = None,
               end: datetime = None,
               warmup_candles: int = 0) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol+timeframe.
    Optionally slice to a date range with warmup buffer.

    Args:
        symbol:         e.g. 'EURUSD'
        timeframe:      e.g. 'M15'
        start:          inclusive start date (None = all)
        end:            inclusive end date (None = all)
        warmup_candles: extra candles before start for indicators

    Returns:
        DataFrame with DatetimeIndex (UTC), columns:
        open, high, low, close, volume, spread
    """
    path = DATA_PATH / symbol / f"{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for {symbol} {timeframe}. "
            f"Run download_history.py first."
        )

    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()

    if start is not None or end is not None:
        # Apply warmup buffer before slicing
        if start is not None and warmup_candles > 0:
            start_idx = df.index.searchsorted(
                pd.Timestamp(start, tz='UTC')
            )
            warmup_start_idx = max(0, start_idx - warmup_candles)
            warmup_start = df.index[warmup_start_idx]
            df = df.loc[warmup_start:]

        if start is not None and warmup_candles == 0:
            df = df.loc[pd.Timestamp(start, tz='UTC'):]

        if end is not None:
            df = df.loc[:pd.Timestamp(end, tz='UTC')]

    return df

def get_available_range(symbol: str, timeframe: str) -> dict:
    """Return the date range available for a symbol+timeframe."""
    path = DATA_PATH / symbol / f"{timeframe}.parquet"
    if not path.exists():
        return {'exists': False}

    df = pd.read_parquet(path, columns=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return {
        'exists': True,
        'start': df['timestamp'].min(),
        'end':   df['timestamp'].max(),
        'count': len(df),
    }