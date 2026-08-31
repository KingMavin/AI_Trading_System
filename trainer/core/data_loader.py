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

def merge_historical_and_live_data(df_historical: pd.DataFrame,
                                   df_live: pd.DataFrame,
                                   price_tolerance: float = 0.001) -> pd.DataFrame:
    """
    Merge historical Parquet/CSV dataset with live log dataset into a single, unified OHLCV dataset.

    Rules & Mechanics:
      1. Converts timestamps to DatetimeIndex (UTC) if not already set.
      2. Validates schema: open, high, low, close, volume columns required.
      3. Price Contradiction Check: On overlapping timestamps, verifies price discrepancy does not exceed price_tolerance.
      4. Deduplication: Live log candles take precedence over historical candles on duplicate timestamps.
      5. Chronological Sorting: Sorts output dataset by DatetimeIndex.
      6. Integrity Check: Fails loudly (ValueError) if any NaN values exist in merged OHLCV data (AGENTS.md §3).

    Args:
        df_historical: Primary historical DataFrame.
        df_live: Live engine log or forward-test DataFrame.
        price_tolerance: Maximum allowable price difference on close price for overlapping timestamps.

    Returns:
        pd.DataFrame: Merged, deduplicated, chronologically sorted DataFrame with DatetimeIndex (UTC).
    """
    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'timestamp' in df_out.columns:
            df_out['timestamp'] = pd.to_datetime(df_out['timestamp'], utc=True)
            df_out = df_out.set_index('timestamp')
        elif not isinstance(df_out.index, pd.DatetimeIndex):
            df_out.index = pd.to_datetime(df_out.index, utc=True)
        elif df_out.index.tz is None:
            df_out.index = df_out.index.tz_localize('UTC')
        return df_out.sort_index()

    h = _prep_df(df_historical)
    l = _prep_df(df_live)

    req_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in req_cols:
        if c not in h.columns or c not in l.columns:
            raise ValueError(f"Required OHLCV column '{c}' missing from input datasets")

    # Check overlapping timestamps for price contradictions
    # NOTE: Price-contradiction check compares close price only; full OHLC/tick-level multi-broker reconciliation across all four price fields is deferred to WAVE 4.
    overlap_times = h.index.intersection(l.index)
    if len(overlap_times) > 0:
        h_close = h.loc[overlap_times, 'close']
        l_close = l.loc[overlap_times, 'close']
        diff = (h_close - l_close).abs()
        max_diff = diff.max()
        if max_diff > price_tolerance:
            import logging
            logging.getLogger(__name__).warning(
                f"Price contradiction detected on {len(overlap_times)} overlapping candles: "
                f"max close diff={max_diff:.5f} > tolerance={price_tolerance:.5f}. "
                f"Live dataset values will overwrite historical values."
            )

    # Combine with live data appended after historical data so keep='last' preserves live data
    merged = pd.concat([h, l])
    merged = merged[~merged.index.duplicated(keep='last')].sort_index()

    # Fail loudly on NaN values per AGENTS.md §3
    nan_cols = merged[req_cols].isna().sum()
    if nan_cols.sum() > 0:
        raise ValueError(f"NaN values detected in merged OHLCV dataset: {nan_cols.to_dict()}")

    return merged