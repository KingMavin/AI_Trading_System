"""
Download historical OHLCV data from MT5 and save to Parquet.
Idempotent — safe to re-run. Only downloads new candles.

Usage:
  python download_history.py
  python download_history.py --symbol EURUSD --timeframe M15
"""

import MetaTrader5 as mt5
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ── CONFIGURATION ──────────────────────────────────────
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
TIMEFRAMES = {
    'M15': mt5.TIMEFRAME_M15,
    'H1':  mt5.TIMEFRAME_H1,
    'H4':  mt5.TIMEFRAME_H4,
}
START_DATE = datetime(2015, 1, 1)
DATA_PATH  = Path(__file__).parent.parent / 'trainer_data' / 'historical'

def connect_mt5() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        log.error("Cannot get account info — check MT5 login")
        return False
    log.info(f"Connected: account {info.login} | {info.server}")
    return True

def download_symbol_timeframe(symbol: str,
                               timeframe_name: str,
                               timeframe_code: int) -> bool:
    output_path = DATA_PATH / symbol / f"{timeframe_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists — incremental update
    start = START_DATE
    if output_path.exists():
        existing = pq.read_metadata(output_path)
        # Read last timestamp from file
        df_existing = pd.read_parquet(output_path,
                                       columns=['timestamp'])
        last_ts = df_existing['timestamp'].max()
        start = last_ts.to_pydatetime()
        log.info(f"{symbol} {timeframe_name}: "
                 f"existing data to {last_ts.date()}, "
                 f"downloading from {start.date()}")
    else:
        log.info(f"{symbol} {timeframe_name}: "
                 f"fresh download from {start.date()}")

    # Download from MT5
    rates = mt5.copy_rates_from(symbol, timeframe_code,
                                 start, 99999)
    if rates is None or len(rates) == 0:
        log.warning(f"{symbol} {timeframe_name}: "
                    f"no data returned from MT5")
        return False

    # Convert to DataFrame
    df_new = pd.DataFrame(rates)
    df_new['timestamp'] = pd.to_datetime(df_new['time'], unit='s',
                                          utc=True)
    df_new = df_new.rename(columns={
        'open': 'open', 'high': 'high',
        'low': 'low', 'close': 'close',
        'tick_volume': 'volume',
        'real_volume': 'real_volume',
        'spread': 'spread',
    })
    df_new = df_new[['timestamp', 'open', 'high',
                      'low', 'close', 'volume', 'spread']]
    df_new = df_new.set_index('timestamp').sort_index()

    # Merge with existing if present
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(
            keep='last')]
        df_combined = df_combined.sort_index()
    else:
        df_combined = df_new

    # Save to Parquet
    df_combined.reset_index().to_parquet(
        output_path, compression='snappy', index=False
    )
    log.info(f"{symbol} {timeframe_name}: "
             f"saved {len(df_combined):,} candles "
             f"({df_combined.index.min().date()} → "
             f"{df_combined.index.max().date()})")
    return True

def main(symbols=None, timeframes=None):
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    if not connect_mt5():
        return

    results = {}
    for symbol in symbols:
        for tf_name, tf_code in timeframes.items():
            success = download_symbol_timeframe(
                symbol, tf_name, tf_code
            )
            results[f"{symbol}_{tf_name}"] = success

    mt5.shutdown()

    passed = sum(results.values())
    total = len(results)
    log.info(f"Download complete: {passed}/{total} successful")
    for key, success in results.items():
        status = "✓" if success else "✗"
        log.info(f"  {status} {key}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help='Single symbol to download')
    parser.add_argument('--timeframe', help='Single timeframe')
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else None
    tfs = ({args.timeframe: TIMEFRAMES[args.timeframe]}
           if args.timeframe else None)
    main(symbols=syms, timeframes=tfs)