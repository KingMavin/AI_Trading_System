r"""
Load MetaTrader exported CSV data and convert to Parquet.
Handles space/tab separated MT format.
Resamples M1 data up to M15, H1, H4 automatically.

Your file format:
  <DATE>     <TIME>     <OPEN>  <HIGH>  <LOW>   <CLOSE> <TICKVOL> <VOL> <SPREAD>
  1985.05.20 00:00:00   0.6445  0.6457  0.6442  0.6445  301       0     50

Usage:
  python load_csv_data.py
  python load_csv_data.py --input C:\path\to\EURUSD_M1.csv --symbol EURUSD
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# ── CONFIGURATION ──────────────────────────────────────
# UPDATE THESE TO MATCH YOUR FILE LOCATIONS
CSV_FILES = {
    'EURUSD': Path(r'D:\work\sy\ats\data\EURUSD_M1.csv'),
    'GBPUSD': Path(r'D:\work\sy\ats\data\GBPUSD_M1.csv'),
    'USDJPY': Path(r'D:\work\sy\ats\data\USDJPY_M1.csv'),
}

# Where Parquet files will be saved
OUTPUT_DIR = Path(__file__).parent.parent / 'trainer_data' / 'historical'

# Timeframes to generate from M1 data
RESAMPLE_TARGETS = {
    'M15': '15min',
    'H1':  '1h',
    'H4':  '4h',
}


def load_m1_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a MetaTrader M1 CSV file.
    Handles space/tab separated format with <COLUMN> headers.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    log.info(f"Reading {filepath.name} ...")

    # Read with flexible whitespace separator
    df = pd.read_csv(
        filepath,
        sep=r'\s+',        # handles any whitespace (spaces or tabs)
        header=0,
        engine='python'
    )

    # Clean column names — strip < > and lowercase
    df.columns = [c.strip().strip('<>').lower()
                  for c in df.columns]

    log.info(f"Columns: {list(df.columns)}")
    log.info(f"Raw rows: {len(df):,}")

    # Build timestamp
    df['timestamp'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str),
        format='%Y.%m.%d %H:%M:%S',
        utc=True,
        errors='coerce'
    )

    # Drop rows where timestamp parsing failed
    bad_ts = df['timestamp'].isna().sum()
    if bad_ts > 0:
        log.warning(f"Dropping {bad_ts} rows with unparseable timestamps")
        df = df.dropna(subset=['timestamp'])

    # Rename columns
    df = df.rename(columns={
        'tickvol': 'volume',
        'vol':     'real_volume',
        'spread':  'spread',
    })

    # Keep core columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'spread' in df.columns:
        cols.append('spread')
    df = df[cols].copy()

    # Set index and sort
    df = df.set_index('timestamp').sort_index()

    # Remove duplicates (keep first)
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    if len(df) < before:
        log.warning(f"Removed {before - len(df)} duplicate timestamps")

    # Remove clearly bad OHLC rows
    bad = (
        (df['high'] < df['low']) |
        (df['open'] <= 0) |
        (df['close'] <= 0) |
        (df['high'] <= 0) |
        (df['low'] <= 0)
    )
    if bad.sum() > 0:
        log.warning(f"Removing {bad.sum()} rows with invalid OHLC")
        df = df[~bad]

    log.info(
        f"Clean M1 rows: {len(df):,} | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    return df


def resample_ohlcv(df: pd.DataFrame,
                   rule: str) -> pd.DataFrame:
    """
    Resample M1 OHLCV data to a higher timeframe.

    OHLCV resampling rules:
      open:   first value in period
      high:   max value in period
      low:    min value in period
      close:  last value in period
      volume: sum of all M1 volumes in period
    """
    agg = {
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }
    if 'spread' in df.columns:
        agg['spread'] = 'mean'

    resampled = df.resample(rule, label='left',
                             closed='left').agg(agg)

    # Drop incomplete periods (NaN close = no data in that period)
    before = len(resampled)
    resampled = resampled.dropna(subset=['close'])
    dropped = before - len(resampled)
    if dropped > 0:
        log.info(f"Dropped {dropped} empty periods after resampling")

    # Remove weekend candles (Saturday=5, Sunday=6)
    resampled = resampled[resampled.index.dayofweek < 5]

    return resampled


def save_parquet(df: pd.DataFrame,
                 symbol: str,
                 timeframe: str) -> Path:
    """Save DataFrame to Parquet."""
    out_dir = OUTPUT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timeframe}.parquet"

    df.reset_index().to_parquet(
        out_path,
        compression='snappy',
        index=False
    )
    log.info(
        f"Saved {symbol} {timeframe}: "
        f"{len(df):,} candles → {out_path}"
    )
    return out_path


def process_symbol(symbol: str,
                   csv_path: Path) -> dict:
    """
    Full pipeline for one symbol:
    1. Load M1 CSV
    2. Resample to M15, H1, H4
    3. Save each as Parquet
    """
    results = {}

    try:
        df_m1 = load_m1_csv(csv_path)
    except Exception as e:
        log.error(f"Failed to load {csv_path}: {e}")
        return {tf: f'LOAD_ERROR: {e}' for tf in RESAMPLE_TARGETS}

    for tf_name, tf_rule in RESAMPLE_TARGETS.items():
        try:
            log.info(f"Resampling {symbol} M1 → {tf_name}...")
            df_resampled = resample_ohlcv(df_m1, tf_rule)
            save_parquet(df_resampled, symbol, tf_name)

            results[tf_name] = {
                'status':  'OK',
                'candles': len(df_resampled),
                'start':   str(df_resampled.index.min().date()),
                'end':     str(df_resampled.index.max().date()),
            }
        except Exception as e:
            log.error(f"Failed to resample {symbol} {tf_name}: {e}")
            results[tf_name] = {'status': f'ERROR: {e}'}

    return results


def main(symbol_filter: str = None,
         csv_override: Path = None):
    """Run conversion for all configured symbols."""

    files = CSV_FILES
    if symbol_filter and csv_override:
        files = {symbol_filter: csv_override}
    elif symbol_filter:
        files = {symbol_filter: CSV_FILES[symbol_filter]}

    all_results = {}
    for symbol, csv_path in files.items():
        log.info(f"\n{'='*50}")
        log.info(f"Processing {symbol} from {csv_path.name}")
        log.info(f"{'='*50}")
        results = process_symbol(symbol, csv_path)
        all_results[symbol] = results

    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    for symbol, tf_results in all_results.items():
        print(f"\n{symbol}:")
        for tf, result in tf_results.items():
            if isinstance(result, dict) and result['status'] == 'OK':
                print(f"  ✓ {tf}: {result['candles']:,} candles "
                      f"({result['start']} → {result['end']})")
            else:
                print(f"  ✗ {tf}: {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--symbol',
        help='Process single symbol (e.g. EURUSD)',
        default=None
    )
    parser.add_argument(
        '--input',
        help='Path to CSV file (used with --symbol)',
        default=None
    )
    args = parser.parse_args()

    csv_path = Path(args.input) if args.input else None
    main(symbol_filter=args.symbol, csv_override=csv_path)