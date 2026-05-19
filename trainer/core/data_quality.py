"""
Data quality checks for OHLCV Parquet files.
Run after download to verify data integrity.

Usage:
  python data_quality.py
  python data_quality.py --symbol EURUSD --timeframe M15
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging

log = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {'M15': 15, 'H1': 60, 'H4': 240}
DATA_PATH = Path(__file__).parent.parent / 'trainer_data' / 'historical'

def check_dataset(symbol: str, timeframe: str) -> dict:
    path = DATA_PATH / symbol / f"{timeframe}.parquet"
    if not path.exists():
        return {'status': 'MISSING', 'symbol': symbol,
                'timeframe': timeframe}

    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()

    report = {
        'symbol': symbol, 'timeframe': timeframe,
        'total_candles': len(df),
        'date_start': str(df.index.min().date()),
        'date_end': str(df.index.max().date()),
        'issues': [], 'warnings': [], 'critical': []
    }

    # Check 1: Duplicates
    dups = df.index.duplicated().sum()
    if dups > 0:
        report['issues'].append(f"Duplicates: {dups} rows")

    # Check 2: OHLC sanity
    ohlc_violations = (
        (df['high'] < df[['open', 'close']].max(axis=1)) |
        (df['low']  > df[['open', 'close']].min(axis=1)) |
        (df['high'] < df['low'])
    ).sum()
    if ohlc_violations > 0:
        report['issues'].append(
            f"OHLC violations: {ohlc_violations} rows"
        )

    # Check 3: Gaps (excluding weekends)
    tf_mins = TIMEFRAME_MINUTES[timeframe]
    expected_interval = pd.Timedelta(minutes=tf_mins)
    diffs = df.index.to_series().diff().dropna()
    business_diffs = diffs[
        ~((diffs > pd.Timedelta(hours=48)) &
          (df.index[1:].dayofweek.isin([5, 6])))
    ]
    gaps = business_diffs[
        business_diffs > expected_interval * 1.5
    ]
    if len(gaps) > 0:
        small_gaps = gaps[gaps <= expected_interval * 5]
        large_gaps = gaps[gaps > expected_interval * 5]
        if len(small_gaps) > 0:
            report['warnings'].append(
                f"Small gaps (< 5 candles): {len(small_gaps)}"
            )
        if len(large_gaps) > 0:
            report['critical'].append(
                f"Large gaps (>= 5 candles): {len(large_gaps)}"
            )

    # Check 4: Zero volume
    zero_vol = (df['volume'] == 0).sum()
    zero_vol_pct = zero_vol / len(df) * 100
    if zero_vol_pct > 5:
        report['warnings'].append(
            f"Zero volume: {zero_vol} candles ({zero_vol_pct:.1f}%)"
        )

    # Check 5: Minimum data
    years = (df.index.max() - df.index.min()).days / 365
    if years < 3:
        report['critical'].append(
            f"Insufficient history: {years:.1f} years (need 3+)"
        )

    # Overall status
    if report['critical']:
        report['status'] = 'CRITICAL'
    elif report['issues']:
        report['status'] = 'ISSUES'
    elif report['warnings']:
        report['status'] = 'WARNINGS'
    else:
        report['status'] = 'OK'

    return report

def run_all_checks(symbols=None, timeframes=None):
    symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']
    timeframes = timeframes or ['M15', 'H1', 'H4']

    all_reports = []
    for symbol in symbols:
        for tf in timeframes:
            report = check_dataset(symbol, tf)
            all_reports.append(report)
            status = report['status']
            candles = report.get('total_candles', 0)
            print(f"  [{status:8}] {symbol} {tf}: "
                  f"{candles:,} candles")
            for issue in report.get('critical', []):
                print(f"             ✗ CRITICAL: {issue}")
            for issue in report.get('issues', []):
                print(f"             ✗ ISSUE: {issue}")
            for warn in report.get('warnings', []):
                print(f"             ⚠ WARNING: {warn}")

    critical_count = sum(
        1 for r in all_reports if r['status'] == 'CRITICAL'
    )
    print(f"\nResult: {critical_count} critical issues found")
    return all_reports

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol')
    parser.add_argument('--timeframe')
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else None
    tfs  = [args.timeframe] if args.timeframe else None
    run_all_checks(symbols=syms, timeframes=tfs)