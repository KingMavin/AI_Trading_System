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
        'symbol':        symbol,
        'timeframe':     timeframe,
        'total_candles': len(df),
        'date_start':    str(df.index.min().date()),
        'date_end':      str(df.index.max().date()),
        'years_covered': round(
            (df.index.max() - df.index.min()).days / 365, 1
        ),
        'issues':   [],
        'warnings': [],
        'critical': []
    }

    # Check 1: Duplicates
    dups = df.index.duplicated().sum()
    if dups > 0:
        report['issues'].append(f"Duplicates: {dups} rows")

    # Check 2: OHLC sanity
    violations = (
        (df['high'] < df[['open', 'close']].max(axis=1)) |
        (df['low']  > df[['open', 'close']].min(axis=1)) |
        (df['high'] < df['low'])
    ).sum()
    if violations > 0:
        report['issues'].append(
            f"OHLC violations: {violations} rows"
        )

    # Check 3: Smart gap detection
    # Excludes weekends, daily rollover, and known holidays
    tf_mins = TIMEFRAME_MINUTES[timeframe]
    expected = pd.Timedelta(minutes=tf_mins)

    diffs = df.index.to_series().diff().dropna()
    all_gaps = diffs[diffs > expected * 1.5]

    if len(all_gaps) > 0:
        gap_start_ts = pd.DatetimeIndex(
            all_gaps.index - all_gaps.values
        )
        gap_end_ts = pd.DatetimeIndex(all_gaps.index)

        # Expected gap types:
        # 1. Weekend: gap starts on Friday (dayofweek=4)
        is_friday = gap_start_ts.dayofweek == 4

        # 2. Weekend: gap starts on Saturday (dayofweek=5)
        #    (some brokers close mid-Saturday)
        is_saturday = gap_start_ts.dayofweek == 5

        # 3. Daily broker rollover: gap <= 4 hours
        is_short = all_gaps <= pd.Timedelta(hours=4)

        # 4. Holiday gaps: gaps that END on Dec 26, 27, 28
        #    or Jan 2, 3, 4 (Christmas + New Year period)
        #    These are always legitimate
        is_xmas_gap = (
            (gap_end_ts.month == 12) &
            (gap_end_ts.day.isin([26, 27, 28]))
        )
        is_newyear_gap = (
            (gap_end_ts.month == 1) &
            (gap_end_ts.day.isin([2, 3, 4]))
        )

        # 5. Any gap <= 2 days is almost certainly a holiday
        #    (Easter, Thanksgiving, national holidays)
        is_short_holiday = all_gaps <= pd.Timedelta(days=5)

        is_expected = (
            is_friday | is_saturday | is_short |
            is_xmas_gap | is_newyear_gap | is_short_holiday
        )

        unexpected = all_gaps[~is_expected]

        if len(unexpected) > 0:
            report['critical'].append(
                f"Genuinely unexpected gaps: {len(unexpected)} "
                f"— investigate manually"
            )

    # Check 4: Zero volume candles
    zero_vol_pct = (df['volume'] == 0).sum() / len(df) * 100
    if zero_vol_pct > 5:
        report['warnings'].append(
            f"High zero-volume rate: {zero_vol_pct:.1f}%"
        )

    # Check 5: Minimum data requirement
    years = (df.index.max() - df.index.min()).days / 365
    if years < 3:
        report['critical'].append(
            f"Only {years:.1f} years of data (minimum: 3)"
        )

    # Status
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

def investigate_gaps(symbol: str, timeframe: str, n: int = 20):
    """Show the largest unexpected gaps for investigation."""
    path = DATA_PATH / symbol / f"{timeframe}.parquet"
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()

    tf_mins = TIMEFRAME_MINUTES[timeframe]
    expected = pd.Timedelta(minutes=tf_mins)
    diffs = df.index.to_series().diff().dropna()
    all_gaps = diffs[diffs > expected * 1.5]

    gap_start_ts = all_gaps.index - all_gaps.values
    is_friday = pd.DatetimeIndex(gap_start_ts).dayofweek == 4
    is_short  = all_gaps <= pd.Timedelta(hours=4)
    unexpected = all_gaps[~is_friday & ~is_short]

    print(f"\nTop {n} unexpected gaps in {symbol} {timeframe}:")
    print(f"{'Gap Start':<35} {'Gap End':<35} {'Duration':<20} {'Day'}")
    print("-" * 100)
    for end_ts, gap in unexpected.nlargest(n).items():
        start_ts = end_ts - gap
        day_name = start_ts.strftime('%A')
        print(f"{str(start_ts):<35} {str(end_ts):<35} {str(gap):<20} {day_name}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol')
    parser.add_argument('--timeframe')
    parser.add_argument('--investigate', action='store_true',
                        help='Show largest unexpected gaps')
    args = parser.parse_args()

    if args.investigate:
        sym = args.symbol or 'EURUSD'
        tf  = args.timeframe or 'M15'
        investigate_gaps(sym, tf)
    else:
        syms = [args.symbol] if args.symbol else None
        tfs  = [args.timeframe] if args.timeframe else None
        run_all_checks(symbols=syms, timeframes=tfs)