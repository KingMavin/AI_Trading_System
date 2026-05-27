"""
ATS Trainer — main entry point.

Commands:
  run     Run a full training pipeline
  check   Run readiness checks only
  status  Show knowledge base status
  rollback Roll back to previous strategy

Usage:
  python trainer/trainer.py run
  python trainer/trainer.py run --symbol EURUSD
  python trainer/trainer.py run --quick
  python trainer/trainer.py check
  python trainer/trainer.py status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from trainer.core.trainer_runner import (
    TrainerRunner, DEFAULT_CONFIG, QUICK_CONFIG
)
from trainer.core.knowledge_base import KnowledgeBase


def cmd_run(args):
    """Run a full training pipeline."""
    config = QUICK_CONFIG.copy() if args.quick \
             else DEFAULT_CONFIG.copy()

    if args.symbol:
        config['symbols'] = [args.symbol]
    if args.start:
        config['data_start'] = args.start
    if args.end:
        config['data_end'] = args.end

    runner = TrainerRunner(
        config=config,
        quick_mode=args.quick
    )
    record = runner.run()

    # Exit code reflects outcome
    if record.outcome == 'COMPLETED':
        sys.exit(0)
    else:
        sys.exit(1)


def cmd_check(args):
    """Run readiness checks."""
    print("\nATS Trainer — Readiness Check")
    print("=" * 50)

    checks = []

    # Check 1: Data files exist
    from trainer.core.data_loader import get_available_range
    symbols    = ['EURUSD', 'GBPUSD', 'USDJPY']
    timeframes = ['M15', 'H1', 'H4']

    data_ok = True
    for sym in symbols:
        for tf in timeframes:
            info = get_available_range(sym, tf)
            if info.get('exists'):
                print(f"  ✓ {sym} {tf}: "
                      f"{info['count']:,} candles "
                      f"({info['start'].date()} → "
                      f"{info['end'].date()})")
            else:
                print(f"  ✗ {sym} {tf}: MISSING")
                data_ok = False

    checks.append(('Data files', data_ok))

    # Check 2: Indicators importable
    try:
        from shared.indicators import calculate_sma
        checks.append(('Indicators module', True))
        print(f"  ✓ indicators.py importable")
    except ImportError as e:
        checks.append(('Indicators module', False))
        print(f"  ✗ indicators.py: {e}")

    # Check 3: Knowledge base accessible
    try:
        kb_path = (
            Path(__file__).parent /
            'trainer_data' / 'history' / 'knowledge_base.json'
        )
        kb    = KnowledgeBase(base_path=str(kb_path))
        stats = kb.get_stats()
        print(
            f"  ✓ Knowledge base: "
            f"{stats['total_runs']} runs, "
            f"{stats['candidates_tested']} candidates"
        )
        checks.append(('Knowledge base', True))
    except Exception as e:
        print(f"  ✗ Knowledge base: {e}")
        checks.append(('Knowledge base', False))

    # Check 4: Required packages
    packages = [
        'pandas', 'numpy', 'pyarrow',
        'pandas_ta', 'MetaTrader5'
    ]
    pkg_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}: not installed")
            pkg_ok = False
    checks.append(('Packages', pkg_ok))

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in checks if ok)
    total  = len(checks)
    print(f"  Result: {passed}/{total} checks passed")

    if passed == total:
        print(f"  Status: READY")
        print(f"  Run: python trainer/trainer.py run")
    else:
        print(f"  Status: NOT READY")
        print(f"  Fix failing checks before running.")
    print("=" * 50 + "\n")

    sys.exit(0 if passed == total else 1)


def cmd_status(args):
    """Show knowledge base status."""
    kb_path = (
        Path(__file__).parent /
        'trainer_data' / 'history' / 'knowledge_base.json'
    )
    kb = KnowledgeBase(base_path=str(kb_path))
    kb.print_summary()


def cmd_rollback(args):
    """Roll back to a previous strategy."""
    print("\nRollback not yet implemented.")
    print("Archive location: engine/strategy/archive/")
    print("Manually copy the desired .json file to "
          "engine/strategy/active_strategy.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ATS Trainer — adaptive strategy training'
    )
    subparsers = parser.add_subparsers(dest='command')

    # run command
    run_parser = subparsers.add_parser(
        'run', help='Run full training pipeline'
    )
    run_parser.add_argument(
        '--symbol',
        choices=['EURUSD', 'GBPUSD', 'USDJPY'],
        help='Single symbol to train on'
    )
    run_parser.add_argument(
        '--start',
        help='Data start date YYYY-MM-DD'
    )
    run_parser.add_argument(
        '--end',
        help='Data end date YYYY-MM-DD'
    )
    run_parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode — smaller grid, shorter period'
    )
    run_parser.set_defaults(func=cmd_run)

    # check command
    check_parser = subparsers.add_parser(
        'check', help='Run readiness checks'
    )
    check_parser.set_defaults(func=cmd_check)

    # status command
    status_parser = subparsers.add_parser(
        'status', help='Show knowledge base status'
    )
    status_parser.set_defaults(func=cmd_status)

    # rollback command
    rollback_parser = subparsers.add_parser(
        'rollback', help='Roll back to previous strategy'
    )
    rollback_parser.set_defaults(func=cmd_rollback)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)