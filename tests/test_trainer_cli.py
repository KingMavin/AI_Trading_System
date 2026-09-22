import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import argparse
from trainer.trainer import build_parser

def test_trainer_cli_valid_symbols():
    parser = build_parser()
    # Test all 5 valid symbols
    valid_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'XAUUSD']
    for sym in valid_symbols:
        args = parser.parse_args(['run', '--symbol', sym])
        assert args.symbol == sym
        assert args.command == 'run'

def test_trainer_cli_invalid_symbol(capsys):
    parser = build_parser()
    # Test invalid symbol rejection
    with pytest.raises(SystemExit):
        parser.parse_args(['run', '--symbol', 'INVALID'])

    captured = capsys.readouterr()
    assert 'invalid choice' in captured.err


def test_trainer_cli_extended_args(monkeypatch):
    parser = build_parser()
    args = parser.parse_args([
        'run', '--symbol', 'EURUSD',
        '--start', '2021-01-01', '--end', '2023-12-31',
        '--opt-months', '6', '--test-months', '1',
        '--held-out-months', '2', '--min-trades-opt', '30',
        '--min-trades-test', '5', '--top-n', '5'
    ])
    assert args.symbol == 'EURUSD'
    assert args.start == '2021-01-01'
    assert args.end == '2023-12-31'
    assert args.opt_months == 6
    assert args.test_months == 1
    assert args.held_out_months == 2
    assert args.min_trades_opt == 30
    assert args.min_trades_test == 5
    assert args.top_n == 5

    captured_config = {}
    class DummyRunner:
        def __init__(self, config, quick_mode=False):
            nonlocal captured_config
            captured_config = config
        def run(self):
            class DummyRecord:
                outcome = 'COMPLETED'
            return DummyRecord()

    import trainer.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, 'TrainerRunner', DummyRunner)
    with pytest.raises(SystemExit) as exc_info:
        trainer_mod.cmd_run(args)
    assert exc_info.value.code == 0
    assert captured_config['symbols'] == ['EURUSD']
    assert captured_config['data_start'] == '2021-01-01'
    assert captured_config['data_end'] == '2023-12-31'
    assert captured_config['opt_months'] == 6
    assert captured_config['test_months'] == 1
    assert captured_config['held_out_months'] == 2
    assert captured_config['min_trades_opt'] == 30
    assert captured_config['min_trades_test'] == 5
    assert captured_config['top_n_candidates'] == 5

