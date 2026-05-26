"""
Performance metrics for backtesting results.
Used by both Trainer (offline) and Engine (degradation monitor).

All functions accept a list of trade dictionaries and return
a single float or dict of floats.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def calculate_metrics(trades: List[Dict],
                      equity_curve: List[float],
                      initial_equity: float = 10000.0) -> Dict:
    """
    Compute all performance metrics from a list of closed trades.

    Args:
        trades:         list of trade dicts (see backtester output)
        equity_curve:   list of equity values at each candle
        initial_equity: starting account equity

    Returns:
        dict of all computed metrics
    """
    if len(trades) == 0:
        return _empty_metrics()

    # ── Basic counts ───────────────────────────────────
    total_trades  = len(trades)
    net_pnls      = [t['net_pnl'] for t in trades]
    winners       = [p for p in net_pnls if p > 0]
    losers        = [p for p in net_pnls if p <= 0]

    win_count     = len(winners)
    loss_count    = len(losers)
    win_rate      = win_count / total_trades if total_trades > 0 else 0

    # ── P&L ───────────────────────────────────────────
    gross_profit  = sum(winners) if winners else 0
    gross_loss    = abs(sum(losers)) if losers else 0
    net_profit    = sum(net_pnls)

    profit_factor = (gross_profit / gross_loss
                     if gross_loss > 0 else float('inf'))

    avg_win       = np.mean(winners) if winners else 0
    avg_loss      = abs(np.mean(losers)) if losers else 0
    avg_win_loss  = (avg_win / avg_loss
                     if avg_loss > 0 else float('inf'))

    # ── Drawdown ──────────────────────────────────────
    equity_arr    = np.array(equity_curve)
    max_drawdown  = _calculate_max_drawdown(equity_arr)
    max_dd_pct    = max_drawdown / initial_equity * 100

    # ── Returns ───────────────────────────────────────
    total_return_pct = (net_profit / initial_equity) * 100

    # Annualised return — estimate from trade durations
    if len(trades) > 0:
        durations = [t.get('duration_candles', 1) for t in trades]
        # Assume M15 — 4 candles per hour, 96 per day, 252 trading days
        total_candles_traded = sum(durations)
        years = total_candles_traded / (96 * 252)
        years = max(years, 1/252)  # minimum 1 trading day
        annualised_return = (
            (1 + total_return_pct/100) ** (1/years) - 1
        ) * 100
    else:
        annualised_return = 0

    # ── Calmar Ratio ──────────────────────────────────
    calmar = (annualised_return / max_dd_pct
              if max_dd_pct > 0 else float('inf'))

    # ── Sharpe Ratio ──────────────────────────────────
    sharpe = _calculate_sharpe(net_pnls)

    # ── Consecutive losses ────────────────────────────
    max_consec_losses = _max_consecutive_losses(net_pnls)

    # ── Ulcer Index ───────────────────────────────────
    ulcer_index = _calculate_ulcer_index(equity_arr)

    return {
        # Counts
        'total_trades':         total_trades,
        'win_count':            win_count,
        'loss_count':           loss_count,
        'win_rate':             round(win_rate, 4),

        # P&L
        'gross_profit':         round(gross_profit, 2),
        'gross_loss':           round(gross_loss, 2),
        'net_profit':           round(net_profit, 2),
        'profit_factor':        round(profit_factor, 4),
        'avg_win':              round(avg_win, 2),
        'avg_loss':             round(avg_loss, 2),
        'avg_win_loss_ratio':   round(avg_win_loss, 4),

        # Returns
        'total_return_pct':     round(total_return_pct, 4),
        'annualised_return_pct':round(annualised_return, 4),

        # Risk
        'max_drawdown_usd':     round(max_drawdown, 2),
        'max_drawdown_pct':     round(max_dd_pct, 4),
        'max_consecutive_losses': max_consec_losses,
        'ulcer_index':          round(ulcer_index, 4),

        # Risk-adjusted
        'calmar_ratio':         round(calmar, 4),
        'sharpe_ratio':         round(sharpe, 4),
    }


def _calculate_max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in currency units."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(np.max(drawdown))


def _calculate_sharpe(pnls: List[float],
                       risk_free: float = 0.0) -> float:
    """Sharpe ratio. Risk-free rate = 0 for simplicity."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    mean = np.mean(arr) - risk_free
    std  = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    # Annualise assuming M15 — sqrt of candles per year
    candles_per_year = 96 * 252
    return float(mean / std * np.sqrt(candles_per_year))


def _max_consecutive_losses(pnls: List[float]) -> int:
    """Count the longest streak of losing trades."""
    max_streak = 0
    current    = 0
    for p in pnls:
        if p <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _calculate_ulcer_index(equity: np.ndarray) -> float:
    """
    Ulcer Index — measures depth and duration of drawdowns.
    Lower is better. Penalises prolonged drawdowns more than Calmar.
    """
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    pct_drawdown = (peak - equity) / peak * 100
    return float(np.sqrt(np.mean(pct_drawdown ** 2)))


def _empty_metrics() -> Dict:
    """Return zero metrics when no trades exist."""
    return {
        'total_trades': 0, 'win_count': 0, 'loss_count': 0,
        'win_rate': 0, 'gross_profit': 0, 'gross_loss': 0,
        'net_profit': 0, 'profit_factor': 0, 'avg_win': 0,
        'avg_loss': 0, 'avg_win_loss_ratio': 0,
        'total_return_pct': 0, 'annualised_return_pct': 0,
        'max_drawdown_usd': 0, 'max_drawdown_pct': 0,
        'max_consecutive_losses': 0, 'ulcer_index': 0,
        'calmar_ratio': 0, 'sharpe_ratio': 0,
    }


def print_metrics(metrics: Dict, title: str = "Performance") -> None:
    """Pretty print metrics to console."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Trades:          {metrics['total_trades']}")
    print(f"  Win Rate:        {metrics['win_rate']*100:.1f}%")
    print(f"  Profit Factor:   {metrics['profit_factor']:.3f}")
    print(f"  Net Profit:      ${metrics['net_profit']:,.2f}")
    print(f"  Total Return:    {metrics['total_return_pct']:.2f}%")
    print(f"  Ann. Return:     {metrics['annualised_return_pct']:.2f}%")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio:    {metrics['calmar_ratio']:.3f}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
    print(f"  Avg Win:         ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss:        ${metrics['avg_loss']:,.2f}")
    print(f"  Max Consec Loss: {metrics['max_consecutive_losses']}")
    print(f"{'='*50}\n")