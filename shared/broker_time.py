"""
shared/broker_time.py

DST-aware broker-time <-> UTC conversion.

WHY THIS EXISTS (see Engine PRD §4.5, Trainer PRD §5.2):
MetaQuotes-Demo's server clock runs broker time, not UTC. Server time was
observed to align with EET/EEST (UTC+2 winter, UTC+3 during EU daylight
saving). MT5 returns all timestamps (candles, ticks, deals, orders) in
this broker-server time with NO timezone marker attached. Treating that
raw value as UTC silently shifts every session label (LONDON, NEW_YORK,
ASIAN, LONDON_NY_OVERLAP, DEAD_ZONE) by 2-3 hours, and shifts the Dead
Zone block window by the same amount.

This module is the single source of truth for that conversion. It is used
by:
  - data_loader.py / load_csv_data.py at ingestion (Trainer, historical)
  - engine/core/engine.py at candle fetch (Engine, live)
Both call sites MUST use this module rather than re-implementing the
offset logic, or live and historical session labels will disagree again.

DESIGN NOTES
- The IANA tz database has no zone literally named "EET/EEST" as MT5
  reports it, but several zones follow the identical EU DST transition
  rule (last Sunday of March -> last Sunday of October) at a UTC+2/+3
  base offset: Europe/Bucharest, Europe/Helsinki, Europe/Kyiv, Europe/Sofia,
  Europe/Athens, Asia/Nicosia. We use Europe/Bucharest as the reference
  because it carries no unusual historical offset changes.
- This encodes an ASSUMPTION (broker == EU EET/EEST DST calendar), not a
  guarantee. It must be validated against a live MT5 timestamp during
  market hours (see verify_live_offset() below) before being trusted for
  real trading. Per the state file: this was pending confirmation as of
  the last session (Monday market-hours check, not yet run).
- All functions here are naive-datetime in, naive-datetime out, by design:
  MT5 returns naive datetimes/timestamps with no tzinfo, and Parquet
  columns in this project are naive-UTC-labeled (not tz-aware pandas
  columns), matching existing pipeline conventions. Do not introduce
  tz-aware pandas Timestamps here; convert via the offset instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# Reference IANA zone that follows the EU EET/EEST DST rule assumed for the
# broker server. Swap this in one place if the broker turns out to follow a
# different calendar (e.g. a non-EU DST rule, or no DST at all).
_BROKER_TZ_NAME = "Europe/Bucharest"
_BROKER_TZ = ZoneInfo(_BROKER_TZ_NAME)
_UTC = ZoneInfo("UTC")


class BrokerTimeError(Exception):
    """Raised when a broker-time conversion cannot be performed safely."""


def broker_offset_hours(reference_broker_time: datetime) -> float:
    """
    Return the broker's UTC offset in hours (2.0 or 3.0 for EET/EEST) that
    is in effect at the given broker-local naive datetime.

    `reference_broker_time` must be a naive datetime interpreted as
    broker-server local time (this is what MT5 gives you).
    """
    if reference_broker_time.tzinfo is not None:
        raise BrokerTimeError(
            "reference_broker_time must be naive (no tzinfo) -- it is "
            "interpreted as broker-local time, not any absolute instant."
        )
    localized = reference_broker_time.replace(tzinfo=_BROKER_TZ)
    offset = localized.utcoffset()
    if offset is None:
        raise BrokerTimeError("Could not resolve UTC offset for broker time.")
    return offset.total_seconds() / 3600.0


def broker_to_utc(broker_dt: datetime) -> datetime:
    """
    Convert a naive broker-server-time datetime to a naive true-UTC
    datetime.

    Example: broker_dt = 2026-07-15 12:00:00 (server, EEST, UTC+3)
             returns    2026-07-15 09:00:00 (true UTC, naive)
    """
    if broker_dt.tzinfo is not None:
        raise BrokerTimeError(
            "broker_to_utc expects a naive datetime (broker-local, no "
            "tzinfo attached). Strip tzinfo before calling."
        )
    localized = broker_dt.replace(tzinfo=_BROKER_TZ)
    as_utc = localized.astimezone(_UTC)
    return as_utc.replace(tzinfo=None)


def utc_to_broker(utc_dt: datetime) -> datetime:
    """
    Convert a naive true-UTC datetime to naive broker-server-time.
    Inverse of broker_to_utc(). Provided for symmetry / debugging tools
    (e.g. "what will the broker clock read when I want to place an order
    at true UTC time X").
    """
    if utc_dt.tzinfo is not None:
        raise BrokerTimeError(
            "utc_to_broker expects a naive datetime (true UTC, no tzinfo "
            "attached). Strip tzinfo before calling."
        )
    localized = utc_dt.replace(tzinfo=_UTC)
    as_broker = localized.astimezone(_BROKER_TZ)
    return as_broker.replace(tzinfo=None)


def broker_series_to_utc(broker_times: pd.Series) -> pd.Series:
    """
    Vectorised conversion for an entire column of naive broker-time
    timestamps (e.g. the <DATE>+<TIME> column parsed from an MT5 CSV
    export, or a DataFrame index).

    Correct across the DST boundary within the same series: each
    timestamp is resolved to its own correct offset (2h or 3h), not a
    single offset applied to the whole column. This matters because a
    single load_csv_data.py run frequently spans multiple years and
    therefore multiple DST transitions.

    Input: pandas Series of naive datetime64[ns] (or convertible).
    Output: pandas Series of naive datetime64[ns], true UTC.
    """
    s = pd.to_datetime(broker_times)
    if s.dt.tz is not None:
        raise BrokerTimeError(
            "broker_series_to_utc expects a naive (tz-unaware) series."
        )
    # Localize to broker tz (resolves DST per-row), convert to UTC, then
    # drop tzinfo again to match this project's naive-UTC convention.
    localized = s.dt.tz_localize(
        _BROKER_TZ_NAME,
        ambiguous="infer",       # autumn fall-back hour (rare in FX data, but handled)
        nonexistent="shift_forward",  # spring-forward gap hour does not exist; shift
    )
    as_utc = localized.dt.tz_convert("UTC")
    return as_utc.dt.tz_localize(None)


def verify_live_offset(mt5_module, symbol: str = "EURUSD") -> dict:
    """
    Live sanity check to run during market hours (per state file: Monday
    check, not yet performed as of last session). Compares the assumed
    EET/EEST offset against a real MT5 tick timestamp vs true UTC "now".

    Requires an active MT5 connection (mt5.initialize() already called).
    Does NOT get imported/executed by data_loader.py or engine.py -- this
    is a standalone diagnostic, meant to be called from capture_specs.py
    or a small one-off script, exactly like the tick_value re-verification
    for XAUUSD.

    Returns a dict suitable for logging/printing:
        {
            "symbol": ...,
            "broker_tick_time": <naive broker-server datetime from MT5>,
            "true_utc_now": <naive true UTC datetime, from system clock>,
            "assumed_offset_hours": <2.0 or 3.0, per EET/EEST calendar>,
            "implied_offset_hours": <measured, from tick_time vs utc_now>,
            "match": <bool, True if assumed and implied agree within 5 min>,
        }

    IMPORTANT: only meaningful with the market open and a fresh tick
    (weekend/holiday snapshots give a stale last-traded-Friday timestamp
    and will produce a meaningless large offset -- this mirrors the
    -34h weekend result already seen and logged in the state file).
    """
    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        raise BrokerTimeError(
            f"No tick data for {symbol}. Is the symbol available and MT5 connected?"
        )
    broker_tick_time = datetime.fromtimestamp(tick.time)
    true_utc_now = datetime.utcnow()

    assumed_offset = broker_offset_hours(broker_tick_time)
    implied_offset = (broker_tick_time - true_utc_now).total_seconds() / 3600.0

    return {
        "symbol": symbol,
        "broker_tick_time": broker_tick_time,
        "true_utc_now": true_utc_now,
        "assumed_offset_hours": assumed_offset,
        "implied_offset_hours": implied_offset,
        "match": abs(assumed_offset - implied_offset) <= (5 / 60),  # 5 min tolerance
    }