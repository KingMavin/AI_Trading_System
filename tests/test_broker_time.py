"""
tests/test_broker_time.py

Unit tests for shared/broker_time.py. All tests are pure logic -- no MT5
connection required, per Engine PRD §4.5 ("conversion logic must be
unit-testable independent of MT5 connectivity").
"""

from datetime import datetime

import pandas as pd
import pytest

from shared.broker_time import (
    BrokerTimeError,
    broker_offset_hours,
    broker_series_to_utc,
    broker_to_utc,
    utc_to_broker,
)


# ---------------------------------------------------------------------------
# broker_offset_hours
# ---------------------------------------------------------------------------

class TestBrokerOffsetHours:
    def test_winter_offset_is_2(self):
        # Mid-January, well clear of any DST transition.
        assert broker_offset_hours(datetime(2026, 1, 15, 12, 0)) == 2.0

    def test_summer_offset_is_3(self):
        # Mid-July, well clear of any DST transition.
        assert broker_offset_hours(datetime(2026, 7, 15, 12, 0)) == 3.0

    def test_rejects_tz_aware_input(self):
        from zoneinfo import ZoneInfo

        aware = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
        with pytest.raises(BrokerTimeError):
            broker_offset_hours(aware)


# ---------------------------------------------------------------------------
# broker_to_utc / utc_to_broker (round trip + known fixed points)
# ---------------------------------------------------------------------------

class TestBrokerToUtc:
    def test_winter_conversion(self):
        # Broker (EET, UTC+2) 12:00 -> UTC 10:00
        broker_dt = datetime(2026, 1, 15, 12, 0, 0)
        assert broker_to_utc(broker_dt) == datetime(2026, 1, 15, 10, 0, 0)

    def test_summer_conversion(self):
        # Broker (EEST, UTC+3) 12:00 -> UTC 09:00
        broker_dt = datetime(2026, 7, 15, 12, 0, 0)
        assert broker_to_utc(broker_dt) == datetime(2026, 7, 15, 9, 0, 0)

    def test_non_stale_winter_evening_conversion(self):
        # Sanity check at a time of day close to the state file's weekend
        # observation, but on a normal trading day (no stale-tick issue).
        # Note: the state file's weekend "-34h" and "23:59 broker / 09:59
        # UTC" readings were caused by a stale last-traded-Friday tick, not
        # by conversion logic -- that staleness case is exercised live via
        # verify_live_offset() during market hours, not reproducible here.
        broker_dt = datetime(2026, 3, 1, 23, 59, 0)  # winter, EET, UTC+2
        utc_dt = broker_to_utc(broker_dt)
        assert utc_dt == datetime(2026, 3, 1, 21, 59, 0)

    def test_round_trip_winter(self):
        original = datetime(2026, 1, 15, 12, 0, 0)
        assert utc_to_broker(broker_to_utc(original)) == original

    def test_round_trip_summer(self):
        original = datetime(2026, 7, 15, 12, 0, 0)
        assert utc_to_broker(broker_to_utc(original)) == original

    def test_rejects_tz_aware_input(self):
        from zoneinfo import ZoneInfo

        aware = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
        with pytest.raises(BrokerTimeError):
            broker_to_utc(aware)


# ---------------------------------------------------------------------------
# EU DST transition boundaries (last Sunday of March / October, 01:00 UTC)
# 2026: spring forward Mar 29 01:00 UTC, fall back Oct 25 01:00 UTC.
# ---------------------------------------------------------------------------

class TestDstTransitions:
    def test_just_before_spring_forward(self):
        # Broker still at EET (UTC+2). 2026-03-29 02:59 broker -> 00:59 UTC.
        broker_dt = datetime(2026, 3, 29, 2, 59, 0)
        assert broker_to_utc(broker_dt) == datetime(2026, 3, 29, 0, 59, 0)

    def test_just_after_spring_forward(self):
        # Clocks jump 03:00 -> 04:00 broker-local at the instant UTC hits
        # 01:00. Broker 04:00:01 (EEST, UTC+3) -> UTC 01:00:01.
        broker_dt = datetime(2026, 3, 29, 4, 0, 1)
        assert broker_to_utc(broker_dt) == datetime(2026, 3, 29, 1, 0, 1)

    def test_just_before_fall_back(self):
        # Still EEST (UTC+3) right up to the transition instant.
        broker_dt = datetime(2026, 10, 25, 3, 59, 0)
        assert broker_to_utc(broker_dt) == datetime(2026, 10, 25, 0, 59, 0)

    def test_just_after_fall_back(self):
        # 03:00-03:59 broker-local is ambiguous (occurs twice: once as
        # EEST, once as EET) -- see TestBrokerSeriesToUtc for that case.
        # Once local time reaches 05:00 there is no more ambiguity: EET
        # (UTC+2) is unambiguously in effect.
        broker_dt = datetime(2026, 10, 25, 5, 0, 1)
        assert broker_to_utc(broker_dt) == datetime(2026, 10, 25, 3, 0, 1)

    def test_offset_flips_across_spring_boundary(self):
        before = broker_offset_hours(datetime(2026, 3, 29, 2, 59, 0))
        after = broker_offset_hours(datetime(2026, 3, 29, 4, 0, 0))
        assert before == 2.0
        assert after == 3.0

    def test_offset_flips_across_autumn_boundary(self):
        before = broker_offset_hours(datetime(2026, 10, 25, 3, 59, 0))  # ambiguous hour, fold=0 default -> EEST
        after = broker_offset_hours(datetime(2026, 10, 25, 5, 0, 0))    # past ambiguity -> EET
        assert before == 3.0
        assert after == 2.0


# ---------------------------------------------------------------------------
# broker_series_to_utc (vectorised, spans DST transitions within one series)
# ---------------------------------------------------------------------------

class TestBrokerSeriesToUtc:
    def test_series_spanning_spring_transition(self):
        broker_times = pd.Series(
            pd.to_datetime(
                [
                    "2026-03-29 02:00:00",  # winter side, UTC+2 -> 00:00 UTC
                    "2026-03-29 04:15:00",  # summer side, UTC+3 -> 01:15 UTC
                ]
            )
        )
        result = broker_series_to_utc(broker_times)
        expected = pd.Series(
            pd.to_datetime(["2026-03-29 00:00:00", "2026-03-29 01:15:00"])
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_series_spanning_autumn_transition_unambiguous_points(self):
        broker_times = pd.Series(
            pd.to_datetime(
                [
                    "2026-10-25 02:30:00",  # before the fold: EEST, UTC+3
                    "2026-10-25 05:00:00",  # after the fold: EET, UTC+2
                ]
            )
        )
        result = broker_series_to_utc(broker_times)
        expected = pd.Series(
            pd.to_datetime(["2026-10-24 23:30:00", "2026-10-25 03:00:00"])
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_series_infers_full_duplicated_hour_correctly(self):
        # This is the realistic production case: continuous M1 data across
        # the fall-back transition contains the local hour 03:00-03:59
        # TWICE in true chronological order (once as EEST, once as EET).
        # `ambiguous="infer"` resolves this correctly given the repeated,
        # ordered block -- this is exactly the shape a real MT5 CSV export
        # takes across an autumn DST rollover.
        broker_times = pd.Series(
            pd.to_datetime(
                [
                    "2026-10-25 03:00:00",  # 1st pass, EEST (+3)
                    "2026-10-25 03:30:00",  # 1st pass, EEST (+3)
                    "2026-10-25 03:00:00",  # 2nd pass, EET (+2)
                    "2026-10-25 03:30:00",  # 2nd pass, EET (+2)
                ]
            )
        )
        result = broker_series_to_utc(broker_times)
        expected = pd.Series(
            pd.to_datetime(
                [
                    "2026-10-25 00:00:00",
                    "2026-10-25 00:30:00",
                    "2026-10-25 01:00:00",
                    "2026-10-25 01:30:00",
                ]
            )
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)
        # Critically: the result must be strictly increasing (true UTC time
        # never repeats or goes backwards), even though the broker-local
        # labels did.
        assert result.is_monotonic_increasing

    def test_series_multi_year_mixed_seasons(self):
        # A realistic ingestion case: one CSV load spans several years,
        # crossing many summer/winter boundaries. Spot-check a handful.
        broker_times = pd.Series(
            pd.to_datetime(
                [
                    "2020-01-10 09:00:00",  # winter -> UTC+2
                    "2022-07-04 09:00:00",  # summer -> UTC+3
                    "2024-12-25 09:00:00",  # winter -> UTC+2
                ]
            )
        )
        result = broker_series_to_utc(broker_times)
        expected = pd.Series(
            pd.to_datetime(
                [
                    "2020-01-10 07:00:00",
                    "2022-07-04 06:00:00",
                    "2024-12-25 07:00:00",
                ]
            )
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_rejects_tz_aware_series(self):
        aware_series = pd.Series(
            pd.to_datetime(["2026-01-15 12:00:00"])
        ).dt.tz_localize("UTC")
        with pytest.raises(BrokerTimeError):
            broker_series_to_utc(aware_series)

    def test_output_dtype_is_naive_datetime64(self):
        broker_times = pd.Series(pd.to_datetime(["2026-01-15 12:00:00"]))
        result = broker_series_to_utc(broker_times)
        assert result.dt.tz is None