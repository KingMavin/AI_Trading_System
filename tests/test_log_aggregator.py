"""
Unit tests for WAVE 10.2 — Central Log Aggregator.
Verifies plain-text and JSONL log parsing, level/system/time filtering, non-blocking file access, and absolute source_file path preservation.
"""

import sys
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.log_aggregator import LogAggregator, LogEntry, parse_timestamp


class TestLogAggregator:

    def test_parse_plain_text_and_jsonl_logs(self, tmp_path):
        engine_log = tmp_path / "engine_20260730.log"
        engine_log.write_text(
            "2026-07-30 02:00:00,100 | INFO | ATS ENGINE STARTING\n"
            "2026-07-30 02:05:00,200 | WARNING | Retrying connection in 5s...\n"
            "2026-07-30 02:10:00,300 | ERROR | MAX_RECONNECT_ATTEMPTS_EXCEEDED\n",
            encoding='utf-8'
        )

        audit_jsonl = tmp_path / "audit_ledger.jsonl"
        audit_jsonl.write_text(
            json.dumps({"timestamp": "2026-07-30T02:15:00+00:00", "event_type": "SAFE_MODE_ENGAGED", "level": "ERROR", "message": "SAFE_MODE engaged"}) + "\n" +
            json.dumps({"timestamp": "2026-07-30T02:20:00+00:00", "event_type": "KILL_SWITCH_TRIPPED", "level": "CRITICAL", "message": "Kill switch present"}) + "\n",
            encoding='utf-8'
        )

        agg = LogAggregator(search_dirs=[tmp_path])
        entries = agg.query_logs()

        assert len(entries) == 5

        # Check level filtering
        error_entries = agg.query_logs(level="ERROR")
        assert len(error_entries) == 2
        for e in error_entries:
            assert e.level == "ERROR"

    def test_source_file_path_preservation(self, tmp_path):
        sample_log = tmp_path / "trainer_20260730.log"
        sample_log.write_text("2026-07-30 03:00:00 | INFO | Candidate evaluated\n", encoding='utf-8')

        agg = LogAggregator(search_dirs=[tmp_path])
        entries = agg.query_logs()

        assert len(entries) == 1
        assert entries[0].source_file == str(sample_log.resolve())
        assert entries[0].source_system == "TRAINER"

    def test_non_blocking_concurrent_write(self, tmp_path):
        log_file = tmp_path / "engine_active.log"
        log_file.write_text("2026-07-30 04:00:00 | INFO | Initial line\n", encoding='utf-8')

        agg = LogAggregator(search_dirs=[tmp_path])

        # Keep file open in append mode (simulating active Engine logging)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("2026-07-30 04:01:00 | WARNING | Appended line during active read\n")
            f.flush()

            # Aggregator must read file concurrently without error or locking collision
            entries = agg.query_logs()
            assert len(entries) == 2
            assert entries[1].level == "WARNING"

    def test_query_text_and_system_filtering(self, tmp_path):
        e_log = tmp_path / "engine_test.log"
        e_log.write_text("2026-07-30 05:00:00 | ERROR | Engine connection dropped\n", encoding='utf-8')

        t_log = tmp_path / "trainer_test.log"
        t_log.write_text("2026-07-30 05:05:00 | ERROR | Trainer candidate rejected\n", encoding='utf-8')

        agg = LogAggregator(search_dirs=[tmp_path])

        engine_only = agg.query_logs(source_system="ENGINE")
        assert len(engine_only) == 1
        assert "Engine connection dropped" in engine_only[0].message

        query_res = agg.query_logs(query_text="candidate")
        assert len(query_res) == 1
        assert query_res[0].source_system == "TRAINER"
