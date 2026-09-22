"""
Central Log Aggregator for ATS — Fix 10.2.
Read-only aggregation and search tool for all system logs, audit ledgers, state snapshots, and reports.
Preserves exact source_file absolute paths on all returned entries for deep audit.
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any

from shared.config import LOG_DIR, STATE_DIR, REPORT_DIR, DECISION_DIR, KB_DIR, OUTPUT_ROOT

log = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Standardized log entry schema across all system sources."""
    timestamp: str
    source_file: str
    source_system: str
    level: str
    event_type: str
    message: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Regular expression for parsing standard ATS plain text log lines:
# Format: 2026-07-30 02:30:00,123 | INFO | Message text...
# Or: 2026-07-30 02:30:00 - INFO - Message text...
LOG_LINE_REGEX = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s*(?:\||-)\s*(?P<level>[A-Z]+)\s*(?:\||-)\s*(?P<message>.*)$'
)


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse various timestamp formats to UTC datetime object."""
    if not ts_str:
        return None
    try:
        # Try ISO 8601
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    for fmt in ('%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class LogAggregator:
    """
    Read-only log aggregator. Parses plain text logs, JSONL audit files, and JSON state files.
    Genuinely read-only: uses standard non-locking file reads.
    """

    def __init__(self, search_dirs: Optional[List[Path]] = None):
        self.search_dirs = search_dirs or [OUTPUT_ROOT, LOG_DIR, STATE_DIR, DECISION_DIR, KB_DIR, REPORT_DIR]

    def _determine_source_system(self, file_path: Path) -> str:
        name_lower = file_path.name.lower()
        path_str = str(file_path).lower()
        if 'engine' in name_lower or 'engine' in path_str:
            return 'ENGINE'
        elif 'trainer' in name_lower or 'trainer' in path_str:
            return 'TRAINER'
        return 'UNKNOWN'

    def parse_file(self, file_path: Path) -> List[LogEntry]:
        """Parse a single log/state file into LogEntry objects. Read-only and exception-safe."""
        if not file_path.exists() or not file_path.is_file():
            return []

        entries: List[LogEntry] = []
        source_file = str(file_path.resolve())
        source_system = self._determine_source_system(file_path)

        # Handle .jsonl (JSON Lines)
        if file_path.suffix == '.jsonl':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            ts = data.get('timestamp') or data.get('created_at') or datetime.now(timezone.utc).isoformat()
                            # Note: Level detection below is best-effort keyword matching for un-leveled JSONL entries; intended as a starting point for investigation, not an authoritative guarantee.
                            level = data.get('level') or ('ERROR' if 'error' in data or 'failure' in str(data.get('event_type')).lower() else 'INFO')
                            event_type = data.get('event_type') or data.get('metric_type') or 'JSONL_RECORD'
                            msg = data.get('message') or data.get('details') or data
                            entries.append(LogEntry(
                                timestamp=ts,
                                source_file=source_file,
                                source_system=source_system,
                                level=str(level).upper(),
                                event_type=str(event_type),
                                message=msg,
                            ))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                log.warning(f"Error reading JSONL file {file_path}: {e}")

        # Handle .log or .txt (Plain Text Logs)
        elif file_path.suffix in ('.log', '.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        match = LOG_LINE_REGEX.match(line)
                        if match:
                            ts = match.group('timestamp')
                            level = match.group('level').upper()
                            msg = match.group('message')
                            entries.append(LogEntry(
                                timestamp=ts,
                                source_file=source_file,
                                source_system=source_system,
                                level=level,
                                event_type='LOG_LINE',
                                message=msg,
                            ))
                        else:
                            # Fallback unformatted text line
                            entries.append(LogEntry(
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                source_file=source_file,
                                source_system=source_system,
                                level='INFO',
                                event_type='LOG_LINE_RAW',
                                message=line,
                            ))
            except Exception as e:
                log.warning(f"Error reading log file {file_path}: {e}")

        # Handle .json (State Snapshots / Checkpoints)
        elif file_path.suffix == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    ts = data.get('updated_at') or data.get('started_at') or datetime.now(timezone.utc).isoformat()
                    level = 'ERROR' if data.get('status') == 'FAILED' or 'error' in data else 'INFO'
                    event_type = 'STATE_SNAPSHOT'
                    entries.append(LogEntry(
                        timestamp=ts,
                        source_file=source_file,
                        source_system=source_system,
                        level=level,
                        event_type=event_type,
                        message=data,
                    ))
            except Exception as e:
                log.warning(f"Error reading JSON file {file_path}: {e}")

        return entries

    def query_logs(self,
                   level: Optional[str] = None,
                   source_system: Optional[str] = None,
                   time_from: Optional[datetime] = None,
                   time_to: Optional[datetime] = None,
                   query_text: Optional[str] = None) -> List[LogEntry]:
        """Query aggregated log entries across all inventory sources."""
        all_entries: List[LogEntry] = []

        seen_files = set()
        for sdir in self.search_dirs:
            if not sdir.exists():
                continue
            for ext in ('*.log', '*.jsonl', '*.json'):
                for fpath in sdir.rglob(ext):
                    abs_path = str(fpath.resolve())
                    if abs_path in seen_files:
                        continue
                    seen_files.add(abs_path)
                    entries = self.parse_file(fpath)
                    all_entries.extend(entries)

        filtered: List[LogEntry] = []
        for entry in all_entries:
            if level and entry.level.upper() != level.upper():
                continue
            if source_system and entry.source_system.upper() != source_system.upper():
                continue

            entry_dt = parse_timestamp(entry.timestamp)
            if entry_dt:
                if time_from and entry_dt < time_from:
                    continue
                if time_to and entry_dt > time_to:
                    continue

            if query_text:
                q_lower = query_text.lower()
                msg_str = json.dumps(entry.message).lower() if isinstance(entry.message, dict) else str(entry.message).lower()
                if q_lower not in msg_str and q_lower not in entry.event_type.lower():
                    continue

            filtered.append(entry)

        # Sort by timestamp
        def sort_key(e: LogEntry):
            dt = parse_timestamp(e.timestamp)
            return dt or datetime.min.replace(tzinfo=timezone.utc)

        filtered.sort(key=sort_key)
        return filtered


def main():
    parser = argparse.ArgumentParser(description='ATS Central Log Aggregator')
    parser.add_argument('--level', help='Filter by log level (INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--system', help='Filter by system (ENGINE, TRAINER)')
    parser.add_argument('--query', help='Free text search in log messages')
    parser.add_argument('--since-hours', type=float, help='Filter logs from last N hours')
    args = parser.parse_args()

    time_from = None
    if args.since_hours:
        time_from = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)

    agg = LogAggregator()
    results = agg.query_logs(
        level=args.level,
        source_system=args.system,
        time_from=time_from,
        query_text=args.query
    )

    print(f"\nFound {len(results)} matching log entries:")
    print("=" * 80)
    for entry in results[-50:]:  # Print last 50 matches
        print(f"[{entry.timestamp}] [{entry.source_system}] [{entry.level}] ({entry.event_type})")
        print(f"  Source File: {entry.source_file}")
        print(f"  Message:     {entry.message}")
        print("-" * 80)


if __name__ == '__main__':
    main()
