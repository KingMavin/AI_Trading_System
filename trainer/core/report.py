"""
Report Generator — Layer 10 (PRD Section 13).

Generates persistent HTML and Markdown reports for every completed or partial
training run and saves them to D:\\work\\files\\reports (or a custom output directory).
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = r"D:\work\files\reports"


class ReportGenerator:
    """Generates persistent HTML and Markdown training reports using atomic file writes."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(DEFAULT_REPORT_DIR)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create report directory {self.output_dir}: {e}")
            raise IOError(f"Could not create report directory {self.output_dir}: {e}")

    def generate(self,
                 run_record: Any,
                 candidates: List[Any],
                 decisions: List[Any],
                 wf_results_by_candidate: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
        """
        Generate both Markdown and HTML reports for a training run using atomic file writing (.tmp + rename).

        Returns:
            Tuple[Path, Path]: (markdown_file_path, html_file_path)
        """
        run_id = getattr(run_record, 'run_id', f"run_{int(datetime.now(timezone.utc).timestamp())}")
        md_path = self.output_dir / f"report_{run_id}.md"
        html_path = self.output_dir / f"report_{run_id}.html"

        md_content = self._build_markdown(run_record, candidates, decisions, wf_results_by_candidate)
        html_content = self._build_html(run_record, candidates, decisions, wf_results_by_candidate, md_content)

        # Atomic write pattern for Markdown report
        tmp_md_path = md_path.with_suffix('.md.tmp')
        with open(tmp_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        if md_path.exists():
            md_path.unlink()
        tmp_md_path.rename(md_path)

        # Atomic write pattern for HTML report
        tmp_html_path = html_path.with_suffix('.html.tmp')
        with open(tmp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        if html_path.exists():
            html_path.unlink()
        tmp_html_path.rename(html_path)

        log.info(f"Training report saved atomically to {md_path} and {html_path}")
        return md_path, html_path

    def _extract_run_meta(self, run_record: Any) -> Tuple[str, str, str, float]:
        run_id = run_record.run_id

        # Symbol resolution (reads real config['symbols'] list from RunRecord)
        if hasattr(run_record, 'config') and isinstance(run_record.config, dict):
            syms = run_record.config.get('symbols', ['ALL'])
            symbol = ", ".join(syms) if isinstance(syms, list) else str(syms)
        else:
            symbol = getattr(run_record, 'symbol', 'ALL')

        status = getattr(run_record, 'outcome', 'COMPLETED')
        if status == 'RUNNING':
            status = 'COMPLETED'

        duration = 0.0
        if hasattr(run_record, 'completed_at') and hasattr(run_record, 'started_at'):
            if run_record.completed_at and run_record.started_at:
                duration = (run_record.completed_at - run_record.started_at).total_seconds()
        elif hasattr(run_record, 'duration_seconds'):
            duration = run_record.duration_seconds

        return run_id, symbol, status, duration

    def _build_markdown(self,
                        run_record: Any,
                        candidates: List[Any],
                        decisions: List[Any],
                        wf_results_by_candidate: Optional[Dict[str, Any]] = None) -> str:
        lines = []
        run_id, symbol, status, duration = self._extract_run_meta(run_record)
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        lines.append(f"# ATS Training Run Report: `{run_id}`")
        lines.append(f"**Generated:** {timestamp}  ")
        lines.append(f"**Symbol:** `{symbol}` | **Pipeline Outcome:** `{status}` | **Duration:** `{duration:.2f}s`\n")

        lines.append("## 1. Run Overview & Configuration")
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        lines.append(f"| Run ID | `{run_id}` |")
        lines.append(f"| Symbol | `{symbol}` |")
        lines.append(f"| Candidates Evaluated | {len(candidates)} |")
        lines.append(f"| Decisions Rendered | {len(decisions)} |")
        
        promoted = getattr(run_record, 'candidates_promoted', 0)
        flagged = getattr(run_record, 'candidates_flagged', 0)
        rejected = getattr(run_record, 'candidates_rejected', 0)
        lines.append(f"| Promoted / Flagged / Rejected | {promoted} / {flagged} / {rejected} |")
        lines.append(f"| Deployed Strategy ID | `{getattr(run_record, 'deployed_strategy_id', 'None')}` |\n")

        # Section 2: Candidate Performance & Composite Scores
        lines.append("## 2. Candidate Evaluation & Composite Scores")
        if not candidates:
            lines.append("_No candidates evaluated in this run._\n")
        else:
            lines.append("| Candidate ID | Template | Gate Status | Spec Source | Composite Score |")
            lines.append("|---|---|---|---|---|")
            for c in candidates:
                cid = c.candidate_id
                tpl = getattr(c, 'template', 'N/A')
                gst = c.gate_status
                spec_src = getattr(c, 'spec_source', 'CAPTURED')
                spec_str = f"**{spec_src}**" if spec_src == "SIMULATION_DEFAULT" else f"`{spec_src}`"
                cscore = c.composite_score
                score_str = f"{cscore:.4f}" if cscore is not None else "N/A"
                lines.append(f"| `{cid}` | `{tpl}` | `{gst}` | {spec_str} | {score_str} |")
            lines.append("")

        # Section 3: Promotion Decisions
        lines.append("## 3. Promotion Decisions")
        if not decisions:
            lines.append("_No promotion decisions recorded._\n")
        else:
            lines.append("| Strategy ID | Outcome | Improvement % | Decision Reasoning |")
            lines.append("|---|---|---|---|")
            for d in decisions:
                sid = getattr(d, 'candidate_id', 'N/A')
                out = d.decision_outcome
                imp = d.improvement_pct
                imp_str = f"{imp:+.2f}%" if imp is not None else "N/A"
                reason = getattr(d, 'recommendation', '')
                reason_clean = str(reason).replace('\n', ' ')
                lines.append(f"| `{sid}` | **{out}** | {imp_str} | {reason_clean} |")
            lines.append("")

        # Section 4: Pipeline Layer Execution Trace
        lines.append("## 4. Pipeline Layer Execution Trace")
        layers = getattr(run_record, 'layers', {})
        if not layers:
            lines.append("_No layer trace available._\n")
        else:
            lines.append("| Layer Name | Status | Details |")
            lines.append("|---|---|---|")
            for lname, ldata in layers.items():
                lstat = ldata.get('status', 'UNKNOWN')
                ldet = str(ldata.get('summary', ldata.get('details', ldata.get('reason', ''))))
                lines.append(f"| `{lname}` | `{lstat}` | {ldet} |")
            lines.append("")

        return "\n".join(lines)

    def _build_html(self,
                    run_record: Any,
                    candidates: List[Any],
                    decisions: List[Any],
                    wf_results_by_candidate: Optional[Dict[str, Any]],
                    md_content: str) -> str:
        run_id, symbol, status, duration = self._extract_run_meta(run_record)
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ATS Training Report - {run_id}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
        margin: 0;
        padding: 24px;
        line-height: 1.6;
    }}
    .container {{
        max-width: 1050px;
        margin: 0 auto;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 32px;
    }}
    h1 {{
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 12px;
        margin-top: 0;
    }}
    h2 {{
        color: #79c0ff;
        margin-top: 28px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
    }}
    .meta-bar {{
        background: #21262d;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 24px;
        font-size: 14px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 14px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #30363d;
    }}
    th {{
        background-color: #21262d;
        color: #8b949e;
        font-weight: 600;
    }}
    tr:hover {{
        background-color: #1c2128;
    }}
    code {{
        font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
        background: #262c36;
        padding: 2px 6px;
        border-radius: 4px;
        color: #e6edf3;
        font-size: 13px;
    }}
    .badge-promoted {{ color: #3fb950; font-weight: bold; }}
    .badge-flagged {{ color: #d29922; font-weight: bold; }}
    .badge-rejected {{ color: #f85149; font-weight: bold; }}
    .badge-sim-default {{
        background-color: #701a1e;
        color: #f85149;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #da3633;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>ATS Training Run Report</h1>
    <div class="meta-bar">
        <strong>Run ID:</strong> <code>{run_id}</code> | 
        <strong>Symbol:</strong> <code>{symbol}</code> | 
        <strong>Outcome:</strong> <code>{status}</code> | 
        <strong>Generated:</strong> {timestamp}
    </div>
"""
        promoted = getattr(run_record, 'candidates_promoted', 0)
        flagged = getattr(run_record, 'candidates_flagged', 0)
        rejected = getattr(run_record, 'candidates_rejected', 0)

        html += f"""
    <h2>1. Run Overview & Configuration</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Run ID</td><td><code>{run_id}</code></td></tr>
        <tr><td>Symbol</td><td><code>{symbol}</code></td></tr>
        <tr><td>Candidates Evaluated</td><td>{len(candidates)}</td></tr>
        <tr><td>Decisions Rendered</td><td>{len(decisions)}</td></tr>
        <tr><td>Promoted / Flagged / Rejected</td><td>
            <span class="badge-promoted">{promoted} Promoted</span> / 
            <span class="badge-flagged">{flagged} Flagged</span> / 
            <span class="badge-rejected">{rejected} Rejected</span>
        </td></tr>
        <tr><td>Deployed Strategy ID</td><td><code>{getattr(run_record, 'deployed_strategy_id', 'None')}</code></td></tr>
        <tr><td>Pipeline Duration</td><td>{duration:.2f}s</td></tr>
    </table>

    <h2>2. Candidate Evaluation & Composite Scores</h2>
"""
        if not candidates:
            html += "<p><em>No candidates evaluated in this run.</em></p>"
        else:
            html += """
    <table>
        <tr>
            <th>Candidate ID</th>
            <th>Template</th>
            <th>Gate Status</th>
            <th>Spec Source</th>
            <th>Composite Score</th>
        </tr>
"""
            for c in candidates:
                cid = c.candidate_id
                tpl = getattr(c, 'template', 'N/A')
                gst = c.gate_status
                spec_src = getattr(c, 'spec_source', 'CAPTURED')
                if spec_src == "SIMULATION_DEFAULT":
                    spec_cell = '<span class="badge-sim-default">SIMULATION_DEFAULT</span>'
                else:
                    spec_cell = f'<code>{spec_src}</code>'

                cscore = c.composite_score
                score_str = f"{cscore:.4f}" if cscore is not None else "N/A"
                html += f"""
        <tr>
            <td><code>{cid}</code></td>
            <td><code>{tpl}</code></td>
            <td><code>{gst}</code></td>
            <td>{spec_cell}</td>
            <td><strong>{score_str}</strong></td>
        </tr>
"""
            html += "    </table>"

        html += """
    <h2>3. Promotion Decisions</h2>
"""
        if not decisions:
            html += "<p><em>No promotion decisions recorded.</em></p>"
        else:
            html += """
    <table>
        <tr>
            <th>Strategy ID</th>
            <th>Outcome</th>
            <th>Improvement %</th>
            <th>Decision Reasoning</th>
        </tr>
"""
            for d in decisions:
                sid = getattr(d, 'candidate_id', 'N/A')
                out = d.decision_outcome
                cls = "badge-promoted" if out in ("AUTO_PROMOTE", "PROMOTED") else ("badge-flagged" if out in ("FLAGGED", "FLAG_REVIEW") else "badge-rejected")
                imp = d.improvement_pct
                imp_str = f"{imp:+.2f}%" if imp is not None else "N/A"
                reason = getattr(d, 'recommendation', '')
                html += f"""
        <tr>
            <td><code>{sid}</code></td>
            <td><span class="{cls}">{out}</span></td>
            <td>{imp_str}</td>
            <td>{reason}</td>
        </tr>
"""
            html += "    </table>"

        html += """
    <h2>4. Pipeline Layer Execution Trace</h2>
"""
        layers = getattr(run_record, 'layers', {})
        if not layers:
            html += "<p><em>No layer trace available.</em></p>"
        else:
            html += """
    <table>
        <tr><th>Layer Name</th><th>Status</th><th>Details</th></tr>
"""
            for lname, ldata in layers.items():
                lstat = ldata.get('status', 'UNKNOWN')
                ldet = str(ldata.get('summary', ldata.get('details', ldata.get('reason', ''))))
                html += f"""
        <tr><td><code>{lname}</code></td><td><code>{lstat}</code></td><td>{ldet}</td></tr>
"""
            html += "    </table>"

        html += """
</div>
</body>
</html>
"""
        return html


def generate_report(run_record: Any,
                    candidates: List[Any],
                    decisions: List[Any],
                    wf_results_by_candidate: Optional[Dict[str, Any]] = None,
                    output_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """Convenience wrapper for ReportGenerator."""
    generator = ReportGenerator(output_dir=output_dir)
    return generator.generate(run_record, candidates, decisions, wf_results_by_candidate)
