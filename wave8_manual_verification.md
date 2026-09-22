# WAVE 8 — Manual Verification Protocol

This document outlines manual testing procedures for verifying live MT5 network connection drops, heartbeat recovery, and `SAFE_MODE` engagement during market hours.

---

## Manual Test Procedure: Live Network Disconnect & Reconnection

### Objective
Verify that the `Engine` correctly detects MT5 terminal network disconnections, executes exponential backoff retries, engages `SAFE_MODE`, and resumes normal operation upon reconnection.

### Step-by-Step Instructions

1. **Start MT5 & Launch Engine in Paper Mode:**
   - Launch MT5 terminal and verify live connection.
   - Run the Engine in paper trading mode:
     ```powershell
     python -m engine.core.engine --symbol EURUSD --paper
     ```
   - Confirm log displays `ATS ENGINE STARTING`.

2. **Simulate Network Outage:**
   - Disconnect network interface (disable Wi-Fi / unplug Ethernet cable) while the Engine is running.

3. **Verify Heartbeat & Backoff Retries:**
   - Monitor `D:\work\files\logs\engine_YYYYMMDD.log`.
   - Confirm log outputs:
     `CONNECTION_HEALTH_FAILURE (#1): MT5 terminal_info reports disconnected or None`
     `Retrying connection in 5s...`
   - Observe subsequent retries at 10s and 20s.

4. **Verify SAFE_MODE Engagement:**
   - After 3 failed reconnect attempts, verify log outputs:
     `MAX_RECONNECT_ATTEMPTS_EXCEEDED (3/3). Entering SAFE_MODE: Blocking all new trade entries.`
   - Open `D:\work\files\logs\audit_ledger.jsonl` and confirm audit event:
     `{"event_type": "SAFE_MODE_ENGAGED", ...}`

5. **Re-establish Network Connection:**
   - Re-enable Wi-Fi / plug Ethernet cable back in.
   - On the next candle cycle, observe log output:
     `MT5 Connection RESTORED after 3 failed attempt(s).`
   - Confirm `SAFE_MODE` is cleared and normal entry evaluation resumes.
