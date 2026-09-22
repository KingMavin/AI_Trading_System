import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from engine.dashboard.app import app, FASTAPI_AVAILABLE


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDashboardSecurity:

    def test_allowed_host_headers_return_success(self):
        """Requests with allowed Host headers (127.0.0.1, localhost, testserver) succeed."""
        client = TestClient(app)

        # 127.0.0.1
        resp_ip = client.get("/api/health", headers={"Host": "127.0.0.1"})
        assert resp_ip.status_code == 200

        # localhost
        resp_local = client.get("/api/health", headers={"Host": "localhost"})
        assert resp_local.status_code == 200

        # testserver (default TestClient host)
        resp_test = client.get("/api/health", headers={"Host": "testserver"})
        assert resp_test.status_code == 200

    def test_forged_host_header_returns_400_bad_request(self):
        """Requests with forged/spoofed Host headers (e.g. DNS rebinding attack) return 400 Bad Request."""
        client = TestClient(app)

        resp_forged = client.get("/api/health", headers={"Host": "rebound.attacker.com"})
        assert resp_forged.status_code == 400

        resp_ip_spoof = client.get("/api/health", headers={"Host": "192.168.1.100"})
        assert resp_ip_spoof.status_code == 400
