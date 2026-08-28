"""HTTP-level tests for the UI dashboard server (issue #42).

Tests the real DashboardHandler in a thread on an ephemeral port,
asserting over urllib.request — no mocking, no browser, no JS execution.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request

from ui.server import _HTML, DashboardHandler
from ui.trial_reader import status_pt

# ── Helpers ────────────────────────────────────────────────────────────────


def _start_server() -> tuple[int, http.server.HTTPServer, threading.Thread]:
    """Start DashboardHandler on a free port. Returns (port, server, thread)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.HTTPServer(("127.0.0.1", port), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return port, server, t


def _stop_server(server: http.server.HTTPServer, t: threading.Thread) -> None:
    server.shutdown()
    t.join(timeout=2)


# ── HTML Shell ─────────────────────────────────────────────────────────────


def test_root_returns_200_with_pt_br():
    """GET / → 200, lang=pt-BR, contains AUTOTUNING wordmark."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert 'lang="pt-BR"' in body
        assert "AUTOTUNING" in body or "AUTO" in body
        assert "Baseline" in body
        assert "Últimos Trials" in body
        assert "Log do servidor" in body
    finally:
        _stop_server(server, t)


def test_html_contains_static_css_link():
    """HTML shell references /static/style.css."""
    assert 'href="/static/style.css"' in _HTML


def test_html_contains_log_pin_button():
    """HTML shell has the log pin toggle button."""
    assert 'id="pin-toggle"' in _HTML


def test_html_contains_stale_banner():
    """HTML shell has the stale-data banner."""
    assert "stale-banner" in _HTML


# ── /api/status ────────────────────────────────────────────────────────────


def test_api_status_returns_json():
    """GET /api/status → 200 JSON with expected top-level keys."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "run_state" in data
        assert "log_tail" in data
        assert "baseline" in data
        assert "trials" in data
    finally:
        _stop_server(server, t)


def test_api_status_trial_rows_have_status_pt():
    """Trial rows in /api/status carry status_pt presentation field."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
        data = json.loads(resp.read())
        trials = data.get("trials") or []
        if trials:
            assert "status_pt" in trials[0]
    finally:
        _stop_server(server, t)


# ── Static Assets ──────────────────────────────────────────────────────────


def test_static_css_returns_200():
    """GET /static/style.css → 200, text/css, contains token #171717."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/style.css")
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert "text/css" in ct
        body = resp.read().decode("utf-8")
        assert "#171717" in body
    finally:
        _stop_server(server, t)


def test_static_font_returns_200():
    """GET /static/fonts/Inter-Regular.woff2 → 200."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/fonts/Inter-Regular.woff2")
        assert resp.status == 200
    finally:
        _stop_server(server, t)


# ── 404 ────────────────────────────────────────────────────────────────────


def test_unknown_path_returns_404():
    """GET /nope → 404."""
    port, server, t = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_server(server, t)


# ── status_pt helper ───────────────────────────────────────────────────────


def test_status_pt_maps_on_front():
    assert status_pt("on_front") == "na fronteira"


def test_status_pt_maps_dominated():
    assert status_pt("dominated") == "dominado"


def test_status_pt_maps_incomplete():
    assert status_pt("incomplete") == "incompleto"


def test_status_pt_maps_rejected():
    assert status_pt("rejected") == "rejeitado"


def test_status_pt_passes_unknown_through():
    assert status_pt("some_unknown") == "some_unknown"
