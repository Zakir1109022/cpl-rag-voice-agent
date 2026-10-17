"""pytest configuration and shared fixtures.

- `sandbox_site` builds a tiny static site in a tmp dir and serves it on
  a free localhost port. Use this URL in browser tool tests.
- Skip the Playwright-dependent tests when the browser binary is missing.
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Skip marker: Playwright + Chromium must be installed
# ---------------------------------------------------------------------------


def _chromium_installed() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        return False
    candidates = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_path:
        candidates.append(env_path)
    # Windows default install location.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(os.path.join(local_app_data, "ms-playwright"))
    # Linux/macOS default install location.
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "ms-playwright"))
    # Quick heuristic - if none of the directories exist or hold a chromium
    # build, assume the browser isn't available.
    for home in candidates:
        if os.path.isdir(home) and any("chromium" in name for name in os.listdir(home)):
            return True
    return False


requires_chromium = pytest.mark.skipif(
    not _chromium_installed(),
    reason="Playwright Chromium is not installed; run `playwright install chromium`",
)


# ---------------------------------------------------------------------------
# Sandbox HTTP server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SITE_HTML = {
    "index.html": """<!doctype html>
<html><head><title>Sandbox Index</title></head>
<body>
  <h1>Sandbox Site</h1>
  <p>Hello, this is the index page used for browser-tool tests.</p>
  <nav>
    <a href="form.html" id="link-form">Form</a>
    <a href="login.html" id="link-login">Login</a>
  </nav>
  <button id="go-btn" type="button">Go</button>
</body></html>""",
    "form.html": """<!doctype html>
<html><head><title>Sandbox Form</title></head>
<body>
  <h1>Form Page</h1>
  <form>
    <label for="name">Name</label>
    <input id="name" name="name" type="text" />
    <label for="plan">Plan</label>
    <select id="plan" name="plan">
      <option value="basic">Basic</option>
      <option value="premium">Premium</option>
    </select>
    <button id="submit-btn" type="submit">Submit</button>
  </form>
</body></html>""",
    "login.html": """<!doctype html>
<html><head><title>Sandbox Login</title></head>
<body>
  <h1>Login</h1>
  <label for="user">Username</label>
  <input id="user" name="user" type="text" />
  <label for="pw">Password</label>
  <input id="pw" name="pw" type="password" />
  <button id="login-btn" type="button">Sign in</button>
  <p id="result" hidden>Signed in as <span id="who"></span></p>
  <script>
    document.getElementById('login-btn').addEventListener('click', function () {
      var u = document.getElementById('user').value;
      document.getElementById('who').textContent = u;
      document.getElementById('result').hidden = false;
    });
  </script>
</body></html>""",
}


@pytest.fixture
def sandbox_site(tmp_path):
    """Create a tiny static site and serve it on a free port.

    Yields a dict with:
      - url: the base URL (e.g. "http://127.0.0.1:8123")
      - index_url, form_url, login_url: convenience
      - dir: the on-disk path
    """
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    for name, body in SITE_HTML.items():
        (site_dir / name).write_text(body, encoding="utf-8")

    port = _free_port()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(site_dir), **kw)
    # Suppress the noisy stderr "code 404" lines during tests.
    handler.log_message = lambda *args, **kwargs: None  # type: ignore

    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Give the server a moment to bind.
    time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    try:
        yield {
            "url": base,
            "index_url": f"{base}/index.html",
            "form_url": f"{base}/form.html",
            "login_url": f"{base}/login.html",
            "dir": site_dir,
        }
    finally:
        httpd.shutdown()
        httpd.server_close()
