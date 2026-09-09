"""Ensure the animated demo's final LLM explanation stays fully in-frame."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

pytest.importorskip("playwright")

STATIC_DIR = Path(__file__).resolve().parents[1] / "recipe_opt_web" / "static"


def _cache_busted_html(filename: str, asset_names: tuple[str, ...]) -> HTMLResponse:
    html = (STATIC_DIR / filename).read_text(encoding="utf-8")
    for name in asset_names:
        path = STATIC_DIR / name
        ver = int(path.stat().st_mtime) if path.is_file() else 0
        html = html.replace(f"/static/{name}", f"/static/{name}?v={ver}")
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def demo_base_url():
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/animated-demo")
    def animated_demo():
        return _cache_busted_html("animated_demo.html", ("animated_demo.css", "animated_demo.js"))

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("animated demo static server failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_result_explain_stays_in_frame(demo_base_url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(
            f"{demo_base_url}/animated-demo?record=1&autoplay=0",
            wait_until="networkidle",
        )
        page.wait_for_function("() => window.__recordReady === true", timeout=30000)
        page.wait_for_function("() => typeof window.__measureResultExplainFit === 'function'")
        fit = page.evaluate("() => window.__measureResultExplainFit()")
        browser.close()

    assert fit.get("text"), "expected LLM explanation text to be present"
    assert fit.get("ok") is True, (
        "result_explain clipped by the stage frame: "
        f"overflowTop={fit.get('overflowTop')} overflowBottom={fit.get('overflowBottom')} "
        f"explain={fit.get('explain')} frame={fit.get('frame')} zoom={fit.get('zoom')}"
    )
    assert fit.get("overflowBottom", 99) <= 1.5
    assert fit.get("overflowTop", 99) <= 1.5
