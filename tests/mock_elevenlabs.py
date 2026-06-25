"""In-process MOCK ElevenLabs server for hermetic e2e tests.

Serves HTTP and WebSocket on ONE localhost port (no TLS) so the real dragonTTS
``ElevenLabsProvider`` can be driven end-to-end WITHOUT touching the network:

- HTTP ``POST /v1/text-to-speech/{voice_id}``
    -> 200 with a deterministic raw ``pcm_s16le@16000`` body (~1s sine ramp).
       Ignores ``?output_format=`` (the provider always asks ``pcm_16000``) and
       accepts any ``xi-api-key``. Increments a module-level HTTP call counter.

- WS ``/v1/text-to-speech/{voice_id}/multi-stream-input``
    Implements ElevenLabs' multi-context protocol as the dragonTTS pool drives it
    (see app/providers/elevenlabs_pool.py):
        client sends  -> {text:' ', context_id, voice_settings?}   (init)
                      -> {text, context_id}                        (real text)
                      -> {context_id, flush:true}                   (flush)
                      -> {context_id, close_context:true}           (end)
        server replies-> {contextId, audio:<b64>} x N              (audio chunks)
                      -> {contextId, isFinal:true}                 (end marker)
    On the real-text frame the mock emits exactly 2 base64 audio chunks then an
    ``isFinal`` frame for that ``context_id``. Init/flush/close frames are ignored.

Startable in-process: ``start(port)`` runs the server in a daemon thread hosting
a uvicorn ``Server`` + ``Config``; ``stop()`` signals shutdown and joins.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from typing import Optional

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

# Module-level observable state (read by the e2e harness to assert "called once").
HTTP_CALL_COUNT = 0
WS_CALL_COUNT = 0
WS_FRAMES_RECEIVED: list[dict] = []
_lock = threading.Lock()


def reset_counters() -> None:
    """Zero the observable counters (call at the start of a fresh run)."""
    global HTTP_CALL_COUNT, WS_CALL_COUNT, WS_FRAMES_RECEIVED
    with _lock:
        HTTP_CALL_COUNT = 0
        WS_CALL_COUNT = 0
        WS_FRAMES_RECEIVED = []


def http_call_count() -> int:
    with _lock:
        return HTTP_CALL_COUNT


def ws_call_count() -> int:
    with _lock:
        return WS_CALL_COUNT


# --- deterministic audio body -------------------------------------------------
# A fixed ~1s 220Hz sine as int16 LE @16kHz. Stable per run so cached/HIT bytes
# are reproducible; cheap to synthesize (no I/O).
_SR = 16000
_DUR_S = 1.0


def _make_pcm() -> bytes:
    t = np.linspace(0.0, _DUR_S, int(_SR * _DUR_S), endpoint=False, dtype=np.float32)
    wave = 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    # gentle linear ramp so it's not pure-DC / looks like real frames
    env = np.minimum(1.0, np.linspace(0.0, 1.0, len(wave)) * 8.0)
    wave = wave * env
    return (np.clip(wave, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


_PCM_BODY = _make_pcm()


# --- HTTP handler -------------------------------------------------------------

async def http_synthesize(request: Request) -> Response:
    """ElevenLabs one-shot synth. Deterministic PCM body; counts every call."""
    global HTTP_CALL_COUNT
    body_bytes = await request.body()
    # Best-effort parse (we don't depend on the payload shape); ignore failure.
    try:
        import json as _json
        payload = _json.loads(body_bytes) if body_bytes else {}
    except Exception:
        payload = {}
    with _lock:
        HTTP_CALL_COUNT += 1
        WS_FRAMES_RECEIVED.append({"kind": "http", "text": str(payload.get("text", ""))[:60]})
    # 200 with raw pcm_16000; accept any xi-api-key; ignore ?output_format=.
    return Response(content=_PCM_BODY, media_type="audio/raw", status_code=200)


# --- WebSocket handler (multi-context protocol) -------------------------------

async def ws_multi_stream(websocket: WebSocket) -> None:
    """ElevenLabs multi-stream-input. Per context_id: 2 b64 audio chunks + isFinal."""
    global WS_CALL_COUNT
    await websocket.accept()
    with _lock:
        WS_CALL_COUNT += 1
    try:
        async for raw in websocket.iter_text():
            import json as _json
            try:
                msg = _json.loads(raw)
            except Exception:
                continue  # ignore malformed frames
            ctx = msg.get("context_id") or msg.get("contextId")
            if not ctx:
                continue
            text = msg.get("text", "")
            # Ignore init (bare space), flush, and close_context frames — only the
            # real-text frame triggers generation (mirrors the pool's expectation
            # and tests/test_elevenlabs_pool.py's _ResponderWS).
            if not text or text.strip() == "":
                with _lock:
                    WS_FRAMES_RECEIVED.append({"ctx": ctx, "frame": "control", "text": text})
                continue
            with _lock:
                WS_FRAMES_RECEIVED.append({"ctx": ctx, "frame": "text", "text": text[:60]})
            # Emit 2 base64 audio chunks then isFinal (the pool ends the stream on
            # the 0.8s idle gap after audio, but we still send isFinal promptly).
            half = len(_PCM_BODY) // 2
            chunks = [_PCM_BODY[:half], _PCM_BODY[half:]]
            for c in chunks:
                await websocket.send_text(
                    _json.dumps({"contextId": ctx, "audio": base64.b64encode(c).decode()})
                )
            await websocket.send_text(_json.dumps({"contextId": ctx, "isFinal": True}))
    except Exception:
        # Client disconnect / close: just drop the connection quietly.
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --- app factory + start/stop -------------------------------------------------

def _build_app() -> Starlette:
    routes = [
        Route("/v1/text-to-speech/{voice_id}", http_synthesize, methods=["POST"]),
        WebSocketRoute("/v1/text-to-speech/{voice_id}/multi-stream-input", ws_multi_stream),
    ]
    return Starlette(routes=routes)


class MockElevenLabs:
    """Runs the mock on ONE localhost port in a daemon thread."""

    def __init__(self) -> None:
        self.port: Optional[int] = None
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def base_url(self) -> str:
        assert self.port is not None, "start() the mock first"
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_host(self) -> str:
        # The pool derives ws host from base_url via http->ws replace.
        assert self.port is not None, "start() the mock first"
        return f"ws://127.0.0.1:{self.port}"

    def start(self, port: int) -> None:
        """Run uvicorn hosting the Starlette app on ``port`` in a daemon thread."""
        self.port = port
        app = _build_app()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            lifespan="on",  # Starlette runs its own minimal startup/shutdown
        )
        self._server = uvicorn.Server(config)
        # Prevent signal handlers (installed only in the main thread) from breaking.
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]

        def _run() -> None:
            # uvicorn.Server.run owns the loop; signal when started (or failed).
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                config.load()
                self._ready.set()
                self._server.run()
            except Exception:
                self._ready.set()
                raise

        self._thread = threading.Thread(target=_run, name="mock-elevenlabs", daemon=True)
        self._thread.start()
        # Block until the loop has been created (best-effort readiness).
        self._ready.wait(timeout=5.0)

    def wait_until_listening(self, timeout: float = 10.0) -> None:
        """Poll the TCP port until uvicorn accepts a connection (or timeout)."""
        import socket

        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError as e:
                last_err = e
                time.sleep(0.05)
        raise RuntimeError(
            f"mock elevenlabs did not start listening on port {self.port}: {last_err}"
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)


def start(port: int) -> MockElevenLabs:
    """Convenience: create + start a mock, returning it for later stop()."""
    m = MockElevenLabs()
    m.start(port)
    m.wait_until_listening()
    return m


if __name__ == "__main__":
    # Manual smoke: run on :9999, print the counter, idle forever.
    import sys

    p = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    reset_counters()
    srv = MockElevenLabs()
    srv.start(p)
    srv.wait_until_listening()
    print(f"mock elevenlabs on http://127.0.0.1:{p} (ctrl-C to stop)")
    print(f"  base_url={srv.base_url}  ws_host={srv.ws_host}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        srv.stop()
