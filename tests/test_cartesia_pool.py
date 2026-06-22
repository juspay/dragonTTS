"""Warm Cartesia socket pool — routing, multiplexing, reuse, error, transient.

Uses a reactive fake socket: on send() it enqueues chunk/done/error responses
tagged with the caller's context_id, so routing is exercised for real without
the network.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.providers import cartesia_pool
from app.providers.base import AudioResult, ProviderError
from app.providers.cartesia import CartesiaProvider


class _ResponderWS:
    """Fake Cartesia socket. Responds to each send() with chunk+done messages."""

    def __init__(self, *, error_on_transcript: str | None = None):
        self.sent: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._error_on = error_on_transcript

    async def send(self, msg_str: str) -> None:
        self.sent.append(msg_str)
        m = json.loads(msg_str)
        if m.get("cancel"):  # cancel control message: no audio response
            return
        ctx = m["context_id"]
        text = m.get("transcript", "")
        if self._error_on is not None and text == self._error_on:
            await self._queue.put(json.dumps({"type": "error", "context_id": ctx, "error": "boom"}))
            return
        mid = max(1, len(text) // 2)
        for part in (text[:mid], text[mid:]):
            await self._queue.put(
                json.dumps(
                    {"type": "chunk", "context_id": ctx, "data": base64.b64encode(part.encode()).decode()}
                )
            )
        await self._queue.put(json.dumps({"type": "done", "context_id": ctx}))

    async def close(self) -> None:
        return None

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        while True:  # runs until the connection task is cancelled at aclose()
            yield await self._queue.get()


class _FakeCM:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


class _FakeConnect:
    """Connector factory: records every socket it creates."""

    def __init__(self, ws_factory):
        self._ws_factory = ws_factory
        self.created: list[_ResponderWS] = []

    def __call__(self, uri):
        ws = self._ws_factory()
        self.created.append(ws)
        return _FakeCM(ws)


def _msg(text: str) -> dict:
    return {"transcript": text, "continue": False, "model_id": "sonic-3.5"}


async def _collect(gen):
    return [c async for c in gen]


async def test_pool_streams_chunks_until_done():
    connect = _FakeConnect(lambda: _ResponderWS())
    pool = cartesia_pool.CartesiaStreamPool("k", min_size=1, connect_fn=connect)
    await pool.start()
    try:
        chunks = await _collect(pool.stream(_msg("hello world")))
        assert b"".join(chunks) == b"hello world"
    finally:
        await pool.aclose()


async def test_pool_reuses_socket_across_utterances():
    connect = _FakeConnect(lambda: _ResponderWS())
    pool = cartesia_pool.CartesiaStreamPool("k", min_size=1, connect_fn=connect)
    await pool.start()
    try:
        await _collect(pool.stream(_msg("first")))
        await _collect(pool.stream(_msg("second")))
        # Both utterances served by ONE socket => no per-miss handshake.
        assert len(connect.created) == 1
    finally:
        await pool.aclose()


async def test_pool_multiplexes_concurrent_utterances():
    connect = _FakeConnect(lambda: _ResponderWS())
    pool = cartesia_pool.CartesiaStreamPool("k", min_size=1, connect_fn=connect)
    await pool.start()
    try:
        a, b = await asyncio.gather(
            _collect(pool.stream(_msg("apple"))),
            _collect(pool.stream(_msg("banana"))),
        )
        # Each consumer received only its own utterance (context_id routing).
        assert b"".join(a) == b"apple"
        assert b"".join(b) == b"banana"
    finally:
        await pool.aclose()


async def test_pool_error_message_raises_provider_error():
    connect = _FakeConnect(lambda: _ResponderWS(error_on_transcript="boom"))
    pool = cartesia_pool.CartesiaStreamPool("k", min_size=1, connect_fn=connect)
    await pool.start()
    try:
        with pytest.raises(ProviderError):
            await _collect(pool.stream(_msg("boom")))
    finally:
        await pool.aclose()


async def test_pool_cancel_sends_cartesia_cancel():
    ws = _ResponderWS()
    connect = _FakeConnect(lambda: ws)
    pool = cartesia_pool.CartesiaStreamPool("k", min_size=1, connect_fn=connect)
    await pool.start()
    try:
        gen = pool.stream(_msg("abandoned"))
        await gen.__anext__()  # take one chunk, then abandon
        await gen.aclose()
        # Abandoning mid-stream must tell Cartesia to cancel that context.
        cancel_msgs = [json.loads(m) for m in ws.sent if json.loads(m).get("cancel")]
        assert cancel_msgs and all("context_id" in m for m in cancel_msgs)
    finally:
        await pool.aclose()


# -- circuit breaker + HTTP fallback -----------------------------------------


class _FailCM:
    async def __aenter__(self):
        raise OSError("connect refused")

    async def __aexit__(self, *exc):
        return False


def _fail_connect(_uri):
    return _FailCM()


async def test_breaker_opens_then_fails_fast():
    pool = cartesia_pool.CartesiaStreamPool(
        "k", min_size=1, connect_fn=_fail_connect,
        acquire_timeout=0.1, failure_threshold=1, cooldown=10.0,
    )
    await pool.start()
    try:
        with pytest.raises(cartesia_pool.SocketUnavailable):
            await pool.acquire()  # waits out acquire_timeout, trips breaker
        # Second acquire is in cooldown -> fails immediately (no wait).
        import time
        t = time.monotonic()
        with pytest.raises(cartesia_pool.SocketUnavailable):
            await pool.acquire()
        assert time.monotonic() - t < 0.5
    finally:
        await pool.aclose()


async def test_provider_falls_back_to_http_when_ws_unavailable(monkeypatch):
    """stream_synth must serve audio via one-shot synth when the WS pool is down."""
    monkeypatch.setattr("app.providers.cartesia.settings.cartesia_stream_pool_size", 2)
    prov = CartesiaProvider(api_key="fake-key")

    class _BadPool:
        async def stream(self, msg):
            raise cartesia_pool.SocketUnavailable("no warm socket")
            yield b""  # makes this an async generator

    prov._get_pool = lambda: _BadPool()  # type: ignore[method-assign]

    async def fake_synth(**kw):
        return AudioResult(b"FALLBACK-AUDIO", "raw", "pcm_s16le", 16000)

    prov.synth = fake_synth  # type: ignore[method-assign]

    chunks = await _collect(
        prov.stream_synth(text="x", voice_id="v", model="m", language="en", params={})
    )
    assert chunks == [b"FALLBACK-AUDIO"]


def test_connection_dispatch_routes_by_context_id():
    """Unit-test the receive demux directly."""
    conn = cartesia_pool._CartesiaConnection("uri", lambda uri: _FakeCM(_ResponderWS()))
    q: asyncio.Queue = asyncio.Queue()
    ctx = "ctx-9"
    conn.contexts[ctx] = q

    conn._dispatch(json.dumps({"type": "chunk", "context_id": ctx, "data": base64.b64encode(b"ab").decode()}))
    conn._dispatch(json.dumps({"type": "chunk", "context_id": "other", "data": "AA=="}))  # ignored
    conn._dispatch(json.dumps({"type": "done", "context_id": ctx}))

    assert q.get_nowait() == b"ab"
    assert q.get_nowait() is cartesia_pool._DONE
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()  # the "other" context message was not routed here
