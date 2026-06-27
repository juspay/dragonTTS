"""Warm Sarvam socket pool — single-stream ordering, exclusive use, growth, error.

Sarvam is NOT multiplexed (one utterance per socket at a time), so the pool
hands out an exclusive warm socket per miss. Uses a reactive fake socket that
answers each ``flush`` with base64 ``audio`` chunks + a ``final`` event tagged
with the text that was sent — exercising the read path for real, no network.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.providers import sarvam_pool
from app.providers.base import ProviderError


class _ResponderWS:
    """Fake Sarvam single-stream socket. Buffers the ``text`` message and, on
    ``flush``, emits audio chunks + a final event."""

    def __init__(self):
        self.sent: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._text = ""
        self._closed = False

    async def send(self, msg_str: str) -> None:
        self.sent.append(msg_str)
        m = json.loads(msg_str)
        t = m.get("type")
        if t == "text":
            self._text = m.get("data", {}).get("text", "")
        elif t == "flush":
            mid = max(1, len(self._text) // 2)
            for part in (self._text[:mid], self._text[mid:]):
                await self._queue.put(
                    json.dumps(
                        {
                            "type": "audio",
                            "data": {"audio": base64.b64encode(part.encode()).decode()},
                        }
                    )
                )
            await self._queue.put(json.dumps({"type": "event", "data": {"event_type": "final"}}))

    async def recv(self) -> str:
        return await self._queue.get()

    async def close(self) -> None:
        self._closed = True


class _FakeCM:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


def _connect_factory():
    """A connect_fn that hands out a FRESH fake socket per connection (so the
    pool can be observed growing distinct sockets). Returns (connect_fn, list)."""
    created: list[_ResponderWS] = []

    def connect_fn(uri, headers):
        ws = _ResponderWS()
        created.append(ws)
        return _FakeCM(ws)

    return connect_fn, created


async def _wait_ready(pool, count=1):
    for _ in range(200):
        if sum(1 for c in pool._conns if c.ready.is_set()) >= count:
            return
        await asyncio.sleep(0.01)


async def _make_pool(min_size=1, max_size=4):
    connect_fn, created = _connect_factory()
    pool = sarvam_pool.SarvamStreamPool(
        api_key="k", model="bulbul:v2", connect_fn=connect_fn,
        min_size=min_size, max_size=max_size,
    )
    await pool.start()
    await _wait_ready(pool, min_size)
    return pool, created


async def _collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


async def test_stream_yields_audio_in_order():
    pool, _created = await _make_pool()
    try:
        chunks = await _collect(pool.stream({"speaker": "s"}, "hello"))
        assert b"".join(chunks) == b"hello"  # two halves concatenated
    finally:
        await pool.aclose()


async def test_two_concurrent_streams_use_separate_sockets():
    # NOT multiplexed: two concurrent utterances need two exclusive sockets.
    pool, _created = await _make_pool(min_size=2, max_size=4)
    try:
        a, b = await asyncio.gather(
            _collect(pool.stream({"speaker": "s"}, "AAAA")),
            _collect(pool.stream({"speaker": "s"}, "BBBB")),
        )
        assert b"".join(a) == b"AAAA"
        assert b"".join(b) == b"BBBB"
    finally:
        await pool.aclose()


async def test_busy_socket_not_reused_grows_pool():
    # min_size=1: occupy the one socket, then acquire must open a second
    # (exclusive use — never over-subscribe a busy socket).
    pool, _created = await _make_pool(min_size=1, max_size=4)
    try:
        c0 = pool._conns[0]
        c0._busy = True  # simulate an in-flight utterance
        conn = await pool.acquire()
        assert conn is not c0            # a different (newly grown) socket
        assert len(pool._conns) == 2     # pool grew
    finally:
        await pool.aclose()


async def test_error_frame_fails_the_stream():
    pool, created = await _make_pool()
    try:
        ws = created[0]
        # queue an error frame the next recv() will return
        await ws._queue.put(json.dumps({"type": "error", "data": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            async for _ in pool.stream({"speaker": "s"}, "x"):
                pass
    finally:
        await pool.aclose()


def test_model_sample_rate():
    assert sarvam_pool.model_sample_rate("bulbul:v2") == 22050
    assert sarvam_pool.model_sample_rate("bulbul:v3") == 24000
    assert sarvam_pool.model_sample_rate("unknown") == 24000  # safe default


async def test_aborted_utterance_resets_socket_not_reused_dirty():
    """Regression: an utterance that ends WITHOUT a clean ``final`` (timeout /
    error / client cancel) must mark its socket dirty + clear ``ready``, and
    ``run`` must then drop + reconnect it. Sarvam is single-stream with no
    per-utterance routing, so without the reset the next caller would ``recv``
    this transcript's leftover frames as its own audio (silent cross-contam)."""

    class _BlockingWS:
        def __init__(self):
            self.sent = []

        async def send(self, msg_str):
            self.sent.append(msg_str)

        async def recv(self):
            await asyncio.Event().wait()  # never returns -> recv() times out

        async def close(self):
            return None

    created: list[_BlockingWS] = []

    def connect_fn(uri, headers):
        class _CM:
            async def __aenter__(self):
                self.ws = _BlockingWS()
                created.append(self.ws)
                return self.ws

            async def __aexit__(self, *exc):
                return False

        return _CM()

    pool = sarvam_pool.SarvamStreamPool(
        api_key="k", model="bulbul:v2", connect_fn=connect_fn, min_size=1, max_size=4,
    )
    await pool.start()
    await _wait_ready(pool, 1)

    conn = pool._conns[0]
    conn._recv_timeout = 0.2  # don't wait the 30s default for the timeout

    with pytest.raises(ProviderError):
        async for _ in pool.stream({"speaker": "s"}, "hello"):
            pass

    # The aborted socket is excluded from the pool right away...
    assert conn._dirty is True
    assert conn.ready.is_set() is False
    # ...and run() drops it + reconnects to a FRESH socket.
    for _ in range(300):
        if len(created) >= 2 and conn.ready.is_set():
            break
        await asyncio.sleep(0.01)
    assert len(created) == 2          # reconnected exactly once
    assert conn.ready.is_set() is True
    assert conn._dirty is False       # fresh socket, flag cleared on connect

    await pool.aclose()


async def test_clean_utterance_keeps_socket_reusable():
    """A normal (final-terminated) utterance must NOT trip the reset — the socket
    stays ready and is handed back out. Guards the reset fix from over-firing."""
    pool, _created = await _make_pool(min_size=1)
    try:
        chunks = await _collect(pool.stream({"speaker": "s"}, "hello"))
        assert b"".join(chunks) == b"hello"
        conn = pool._conns[0]
        assert conn._dirty is False
        assert conn.ready.is_set() is True
        assert await pool.acquire() is conn   # same socket reused for the next call
    finally:
        await pool.aclose()
