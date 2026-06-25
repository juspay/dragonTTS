"""Warm ElevenLabs socket pool — routing, multiplexing, cap, error.

Uses a reactive fake socket: on send() it enqueues base64 ``audio`` + ``isFinal``
messages tagged with the caller's ``context_id``, so demux/multiplexing is
exercised for real without the network. Mirrors tests/test_cartesia_pool.py.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.providers import elevenlabs_pool
from app.providers.base import ProviderError


class _ResponderWS:
    """Fake ElevenLabs multi-context socket. Responds to each send() with
    audio chunks + an isFinal end marker, tagged with the caller's context_id."""

    def __init__(self):
        self.sent: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, msg_str: str) -> None:
        self.sent.append(msg_str)
        m = json.loads(msg_str)
        ctx = m["context_id"]
        text = m.get("text", "")
        # Only the real-text message produces audio; the init (space), flush,
        # and close_context frames carry empty/absent text.
        if not text or text == " ":
            return
        mid = max(1, len(text) // 2)
        for part in (text[:mid], text[mid:]):
            await self._queue.put(
                json.dumps(
                    {"context_id": ctx, "audio": base64.b64encode(part.encode()).decode()}
                )
            )
        await self._queue.put(json.dumps({"context_id": ctx, "isFinal": True}))

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


def _connect_factory():
    """Returns a connect_fn that hands out a shared fake WS (so all pool sockets
    are the same fake, letting us assert multiplexing on one socket)."""
    ws = _ResponderWS()

    def connect_fn(uri, headers):
        return _FakeCM(ws)

    return connect_fn, ws


async def _make_pool(min_size=1, max_size=4):
    connect_fn, ws = _connect_factory()
    pool = elevenlabs_pool.ElevenLabsStreamPool(
        api_key="k", voice_id="v1", model_id="eleven_flash_v2_5",
        connect_fn=connect_fn, min_size=min_size, max_size=max_size,
    )
    await pool.start()
    # wait for the socket to be ready
    for _ in range(100):
        if elevenlabs_pool._ElevenLabsConnection and any(c.ready.is_set() for c in pool._conns):
            break
        await asyncio.sleep(0.01)
    return pool, ws


async def test_stream_yields_audio_in_order():
    pool, _ws = await _make_pool()
    try:
        chunks = [c async for c in pool.stream({"text": "hello", "voice_settings": {}})]
        assert b"".join(chunks) == b"hello"   # two halves concatenated
    finally:
        await pool.aclose()


async def test_multiplex_two_contexts_one_socket():
    pool, ws = await _make_pool(min_size=1)
    try:
        # two concurrent utterances on the same warm socket, demuxed by context_id
        a, b = await asyncio.gather(
            _collect(pool.stream({"text": "AAAA"})),
            _collect(pool.stream({"text": "BBBB"})),
        )
        assert b"".join(a) == b"AAAA"
        assert b"".join(b) == b"BBBB"
        # Both utterances were multiplexed over the SAME (only) socket: each
        # stream sends init+text+flush+close_context (4 msgs), so 2 streams
        # => 8 sends, but only 2 distinct context_ids — that's the real demux proof.
        ctx_ids = {
            json.loads(m).get("context_id") for m in ws.sent if "context_id" in m
        }
        assert len(ctx_ids) == 2
        assert len(ws.sent) == 8  # 2 streams * (init + text + flush + close_context)
    finally:
        await pool.aclose()


async def test_cap_opens_second_socket_when_first_full():
    # min_size=1, max_size=4; fill the one socket to the 5-context cap, then a 6th
    # acquire must open a second socket rather than over-subscribe.
    connect_fn, _ws = _connect_factory()
    pool = elevenlabs_pool.ElevenLabsStreamPool(
        api_key="k", voice_id="v1", model_id="eleven_flash_v2_5",
        connect_fn=connect_fn, min_size=1, max_size=4,
    )
    await pool.start()
    for _ in range(100):
        if any(c.ready.is_set() for c in pool._conns):
            break
        await asyncio.sleep(0.01)
    try:
        c0 = pool._conns[0]
        c0.inflight = elevenlabs_pool._MAX_CONTEXTS_PER_SOCKET  # saturate it
        conn = await pool.acquire()
        assert conn is not c0                      # a different (newly added) socket
        assert len(pool._conns) == 2               # pool grew
        assert conn.inflight < elevenlabs_pool._MAX_CONTEXTS_PER_SOCKET
    finally:
        await pool.aclose()


async def test_error_message_fails_the_stream():
    connect_fn, _ws = _connect_factory()
    pool = elevenlabs_pool.ElevenLabsStreamPool(
        api_key="k", voice_id="v1", model_id="eleven_flash_v2_5",
        connect_fn=connect_fn, min_size=1,
    )
    await pool.start()
    for _ in range(100):
        if any(c.ready.is_set() for c in pool._conns):
            break
        await asyncio.sleep(0.01)
    try:
        conn = pool._conns[0]
        # inject an error frame for a context we're about to start
        ctx = "dead-ctx"
        q: asyncio.Queue = asyncio.Queue()
        conn.contexts[ctx] = q
        await q.put(elevenlabs_pool._Err(ProviderError("boom")))
        with pytest.raises(ProviderError):
            item = await q.get()
            raise item.exc
    finally:
        await pool.aclose()


async def _collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out
