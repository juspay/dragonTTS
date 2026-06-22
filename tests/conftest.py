"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.providers.base import AudioResult, BaseTTSProvider


class FakeProvider(BaseTTSProvider):
    """Deterministic in-process provider — no network. Returns raw 16kHz PCM."""

    name = "cartesia"

    def __init__(self, audio: bytes | None = None):
        self._audio = audio or (b"\x01\x00" * 400)  # 400 silent-ish PCM frames
        self.calls = 0
        self.stream_calls = 0

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        self.calls += 1
        return AudioResult(
            audio=self._audio, container="raw", encoding="pcm_s16le", sample_rate=16000
        )

    async def stream_synth(self, *, text, voice_id, model, language, params):
        """Yield the audio in 4 pieces to exercise chunk-by-chunk streaming."""
        self.stream_calls += 1
        piece = max(1, len(self._audio) // 4)
        for i in range(0, len(self._audio), piece):
            yield self._audio[i : i + piece]


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Redirect DB + blobs to a per-test temp dir and keep tests hermetic."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "blob_dir", str(tmp_path / "blobs"))
    # Don't let app startup open real Cartesia streaming sockets in tests.
    monkeypatch.setattr(settings, "cartesia_stream_pool_size", 0)
    # Don't let reads trigger predictive warming (tested explicitly elsewhere).
    monkeypatch.setattr(settings, "predictive_warm_enabled", False)
    return tmp_path


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def pcm_request() -> dict:
    return {
        "model_id": "cartesia:sonic-3.5",
        "transcript": "thank you",
        "voice": {"id": "v1"},
        "language": "en",
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    }


@pytest.fixture
def app_client(tmp_storage):
    """FastAPI TestClient with a FakeProvider injected under the 'cartesia' route."""
    from app.main import app

    with TestClient(app) as client:
        fake = FakeProvider()
        app.state.registry._providers["cartesia"] = fake
        client._fake = fake  # type: ignore[attr-defined]
        yield client
