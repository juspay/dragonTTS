"""API integration tests against the FastAPI app with a FakeProvider."""

from __future__ import annotations

import base64


def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_then_read_is_hit(app_client, pcm_request):
    r = app_client.post("/tts/create", json=pcm_request)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "CREATED" and r.json()["source"] == "synth"
    r2 = app_client.post("/tts/bytes", json=pcm_request)
    assert r2.status_code == 200
    assert r2.headers["x-cache"] == "HIT"
    assert app_client._fake.calls == 1


def test_check_flow(app_client, pcm_request):
    assert app_client.post("/tts/check", json=pcm_request).json()["cached"] is False
    app_client.post("/tts/create", json=pcm_request)
    assert app_client.post("/tts/check", json=pcm_request).json()["cached"] is True


def test_create_override(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)
    r = app_client.post("/tts/create", json=pcm_request)
    assert r.json()["status"] == "OVERRIDDEN"


def test_create_from_base64(app_client, pcm_request):
    audio = b"\x00\x01" * 50
    body = {**pcm_request, "audio_base64": base64.b64encode(audio).decode()}
    r = app_client.post("/tts/create", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "CREATED" and j["source"] == "base64"
    assert app_client._fake.calls == 0
    out = app_client.post("/tts/bytes", json=pcm_request)
    assert out.content == audio


def test_delete_flow(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)
    r = app_client.post("/tts/delete", json=pcm_request)
    assert r.json()["status"] == "deleted"
    assert app_client.post("/tts/check", json=pcm_request).json()["cached"] is False
    r = app_client.post("/tts/delete", json=pcm_request)
    assert r.json()["status"] == "not_found"


def test_miss_write_through_then_hit(app_client, pcm_request):
    r1 = app_client.post("/tts/bytes", json=pcm_request)
    assert r1.headers["x-cache"] == "MISS"
    r2 = app_client.post("/tts/bytes", json=pcm_request)
    assert r2.headers["x-cache"] == "HIT"
    assert r1.content == r2.content
    assert app_client._fake.calls == 1


def test_list_get_delete_by_key(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)
    body = app_client.get("/cache").json()
    entries = body["entries"]
    assert len(entries) == 1
    assert body["has_next"] is False
    key = entries[0]["key"]
    got = app_client.get(f"/cache/{key}")
    assert got.status_code == 200
    assert got.headers["x-provider"] == "cartesia"
    assert app_client.delete(f"/cache/{key}").status_code == 200
    assert app_client.get(f"/cache/{key}").status_code == 404


def test_unconfigured_provider_returns_503(app_client):
    req = {
        "model_id": "sarvam:bulbul:v3",
        "transcript": "hi",
        "voice": {"id": "shreya"},
        "language": "en-IN",
    }
    r = app_client.post("/tts/bytes", json=req)
    assert r.status_code == 503
    assert "sarvam" in r.json()["detail"]


def test_bad_model_id_returns_400(app_client):
    req = {"model_id": "no-colon", "transcript": "hi", "voice": {"id": "v1"}}
    r = app_client.post("/tts/bytes", json=req)
    assert r.status_code == 400


def test_upstream_error_maps_to_503(app_client, pcm_request):
    import httpx

    from app.providers.base import BaseTTSProvider

    class RaisingProvider(BaseTTSProvider):
        name = "cartesia"

        async def synth(self, *, text, voice_id, model, language, params):
            raise httpx.ConnectError("boom")

        async def aclose(self):
            pass

    app_client.app.state.registry._providers["cartesia"] = RaisingProvider()
    r = app_client.post("/tts/bytes", json=pcm_request)  # MISS → synth raises
    assert r.status_code == 503
    assert "unreachable" in r.json()["detail"]


def test_stats_bad_date_returns_400(app_client):
    assert app_client.get("/stats?from=not-a-date").status_code == 400
