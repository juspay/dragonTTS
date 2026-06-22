"""Tests for stats / bulk-create / clear / pagination endpoints."""

from __future__ import annotations

import base64
from datetime import datetime, timezone


def test_stats(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)
    app_client.post(
        "/tts/create", json={**pcm_request, "voice": {"id": "v2"}, "transcript": "goodbye"}
    )
    s = app_client.get("/stats").json()
    assert s["entries"] == 2
    assert s["total_bytes"] > 0
    assert "cartesia" in s["by_provider"]
    assert s["by_provider"]["cartesia"]["entries"] == 2
    assert "cartesia" in s["providers_configured"]


def test_stats_session_hit_rate(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)  # create is not a read
    app_client.post("/tts/bytes", json=pcm_request)  # HIT
    app_client.post("/tts/bytes", json={**pcm_request, "voice": {"id": "v9"}})  # MISS
    s = app_client.get("/stats").json()
    assert s["session"]["hits"] == 1
    assert s["session"]["misses"] == 1
    assert s["session"]["hit_rate"] == 0.5


def test_bulk_create_mixed_sources(app_client, pcm_request):
    items = [
        {**pcm_request, "transcript": "hello", "voice": {"id": "v1"}},
        {**pcm_request, "transcript": "world", "voice": {"id": "v2"}},
        {
            **pcm_request,
            "voice": {"id": "v3"},
            "audio_base64": base64.b64encode(b"\x00" * 16).decode(),
        },
    ]
    r = app_client.post("/tts/create/bulk", json=items)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["created"] == 3 and j["errors"] == 0
    assert {x["source"] for x in j["results"]} == {"synth", "base64"}


def test_clear_all_and_filtered(app_client, pcm_request):
    app_client.post("/tts/create", json={**pcm_request, "voice": {"id": "v1"}, "transcript": "a"})
    app_client.post("/tts/create", json={**pcm_request, "voice": {"id": "v2"}, "transcript": "b"})
    assert len(app_client.get("/cache").json()["entries"]) == 2

    r = app_client.post("/cache/clear?voice_id=v1")  # filtered
    assert r.json()["deleted"] == 1
    assert len(app_client.get("/cache").json()["entries"]) == 1

    r = app_client.post("/cache/clear")  # all
    assert r.json()["deleted"] == 1
    assert len(app_client.get("/cache").json()["entries"]) == 0


def test_cache_pagination(app_client, pcm_request):
    for i in range(3):
        app_client.post(
            "/tts/create",
            json={**pcm_request, "voice": {"id": f"v{i}"}, "transcript": f"t{i}"},
        )
    p1 = app_client.get("/cache?limit=2").json()
    assert len(p1["entries"]) == 2
    assert p1["limit"] == 2 and p1["has_next"] is True
    p2 = app_client.get("/cache?limit=2&offset=2").json()
    assert len(p2["entries"]) == 1 and p2["has_next"] is False


def test_cache_list_limit_capped(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)
    body = app_client.get("/cache?limit=9999999").json()
    assert body["limit"] == 1000  # clamped to the hard cap
    assert len(body["entries"]) == 1


def test_stats_durable_metrics_and_date_filter(app_client, pcm_request):
    app_client.post("/tts/create", json=pcm_request)  # create synth → creates=1, synth_calls=1
    app_client.post("/tts/bytes", json=pcm_request)  # HIT → requests=1, hits=1
    app_client.post("/tts/bytes", json={**pcm_request, "voice": {"id": "v9"}})  # MISS
    s = app_client.get("/stats").json()
    assert s["requests"] == 2 and s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert s["creates"] == 1 and s["synth_calls"] == 2  # create + miss synth

    today = datetime.now(timezone.utc).date().isoformat()
    s_today = app_client.get(f"/stats?from={today}&to={today}").json()
    assert s_today["requests"] == 2  # date filter includes today

    s_past = app_client.get("/stats?from=2000-01-01&to=2000-01-02").json()
    assert s_past["requests"] == 0 and s_past["hit_rate"] is None  # outside any activity


def test_stats_snapshot_tracks_create_delete(app_client, pcm_request):
    s0 = app_client.get("/stats").json()
    assert s0["entries"] == 0
    app_client.post("/tts/create", json={**pcm_request, "voice": {"id": "v1"}})
    app_client.post("/tts/create", json={**pcm_request, "voice": {"id": "v2"}})
    s1 = app_client.get("/stats").json()
    assert s1["entries"] == 2 and s1["by_provider"]["cartesia"]["entries"] == 2
    app_client.post("/tts/delete", json={**pcm_request, "voice": {"id": "v1"}})
    s2 = app_client.get("/stats").json()
    assert s2["entries"] == 1  # totals decremented on delete
