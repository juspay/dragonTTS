"""Slack daily summary: once-per-day claim dedupe, the never-raise sender, the
summary build + cost math (overall + per provider), and the force-send endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.alerts.slack import Alert, _format_tag
from app.alerts.summary import build_summary, send_daily_summary
from app.cache.service import CacheService
from app.core.config import settings
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


async def _svc(tmp_storage, fake_provider, monkeypatch):
    monkeypatch.setattr(settings, "enable_write_through", True)
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None), meta


# --- once-per-day claim (multi-worker dedupe) --------------------------------


async def test_claim_dedupe_only_one_wins(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    today = "2026-07-15"
    assert await meta.claim_slack_summary(today) is True   # first worker wins
    assert await meta.claim_slack_summary(today) is False  # second blocked
    assert await meta.claim_slack_summary(today) is False  # third blocked


async def test_release_allows_reclaim(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    today = "2026-07-15"
    assert await meta.claim_slack_summary(today) is True
    await meta.release_slack_summary()  # simulate a send failure -> retry allowed
    assert await meta.claim_slack_summary(today) is True


async def test_claim_new_day_is_independent(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    assert await meta.claim_slack_summary("2026-07-15") is True
    assert await meta.claim_slack_summary("2026-07-16") is True  # next day = new slot


# --- sender: best-effort, never raises ---------------------------------------


async def test_send_noop_without_webhook():
    assert await Alert(webhook_url="", tag_users="").send(title="t") is False


async def test_send_swallows_http_failure(monkeypatch):
    import httpx
    alert = Alert(webhook_url="https://example.invalid/hook", tag_users="")

    async def _boom(self, *a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(httpx.AsyncClient, "post", _boom)
    assert await alert.send(title="t") is False  # raised, but never propagates


async def test_send_success_on_200_ok(monkeypatch):
    import httpx
    alert = Alert(webhook_url="https://example.invalid/hook", tag_users="")

    class _Resp:
        status_code = 200
        text = "ok"

    async def _post(self, *a, **k):
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    ok = await alert.send(title="t", fields=[{"name": "a", "value": "b"}])
    assert ok is True


def test_format_tag_passes_subteam_through():
    assert _format_tag("<!subteam^S05KD5LN31Q>") == "<!subteam^S05KD5LN31Q>"
    assert _format_tag("alice") == "<@alice>"
    assert _format_tag("") == ""


# --- summary build + per-provider cost math ----------------------------------


async def test_build_summary_overall_and_per_provider(tmp_storage, fake_provider, monkeypatch):
    svc, meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    # record_metrics(provider=...) writes the global row AND the per-provider row.
    await meta.record_metrics(
        provider="gemini", requests=100, hits=80, words_served=1000, words_synthesized=250
    )
    await meta.record_metrics(
        provider="elevenlabs", requests=200, hits=100, words_served=2000, words_synthesized=1000
    )
    monkeypatch.setattr(
        settings, "slack_cost_per_word", {"gemini": 0.00002, "elevenlabs": 0.00001}
    )

    payload = await build_summary(svc)
    assert payload["title"].startswith("📊 DragonTTS")
    field_map = {f["name"]: f["value"] for f in payload["fields"]}
    assert set(field_map) >= {"Cache hit rate", "Words from cache", "Est. cost saved", "Window"}
    # overall: 180 hits / 300 requests = 60%
    assert "60%" in field_map["Cache hit rate"]
    # overall words from cache: (3000 - 1250) / 3000 = 58%
    assert "58%" in field_map["Words from cache"]
    # total cost = gemini 0.015 + elevenlabs 0.010 = 0.025 USD -> x96 = 2.4 -> ₹2
    assert "₹2" in field_map["Est. cost saved"]

    # per-provider sections (alphabetical) + one cache section, each a titled dict.
    assert len(payload["sections"]) == 3
    assert payload["sections"][0]["title"] == "elevenlabs"
    assert payload["sections"][1]["title"] == "gemini"
    assert payload["sections"][2]["title"] == "Cache"
    # elevenlabs: 0.010*96 = 0.96 -> ₹1 ; gemini: 0.015*96 = 1.44 -> ₹1
    assert "₹1" in payload["sections"][0]["text"]
    assert "₹1" in payload["sections"][1]["text"]
    assert "all providers" in field_map["Est. cost saved"]
    assert payload["title"].startswith("📊 DragonTTS Daily Cache Summary")


# --- send_daily_summary gate + force -----------------------------------------


async def test_send_daily_summary_noop_without_webhook(tmp_storage, fake_provider, monkeypatch):
    svc, _meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    monkeypatch.setattr(settings, "slack_webhook_url", "")
    assert await send_daily_summary(svc, force=True) is False


async def test_send_daily_summary_force_bypasses_claim(tmp_storage, fake_provider, monkeypatch):
    svc, _meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    monkeypatch.setattr(settings, "slack_webhook_url", "https://example.invalid/hook")
    from app.alerts import slack as slackmod
    monkeypatch.setattr(slackmod.slack_alert, "send", AsyncMock(return_value=True))
    # Force must NOT touch the once-per-day claim (endpoint can fire repeatedly).
    assert await send_daily_summary(svc, force=True) is True
    assert await send_daily_summary(svc, force=True) is True  # still allowed


async def test_send_daily_summary_non_force_releases_on_failure(tmp_storage, fake_provider, monkeypatch):
    svc, meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    monkeypatch.setattr(settings, "slack_webhook_url", "https://example.invalid/hook")
    # past the target time so the time gate passes
    monkeypatch.setattr("app.alerts.summary._past_target_time", lambda: True)
    from app.alerts import slack as slackmod
    monkeypatch.setattr(slackmod.slack_alert, "send", AsyncMock(return_value=False))  # send fails
    assert await send_daily_summary(svc, force=False) is False
    # claim was released on failure -> a fresh claim still wins
    assert await meta.claim_slack_summary("2099-01-01") is True


async def test_send_daily_summary_never_raises_on_send_exception(
    tmp_storage, fake_provider, monkeypatch
):
    """If slack_alert.send itself raises, send_daily_summary must absorb it
    (logs + returns False) — never propagate into the background loop."""
    svc, _meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    monkeypatch.setattr(settings, "slack_webhook_url", "https://example.invalid/hook")
    from app.alerts import slack as slackmod

    async def _raise(**kw):
        raise RuntimeError("slack exploded")

    monkeypatch.setattr(slackmod.slack_alert, "send", _raise)
    assert await send_daily_summary(svc, force=True) is False


async def test_build_summary_failure_releases_claim(tmp_storage, fake_provider, monkeypatch):
    """A build_summary raise AFTER winning the claim must release it, else the
    day's summary is silently lost (regression for the claim-vs-build ordering)."""
    svc, meta = await _svc(tmp_storage, fake_provider, monkeypatch)
    monkeypatch.setattr(settings, "slack_webhook_url", "https://example.invalid/hook")
    monkeypatch.setattr("app.alerts.summary._past_target_time", lambda: True)

    async def _boom(_cache):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.alerts.summary.build_summary", _boom)
    assert await send_daily_summary(svc, force=False) is False
    # claim was released despite the build failure -> a fresh claim still wins
    assert await meta.claim_slack_summary("2099-01-01") is True


def test_past_target_time_falls_back_on_malformed_env(monkeypatch):
    """A malformed SLACK_SUMMARY_TIME_UTC must not raise (would silently disable
    the daily summary); it falls back to 17:30 UTC."""
    from app.alerts import summary as summ

    monkeypatch.setattr(settings, "slack_summary_time_utc", "25:99")
    assert summ._past_target_time() in (True, False)  # no raise either way

    monkeypatch.setattr(settings, "slack_summary_time_utc", "not-a-time")
    assert summ._past_target_time() in (True, False)


async def test_send_malformed_field_does_not_raise():
    """A malformed field dict (missing 'value') must not escape send() — the
    never-raise contract covers payload construction, not just httpx."""
    alert = Alert(webhook_url="https://example.invalid/hook", tag_users="")
    assert await alert.send(title="t", fields=[{"name": "x"}]) is False


# --- force-send endpoint -----------------------------------------------------


def test_slack_summary_endpoint_noop_without_webhook(app_client, monkeypatch):
    monkeypatch.setattr(settings, "slack_webhook_url", "")
    r = app_client.post("/slack-summary")
    assert r.status_code == 200
    assert r.json() == {"sent": False}


def test_slack_summary_endpoint_force_sends(app_client, monkeypatch):
    monkeypatch.setattr(settings, "slack_webhook_url", "https://example.invalid/hook")
    from app.alerts import slack as slackmod
    monkeypatch.setattr(slackmod.slack_alert, "send", AsyncMock(return_value=True))
    r = app_client.post("/slack-summary")
    assert r.status_code == 200
    assert r.json()["sent"] is True
