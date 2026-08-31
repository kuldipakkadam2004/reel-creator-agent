"""Tests for pipeline/instagram.py.

Entirely offline: the Graph API is a dictionary of canned bodies. What is under
test is the part that will actually go wrong in production - matching a reel to
the post it became, surviving Meta renaming a metric, and telling an expired
token apart from a permission that was never granted.

    python -m pytest tests/test_instagram.py -v
    python tests/test_instagram.py          # works without pytest installed
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import instagram, perform  # noqa: E402


# ---------------------------------------------------------------------------
# A Graph API made of dictionaries
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, body, status: int = 200):
        self._body = body
        self.status_code = status
        self.text = str(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Answers by the last path segment, and records what it was asked."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.asked: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.asked.append((url, dict(params or {})))
        for fragment, answer in self.routes.items():
            if fragment in url:
                if callable(answer):
                    return FakeResponse(answer(params or {}))
                if isinstance(answer, list):
                    # A queue: each call gets the next answer.
                    return FakeResponse(answer.pop(0) if answer else {})
                return FakeResponse(answer)
        return FakeResponse({"error": {"message": f"no route for {url}",
                                       "code": 100}})


@pytest.fixture(autouse=True)
def credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INSTAGRAM_ACCESS_TOKEN", "token-abc")
    monkeypatch.setattr(config, "INSTAGRAM_USER_ID", "17800000000000000")
    monkeypatch.setattr(config, "INSTAGRAM_MATCH_HOURS", 96)
    monkeypatch.setattr(config, "PERFORMANCE_FILE", tmp_path / "performance.json")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "token-abc")
    monkeypatch.setenv("INSTAGRAM_USER_ID", "17800000000000000")
    return tmp_path


def _post(media_id: str, caption: str, when: datetime) -> dict:
    return {"id": media_id, "caption": caption,
            "timestamp": when.isoformat(timespec="seconds"),
            "media_type": "VIDEO", "media_product_type": "REELS",
            "permalink": f"https://instagram.com/reel/{media_id}"}


def _insight(name: str, value):
    return {"name": name, "values": [{"value": value}]}


# ---------------------------------------------------------------------------
# Meta's error envelope
# ---------------------------------------------------------------------------

def test_an_expired_token_is_told_apart_from_everything_else(credentials):
    session = FakeSession({"/": {"error": {
        "message": "Error validating access token: Session has expired",
        "code": 190}}})

    with pytest.raises(instagram.InstagramAuthError, match="refresh"):
        instagram.account(session)


def test_a_missing_permission_says_which_one(credentials):
    session = FakeSession({"/": {"error": {
        "message": "(#10) Application does not have permission", "code": 10}}})

    with pytest.raises(instagram.InstagramAuthError, match="insights"):
        instagram.account(session)


def test_being_rate_limited_is_not_an_auth_failure(credentials):
    session = FakeSession({"/": {"error": {
        "message": "Application request limit reached", "code": 4}}})

    with pytest.raises(instagram.InstagramError) as caught:
        instagram.account(session)
    assert not isinstance(caught.value, instagram.InstagramAuthError)
    assert "nothing was recorded" in str(caught.value)


def test_a_non_json_body_is_reported_with_what_came_back(credentials):
    session = FakeSession({"/": "<html>502 Bad Gateway</html>"})

    with pytest.raises(instagram.InstagramError, match="non-JSON"):
        instagram.account(session)


def test_credentials_are_required_before_anything_is_sent(monkeypatch):
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)

    with pytest.raises(config.MissingConfigError, match="INSTAGRAM_ACCESS_TOKEN"):
        instagram.require_credentials()


# ---------------------------------------------------------------------------
# Metric names that churn
# ---------------------------------------------------------------------------

def test_a_metric_this_api_version_refuses_is_dropped_and_the_rest_kept(credentials):
    session = FakeSession({"/insights": [
        {"error": {"message": "(#100) ig_reels_avg_watch_time is not valid for "
                              "this media product type", "code": 100}},
        {"data": [_insight("reach", 900), _insight("saved", 12)]},
    ]})

    metrics, warnings = instagram._insights(session, "1791")

    assert metrics == {"reach": 900, "saves": 12}
    assert any("does not report" in warning for warning in warnings)


def test_an_unreadable_refusal_falls_back_to_the_core_metrics(credentials):
    session = FakeSession({"/insights": [
        {"error": {"message": "something entirely new went wrong", "code": 100}},
        {"data": [_insight("reach", 900)]},
    ]})

    metrics, warnings = instagram._insights(session, "1791")

    assert metrics == {"reach": 900}
    assert any("core four" in warning for warning in warnings)


def test_a_refusal_of_even_the_core_metrics_gives_up_quietly(credentials):
    session = FakeSession({"/insights": {
        "error": {"message": "something entirely new", "code": 100}}})

    metrics, warnings = instagram._insights(session, "1791")

    assert metrics == {}
    assert warnings


def test_plays_and_views_both_land_on_views(credentials):
    session = FakeSession({"/insights": {"data": [_insight("plays", 4200)]}})

    metrics, _ = instagram._insights(session, "1791")

    assert metrics == {"views": 4200}


def test_an_expired_token_is_not_swallowed_by_the_metric_retry(credentials):
    session = FakeSession({"/insights": {"error": {
        "message": "Session has expired", "code": 190}}})

    with pytest.raises(instagram.InstagramAuthError):
        instagram._insights(session, "1791")


# ---------------------------------------------------------------------------
# Matching a reel to its post
# ---------------------------------------------------------------------------

def _features(tmp_path: Path, *, made: datetime, caption: str = "",
              spoken: float | None = 52.69) -> dict:
    run_dir = tmp_path / "out" / "2026-08-24" / "ocean-1a2b3c"
    run_dir.mkdir(parents=True, exist_ok=True)
    if caption:
        (run_dir / "post.txt").write_text(caption, encoding="utf-8")
    return {"run_id": "ocean-1a2b3c", "run_dir": str(run_dir),
            "created_at": made.isoformat(timespec="seconds"),
            "spoken_seconds": spoken}


def test_a_reel_is_matched_by_the_caption_step_6_wrote(credentials, tmp_path):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made,
                         caption="Snailfish live 11 kilometres down.\n\nBody.")
    posts = [
        _post("1791", "Some other post entirely", made + timedelta(hours=2)),
        _post("1792", "Snailfish live 11 kilometres down.\n\nBody. #ocean",
              made + timedelta(days=40)),
    ]

    found = instagram.match(features, posts)

    # The caption wins even though the other post is far closer in time.
    assert found["media_id"] == "1792"
    assert found["how"] == "caption"


def test_without_a_caption_the_nearest_later_post_is_taken(credentials, tmp_path):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made)
    posts = [_post("1791", "a", made + timedelta(hours=30)),
             _post("1792", "b", made + timedelta(hours=2))]

    found = instagram.match(features, posts)

    assert found["media_id"] == "1792"
    assert found["how"] == "timestamp"


def test_a_post_made_before_the_reel_existed_is_not_that_reel(credentials, tmp_path):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made)
    posts = [_post("1791", "a", made - timedelta(hours=1))]

    assert instagram.match(features, posts)["media_id"] == ""


def test_a_post_outside_the_window_is_not_that_reel(credentials, tmp_path):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made)
    posts = [_post("1791", "a", made + timedelta(hours=200))]

    assert instagram.match(features, posts)["how"] == ""


def test_no_posts_at_all_is_not_a_match(credentials, tmp_path):
    features = _features(tmp_path,
                         made=datetime(2026, 8, 24, tzinfo=timezone.utc))

    assert instagram.match(features, [])["media_id"] == ""


# ---------------------------------------------------------------------------
# Watch-through is derived, not reported
# ---------------------------------------------------------------------------

def test_milliseconds_watched_become_a_share_of_the_reel(credentials):
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 32000, "reach": 900}, {"spoken_seconds": 52.69}
    )

    assert metrics["avg_watch_seconds"] == 32.0
    assert metrics["watch_through"] == pytest.approx(0.6073, abs=0.001)
    assert metrics["reach"] == 900


def test_watching_a_loop_twice_does_not_read_as_two_hundred_percent(credentials):
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 90000}, {"spoken_seconds": 40.0}
    )

    assert metrics["watch_through"] == 1.0


def test_without_the_reel_s_length_there_is_no_watch_through(credentials):
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 32000}, {"spoken_seconds": None}
    )

    assert "watch_through" not in metrics
    assert metrics["avg_watch_seconds"] == 32.0


def test_a_post_with_no_watch_time_keeps_its_other_metrics(credentials):
    assert instagram.derive_watch_through({"reach": 12}, {}) == {"reach": 12}


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------

def _wired(tmp_path, monkeypatch, *, insights: dict, posts: list[dict]):
    session = FakeSession({"/insights": insights, "/media": {"data": posts}})
    monkeypatch.setattr(instagram, "_session", lambda: session)
    return session


def test_fetch_returns_metrics_step_9_understands(credentials, tmp_path,
                                                  monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 4200), _insight("saved", 31),
                              _insight("ig_reels_avg_watch_time", 32000)]},
           posts=[_post("1792", "Snailfish live deep. #ocean",
                        made + timedelta(hours=18))])

    out = instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features})

    assert out["ocean-1a2b3c"]["views"] == 4200
    assert out["ocean-1a2b3c"]["saves"] == 31
    assert out["ocean-1a2b3c"]["watch_through"] == pytest.approx(0.607, abs=0.01)
    assert set(out["ocean-1a2b3c"]) <= set(perform.METRICS)


def test_how_the_match_was_made_is_written_down(credentials, tmp_path,
                                                monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made)
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("reach", 900)]},
           posts=[_post("1792", "unrelated", made + timedelta(hours=18))])

    instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features})

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["matched_by"] == "timestamp"
    assert entry["media_id"] == "1792"


def test_a_pinned_media_id_overrules_the_matcher(credentials, tmp_path,
                                                 monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.reels["ocean-1a2b3c"]["media_id"] = "9999"
    ledger.save()
    session = _wired(tmp_path, monkeypatch,
                     insights={"data": [_insight("reach", 900)]},
                     posts=[_post("1792", "Snailfish live deep.",
                                  made + timedelta(hours=18))])

    instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features})

    assert any("9999/insights" in url for url, _ in session.asked)
    assert perform.PerformanceLedger().entry("ocean-1a2b3c")["media_id"] == "9999"


def test_a_reel_that_cannot_be_matched_is_left_pending(credentials, tmp_path,
                                                       monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made)
    _wired(tmp_path, monkeypatch, insights={"data": []},
           posts=[_post("1791", "old", made - timedelta(days=3))])

    assert instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features}) == {}


def test_an_account_with_no_posts_asks_for_no_insights(credentials, tmp_path,
                                                       monkeypatch):
    features = _features(tmp_path,
                         made=datetime(2026, 8, 24, tzinfo=timezone.utc))
    session = _wired(tmp_path, monkeypatch, insights={"data": []}, posts=[])

    assert instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features}) == {}
    assert not any("insights" in url for url, _ in session.asked)


def test_the_provider_plugs_into_step_9(credentials, tmp_path, monkeypatch):
    """The contract perform.fetch expects, exercised through perform."""
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 4200)]},
           posts=[_post("1792", "Snailfish live deep.",
                        made + timedelta(hours=18))])

    recorded, warnings = perform.fetch(provider=instagram.fetch)

    assert recorded == 1
    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["status"] == "posted"
    assert entry["metrics"]["views"] == 4200
    assert entry["source"] == "provider"


# ---------------------------------------------------------------------------
# Pinning and refreshing
# ---------------------------------------------------------------------------

def test_pinning_records_the_media_id_by_hand(credentials, tmp_path):
    run_dir = tmp_path / "out" / "2026-08-24" / "ocean-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / config.VIDEO_FILENAME).write_bytes(b"x")

    assert instagram.pin("1a2b3c", "1792") == []

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert (entry["media_id"], entry["matched_by"]) == ("1792", "pinned")


def test_refreshing_returns_the_new_token(credentials):
    session = FakeSession({"refresh_access_token":
                           {"access_token": "token-xyz", "expires_in": 5183944}})

    assert instagram.refresh_token(session) == "token-xyz"


def test_a_token_too_far_gone_to_refresh_says_what_to_do(credentials):
    session = FakeSession({"refresh_access_token": {}})

    with pytest.raises(instagram.InstagramError, match="Meta dashboard"):
        instagram.refresh_token(session)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def test_cli_check_prints_the_account(credentials, monkeypatch, capsys):
    session = FakeSession({"/": {"id": "178", "username": "oceanfacts",
                                 "media_count": 12}})
    monkeypatch.setattr(instagram, "_session", lambda: session)

    assert instagram.main(["--check"]) == 0
    assert "oceanfacts" in capsys.readouterr().out


def test_cli_without_credentials_says_which_two_are_needed(monkeypatch, capsys):
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_USER_ID", raising=False)

    assert instagram.main(["--check"]) == 1
    err = capsys.readouterr()
    assert "INSTAGRAM_ACCESS_TOKEN" in err.out or "INSTAGRAM_ACCESS_TOKEN" in err.err


def test_cli_reports_an_api_failure_without_a_traceback(credentials,
                                                        monkeypatch, capsys):
    session = FakeSession({"/": {"error": {"message": "Session has expired",
                                           "code": 190}}})
    monkeypatch.setattr(instagram, "_session", lambda: session)

    assert instagram.main(["--media"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_media_lists_what_the_api_sees(credentials, monkeypatch, capsys):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    session = FakeSession({"/media": {"data": [_post("1792", "A hook", made)]}})
    monkeypatch.setattr(instagram, "_session", lambda: session)

    assert instagram.main(["--media"]) == 0
    out = capsys.readouterr().out
    assert "1792" in out and "A hook" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
