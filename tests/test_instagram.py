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
from datetime import date, datetime, timedelta, timezone
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
    monkeypatch.setattr(config, "INSTAGRAM_API_HOST", "")
    monkeypatch.setattr(config, "PERFORMANCE_FILE", tmp_path / "performance.json")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    # The token clock too. Its path is fixed from the real OUTPUT_DIR when
    # config is imported, so moving OUTPUT_DIR above does not move it. Until
    # 2026-10-01 every fetch test wrote "token-abc"'s hash into the real
    # output/token-age.json. The next real read then took the real token for
    # a new one and restarted its sixty days, so the expiry warning could
    # never come due.
    monkeypatch.setattr(config, "INSTAGRAM_TOKEN_STATE", tmp_path / "token-age.json")
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
# Which of Meta's two hosts is asked
# ---------------------------------------------------------------------------

def _asked_host(session) -> str:
    url = session.asked[0][0]
    return "/".join(url.split("/")[:3])


def test_an_instagram_login_token_goes_to_the_instagram_host(credentials,
                                                             monkeypatch):
    monkeypatch.setattr(config, "INSTAGRAM_ACCESS_TOKEN", "IGAAPMOD8Yxt5BZA")
    session = FakeSession({"/": {"id": "178", "username": "reels"}})

    instagram.account(session)

    assert _asked_host(session) == "https://graph.instagram.com"


def test_a_facebook_login_token_goes_to_the_facebook_host(credentials,
                                                          monkeypatch):
    monkeypatch.setattr(config, "INSTAGRAM_ACCESS_TOKEN", "EAAGm0PX4ZCpsBA")
    session = FakeSession({"/": {"id": "178", "username": "reels"}})

    instagram.account(session)

    assert _asked_host(session) == "https://graph.facebook.com"


def test_the_host_can_be_named_outright(credentials, monkeypatch):
    monkeypatch.setattr(config, "INSTAGRAM_ACCESS_TOKEN", "IGAAPMOD8Yxt5BZA")
    monkeypatch.setattr(config, "INSTAGRAM_API_HOST", "graph.example.com")
    session = FakeSession({"/": {"id": "178", "username": "reels"}})

    instagram.account(session)

    assert _asked_host(session) == "https://graph.example.com"


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


def test_a_refusal_listing_what_is_valid_is_read_the_right_way_round(credentials):
    """The live shape of this: Meta names the metrics it *would* have taken."""
    session = FakeSession({"/insights": [
        {"error": {"message":
                   "metric[1] must be one of the following values: "
                   "impressions, shares, comments, likes, saved, reach, "
                   "ig_reels_avg_watch_time, views", "code": 100}},
        {"data": [_insight("views", 13), _insight("reach", 6),
                  _insight("ig_reels_avg_watch_time", 20120)]},
    ]})

    metrics, warnings = instagram._insights(session, "1791")

    # Read the wrong way round this drops the seven good names and keeps the
    # one bad one, and the fetch ends up with the core four at best.
    assert metrics == {"views": 13, "reach": 6, "avg_watch_seconds": 20120}
    assert any("plays" in warning for warning in warnings)
    assert len(session.asked) == 2


def test_follows_and_profile_visits_are_asked_for(credentials):
    """The outcome column. It was in the ledger and nothing ever fetched it."""
    session = FakeSession({"/insights": {"data": [
        _insight("reach", 289), _insight("follows", 2),
        _insight("profile_visits", 11)]}})

    metrics, _ = instagram._insights(session, "1791")

    assert metrics == {"reach": 289, "follows": 2, "profile_visits": 11}
    asked = session.asked[0][1]["metric"]
    assert "follows" in asked and "profile_visits" in asked


def test_an_api_version_without_follows_still_answers_for_the_rest(credentials):
    """Asking is free: a refusal costs a warning, not the fetch."""
    session = FakeSession({"/insights": [
        {"error": {"message":
                   "metric[1] must be one of the following values: "
                   "reach, likes, comments, saved, shares, views",
                   "code": 100}},
        {"data": [_insight("reach", 289), _insight("saved", 0)]},
    ]})

    metrics, warnings = instagram._insights(session, "1791")

    assert metrics == {"reach": 289, "saves": 0}
    assert any("follows" in warning for warning in warnings)


def test_a_metric_answering_with_a_breakdown_is_not_stored_as_one(credentials):
    """A dict written into the ledger is a number every reader would divide by."""
    session = FakeSession({"/insights": {"data": [
        _insight("reach", 289),
        _insight("follows", {"profile": 2, "reel": 1})]}})

    metrics, warnings = instagram._insights(session, "1791")

    assert metrics == {"reach": 289}
    assert any("rather than a count" in warning for warning in warnings)


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


def test_the_loop_that_the_cap_hides_is_kept_beside_it(credentials):
    """Measured: 89.5s of average watch time on a 37.59s reel.

    A share of the reel cannot exceed 1.0, so the cap is right for what
    `watch_through` means - and it made that reel, which had three views, the
    best-scoring reel in the ledger. The uncapped division is what the report
    ranks on.
    """
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 89500}, {"spoken_seconds": 37.59}
    )

    assert metrics["watch_through"] == 1.0
    assert metrics["watch_ratio"] == pytest.approx(2.381, abs=0.001)


def test_below_the_cap_the_two_watch_numbers_are_the_same(credentials):
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 20000}, {"spoken_seconds": 40.0}
    )

    assert metrics["watch_through"] == metrics["watch_ratio"] == 0.5


def test_without_the_reel_s_length_there_is_no_watch_through(credentials):
    metrics = instagram.derive_watch_through(
        {"avg_watch_seconds": 32000}, {"spoken_seconds": None}
    )

    assert "watch_through" not in metrics
    assert metrics["avg_watch_seconds"] == 32.0


def test_a_post_with_no_watch_time_keeps_its_other_metrics(credentials):
    assert instagram.derive_watch_through({"reach": 12}, {}) == {"reach": 12}


# ---------------------------------------------------------------------------
# One post is one reel
# ---------------------------------------------------------------------------

def _row(tmp_path: Path, run_id: str, *, made: datetime,
         caption: str = "") -> dict:
    run_dir = tmp_path / "out" / made.strftime("%Y-%m-%d") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if caption:
        (run_dir / "post.txt").write_text(caption, encoding="utf-8")
    return {"run_id": run_id, "run_dir": str(run_dir),
            "created_at": made.isoformat(timespec="seconds"),
            "spoken_seconds": 50.0}


def test_a_reel_that_was_never_posted_does_not_borrow_the_numbers(credentials,
                                                                  tmp_path):
    """The measured failure: 7 posts, 12 reels, 5 wearing somebody else's."""
    monday = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "posted-1": _row(tmp_path, "posted-1", made=monday,
                         caption="This is Louisiana being born."),
        "shelved-1": _row(tmp_path, "shelved-1", made=monday - timedelta(hours=17)),
        "shelved-2": _row(tmp_path, "shelved-2", made=monday - timedelta(hours=2)),
    }
    posts = [_post("1792", "This is Louisiana being born. #delta",
                   monday + timedelta(hours=8))]

    found = instagram.assign(rows, posts)

    assert found["posted-1"]["media_id"] == "1792"
    assert found["posted-1"]["how"] == "caption"
    assert found["shelved-1"]["media_id"] == ""
    assert found["shelved-2"]["media_id"] == ""


def test_a_caption_takes_its_post_back_off_a_nearer_guess(credentials, tmp_path):
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "guesser": _row(tmp_path, "guesser", made=made + timedelta(hours=1)),
        "author": _row(tmp_path, "author", made=made,
                       caption="A single inch of soil."),
    }
    posts = [_post("1792", "A single inch of soil. #soil",
                   made + timedelta(hours=6)),
             _post("1793", "something else entirely",
                   made + timedelta(hours=30))]

    found = instagram.assign(rows, posts)

    assert found["author"]["media_id"] == "1792"
    # The guess is made after the caption, out of what is left.
    assert found["guesser"]["media_id"] == "1793"
    assert found["guesser"]["how"] == "timestamp"


def test_two_reels_on_one_topic_do_not_both_claim_the_post(credentials,
                                                            tmp_path):
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "retry": _row(tmp_path, "retry", made=made - timedelta(days=3),
                      caption="The sound of the deep ocean."),
        "kept": _row(tmp_path, "kept", made=made,
                     caption="The sound of the deep ocean."),
    }
    posts = [_post("1792", "The sound of the deep ocean. #ocean",
                   made + timedelta(hours=4))]

    found = instagram.assign(rows, posts)

    # Identical captions, so the reel made nearest the post is the one posted.
    assert found["kept"]["media_id"] == "1792"
    assert found["retry"]["media_id"] == ""


def test_a_pin_holds_its_post_against_a_caption(credentials, tmp_path):
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "pinned": _row(tmp_path, "pinned", made=made),
        "author": _row(tmp_path, "author", made=made, caption="Whale song."),
    }
    posts = [_post("1792", "Whale song. #whales", made + timedelta(hours=4))]

    found = instagram.assign(rows, posts, {"pinned": "1792"})

    assert found["pinned"]["media_id"] == "1792"
    assert found["pinned"]["how"] == "pinned"
    assert found["author"]["media_id"] == ""


def test_two_pins_on_one_post_do_not_both_hold_it(credentials, tmp_path):
    """Two pins on one post are a typo, not a fact. The first one wins."""
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "first": _row(tmp_path, "first", made=made),
        "second": _row(tmp_path, "second", made=made),
    }
    posts = [_post("1792", "Whale song.", made + timedelta(hours=4))]

    found = instagram.assign(rows, posts, {"first": "1792", "second": "1792"})

    assert found["first"]["media_id"] == "1792"
    assert found["second"]["media_id"] == ""


def test_a_caption_takes_its_post_off_a_remembered_guess(credentials, tmp_path):
    """The measured failure of 2026-09-17, in the shape it actually had.

    The mole-rat reel held media 18100751702021813 on a timestamp guess made
    before the brain reel existed. The brain reel then matched the same post on
    its caption, and both rows kept it - because `fetch` replayed the stored id
    as a pin and the pinned rung was the one rung that did not check `claimed`.
    """
    made = datetime(2026, 9, 12, 16, 51, tzinfo=timezone.utc)
    rows = {
        "mole": _row(tmp_path, "mole", made=made),
        "brain": _row(tmp_path, "brain", made=made + timedelta(hours=18),
                      caption="Your brain takes the rubbish out at night."),
    }
    posts = [_post("18100751702021813",
                   "Your brain takes the rubbish out at night. #sleep",
                   made + timedelta(hours=19))]

    found = instagram.assign(rows, posts, None,
                             {"mole": "18100751702021813"})

    assert found["brain"]["media_id"] == "18100751702021813"
    assert found["brain"]["how"] == "caption"
    assert found["mole"]["media_id"] == ""


def test_a_post_a_settled_reel_holds_is_not_guessed_onto_again(credentials,
                                                                tmp_path):
    """The other side of "one post is one reel", and how four rows broke it.

    `unsettled` stops offering a reel once its reading is final, so a settled
    reel is never in `features_by_run` and the post it holds used to look
    unspoken-for. The next reel made within the match window then guessed its
    way onto it - which is how two mole-rat runs, a beetle run and a bison run
    came to wear three posts that had already been settled on their captions.
    """
    made = datetime(2026, 9, 14, 5, 41, tzinfo=timezone.utc)
    rows = {"guesser": _row(tmp_path, "guesser", made=made)}
    posts = [_post("18022390649870546", "Collectors have to use a drill.",
                   made + timedelta(hours=36))]

    found = instagram.assign(rows, posts, None, None,
                             {"18022390649870546": "caption"})

    assert found["guesser"]["media_id"] == ""


def test_a_caption_still_takes_a_post_a_settled_guess_is_sitting_on(
        credentials, tmp_path):
    """Held is not sacred: a settled *guess* is still only the nearest post.

    Refusing every held post would put the guess back above the evidence, with
    the ranks reversed - the failure this matcher was rewritten to end.
    """
    made = datetime(2026, 9, 12, 16, 51, tzinfo=timezone.utc)
    rows = {"brain": _row(tmp_path, "brain", made=made,
                          caption="While you sleep, your brain floods itself.")}
    posts = [_post("18100751702021813",
                   "While you sleep, your brain floods itself. #sleep",
                   made + timedelta(hours=19))]

    found = instagram.assign(rows, posts, None, None,
                             {"18100751702021813": "timestamp"})

    assert found["brain"]["media_id"] == "18100751702021813"
    assert found["brain"]["how"] == "caption"


def test_a_timestamp_guess_cannot_take_a_post_off_another_guess(credentials,
                                                                tmp_path):
    """Only evidence may take a held post. A second guess is not evidence."""
    made = datetime(2026, 9, 12, 16, 51, tzinfo=timezone.utc)
    rows = {"guesser": _row(tmp_path, "guesser", made=made)}
    posts = [_post("18100751702021813", "nothing this reel ever wrote",
                   made + timedelta(hours=19))]

    found = instagram.assign(rows, posts, None, None,
                             {"18100751702021813": "timestamp"})

    assert found["guesser"]["media_id"] == ""


def test_a_remembered_match_keeps_its_post_when_nothing_contests_it(
        credentials, tmp_path):
    """The reason the id is stored at all: a reel keeps its post on a re-read."""
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {"r1": _row(tmp_path, "r1", made=made)}
    posts = [_post("1792", "nothing this reel ever wrote",
                   made + timedelta(hours=6))]

    found = instagram.assign(rows, posts, None, {"r1": "1792"})

    assert found["r1"]["media_id"] == "1792"
    assert found["r1"]["how"] == "remembered"


def test_a_remembered_match_does_not_outrank_a_hand_pin(credentials, tmp_path):
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    rows = {
        "guesser": _row(tmp_path, "guesser", made=made),
        "pinned": _row(tmp_path, "pinned", made=made),
    }
    posts = [_post("1792", "Whale song.", made + timedelta(hours=4))]

    found = instagram.assign(rows, posts, {"pinned": "1792"},
                             {"guesser": "1792"})

    assert found["pinned"]["media_id"] == "1792"
    assert found["guesser"]["media_id"] == ""


def test_when_the_post_went_up_comes_back_with_the_match(credentials, tmp_path):
    made = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    posted = made + timedelta(hours=6)
    rows = {"r1": _row(tmp_path, "r1", made=made, caption="Whale song.")}

    found = instagram.assign(rows, [_post("1792", "Whale song.", posted)])

    assert found["r1"]["posted_at"] == posted.isoformat(timespec="seconds")


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


# ---------------------------------------------------------------------------
# The profile's own day - what stands in for follows Meta will not give per reel
# ---------------------------------------------------------------------------

def _visits_by_day(counts: dict):
    """An account insights endpoint answering one UTC day per request."""
    def answer(params):
        day = datetime.fromtimestamp(int(params["since"]), timezone.utc)
        value = counts.get(day.date().isoformat())
        if value is None:
            return {"data": []}
        return {"data": [{"name": "profile_views", "period": "day",
                          "total_value": {"value": value}}]}
    return answer


def test_profile_visits_are_read_one_finished_day_at_a_time(credentials):
    session = FakeSession({"/insights": _visits_by_day(
        {"2026-10-11": 0, "2026-10-12": 3, "2026-10-13": 4, "2026-10-14": 9})})

    days = instagram.account_days(session, today=date(2026, 10, 14), days=3)

    assert days == {"2026-10-11": {"profile_visits": 0},
                    "2026-10-12": {"profile_visits": 3},
                    "2026-10-13": {"profile_visits": 4}}
    # Today has not finished, so it is never asked.
    assert len(session.asked) == 3
    assert all(params["metric_type"] == "total_value"
               for _, params in session.asked)


def test_a_day_already_held_is_not_asked_for_again(credentials):
    session = FakeSession({"/insights": _visits_by_day({"2026-10-13": 4})})

    days = instagram.account_days(session, today=date(2026, 10, 14), days=3,
                                  have=["2026-10-11", "2026-10-12"])

    assert days == {"2026-10-13": {"profile_visits": 4}}
    assert len(session.asked) == 1


def test_a_day_with_no_number_is_left_out_rather_than_read_as_nought(credentials):
    session = FakeSession({"/insights": _visits_by_day({"2026-10-13": 4})})

    days = instagram.account_days(session, today=date(2026, 10, 14), days=2)

    assert "2026-10-12" not in days


def test_one_refused_day_ends_the_account_read(credentials):
    """A token that cannot have one day cannot have the other twenty-seven."""
    session = FakeSession({"/insights": {"error": {
        "message": "(#100) Invalid metric", "code": 100}}})

    assert instagram.account_days(session, today=date(2026, 10, 14)) == {}
    assert len(session.asked) == 1


def test_fetch_keeps_the_follower_count_and_the_visits_by_day(credentials,
                                                              tmp_path,
                                                              monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    visits = _visits_by_day({yesterday.isoformat(): 4})

    def insights(params):
        # The account asks with a day; a reel asks without one.
        if "since" in params:
            return visits(params)
        return {"data": [_insight("reach", 900)]}

    session = FakeSession({"/insights": insights,
                           "/media": {"data": [_post(
                               "1792", "Snailfish live deep.",
                               made + timedelta(hours=18))]},
                           "17800000000000000": {
                               "id": "17800000000000000",
                               "username": "reels", "followers_count": 7}})
    monkeypatch.setattr(instagram, "_session", lambda: session)
    monkeypatch.setattr(instagram, "ACCOUNT_DAILY_DAYS", 2)

    instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features})

    daily = perform.PerformanceLedger().audience["daily"]
    today = datetime.now(timezone.utc).date().isoformat()
    assert daily[today] == {"followers": 7}
    assert daily[yesterday.isoformat()] == {"profile_visits": 4}
    assert perform.PerformanceLedger().entry("ocean-1a2b3c")["media_id"] == "1792"


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


def test_step_9_keeps_what_the_provider_wrote_about_the_match(credentials,
                                                             tmp_path,
                                                             monkeypatch):
    """The provider saves the trail, then step 9 saves the numbers.

    Both write the same file. Step 9 read it before the provider ran, so
    without the merge its save reverts the media id, the how, and the posting
    time - and a timestamp guess stops being auditable at all.
    """
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    posted = made + timedelta(hours=18)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 4200)]},
           posts=[_post("1792", "Snailfish live deep.", posted)])

    perform.fetch(ledger=ledger, provider=instagram.fetch)

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["media_id"] == "1792"
    assert entry["matched_by"] == "caption"
    assert entry["permalink"].endswith("1792")
    # And the post's own time is when it was posted, not when the reel was made.
    assert entry["posted_at"] == posted.isoformat(timespec="seconds")
    assert entry["posted_at_assumed"] is False


def test_a_reel_read_again_keeps_the_newer_numbers(credentials, tmp_path,
                                                   monkeypatch):
    """The defect this closes: read once at six hours, judged on it forever."""
    made = datetime.now(timezone.utc) - timedelta(hours=6)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 3)]},
           posts=[_post("1792", "Snailfish live deep.", made)])
    perform.fetch(ledger=ledger, provider=instagram.fetch)

    # A day later, the same reel, and the provider is asked again because the
    # reading it has is younger than the settle age.
    ledger = perform.PerformanceLedger()
    assert [row["run_id"] for row in ledger.unsettled()] == ["ocean-1a2b3c"]
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 412)]},
           posts=[_post("1792", "Snailfish live deep.", made)])
    perform.fetch(ledger=ledger, provider=instagram.fetch)

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["metrics"]["views"] == 412
    assert [reading["metrics"]["views"]
            for reading in entry["measurements"]] == [3, 412]


def test_a_guessed_match_is_not_promoted_to_pinned_by_a_re_read(credentials,
                                                                tmp_path,
                                                                monkeypatch):
    """A timestamp match is a guess, and a guess has to stay auditable."""
    made = datetime.now(timezone.utc) - timedelta(hours=6)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 3)]},
           posts=[_post("1792", "something else entirely",
                        made + timedelta(hours=1))])
    perform.fetch(ledger=ledger, provider=instagram.fetch)
    assert perform.PerformanceLedger().entry("ocean-1a2b3c")["matched_by"] \
        == "timestamp"

    ledger = perform.PerformanceLedger()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 412)]},
           posts=[_post("1792", "something else entirely",
                        made + timedelta(hours=1))])
    perform.fetch(ledger=ledger, provider=instagram.fetch)

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["matched_by"] == "timestamp"
    assert entry["metrics"]["views"] == 412


def test_two_reels_are_never_handed_the_same_numbers(credentials, tmp_path,
                                                     monkeypatch):
    made = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
    posted = _features(tmp_path, made=made, caption="Snailfish live deep.")
    shelved = _row(tmp_path, "canyon-9f9f9f9f", made=made - timedelta(hours=4))
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("views", 4200)]},
           posts=[_post("1792", "Snailfish live deep.",
                        made + timedelta(hours=18))])

    out = instagram.fetch(["canyon-9f9f9f9f", "ocean-1a2b3c"],
                          {"canyon-9f9f9f9f": shelved, "ocean-1a2b3c": posted})

    assert list(out) == ["ocean-1a2b3c"]


def test_a_reel_that_loses_its_post_loses_the_numbers_it_read_off_it(
        credentials, tmp_path, monkeypatch):
    """The corrupting half: a refused claim that keeps its borrowed numbers.

    Reading this back is how the ledger came to say 14 posted against an
    account with 13 posts. The row has to go back to pending with nothing on
    it, because a wrong number is indistinguishable from a real one in the
    report.
    """
    made = datetime.now(timezone.utc) - timedelta(hours=30)
    mole = _row(tmp_path, "mole-98c893d3", made=made)
    brain = _row(tmp_path, "brain-dbbe0d8f", made=made + timedelta(hours=18),
                 caption="Your brain takes the rubbish out at night.")
    ledger = perform.PerformanceLedger()
    ledger.register(mole)
    ledger.register(brain)
    ledger.record("mole-98c893d3", {"reach": 15}, source="provider")
    ledger.reels["mole-98c893d3"]["media_id"] = "1792"
    ledger.reels["mole-98c893d3"]["matched_by"] = "timestamp"
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("reach", 15)]},
           posts=[_post("1792", "Your brain takes the rubbish out at night.",
                        made + timedelta(hours=19))])

    out = instagram.fetch(["mole-98c893d3", "brain-dbbe0d8f"],
                          {"mole-98c893d3": mole, "brain-dbbe0d8f": brain})

    assert list(out) == ["brain-dbbe0d8f"]
    disowned = perform.PerformanceLedger().entry("mole-98c893d3")
    assert disowned["status"] == "pending"
    assert "metrics" not in disowned
    assert "media_id" not in disowned
    # The reel itself is untouched - it was made, whatever it did or did not
    # become.
    assert disowned["features"]["run_id"] == "mole-98c893d3"


def test_a_disowned_row_survives_the_ledger_save_around_it(credentials,
                                                            tmp_path,
                                                            monkeypatch):
    """Through `perform.fetch`, which is the path that actually runs.

    The provider clears the row on disk while `perform.fetch` is holding a copy
    read before it ran, and a disowned reel is not in `answers` - it has no
    numbers, that is the point - so the annotation merge never reaches it and
    the save puts the borrowed numbers straight back.

    The reading here is 10 hours old, which is to say still unsettled. That is
    the only state this heals from: `unsettled` never asks again about a row
    whose reading is already comparable, so a bad match that got a comparable
    reading stays until somebody repairs it by hand. Finding one is a REELS
    count above the AUDIENCE post count in `--report`.
    """
    made = datetime.now(timezone.utc) - timedelta(hours=10)
    mole = _row(tmp_path, "mole-98c893d3", made=made)
    brain = _row(tmp_path, "brain-dbbe0d8f", made=made + timedelta(hours=1),
                 caption="Your brain takes the rubbish out at night.")
    ledger = perform.PerformanceLedger()
    ledger.register(mole)
    ledger.register(brain)
    ledger.record("mole-98c893d3", {"reach": 15}, source="provider")
    ledger.reels["mole-98c893d3"]["media_id"] = "1792"
    ledger.reels["mole-98c893d3"]["matched_by"] = "timestamp"
    ledger.save()
    _wired(tmp_path, monkeypatch,
           insights={"data": [_insight("reach", 15)]},
           posts=[_post("1792", "Your brain takes the rubbish out at night.",
                        made + timedelta(hours=2))])

    perform.fetch(ledger=perform.PerformanceLedger(), provider=instagram.fetch)

    disowned = perform.PerformanceLedger().entry("mole-98c893d3")
    assert disowned["status"] == "pending"
    assert "metrics" not in disowned
    assert perform.PerformanceLedger().entry(
        "brain-dbbe0d8f")["metrics"]["reach"] == 15


def test_numbers_typed_in_by_hand_are_not_dropped_by_the_matcher(
        credentials, tmp_path, monkeypatch):
    """A person saying what a reel did outranks a provider losing the post."""
    made = datetime.now(timezone.utc) - timedelta(hours=30)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.record("ocean-1a2b3c", {"reach": 400}, source="manual")
    ledger.reels["ocean-1a2b3c"]["media_id"] = "1792"
    ledger.reels["ocean-1a2b3c"]["matched_by"] = "timestamp"
    ledger.save()
    _wired(tmp_path, monkeypatch, insights={"data": []},
           posts=[_post("4242", "some other account's post", made)])

    instagram.fetch(["ocean-1a2b3c"], {"ocean-1a2b3c": features})

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["metrics"]["reach"] == 400
    assert entry["status"] == "posted"


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



# ---------------------------------------------------------------------------
# The audience the reach figures are divided by
# ---------------------------------------------------------------------------

def test_the_account_call_asks_for_the_follower_count(credentials):
    session = FakeSession({"/": {"id": "178", "username": "reels",
                                 "followers_count": 2, "follows_count": 0,
                                 "media_count": 8}})
    instagram.account(session)
    asked = session.asked[0][1]["fields"]
    assert "followers_count" in asked


def test_a_field_the_api_refuses_does_not_cost_the_whole_call(credentials):
    """A token that cannot see the follower count must still be able to prove
    itself - proving itself is this call's first job."""
    answers = [
        {"error": {"message": "(#100) followers_count is not valid",
                   "code": 100}},
        {"id": "178", "username": "reels"},
    ]
    session = FakeSession({"/": answers})

    body = instagram.account(session)

    assert body["username"] == "reels"
    assert "followers_count" not in session.asked[-1][1]["fields"]


def test_the_audience_is_read_into_the_shape_step_9_stores(credentials):
    session = FakeSession({"/": {"id": "178", "username": "reels",
                                 "followers_count": 2, "follows_count": 0,
                                 "media_count": 8}})
    found = instagram.audience(session)

    assert found["followers"] == 2
    assert found["following"] == 0
    assert found["posts"] == 8
    assert found["username"] == "reels"
    assert found["read_at"]


def test_an_unreadable_audience_is_empty_rather_than_zero(credentials):
    """Nobody following the account and no permission to see who does are
    different findings, and only one of them is about the reels."""
    session = FakeSession({"/": [
        {"error": {"message": "(#100) followers_count is not valid", "code": 100}},
        {"error": {"message": "(#190) bad token", "code": 190}},
    ]})
    assert instagram.audience(session) == {}


# ---------------------------------------------------------------------------
# How long the token has left
#
# A long-lived token lasts sixty days and can only be refreshed while it is
# still valid. The failure mode is not "the read errors today", it is "the read
# has been erroring for three weeks and nothing said so" - which is what an
# unattended daily task is for and what it was not doing.
# ---------------------------------------------------------------------------

def test_the_first_sight_of_a_token_starts_its_clock(tmp_path):
    state = tmp_path / "token-age.json"

    aged = instagram.token_age("tok-abc", state_path=state,
                               today=date(2026, 9, 17))

    assert aged["first_seen"] == "2026-09-17"
    assert aged["days_old"] == 0
    assert aged["days_left"] == config.INSTAGRAM_TOKEN_DAYS
    assert aged["warn"] is False


def test_the_token_itself_is_never_written_down(tmp_path):
    """Only a truncated hash of it, so the file is not a second copy of .env."""
    state = tmp_path / "token-age.json"

    instagram.token_age("hunter2-secret", state_path=state,
                        today=date(2026, 9, 17))

    assert "hunter2-secret" not in state.read_text(encoding="utf-8")


def test_the_clock_keeps_running_for_the_same_token(tmp_path):
    state = tmp_path / "token-age.json"
    instagram.token_age("tok-abc", state_path=state, today=date(2026, 9, 1))

    aged = instagram.token_age("tok-abc", state_path=state,
                               today=date(2026, 9, 17))

    assert aged["days_old"] == 16
    assert aged["days_left"] == config.INSTAGRAM_TOKEN_DAYS - 16


def test_pasting_a_new_token_restarts_the_clock_by_itself(tmp_path):
    """Nobody has to remember to reset anything - the hash changing is the
    signal, which is the whole reason it is stored."""
    state = tmp_path / "token-age.json"
    instagram.token_age("tok-old", state_path=state, today=date(2026, 7, 1))

    aged = instagram.token_age("tok-new", state_path=state,
                               today=date(2026, 9, 17))

    assert aged["first_seen"] == "2026-09-17"
    assert aged["days_old"] == 0


def test_a_token_inside_the_warning_window_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INSTAGRAM_TOKEN_DAYS", 60)
    monkeypatch.setattr(config, "INSTAGRAM_TOKEN_WARN_DAYS", 14)
    state = tmp_path / "token-age.json"
    instagram.token_age("tok-abc", state_path=state, today=date(2026, 7, 20))

    aged = instagram.token_age("tok-abc", state_path=state,
                               today=date(2026, 9, 17))

    assert aged["days_left"] == 1
    assert aged["warn"] is True


def test_no_token_is_no_answer_rather_than_a_zero(tmp_path, monkeypatch):
    """A hand-typed ledger has no token, and saying it expires today would be
    a warning about nothing."""
    monkeypatch.setattr(config, "INSTAGRAM_ACCESS_TOKEN", "")

    assert instagram.token_age("", state_path=tmp_path / "s.json") == {}


def test_an_unreadable_state_file_costs_the_warning_not_the_read(tmp_path):
    """Losing the file restarts the clock, which errs towards saying nothing
    rather than towards a wrong date."""
    state = tmp_path / "token-age.json"
    state.write_text("{not json", encoding="utf-8")

    aged = instagram.token_age("tok-abc", state_path=state,
                               today=date(2026, 9, 17))

    assert aged["first_seen"] == "2026-09-17"
    assert aged["warn"] is False


# ---------------------------------------------------------------------------
# Follows by day, and the weekly lifetime read
# ---------------------------------------------------------------------------

def _follows_by_day(counts: dict):
    """The account answering follows_and_unfollows for one UTC day a request,
    with no results on a day Meta has not filled in yet."""
    def answer(params):
        day = datetime.fromtimestamp(int(params["since"]), timezone.utc).date()
        breakdown = {"dimension_keys": ["follow_type"]}
        if day.isoformat() in counts:
            follows, unfollows = counts[day.isoformat()]
            breakdown["results"] = [
                {"dimension_values": ["FOLLOWER"], "value": follows},
                {"dimension_values": ["NON_FOLLOWER"], "value": unfollows}]
        return {"data": [{"name": "follows_and_unfollows", "period": "day",
                          "total_value": {"breakdowns": [breakdown]}}]}
    return answer


def test_follows_are_read_one_finished_day_at_a_time(credentials):
    session = FakeSession({"/insights": _follows_by_day(
        {"2026-10-17": (20, 0), "2026-10-18": (34, 1)})})

    days = instagram.account_follow_days(session, today=date(2026, 10, 20),
                                         days=3)

    assert days == {"2026-10-17": {"follows": 20, "unfollows": 0},
                    "2026-10-18": {"follows": 34, "unfollows": 1}}
    assert len(session.asked) == 3
    assert all(params["breakdown"] == "follow_type"
               for _, params in session.asked)


def test_a_day_meta_has_not_filled_in_is_asked_again_not_stored(credentials):
    """On 2026-10-20 the 19th came back with no results at all."""
    session = FakeSession({"/insights": _follows_by_day({"2026-10-18": (34, 1)})})

    days = instagram.account_follow_days(session, today=date(2026, 10, 20),
                                         days=2)

    assert "2026-10-19" not in days
    assert days["2026-10-18"] == {"follows": 34, "unfollows": 1}


def test_a_token_that_cannot_read_follows_loses_nothing_else(credentials):
    session = FakeSession({"/insights": {"error": {
        "message": "(#10) Application does not have permission for this action",
        "code": 10}}})

    assert instagram.account_follow_days(session,
                                         today=date(2026, 10, 20)) == {}
    assert len(session.asked) == 1


def test_a_reel_a_week_old_is_read_again_for_its_lifetime(credentials,
                                                          tmp_path,
                                                          monkeypatch):
    """Into a reading of its own: the one every table compares does not move."""
    made = datetime.now(timezone.utc) - timedelta(days=9)
    posted = made + timedelta(hours=1)
    features = _features(tmp_path, made=made, caption="Snailfish live deep.")
    ledger = perform.PerformanceLedger()
    ledger.register(features)
    ledger.record("ocean-1a2b3c", {"reach": 9200, "shares": 112},
                  posted_at=posted.isoformat(timespec="seconds"),
                  source="provider")
    ledger.reels["ocean-1a2b3c"].update(hours_after_post=39.3,
                                        media_id="1792", matched_by="caption")
    ledger.save()
    session = _wired(tmp_path, monkeypatch,
                     insights={"data": [_insight("reach", 32987),
                                        _insight("shares", 1196)]},
                     posts=[_post("1792", "Snailfish live deep.", posted)])

    perform.fetch(ledger=perform.PerformanceLedger(), provider=instagram.fetch)

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["metrics"]["reach"] == 9200
    assert entry["hours_after_post"] == 39.3
    assert entry["lifetime"]["metrics"] == {"reach": 32987, "shares": 1196}
    asked = [params["metric"] for url, params in session.asked
             if "1792/insights" in url]
    assert asked == [",".join(instagram.LIFETIME_METRICS)]
