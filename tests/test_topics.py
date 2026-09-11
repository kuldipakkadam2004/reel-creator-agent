"""Tests for pipeline/topics.py.

Everything here is offline: the sources are fed canned payloads, the stock
providers are stubbed, the Gemini call is intercepted and the ledger is written
to a tmp_path, so the suite never calls an API, spends a token or touches the
real ledger.

For a live check run `python pipeline/topics.py --source trends --no-phrases`
by hand - the trends feed needs no key.

    python -m pytest tests/test_topics.py -v
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import script_gen, topics, visuals  # noqa: E402
from pipeline.topics import Candidate  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    """Stand-in for requests.Response covering what topics.py touches."""

    def __init__(self, status_code=200, payload=None, body=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = body if payload is None else json.dumps(payload).encode()
        self.headers = dict(headers or {})

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Routes GETs by substring of the URL. Records everything it was asked."""

    def __init__(self, routes=None, post=None):
        self.routes = dict(routes or {})
        self.headers = {}
        self.calls = []
        self.posts = []
        self._post = post

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {},
                           "headers": headers or {}})
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        return FakeResponse(404, body=b"no route")

    def post(self, url, auth=None, data=None, headers=None, timeout=None):
        self.posts.append({"url": url, "auth": auth, "data": data})
        return self._post if self._post is not None else FakeResponse(404)


class FakeProvider:
    """A stock provider that returns whatever it was constructed with."""

    def __init__(self, name, results=(), kinds=("video",), error=None):
        self.name = name
        self.kinds = kinds
        self.results = list(results)
        self.error = error
        self.searches = []

    def search(self, query, kind, limit):
        self.searches.append((query, kind, limit))
        if self.error:
            raise self.error
        return self.results[:limit]


def stock(*descriptions, title=""):
    """Stock results described only by their URL slug, as Pexels serves them."""
    return [
        visuals.Candidate(
            provider="fake", provider_id=str(index), kind="video",
            width=1080, height=1920, download_url="http://x/v.mp4",
            source_url=f"https://www.pexels.com/video/{slug}-{index}000/",
            title=title,
        )
        for index, slug in enumerate(descriptions)
    ]


def _numbered(prompt):
    """The `0. topic` lines a classifier prompt lists, in order.

    Lets a mocked `call_gemini` answer about a named topic instead of about a
    position, so a test does not silently rate the wrong candidate when the
    order the shortlist is built in changes.
    """
    found = re.findall(r"^\s*(\d+)\.\s*(.+)$", prompt, re.MULTILINE)
    return [line for _, line in sorted(found, key=lambda pair: int(pair[0]))]


def candidate(topic, **kwargs):
    kwargs.setdefault("source", "reddit")
    kwargs.setdefault("external_id", topics.slugify(topic) or "x")
    kwargs.setdefault("title", topic)
    return Candidate(topic=topic, **kwargs)


@pytest.fixture
def ledger(tmp_path):
    return topics.TopicLedger(tmp_path / "usage.json")


# ---------------------------------------------------------------------------
# Title cleaning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("TIL that the Dead Sea has no outlet",
     "the Dead Sea has no outlet"),
    ("[OC] ELI5: why the ocean is salty",
     "why the ocean is salty"),
    ("PSA - check your smoke alarm",
     "check your smoke alarm"),
    ("How volcanoes form | National Geographic",
     "How volcanoes form"),
    ("“Why we dream”", "Why we dream"),
    ("   spaced    out   title  ", "spaced out title"),
])
def test_clean_title_strips_the_furniture(raw, expected):
    assert topics.clean_title(raw) == expected


def test_clean_title_handles_stacked_prefixes():
    assert topics.clean_title("[OC] TIL that lightning strikes twice") == \
        "lightning strikes twice"


def test_clean_title_strips_emoji():
    """A YouTube title with an emoji crashed the CLI on a cp1252 console, and
    a thumbnail-bait emoji is not part of what a video is about either."""
    cleaned = topics.clean_title("New piece of Earth being born 🌊 beneath the ocean")
    assert "🌊" not in cleaned
    assert cleaned == "New piece of Earth being born beneath the ocean"


def test_the_cli_output_survives_a_console_that_cannot_encode_a_title(capsys):
    """The board is gathered, probed and paid for before anything is printed,
    so the last line must not be able to throw it away."""
    topics._make_output_lossy()          # must not raise on any stream type


def test_youtube_is_not_judged_against_the_breaking_news_window(monkeypatch):
    """Every YouTube candidate on the first live board was twelve days old and
    every one took the stale penalty - it is asked what has already worked."""
    monkeypatch.setattr(config, "TOPIC_MAX_AGE_HOURS", 72.0)
    monkeypatch.setattr(config, "TOPIC_YOUTUBE_WINDOW_DAYS", 30)
    # Pinned, not inherited: the reddit half of this test needs a window
    # shorter than 306h to be stale at all, and reading that from whatever
    # .env happens to say makes the assertion depend on the operator's config.
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "top")
    monkeypatch.setattr(config, "TOPIC_REDDIT_WINDOW", "week")

    fresh_enough = candidate("how volcanoes form", source="youtube",
                             external_id="v1", age_hours=306)
    assert "past its peak" not in topics.score_candidate(fresh_enough)["components"]

    same_age_on_reddit = candidate("how volcanoes form", age_hours=306)
    assert "past its peak" in topics.score_candidate(same_age_on_reddit)["components"]


def test_youtube_past_its_own_window_is_still_stale():
    stale = candidate("how volcanoes form", source="youtube",
                      external_id="v2", age_hours=30 * 24 + 100)
    assert "past its peak" in topics.score_candidate(stale)["components"]


def test_clean_title_leaves_an_ordinary_headline_alone():
    headline = "Scientists explain why the ocean is salty"
    assert topics.clean_title(headline) == headline


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_velocity_scores_the_rate_not_the_total():
    slow = candidate("why the ocean is salty", reach=8000, rate=8000 / 168,
                     age_hours=168)
    fast = candidate("why the ocean is salty", reach=600, rate=600 / 2,
                     age_hours=2)
    assert (topics.score_candidate(fast)["components"]["velocity"]
            > topics.score_candidate(slow)["components"]["velocity"])


def test_a_fresh_fast_candidate_is_marked_still_climbing():
    parts = topics.score_candidate(
        candidate("why the ocean is salty", reach=600, rate=400, age_hours=1.5)
    )["components"]
    assert parts["still climbing"] == topics.RISING_BONUS


def test_a_day_old_candidate_is_not_climbing_however_fast():
    parts = topics.score_candidate(
        candidate("why the ocean is salty", reach=9000, rate=400, age_hours=30)
    )["components"]
    assert "still climbing" not in parts


def test_an_explainer_beats_a_fixture_on_shape_alone():
    explainer = topics.score_candidate(candidate("why the ocean is salty"))
    fixture = topics.score_candidate(candidate("Braves at Twins on Sunday"))
    assert explainer["components"]["explainer shape"] == topics.EXPLAINER_BONUS
    assert "proper nouns" in fixture["components"]
    assert explainer["score"] > fixture["score"]


def test_proper_noun_penalty_is_capped():
    # A sentence, not a list of surnames: six bare capitalised words are
    # indistinguishable from Title Case and are read as a style instead.
    parts = topics.score_candidate(
        candidate("Diggs signs with Seattle after leaving Dallas "
                  "and joining Denver next season")
    )["components"]
    assert parts["proper nouns"] == topics.PROPER_NOUN_FLOOR


def test_title_case_is_a_house_style_not_a_list_of_names():
    """Every YouTube candidate on the first live board took the full penalty,
    and not one of them contained a name."""
    assert topics._proper_nouns("What If All Oceans Vanished for 30 Days") == []
    assert topics._proper_nouns("How Mangroves Save Coastlines") == []
    # A sentence with real names in it still reads as one.
    assert topics._proper_nouns("Trevon Diggs signs with Seattle") == ["Diggs", "Seattle"]


def _rss(subreddits):
    return topics.RedditRssSource(FakeSession(), subreddits=tuple(subreddits))


def test_a_starved_subreddit_is_named_rather_than_silent():
    """A combined feed is ranked by absolute score, so a small subreddit sharing
    a group with a large one is not read thinly - it is not read at all. That is
    how seven of nine went unread for weeks behind a healthy-looking `fetched
    40`."""
    source = _rss(["todayilearned", "geology", "oceans"])
    found = [candidate("a thing", context="r/todayilearned top #1")]
    notes = source._starved(found)

    assert len(notes) == 1
    assert "geology" in notes[0] and "oceans" in notes[0]
    assert "todayilearned" not in notes[0]


def test_the_starvation_note_matches_the_subreddit_however_it_was_typed():
    """The feed spells a subreddit the way its mods do; TOPIC_SUBREDDITS however
    it was typed. Comparing them exactly reported every one as starved."""
    source = _rss(["earthscience", "marinebiology"])
    found = [candidate("a thing", context="r/EarthScience top #1"),
             candidate("another", context="r/marinebiology top #2")]
    assert source._starved(found) == []


def test_nothing_is_said_when_every_subreddit_was_read():
    source = _rss(["oceans"])
    assert source._starved([candidate("x", context="r/oceans top #1")]) == []
    # And an empty fetch is the failure path's business, not this one's.
    assert source._starved([]) == []


def test_a_short_capitalised_phrase_is_not_treated_as_title_case():
    """Two capitals is a name, not a style - the sample is too small to tell
    otherwise, and treating it as a style would disarm the drop rule."""
    assert topics._proper_nouns("Trevon Diggs") == ["Diggs"]


def test_title_case_holds_back_the_drop_rule(offline, monkeypatch):
    """No reliable name signal means no evidence for the one hard filter."""
    monkeypatch.setattr(
        topics, "_ready_providers",
        lambda session, warnings: [FakeProvider("pexels", stock("a-man-in-a-studio"))],
    )
    install_source(monkeypatch, "reddit", [
        candidate("What If All Oceans Vanished for 30 Days", external_id="t1",
                  rate=300, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))
    assert len(board["candidates"]) == 1, "a 0/8 probe alone must not drop it"


def test_hashtags_and_emoji_do_not_survive_into_a_topic():
    assert topics.clean_title("How Mangroves Save Coastlines #shorts #nature") ==         "How Mangroves Save Coastlines"
    assert topics.clean_title(
        "A new piece of Earth is being born 🌋 beneath the ocean ☄ #nasa"
    ) == "A new piece of Earth is being born beneath the ocean"


def test_a_niche_of_ocean_matches_a_topic_about_oceans():
    """Exact set intersection missed the plural, so every ocean topic on an
    ocean niche scored as off-niche."""
    terms = topics._terms("ocean, earth science and nature")
    parts = topics.score_candidate(
        candidate("What If All Oceans Vanished for 30 Days"), niche_terms=terms
    )["components"]
    assert "off the niche" not in parts
    assert parts["niche match"] > 0


def test_a_short_topic_from_a_long_headline_is_not_drift():
    """Scoring how much of the ORIGINAL survived threw away good rewrites."""
    assert topics._phrase_is_faithful(
        "We've Explored Only 5% of the Ocean... What's Hiding in the Other 95%?",
        "what is hiding in the unexplored ocean",
    )


def test_a_rewrite_built_from_words_that_were_never_there_is_drift():
    assert not topics._phrase_is_faithful(
        "the ocean is salty because of rain", "the best pasta recipes in Rome"
    )


def test_places_and_planets_are_not_the_proper_nouns_that_hurt():
    assert topics._proper_nouns("why the Atlantic is saltier than the Pacific") == []
    assert topics._proper_nouns("how Mars lost its atmosphere") == []
    assert topics._proper_nouns("Trevon Diggs signs") == ["Diggs"]


def test_a_topic_off_the_niche_is_penalised_but_not_dropped():
    terms = topics._terms("ocean science marine biology")
    parts = topics.score_candidate(
        candidate("how sourdough starter works"), niche_terms=terms
    )["components"]
    assert parts["off the niche"] == topics.NICHE_MISS_PENALTY


def test_a_topic_on_the_niche_is_paid_for_it():
    terms = topics._terms("ocean science marine biology")
    parts = topics.score_candidate(
        candidate("why the ocean is salty"), niche_terms=terms
    )["components"]
    assert parts["niche match"] > 0


def test_a_stale_candidate_is_scored_down():
    parts = topics.score_candidate(
        candidate("why the ocean is salty", age_hours=200), max_age_hours=72
    )["components"]
    assert parts["past its peak"] == topics.STALE_PENALTY


def test_a_banned_word_sinks_a_candidate():
    result = topics.score_candidate(
        candidate("why the ocean is salty"), banned=("ocean",)
    )
    assert result["components"]["banned word"] == topics.BANNED_WORD_PENALTY
    assert result["score"] < 0


def test_an_empty_candidate_scores_bottom():
    assert topics.score_candidate(candidate(""))["score"] == -99.0


def test_every_component_is_reported_so_a_board_can_be_read_back():
    result = topics.score_candidate(
        candidate("why the ocean is salty", reach=600, rate=300, age_hours=2)
    )
    assert result["score"] == round(sum(result["components"].values()), 3)
    assert set(result["components"]) >= {"velocity", "explainer shape"}


# ---------------------------------------------------------------------------
# Curiosity
# ---------------------------------------------------------------------------

def test_a_topic_with_a_gap_in_it_outscores_a_flat_one():
    """The rest of the scoring says whether a topic can be made. This is the
    only part that says whether the reel is worth watching."""
    gap = topics.score_candidate(
        candidate("the deepest part of the ocean is not where you would think")
    )
    flat = topics.score_candidate(candidate("the ocean is large and blue"))
    assert gap["score"] > flat["score"]


def test_the_curiosity_component_names_the_signals_that_fired():
    """A board that prints "+1.0" can only be accepted; one that prints
    "curiosity (extreme, scale)" can be argued with."""
    parts = topics.score_candidate(
        candidate("why 90 percent of the deepest ocean is unexplored")
    )["components"]
    name = next(key for key in parts if key.startswith("curiosity"))
    assert "extreme" in name and "scale" in name
    assert parts[name] == 2 * topics.CURIOSITY_SIGNAL_WEIGHT


def test_curiosity_counts_distinct_signals_not_repeated_ones():
    once = topics.curiosity_component("what nobody knows about the sea floor")
    twice = topics.curiosity_component(
        "what nobody knows and cannot explain about the sea floor"
    )
    assert once[0] == twice[0] == topics.CURIOSITY_SIGNAL_WEIGHT


def test_the_off_arm_does_not_charge_curiosity_at_all(ledger):
    """The 'off' arm of PERFORMANCE_EXPLORE=curiosity, at the scorer.

    Turned off rather than reweighted on purpose: the trial asks whether
    CURIOSITY_SIGNAL_WEIGHT earns its place, and any other number to compare
    0.5 against would be a second prior invented to test the first.
    """
    topic = candidate("why 90 percent of the deepest ocean is unexplored")

    on = topics.score_candidate(topic)
    off = topics.score_candidate(topic, curiosity=False)

    assert any(key.startswith("curiosity") for key in on["components"])
    assert not any(key.startswith("curiosity") for key in off["components"])
    assert round(on["score"] - off["score"], 3) == round(
        2 * topics.CURIOSITY_SIGNAL_WEIGHT, 3)


def test_the_on_arm_is_exactly_what_every_other_caller_gets(ledger):
    """None and True both mean "charge it" - there is no third state."""
    topic = candidate("why 90 percent of the deepest ocean is unexplored")

    assert (topics.score_candidate(topic, curiosity=None)["score"]
            == topics.score_candidate(topic)["score"])
    assert (topics.score_candidate(topic, curiosity=True)["score"]
            == topics.score_candidate(topic)["score"])


def test_turning_curiosity_off_leaves_every_other_component_alone():
    """The arm has to move one thing, or its segment answers two questions."""
    topic = candidate("why 90 percent of the deepest ocean is unexplored")

    on = topics.score_candidate(topic)["components"]
    off = topics.score_candidate(topic, curiosity=False)["components"]

    assert {key: value for key, value in on.items()
            if not key.startswith("curiosity")} == off


def test_the_off_arm_can_change_which_candidate_wins():
    """A trial that never changes the pick is not a trial.

    One signal is worth 0.5 against a real board whose top scores sit inside a
    point of each other, so the arm reorders it rather than nudging it - which
    is the whole reason it is worth a day of the rota. The margin here is set
    with `discussion`, the one component that takes a continuous value, so the
    two candidates differ by 0.2 before curiosity is charged and by -0.3
    after.
    """
    # 1.2 explainer + 0.3 discussion, and nothing curiosity can touch.
    plain = candidate("how a river delta builds new land", comments=1, reach=20)
    # 1.2 explainer + 0.5 for one contradiction signal.
    curious = candidate("why the seasons are not caused by distance")

    def winner(**kwargs):
        return max(((topics.score_candidate(item, **kwargs)["score"], item.topic)
                    for item in (plain, curious)))[1]

    assert winner() == curious.topic
    assert winner(curiosity=False) == plain.topic


def test_a_superlative_inside_a_longer_word_is_not_a_superlative():
    """"epic" lives inside "epicentre", and on an earth-science account a
    substring match would fire on half the board."""
    assert topics.curiosity_signals("what the epicentre of a quake is") == []
    assert topics.hype_words("what the epicentre of a quake is") == []


def test_hype_in_a_title_is_scored_down_the_way_it_is_in_a_hook():
    parts = topics.score_candidate(
        candidate("this insane ocean fact will blow your mind")
    )["components"]
    assert parts["hype in the title"] == topics.HYPE_PENALTY


# ---------------------------------------------------------------------------
# The niche vocabulary
# ---------------------------------------------------------------------------

def test_a_search_keyword_does_not_become_part_of_the_subject():
    """TOPIC_KEYWORDS is a query string for the YouTube search, not a
    description of the account. Read as both, a keyword of "natural world" put
    the bare token "world" in the vocabulary - which is what scored a hockey
    world cup a full niche match on an ocean-science board."""
    terms = topics.niche_vocabulary("ocean, earth science and nature",
                                    ("natural world", "how things work"))
    assert "world" not in terms and "things" not in terms
    parts = topics.score_candidate(
        candidate("2026 Men's FIH Hockey World Cup"), niche_terms=terms
    )["components"]
    assert parts["off the niche"] == topics.NICHE_MISS_PENALTY


def test_keywords_are_the_vocabulary_only_when_there_is_no_niche():
    assert topics.niche_vocabulary("", ("marine biology",)) == {"marine", "biology"}


# ---------------------------------------------------------------------------
# The reddit listing
# ---------------------------------------------------------------------------

def test_a_week_of_top_posts_is_not_judged_against_the_news_window(monkeypatch):
    """The same argument YouTube already won: a source asked for the best of
    the week should not be penalised for returning something six days old."""
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "top")
    monkeypatch.setattr(config, "TOPIC_REDDIT_WINDOW", "week")
    monkeypatch.setattr(config, "TOPIC_MAX_AGE_HOURS", 72.0)
    assert topics.max_age_for("redditrss") == 168.0
    parts = topics.score_candidate(
        candidate("why the ocean is salty", source="redditrss", age_hours=140)
    )["components"]
    assert "past its peak" not in parts


def test_hot_still_answers_to_the_news_window(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "hot")
    monkeypatch.setattr(config, "TOPIC_MAX_AGE_HOURS", 72.0)
    assert topics.max_age_for("redditrss") == 72.0


def test_top_of_all_time_has_no_window_to_be_late_to(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "top")
    monkeypatch.setattr(config, "TOPIC_REDDIT_WINDOW", "all")
    parts = topics.score_candidate(
        candidate("why the ocean is salty", source="redditrss", age_hours=20_000)
    )["components"]
    assert "past its peak" not in parts


def test_an_unknown_listing_falls_back_instead_of_breaking_the_source(monkeypatch):
    """A typo in an .env should cost a default, not the whole board."""
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "trending")
    monkeypatch.setattr(config, "TOPIC_REDDIT_WINDOW", "fortnight")
    assert topics.reddit_listing() == ("top", "week")


# ---------------------------------------------------------------------------
# Sourceability
# ---------------------------------------------------------------------------

def test_sourceability_runs_from_the_floor_to_the_bonus():
    needed = config.TOPIC_SOURCEABLE_HITS
    assert topics.sourceability_component(0, needed) == topics.SOURCEABLE_FLOOR
    assert topics.sourceability_component(needed, needed) == topics.SOURCEABLE_BONUS
    assert topics.sourceability_component(needed * 9, needed) == topics.SOURCEABLE_BONUS
    middle = topics.sourceability_component(needed - 1, needed)
    assert topics.SOURCEABLE_FLOOR < middle < topics.SOURCEABLE_BONUS


def test_same_word_matches_a_plural_but_not_a_neighbour():
    assert topics._same_word("salt", "salty")
    assert topics._same_word("volcano", "volcanoes")
    assert not topics._same_word("ocean", "octopus")
    assert not topics._same_word("shawver", "shaving")


def test_a_result_is_read_from_its_url_slug_when_it_has_no_title():
    result = stock("serene-ocean-surface-with-gentle-waves")[0]
    assert "ocean" in topics._result_terms(result)
    assert "waves" in topics._result_terms(result)


def test_one_landing_word_is_enough():
    """Measured: requiring two scored 0/8 for five of six known-good topics,
    because a library describes a clip in its own words, rarely in two of
    yours at once. See PROBE_MIN_TERM_HITS."""
    clip = stock("volcano-lava-flowing-into-the-sea")[0]
    assert topics._result_matches(clip, {"lava", "ocean", "water"})

    clip = stock("serene-ocean-surface")[0]
    assert topics._result_matches(clip, {"ocean", "salty"})


def test_a_result_about_nothing_in_the_query_does_not_count():
    studio = stock("a-guy-modeling-in-a-studio")[0]
    assert not topics._result_matches(studio, {"lava", "ocean", "water"})


def test_probe_reports_hits_against_the_sample():
    provider = FakeProvider("pexels", stock(
        "serene-ocean-surface-with-salty-spray",
        "salty-ocean-waves-crashing",
        "a-man-in-a-studio",
    ))
    report = topics.probe_sourceability("why the ocean is salty", [provider],
                                        sample=3, needed=2)
    assert report["checked"] is True
    assert report["hits"] == 2
    assert report["sample"] == 3
    assert report["provider"] == "pexels"


def test_probe_stops_at_the_first_provider_holding_enough():
    first = FakeProvider("pexels", stock("salty-ocean-waves",
                                         "ocean-salt-flats"))
    second = FakeProvider("openverse", stock("ocean-salt"), kinds=("image",))
    topics.probe_sourceability("why the ocean is salty", [first, second],
                               sample=2, needed=2)
    assert first.searches and not second.searches, \
        "the second provider's quota should not be spent once the first answered"


def test_probe_falls_through_to_the_next_provider_when_the_first_is_thin():
    first = FakeProvider("pexels", stock("a-man-in-a-studio"))
    second = FakeProvider("openverse", stock("salty-ocean-waves",
                                             "ocean-salt-flats"),
                          kinds=("image",))
    report = topics.probe_sourceability("why the ocean is salty",
                                        [first, second], sample=2, needed=2)
    assert first.searches and second.searches
    assert report["provider"] == "openverse"
    assert report["hits"] == 2


def test_a_failing_provider_is_a_note_not_an_exception():
    broken = FakeProvider("pexels", error=visuals.ProviderError("429", provider="pexels"))
    working = FakeProvider("openverse", stock("salty-ocean-waves"),
                           kinds=("image",))
    report = topics.probe_sourceability("why the ocean is salty",
                                        [broken, working], sample=1, needed=1)
    assert report["checked"] is True
    assert report["notes"] and "pexels" in report["notes"][0]
    assert report["provider"] == "openverse"


def test_a_probe_where_everything_failed_is_unchecked_not_zero():
    broken = FakeProvider("pexels", error=visuals.ProviderError("500", provider="pexels"))
    report = topics.probe_sourceability("why the ocean is salty", [broken])
    assert report["checked"] is False
    assert report["hits"] == 0


def test_an_unchecked_probe_does_not_touch_the_score():
    item = candidate("why the ocean is salty")
    item.components = {"velocity": 1.0}
    item.score = 1.0
    item.sourceability = {"checked": False, "hits": 0}
    topics.apply_sourceability(item)
    assert item.score == 1.0 and "footage" not in item.components


def test_the_footage_pass_reads_its_ratings_back(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "footage": 3},
        {"index": 1, "footage": 0},
    ]))
    reports, warnings = topics.classify_sourceability(
        [candidate("why the ocean is salty", external_id="a"),
         candidate("State Farm dividend payout", external_id="b")],
        needed=3,
    )
    assert warnings == []
    assert reports[0]["checked"] is True and reports[0]["hits"] == 3
    assert reports[1]["checked"] is True and reports[1]["hits"] == 0
    assert reports[0]["provider"] == "model"


def test_the_footage_pass_asks_once_for_the_whole_shortlist(monkeypatch):
    """Sixteen requests would be the expensive way to ask one question."""
    calls = []

    def capture(*args, **kwargs):
        calls.append(args)
        return json.dumps([{"index": n, "footage": 2} for n in range(5)])

    monkeypatch.setattr(script_gen, "call_gemini", capture)
    reports, _ = topics.classify_sourceability(
        [candidate(f"why the ocean is salty {n}", external_id=f"n{n}")
         for n in range(5)]
    )
    assert len(calls) == 1
    assert len(reports) == 5


def test_a_failed_footage_call_leaves_the_board_unchecked(monkeypatch):
    """Never a guess: an unchecked candidate scores as neither shootable nor
    not, exactly as a probe where every provider failed does."""
    def boom(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503")

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    reports, warnings = topics.classify_sourceability(
        [candidate("why the ocean is salty")]
    )
    assert reports == {}
    assert warnings and "footage pass failed" in warnings[0]


def test_a_candidate_the_footage_pass_skipped_is_left_unchecked(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "footage": 3},
    ]))
    reports, warnings = topics.classify_sourceability(
        [candidate("why the ocean is salty", external_id="a"),
         candidate("why the sky is blue", external_id="b")]
    )
    assert set(reports) == {0}
    assert warnings and "no rating for 1 of 2" in warnings[0]


def test_the_rungs_run_from_the_floor_to_the_bonus():
    assert topics.sourceability_rung(0) == topics.SOURCEABLE_FLOOR
    assert topics.sourceability_rung(topics.SOURCEABILITY_SCALE) == \
        topics.SOURCEABLE_BONUS
    # Out-of-range answers are clamped rather than trusted.
    assert topics.sourceability_rung(99) == topics.SOURCEABLE_BONUS
    assert topics.sourceability_rung(-4) == topics.SOURCEABLE_FLOOR


def test_a_topic_shot_with_stand_ins_is_not_priced_as_a_near_failure():
    """Rung 2 on the straight line was +0.17, which put ten backlog lines -
    'what happens inside a volcano before it erupts' among them - under
    RUN_TOPIC_MIN_SCORE. The cliff belongs between 1 and 0."""
    assert topics.sourceability_rung(2) > topics.sourceability_rung(1)
    assert topics.sourceability_rung(2) > 0.5
    assert topics.sourceability_rung(2) < topics.SOURCEABLE_BONUS
    # The drop from 1 to 0 is the biggest step on the scale.
    steps = [topics.sourceability_rung(n + 1) - topics.sourceability_rung(n)
             for n in range(topics.SOURCEABILITY_SCALE)]
    assert steps[0] == max(steps)


def test_a_priced_report_is_used_over_the_hits_curve():
    """The model pass names what each rung is worth; the keyword probe reads
    its component off the hit count. One function has to serve both."""
    item = candidate("why the ocean is salty")
    item.components = {"velocity": 1.0}
    item.sourceability = {"checked": True, "hits": 3, "component": 0.9}
    topics.apply_sourceability(item)
    assert item.components["footage"] == 0.9

    probed_item = candidate("why the ocean is salty")
    probed_item.components = {"velocity": 1.0}
    probed_item.sourceability = {"checked": True, "hits": 3}
    topics.apply_sourceability(probed_item, needed=3)
    assert probed_item.components["footage"] == topics.SOURCEABLE_BONUS


def test_only_a_zero_rung_reads_as_no_footage_at_all(monkeypatch):
    """A 1 is thin, not absent, and the hard filter must not read it as a
    zero - that is the difference between scoring down and dropping."""
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "footage": 1},
        {"index": 1, "footage": 0},
    ]))
    reports, _ = topics.classify_sourceability(
        [candidate("thin topic", external_id="a"),
         candidate("absent topic", external_id="b")], needed=3,
    )
    assert reports[0]["hits"] == 3, "a 1 must not trip the hard filter"
    assert reports[1]["hits"] == 0
    assert reports[0]["component"] == topics.sourceability_rung(1)


def test_the_footage_ratings_are_read_tolerantly():
    """Same tolerance as the niche pass: a bare array, a fenced block and a
    stringified number are all things the model actually returns."""
    assert topics._parse_sourceability_json("[3, 0]") == {0: 3, 1: 0}
    assert topics._parse_sourceability_json(
        '```json\n[{"index": 0, "footage": "2"}]\n```') == {0: 2}
    assert topics._parse_sourceability_json(
        '[{"index": 0, "rating": 1}]') == {0: 1}


def test_llm_mode_drops_a_named_subject_the_model_says_cannot_be_shot(
        offline, monkeypatch):
    """The hard filter's whole job, and the reason the keyword probe was
    replaced: 'State Farm dividend' honestly retrieves farm footage."""
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "llm")
    # Keyed off the prompt rather than off position. The shortlist is ordered
    # by a score that no longer charges proper nouns to an unrewritten
    # headline, so "State Farm ..." now reaches the probe above "why the ocean
    # is salty" - and a mock that answered by index was quietly rating the
    # wrong one, which is the model saying the named subject *is* shootable
    # and the filter correctly keeping it.
    monkeypatch.setattr(script_gen, "call_gemini", lambda prompt, **kw: json.dumps([
        {"index": index, "footage": 0 if "State Farm" in line else 3}
        for index, line in enumerate(_numbered(prompt))
    ]))
    install_source(monkeypatch, "reddit", [
        candidate("why the ocean is salty", external_id="t1",
                  rate=300, age_hours=2),
        candidate("State Farm dividend payout is largest in company history",
                  external_id="t2", rate=299, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))
    shown = [row["topic"] for row in board["candidates"]]
    assert "why the ocean is salty" in shown
    assert not any("State Farm" in row for row in shown)
    assert any("no footage" in row["reason"] for row in board["rejected"])


def test_llm_mode_keeps_an_unnamed_topic_the_model_says_cannot_be_shot(
        offline, monkeypatch):
    """Both signals or neither - a wrong drop silently removes a topic that
    could have been made, so a zero alone must not decide it."""
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "llm")
    monkeypatch.setattr(script_gen, "call_gemini",
                        lambda *a, **kw: json.dumps([{"index": 0, "footage": 0}]))
    install_source(monkeypatch, "reddit", [
        candidate("why the ocean is salty", external_id="t1",
                  rate=300, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))
    assert len(board["candidates"]) == 1, "a 0 rating alone must not drop it"


def test_llm_mode_does_not_spend_a_stock_request(offline, monkeypatch):
    """The point of the mode: the question is asked of something that can read
    the topic, not of a keyword search."""
    provider = FakeProvider("pexels", stock("salty-ocean-waves"))
    monkeypatch.setattr(topics, "_ready_providers",
                        lambda session, warnings: [provider])
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "llm")
    monkeypatch.setattr(script_gen, "call_gemini",
                        lambda *a, **kw: json.dumps([{"index": 0, "footage": 3}]))
    install_source(monkeypatch, "reddit", [
        candidate("why the ocean is salty", external_id="t1",
                  rate=300, age_hours=2),
    ])
    topics.find_topics("", sources=("reddit",))
    assert provider.searches == []


def test_probe_query_drops_possessives():
    assert "'" not in topics.probe_query("State Farm's dividend payout")


def test_probe_query_drops_words_no_library_has_footage_of():
    """'how deep the ocean actually goes' probed as 'deep ocean actually' and
    found nothing - "actually" outscores "ocean" on length alone."""
    query = topics.probe_query("how deep the ocean actually goes")
    assert "actually" not in query
    assert "ocean" in query and "deep" in query


def test_dropping_a_dead_word_promotes_a_real_one():
    """The query stays PROBE_QUERY_WORDS long rather than losing a slot."""
    query = topics.probe_query("what happens to a raindrop before it lands")
    assert "happens" not in query and "before" not in query
    assert "raindrop" in query and "lands" in query


def test_a_topic_of_nothing_but_dead_words_still_gets_a_query():
    assert topics.probe_query("what actually happens") != ""


def test_reddit_explains_a_403_instead_of_quoting_its_block_page():
    """The refusal arrives as several kilobytes of stylesheet; the status is
    the whole of the information in it."""
    raw = topics.SourceError("reddit rejected the API key [403]: <body class="
                             "theme-beta><div><style>.theme-light,:root{--rem",
                             source="reddit")
    explained = str(topics.RedditSource._explain(raw, authenticated=False))
    assert "REDDIT_CLIENT_ID" in explained
    assert "prefs/apps" in explained
    assert "<style>" not in explained


def test_reddit_blames_the_credentials_when_there_were_some():
    raw = topics.SourceError("reddit rejected the API key [401]: bad",
                             source="reddit")
    explained = str(topics.RedditSource._explain(raw, authenticated=True))
    assert "REDDIT_CLIENT_ID" in explained and "rate limited" in explained


def test_an_unrelated_reddit_failure_is_passed_through_untouched():
    raw = topics.SourceError("could not reach reddit: ConnectionError",
                             source="reddit")
    assert topics.RedditSource._explain(raw, authenticated=False) is raw


# ---------------------------------------------------------------------------
# How long it should be
# ---------------------------------------------------------------------------

def probed(topic, hits, **kwargs):
    """A candidate carrying a finished footage probe."""
    made = candidate(topic, **kwargs)
    made.sourceability = {"checked": True, "hits": hits, "sample": 8,
                          "query": topic, "provider": "fake"}
    return made


def rated(topic, rating, **kwargs):
    """A candidate carrying a finished model footage rating."""
    made = candidate(topic, **kwargs)
    made.sourceability = {"checked": True, "rating": rating,
                          "hits": 0 if rating == 0 else 3, "sample": 3,
                          "component": topics.sourceability_rung(rating),
                          "query": topic, "provider": "model"}
    return made


def test_the_length_reads_the_rung_rather_than_the_binary_hits():
    """Under the model pass `hits` is only ever zero or the threshold, so a
    topic rated filler would otherwise be given the seconds of one the library
    plainly holds."""
    rich = topics.suggest_duration(rated("why the ocean is salty", 3))
    stands_in = topics.suggest_duration(rated("why the ocean is salty", 2))
    thin = topics.suggest_duration(rated("why the ocean is salty", 1))
    assert rich["components"]["footage"] == topics.DURATION_FOOTAGE_RICH
    assert thin["components"]["footage"] < stands_in["components"]["footage"]
    assert stands_in["components"]["footage"] < rich["components"]["footage"]


def test_a_probe_without_a_rung_still_reads_its_hits():
    """The keyword probe is still selectable and still scores the old way."""
    rich = topics.suggest_duration(probed("why the ocean is salty", 8))
    assert rich["components"]["footage"] == topics.DURATION_FOOTAGE_RICH


def test_a_mechanism_is_given_longer_than_a_fixture():
    mechanism = topics.suggest_duration(candidate("why the ocean is salty"))
    fixture = topics.suggest_duration(candidate("Braves at Twins on Sunday"))
    assert mechanism["components"]["explainer"] == topics.DURATION_EXPLAINER_BONUS
    assert fixture["components"]["a named subject"] == topics.DURATION_FIXTURE_PENALTY
    assert mechanism["seconds"] > fixture["seconds"]


def test_a_topic_with_nothing_to_say_either_way_takes_the_default():
    result = topics.suggest_duration(candidate("the deep sea has its own light"))
    assert result["seconds"] == config.SCRIPT_DEFAULT_SECONDS
    assert set(result["components"]) == {"base"}


def test_two_names_do_not_shorten_a_topic_twice():
    """Unlike the score's own penalty this is not cumulative - two names do not
    make a topic half as long again as one."""
    one = topics.suggest_duration(candidate("the deal Diggs signed"))
    two = topics.suggest_duration(candidate("the deal Diggs signed with Seattle"))
    assert (one["components"]["a named subject"]
            == two["components"]["a named subject"]
            == topics.DURATION_FIXTURE_PENALTY)


def test_thin_footage_shortens_the_reel_and_plenty_does_not():
    needed = config.TOPIC_SOURCEABLE_HITS
    thin = topics.suggest_duration(probed("why the ocean is salty", 0))
    rich = topics.suggest_duration(probed("why the ocean is salty", needed * 3))
    assert thin["components"]["footage"] == topics.DURATION_FOOTAGE_THIN
    assert rich["components"]["footage"] == topics.DURATION_FOOTAGE_RICH
    assert thin["seconds"] < rich["seconds"]


def test_thin_footage_shortens_rather_than_vetoes():
    """The probe is a weak negative - a miss may only be a synonym the library
    knew and this does not - so it may not drive a reel to the floor on its
    own. The conclusive case has its own hard filter in find_topics."""
    thin = topics.suggest_duration(probed("why the ocean is salty", 0))
    assert thin["seconds"] > config.SCRIPT_MIN_SECONDS


def test_an_unchecked_probe_does_not_touch_the_length():
    unchecked = candidate("why the ocean is salty")
    unchecked.sourceability = {"checked": False, "hits": 0}
    assert "footage" not in topics.suggest_duration(unchecked)["components"]


def test_an_argument_earns_a_middle_to_put_it_in():
    quiet = topics.suggest_duration(candidate("why the ocean is salty", reach=600))
    loud = topics.suggest_duration(
        candidate("why the ocean is salty", reach=600, comments=200)
    )
    assert loud["components"]["discussion"] > 0
    assert loud["seconds"] >= quiet["seconds"]


def test_a_suggestion_lands_on_the_step():
    step = config.SCRIPT_DURATION_STEP_SECONDS
    for topic in ("why the ocean is salty", "Braves at Twins on Sunday",
                  "the deep sea has its own light", "how mangroves hold a coast"):
        assert topics.suggest_duration(candidate(topic))["seconds"] % step == 0


def test_a_suggestion_stays_inside_the_configured_range(monkeypatch):
    monkeypatch.setattr(config, "SCRIPT_MIN_SECONDS", 25)
    monkeypatch.setattr(config, "SCRIPT_MAX_SECONDS", 35)
    floor = topics.suggest_duration(probed("Braves at Twins on Sunday", 0))
    ceiling = topics.suggest_duration(
        probed("why the ocean is salty", 9, reach=600, comments=200)
    )
    assert floor["seconds"] == 25
    assert ceiling["seconds"] == 35


def test_every_part_of_a_length_is_reported_so_it_can_be_read_back():
    result = topics.suggest_duration(probed("why the ocean is salty", 1,
                                            reach=600, comments=200))
    assert set(result["components"]) == {"base", "explainer", "discussion",
                                         "footage"}
    assert result["components"]["base"] == config.SCRIPT_DEFAULT_SECONDS


def test_the_length_is_decided_after_the_phrase_pass(offline, monkeypatch):
    """The rewrite is what turns a fact into a mechanism, and the length turns
    on which of the two it is - so reading the headline would score the wrong
    text. This is a fact going in and an explainer coming out."""
    raw = "TIL that the ocean is salty because rivers carry minerals into it"
    install_source(monkeypatch, "reddit",
                   [candidate(raw, rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    monkeypatch.setattr(topics, "polish_phrases", _rewrite_to("why the ocean is salty"))

    board = topics.find_topics(sourceability="off")
    only = board["candidates"][0]
    assert only["topic"] == "why the ocean is salty"
    assert "explainer" in only["duration_components"]


def _rewrite_to(phrase):
    def polish(candidates, *, mode=None):
        for one in candidates:
            one.topic = phrase
            # The real pass clears this for everything it is handed, and
            # the re-score behind it reads the flag. A stub that left it
            # set would have the board scoring a settled wording as
            # though it were still a headline.
            one.provisional = False
        return []
    return polish


def test_the_board_carries_the_length_it_decided(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    board = topics.find_topics(sourceability="off", phrases="heuristic")
    only = board["candidates"][0]
    assert only["duration"] == topics.suggest_duration(
        candidate("why the ocean is salty", rate=300, age_hours=2)
    )["seconds"]
    assert only["duration_components"]["base"] == config.SCRIPT_DEFAULT_SECONDS


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def reddit_payload(*posts):
    return {"data": {"children": [{"data": post} for post in posts]}}


def test_reddit_parses_a_listing_into_candidates(monkeypatch):
    monkeypatch.setattr(config, "REDDIT_CLIENT_ID", None)
    session = FakeSession({"reddit.com": FakeResponse(payload=reddit_payload(
        {"id": "abc", "title": "TIL that the ocean is salty", "score": 1200,
         "num_comments": 80, "created_utc": 1_700_000_000,
         "permalink": "/r/todayilearned/comments/abc/", "subreddit": "todayilearned"},
    ))})
    source = topics.RedditSource(session, subreddits=("todayilearned",))
    found = source.fetch("ocean", 25)

    assert len(found) == 1
    assert found[0].topic == "the ocean is salty"
    assert found[0].title == "TIL that the ocean is salty"
    assert found[0].context == "r/todayilearned top"
    assert found[0].url.endswith("/r/todayilearned/comments/abc/")
    assert found[0].rate == pytest.approx(1200 / found[0].age_hours)


def test_reddit_skips_stickies_and_nsfw(monkeypatch):
    monkeypatch.setattr(config, "REDDIT_CLIENT_ID", None)
    session = FakeSession({"reddit.com": FakeResponse(payload=reddit_payload(
        {"id": "1", "title": "Monthly megathread", "stickied": True},
        {"id": "2", "title": "Something explicit", "over_18": True},
        {"id": "3", "title": "why the ocean is salty", "score": 10,
         "created_utc": 1_700_000_000},
    ))})
    found = topics.RedditSource(session, subreddits=("x",)).fetch("", 25)
    assert [c.external_id for c in found] == ["3"]


def test_a_subreddit_override_makes_reddit_ready_without_env(monkeypatch):
    """--subreddit has to reach the readiness check, or the source is skipped
    before the override is ever applied."""
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ())
    assert topics.RedditSource.readiness()[0] is False
    assert topics.RedditSource.readiness(subreddits=("askscience",))[0] is True


def test_reddit_without_subreddits_says_so(monkeypatch):
    # Pinned, or this passes only on a machine whose .env has no subreddits.
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ())
    with pytest.raises(topics.SourceError, match="TOPIC_SUBREDDITS"):
        topics.RedditSource(FakeSession(), subreddits=()).fetch("", 25)


def test_reddit_falls_back_to_anonymous_when_the_token_call_fails(monkeypatch):
    monkeypatch.setattr(config, "REDDIT_CLIENT_ID", "id")
    monkeypatch.setattr(config, "REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setattr(config, "get_env",
                        lambda name, default=None, **kw: {
                            "REDDIT_CLIENT_ID": "id",
                            "REDDIT_CLIENT_SECRET": "secret",
                        }.get(name, default))
    session = FakeSession(
        {"www.reddit.com": FakeResponse(payload=reddit_payload(
            {"id": "9", "title": "why the ocean is salty", "score": 5,
             "created_utc": 1_700_000_000}))},
        post=FakeResponse(500, body=b"nope"),
    )
    found = topics.RedditSource(session, subreddits=("x",)).fetch("", 5)
    assert session.posts, "the token was attempted"
    assert found and "www.reddit.com" in session.calls[0]["url"], \
        "a failed token falls back to the public endpoint rather than failing"


def test_youtube_ranks_on_views_per_day(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda name, default=None, **kw:
                        "key" if name == "YOUTUBE_API_KEY" else default)
    published = "2026-08-09T00:00:00Z"      # ten days before the frozen 'now'
    session = FakeSession({
        "youtube/v3/search": FakeResponse(payload={"items": [
            {"id": {"videoId": "vid1"},
             "snippet": {"title": "How volcanoes form", "publishedAt": published,
                         "channelTitle": "Science"}},
        ]}),
        "youtube/v3/videos": FakeResponse(payload={"items": [
            {"id": "vid1", "statistics": {"viewCount": "500000"}},
        ]}),
    })
    found = topics.YouTubeSource(session, keywords=("volcano",)).fetch("", 25)

    assert len(found) == 1
    assert found[0].reach == 500_000
    assert found[0].rate_unit == "views/day"
    # 500k over roughly ten days, whatever the clock says when this runs.
    assert found[0].rate == pytest.approx(500_000 / (found[0].age_hours / 24), rel=0.01)
    assert found[0].url == "https://www.youtube.com/watch?v=vid1"


def test_youtube_asks_only_for_the_recent_window(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda name, default=None, **kw:
                        "key" if name == "YOUTUBE_API_KEY" else default)
    session = FakeSession({"youtube/v3/search": FakeResponse(payload={"items": []})})
    topics.YouTubeSource(session, keywords=("volcano",), window_days=30).fetch("", 5)
    params = session.calls[0]["params"]
    assert params["order"] == "viewCount"
    assert params["publishedAfter"].endswith("Z")


def test_youtube_caps_how_many_searches_one_scout_spends(monkeypatch):
    monkeypatch.setattr(config, "get_env",
                        lambda name, default=None, **kw:
                        "key" if name == "YOUTUBE_API_KEY" else default)
    session = FakeSession({"youtube/v3/search": FakeResponse(payload={"items": []})})
    topics.YouTubeSource(
        session, keywords=tuple(f"k{n}" for n in range(20))
    ).fetch("", 5)
    assert len(session.calls) == topics.MAX_YOUTUBE_SEARCHES


TRENDS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:ht="https://trends.google.com/trending/rss">
  <channel>
    <item>
      <title>bioluminescence</title>
      <pubDate>Wed, 19 Aug 2026 06:00:00 +0000</pubDate>
      <ht:approx_traffic>20,000+</ht:approx_traffic>
      <ht:news_item>
        <ht:news_item_title>Why the sea glows off the coast of Sorana Cîrstea</ht:news_item_title>
        <ht:news_item_url>https://example.com/a</ht:news_item_url>
      </ht:news_item>
    </item>
  </channel>
</rss>
"""


def test_trends_parses_the_feed_and_prefers_the_headline():
    session = FakeSession({"trends.google.com":
                           FakeResponse(body=TRENDS_RSS.encode("utf-8"))})
    found = topics.TrendsSource(session, geo="US").fetch("", 25)

    assert len(found) == 1
    assert found[0].reach == 20_000
    assert found[0].title == "bioluminescence"
    # The two-word search term does not say what the story is; the headline does.
    assert found[0].topic.startswith("Why the sea glows")


def test_trends_decodes_utf8_rather_than_guessing_latin1():
    """requests guesses latin-1 for an XML body with no charset header, which
    turns every accented name in the feed into mojibake."""
    session = FakeSession({"trends.google.com":
                           FakeResponse(body=TRENDS_RSS.encode("utf-8"))})
    found = topics.TrendsSource(session, geo="US").fetch("", 25)
    assert "Cîrstea" in found[0].topic
    assert "�" not in found[0].topic


def test_trends_survives_a_namespace_change():
    """Matched on local tag names, because Google has moved the URI before."""
    moved = TRENDS_RSS.replace("https://trends.google.com/trending/rss",
                               "https://trends.google.com/trends/rss/v2")
    session = FakeSession({"trends.google.com":
                           FakeResponse(body=moved.encode("utf-8"))})
    found = topics.TrendsSource(session, geo="US").fetch("", 25)
    assert found and found[0].reach == 20_000


@pytest.mark.parametrize("raw, expected", [
    ("20,000+", 20_000), ("2K+", 2_000), ("1.5M+", 1_500_000),
    ("", 0), ("lots", 0),
])
def test_approx_traffic_reads_googles_shorthand(raw, expected):
    assert topics._approx_traffic(raw) == expected


def test_a_malformed_feed_is_a_source_error_not_a_crash():
    session = FakeSession({"trends.google.com": FakeResponse(body=b"<rss")})
    with pytest.raises(topics.SourceError, match="valid XML"):
        topics.TrendsSource(session).fetch("", 5)


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def test_the_same_story_from_two_sources_takes_one_slot():
    both = [candidate("why the ocean is salty", source="reddit", reach=100),
            candidate("why the ocean is salty", source="trends",
                      external_id="t1", reach=9000)]
    kept, dropped = topics._dedupe(both)
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].source == "trends", "the wider-reaching version survives"
    assert any("reddit" in note for note in kept[0].notes)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_a_board_marks_its_candidates_seen(ledger):
    item = candidate("why the ocean is salty")
    assert not ledger.seen_recently(item.key)
    ledger.record_board([item])
    assert ledger.seen_recently(item.key)
    assert not ledger.was_used(item.key)


def test_a_taken_topic_is_used_not_merely_seen(ledger):
    item = candidate("why the ocean is salty")
    ledger.record_board([item])
    ledger.mark_used(item.key, item.topic)
    assert ledger.was_used(item.key)


def test_the_seen_penalty_expires_after_the_history_window(ledger):
    item = candidate("why the ocean is salty")
    ledger.record_board([item])
    for _ in range(5):
        ledger.record_board([candidate("something else", external_id="other")])
    assert not ledger.seen_recently(item.key, within=3)
    assert ledger.seen_recently(item.key, within=50)


def test_the_ledger_round_trips_through_disk(tmp_path):
    path = tmp_path / "usage.json"
    first = topics.TopicLedger(path)
    item = candidate("why the ocean is salty")
    first.record_board([item])
    first.mark_used(item.key, item.topic)
    assert first.save() == []

    second = topics.TopicLedger(path)
    assert second.was_used(item.key)
    assert second.board_index == 1


def test_pruning_forgets_the_passed_over_but_never_what_was_made(tmp_path):
    ledger = topics.TopicLedger(tmp_path / "usage.json")
    made = candidate("why the ocean is salty", external_id="made")
    passed = candidate("how volcanoes form", external_id="passed")
    ledger.record_board([made, passed])
    ledger.mark_used(made.key, made.topic)

    ledger._data["boards"] = topics.LEDGER_KEEP_BOARDS + 50
    ledger._prune()

    assert made.key in ledger._data["topics"]
    assert passed.key not in ledger._data["topics"]


def test_take_topic_persists(tmp_path):
    ledger = topics.TopicLedger(tmp_path / "usage.json")
    assert topics.take_topic("reddit:abc", "why the ocean is salty",
                             ledger=ledger) == []
    assert topics.TopicLedger(tmp_path / "usage.json").was_used("reddit:abc")


def test_a_refused_script_is_remembered_against_its_topic(ledger):
    """It was not, and the same topic bought the same Pro call five times."""
    item = candidate("what the sound of the deep ocean actually is")
    ledger.record_board([item])

    ledger.mark_refused(item.key, item.topic)

    assert ledger.refusals(item.key) == 1
    assert not ledger.was_used(item.key), "a refusal is not a reel"


def test_a_topic_refused_to_the_limit_is_dropped_not_merely_priced(ledger,
                                                                   monkeypatch):
    monkeypatch.setattr(config, "TOPIC_MAX_REFUSALS", 2)
    item = candidate("what the sound of the deep ocean actually is")

    ledger.mark_refused(item.key, item.topic)
    assert not ledger.was_refused_out(item.key)

    ledger.mark_refused(item.key, item.topic)
    assert ledger.was_refused_out(item.key)


def test_refusals_can_be_switched_off_entirely(ledger, monkeypatch):
    monkeypatch.setattr(config, "TOPIC_MAX_REFUSALS", 0)
    item = candidate("why the ocean is salty")
    ledger.mark_refused(item.key, item.topic)

    assert ledger.was_refused_out(item.key) is False


def test_one_refusal_costs_a_retry_and_five_do_not_cost_five(ledger):
    """Charged once however many times, so it cannot stack into a deadlock.

    Board #41 deadlocked on two penalties priced independently. What ends a
    topic here is the count, which is a filter and cannot reach the score.
    """
    item = candidate("why the ocean is salty")

    once = topics.score_candidate(item, refused=1)["components"]
    many = topics.score_candidate(item, refused=5)["components"]

    assert sum(once.values()) == sum(many.values())
    assert topics.REFUSED_PENALTY in once.values()


def test_a_refusal_survives_the_pruning_that_forgets_a_passed_over_topic(tmp_path):
    ledger = topics.TopicLedger(tmp_path / "usage.json")
    refused = candidate("why the ocean is salty", external_id="refused")
    passed = candidate("how volcanoes form", external_id="passed")
    ledger.record_board([refused, passed])
    ledger.mark_refused(refused.key, refused.topic)

    ledger._data["boards"] = topics.LEDGER_KEEP_BOARDS + 50
    ledger._prune()

    # Forgetting it is buying the refusal a second time.
    assert refused.key in ledger._data["topics"]
    assert passed.key not in ledger._data["topics"]


def test_refuse_topic_persists(tmp_path):
    ledger = topics.TopicLedger(tmp_path / "usage.json")
    assert topics.refuse_topic("evergreen:deep-ocean", "the deep ocean",
                               ledger=ledger) == []
    assert topics.TopicLedger(tmp_path / "usage.json").refusals(
        "evergreen:deep-ocean") == 1


# ---------------------------------------------------------------------------
# The niche pass
# ---------------------------------------------------------------------------

def test_a_verdict_keeps_a_topic_that_shares_no_word_with_the_niche():
    """The miss that made the pass necessary. "caves beneath the Moon" is
    squarely earth science and has not one word in common with it."""
    terms = topics.niche_vocabulary("ocean, earth science and nature")
    topic = "NASA has found evidence of caves beneath the Moon"

    without = topics.score_candidate(candidate(topic), niche_terms=terms)
    with_verdict = topics.score_candidate(candidate(topic), niche_terms=terms,
                                          on_niche=True)

    assert without["components"]["off the niche"] == topics.NICHE_MISS_PENALTY
    assert with_verdict["components"]["niche match"] == (
        topics.NICHE_MATCH_WEIGHT * topics.NICHE_VERDICT_SHARE
    )


def test_a_verdict_also_overrules_a_word_that_matched_by_accident():
    """The error in the other direction, and the same error underneath."""
    terms = topics.niche_vocabulary("ocean, earth science and nature")
    parts = topics.score_candidate(
        candidate("2026 Men's FIH Hockey World Cup"),
        niche_terms=terms | {"world"}, on_niche=False,
    )["components"]
    assert parts["off the niche"] == topics.NICHE_MISS_PENALTY


def test_shared_words_still_decide_how_squarely_it_sits():
    """The verdict is membership; the words are strength. A topic the model
    kept and the description names outright is worth more than one it only
    kept."""
    terms = topics.niche_vocabulary("ocean science")
    squarely = topics.score_candidate(
        candidate("why ocean science keeps missing the deep sea"),
        niche_terms=terms, on_niche=True,
    )["components"]["niche match"]
    loosely = topics.score_candidate(
        candidate("what lives under the ice in Antarctica"),
        niche_terms=terms, on_niche=True,
    )["components"]["niche match"]
    assert squarely > loosely


def test_the_niche_pass_reads_its_verdicts_back(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "on_topic": True},
        {"index": 1, "on_topic": False},
    ]))
    verdicts, warnings = topics.classify_niche(
        [candidate("why the ocean is salty", external_id="a"),
         candidate("a new pasta shape", external_id="b")],
        "ocean science", mode="llm",
    )
    assert verdicts == {0: True, 1: False}
    assert warnings == []


def test_a_failed_niche_call_falls_back_to_words_rather_than_a_guess(monkeypatch):
    def boom(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503")

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    verdicts, warnings = topics.classify_niche(
        [candidate("why the ocean is salty")], "ocean science", mode="llm"
    )
    assert verdicts == {}
    assert warnings and "niche pass failed" in warnings[0]


def test_a_candidate_the_model_skipped_falls_back_on_its_own(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "on_topic": True},
    ]))
    verdicts, warnings = topics.classify_niche(
        [candidate("why the ocean is salty", external_id="a"),
         candidate("a new pasta shape", external_id="b")],
        "ocean science", mode="llm",
    )
    assert verdicts == {0: True}
    assert warnings and "no verdict for 1 of 2" in warnings[0]


def test_the_niche_pass_is_skipped_entirely_in_terms_mode(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("no Gemini call should be made")

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    assert topics.classify_niche([candidate("x")], "ocean", mode="terms") == ({}, [])


def test_the_niche_pass_is_not_shown_the_scores(monkeypatch):
    """It answers a closed question about subject matter. It does not rank,
    and it cannot rank what it was never given."""
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return json.dumps([{"index": 0, "on_topic": True}])

    monkeypatch.setattr(script_gen, "call_gemini", capture)
    scored = candidate("why the ocean is salty")
    scored.score = 4.2
    scored.components = {"velocity": 1.8}
    topics.classify_niche([scored], "ocean science", mode="llm")

    assert "ocean science" in seen["prompt"]
    assert "why the ocean is salty" in seen["prompt"]
    assert "4.2" not in seen["prompt"] and "velocity" not in seen["prompt"]


def test_niche_json_is_read_tolerantly():
    assert topics._parse_niche_json(
        '```json\n[{"index": 1, "on_topic": "yes"}]\n```'
    ) == {1: True}


def test_the_board_notes_a_candidate_kept_on_judgement_alone(offline, monkeypatch):
    monkeypatch.setattr(config, "TOPIC_NICHE_MATCH", "llm")
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "on_topic": True},
    ]))
    install_source(monkeypatch, "reddit", [
        candidate("what lives under the ice in Antarctica", external_id="a",
                  rate=200, age_hours=2),
    ])
    board = topics.find_topics("ocean science", sources=("reddit",))
    only = board["candidates"][0]

    assert "niche match" in only["components"]
    assert any("shares no word" in note for note in only["notes"])
    assert board["niche_verdicts"] == 1


# ---------------------------------------------------------------------------
# The phrase pass
# ---------------------------------------------------------------------------

def test_phrases_are_rewritten_in_place(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "topic": "why the ocean is salty"},
    ]))
    board = [candidate("the ocean is salty because of rain",
                       title="TIL the ocean is salty because of rain")]
    assert topics.polish_phrases(board, mode="llm") == []
    assert board[0].topic == "why the ocean is salty"


def test_a_rewrite_that_drifted_off_its_original_is_discarded(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **kw: json.dumps([
        {"index": 0, "topic": "the best pasta recipes in Rome"},
    ]))
    board = [candidate("the ocean is salty because of rain",
                       title="TIL the ocean is salty because of rain")]
    warnings = topics.polish_phrases(board, mode="llm")

    assert board[0].topic == "the ocean is salty because of rain"
    assert warnings and "drifted" in warnings[0]
    assert any("shares too little" in note for note in board[0].notes)


def test_a_failed_phrase_call_leaves_the_headlines_usable(monkeypatch):
    def boom(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503")

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    board = [candidate("why the ocean is salty")]
    warnings = topics.polish_phrases(board, mode="llm")

    assert board[0].topic == "why the ocean is salty"
    assert warnings and "phrase pass failed" in warnings[0]


def test_the_phrase_pass_is_skipped_entirely_in_heuristic_mode(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("no Gemini call should be made")

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    assert topics.polish_phrases([candidate("x")], mode="heuristic") == []


# ---------------------------------------------------------------------------
# Wording is settled before anything reads it
# ---------------------------------------------------------------------------

def _fake_gemini(rewrites=None, footage=3):
    """One `call_gemini` standing in for both model passes in step 0.

    Dispatches on the prompt, because the phrase pass and the footage pass go
    through the same function and a board test now runs both.
    """
    def call(prompt, **kwargs):
        lines = _numbered(prompt)
        if "Rewrite each headline" in prompt:
            return json.dumps([
                {"index": index, "topic": (rewrites or {}).get(line, line)}
                for index, line in enumerate(lines)
            ])
        return json.dumps([
            {"index": index, "footage": footage} for index, _ in enumerate(lines)
        ])
    return call


def test_an_unrewritten_headline_is_not_charged_for_its_shape():
    """The three components the phrase pass exists to fix. Charging them to a
    headline marks a candidate down for a shape it is about to lose."""
    raw = ("TIL that Naked mole-rats can survive 18 minutes without oxygen, "
           "a trick otherwise only seen in plants")
    one = candidate(raw, rate=300, age_hours=2)

    one.provisional = True
    charged = topics.score_candidate(one)["components"]
    assert "explainer shape" not in charged
    assert "too long" not in charged
    assert "proper nouns" not in charged

    one.provisional = False
    assert {"too long", "proper nouns"} & set(
        topics.score_candidate(one)["components"])


def test_content_components_are_charged_either_way():
    """The line the flag draws is form against content: the phrase pass is told
    to keep the subject, so what a topic is *about* survives the rewrite."""
    one = candidate("the largest waterfall on earth is underwater",
                    rate=300, age_hours=2)
    one.provisional = True
    provisional = topics.score_candidate(one, niche_terms={"ocean", "earth"})
    one.provisional = False
    final = topics.score_candidate(one, niche_terms={"ocean", "earth"})

    for part in ("velocity", "niche match", "curiosity (extreme)"):
        assert provisional["components"][part] == final["components"][part]


def test_the_backlog_is_final_wording_and_a_feed_is_not():
    """Set on the source class, so it is one fact in one place rather than a
    flag repeated at six Candidate(...) sites."""
    assert topics.EvergreenSource.wording_is_final is True
    assert topics.RedditRssSource.wording_is_final is False


def test_the_footage_probe_is_asked_about_the_rewritten_topic(
        offline, monkeypatch):
    """The probe searches for what the topic says. Asked before the rewrite it
    handed the stock libraries a whole reddit headline - a sentence no library
    has ever been asked for."""
    raw = ("Naked mole-rats can survive up to 18 minutes without oxygen by "
           "switching their metabolism to run on fructose instead of glucose")
    phrase = "how naked mole-rats survive without oxygen"
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "llm")
    monkeypatch.setattr(script_gen, "call_gemini", _fake_gemini({raw: phrase}))
    install_source(monkeypatch, "reddit",
                   [candidate(raw, external_id="t1", rate=300, age_hours=2)])

    board = topics.find_topics("", sources=("reddit",), phrases="llm")
    only = board["candidates"][0]
    assert only["topic"] == phrase
    assert only["sourceability"]["query"] == phrase


def test_a_rewritten_headline_is_scored_on_the_shape_it_ships_with(
        offline, monkeypatch):
    """Board #33: the one live topic on the board sat third behind two backlog
    lines, carrying `too long` and no explainer bonus - both read off a 28-word
    headline it was never going to ship as."""
    raw = ("Naked mole-rats can survive up to 18 minutes without oxygen by "
           "switching their metabolism to run on fructose instead of glucose")
    phrase = "how naked mole-rats survive without oxygen"
    monkeypatch.setattr(script_gen, "call_gemini", _fake_gemini({raw: phrase}))
    install_source(monkeypatch, "reddit",
                   [candidate(raw, external_id="t1", rate=300, age_hours=2)])

    board = topics.find_topics("", sources=("reddit",), sourceability="off",
                               phrases="llm")
    only = board["candidates"][0]
    assert only["components"]["explainer shape"] == topics.EXPLAINER_BONUS
    assert "too long" not in only["components"]


def test_the_phrase_pass_settles_the_wording_on_every_path(monkeypatch):
    """The flag means a rewrite is still coming, so every way of not getting one
    has to clear it - or the score never reads the wording that ships."""
    def boom(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503")

    off = candidate("why the ocean is salty")
    off.provisional = True
    topics.polish_phrases([off], mode="heuristic")
    assert off.provisional is False

    monkeypatch.setattr(script_gen, "call_gemini", boom)
    failed = candidate("why the ocean is salty")
    failed.provisional = True
    topics.polish_phrases([failed], mode="llm")
    assert failed.provisional is False

    monkeypatch.setattr(script_gen, "call_gemini",
                        lambda *a, **kw: json.dumps([{"index": 0, "topic": ""}]))
    empty = candidate("why the ocean is salty")
    empty.provisional = True
    topics.polish_phrases([empty], mode="llm")
    assert empty.provisional is False


def test_the_phrase_budget_grows_with_the_batch(monkeypatch):
    """It runs on the probe shortlist now, not on the eight-row board the fixed
    1024 was sized for. A truncated array is silently kept as headlines."""
    seen = {}

    def capture(prompt, **kwargs):
        seen.update(kwargs)
        return json.dumps([])

    monkeypatch.setattr(script_gen, "call_gemini", capture)
    topics.polish_phrases([candidate(f"topic number {n}") for n in range(40)],
                          mode="llm")
    assert seen["max_output_tokens"] >= 40 * topics.PHRASE_TOKENS_PER_PHRASE


def test_a_winner_that_missed_the_shortlist_is_still_rewritten(
        offline, monkeypatch):
    """The sorts after the shortlist read every candidate, so one that was never
    on it can still finish inside `wanted`. Without the backstop it would ship
    to step 2 as a raw headline."""
    raw = "TIL that the ocean is salty because rivers carry minerals into it"
    monkeypatch.setattr(config, "TOPIC_PROBE_LIMIT", 1)
    monkeypatch.setattr(script_gen, "call_gemini",
                        _fake_gemini({raw: "why the ocean is salty"}))
    install_source(monkeypatch, "reddit", [
        candidate("how a canyon is carved", external_id="t1",
                  rate=900, age_hours=2),
        candidate(raw, external_id="t2", rate=300, age_hours=2),
    ])

    board = topics.find_topics("", sources=("reddit",), sourceability="off",
                               phrases="llm")
    shown = [row["topic"] for row in board["candidates"]]
    assert "why the ocean is salty" in shown
    assert not any(row.startswith("TIL") for row in shown)


def test_phrase_json_is_read_tolerantly():
    parsed = topics._parse_phrase_json(
        '```json\n[{"index": 1, "topic": "why the ocean is salty."}]\n```', 2
    )
    assert parsed == {1: "why the ocean is salty"}


def test_phrase_json_ignores_an_index_off_the_end():
    assert topics._parse_phrase_json('[{"index": 7, "topic": "x y z"}]', 2) == {}


# ---------------------------------------------------------------------------
# find_topics
# ---------------------------------------------------------------------------

@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network, no Gemini, no real ledger."""
    monkeypatch.setattr(topics, "_new_session", lambda: FakeSession())
    monkeypatch.setattr(topics, "_ready_providers", lambda session, warnings: [])
    monkeypatch.setattr(config, "TOPIC_USAGE_FILE", tmp_path / "usage.json")
    # Every path the scout writes to belongs here. `main` saves the board it
    # printed so `--take` can act on it, and a test that called main without
    # this would overwrite the real one - which it did, and which then handed
    # a fixture topic to the next real --take.
    monkeypatch.setattr(config, "TOPIC_BOARD_FILE", tmp_path / "last_board.json")
    monkeypatch.setattr(config, "TOPIC_PHRASE_MODE", "heuristic")
    # The niche pass is the other Gemini call in step 0, and a suite that left
    # it on would spend a token per board and take a network round trip per
    # test. Tests that want it stub `call_gemini` and switch this back.
    monkeypatch.setattr(config, "TOPIC_NICHE_MATCH", "terms")
    # And the footage pass is the third. It defaults to `llm`, so leaving it
    # alone would send every board test to Gemini and stop the keyword probe -
    # which most of these tests are about - from running at all. Tests for the
    # llm mode stub `call_gemini` and switch this back.
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "probe")
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("todayilearned",))
    return tmp_path


def install_source(monkeypatch, name, candidates=None, error=None):
    """Replace one entry in the source registry with a canned one."""

    class Canned(topics.Source):
        pass

    Canned.name = name
    Canned.key_env = ()
    Canned.rate_unit = "per hour"

    def fetch(self, niche, limit):
        if error:
            raise error
        return [c for c in (candidates or [])]

    Canned.fetch = fetch
    monkeypatch.setitem(topics._SOURCES, name, Canned)


def test_find_topics_ranks_and_reports(offline, monkeypatch):
    install_source(monkeypatch, "reddit", [
        candidate("why the ocean is salty", reach=900, rate=300, age_hours=3),
        candidate("Braves at Twins on Sunday", external_id="b1",
                  reach=200, rate=20, age_hours=3),
    ])
    board = topics.find_topics("ocean science", sources=("reddit",))

    assert [c["topic"] for c in board["candidates"]][0] == "why the ocean is salty"
    assert board["stats"]["shown"] == 2
    assert board["candidates"][0]["components"]["velocity"] > 0
    assert board["sources"][0]["fetched"] == 2


# ---------------------------------------------------------------------------
# The niche gate
# ---------------------------------------------------------------------------

def _two_candidates():
    """One on an ocean niche and slow, one off it and moving fast.

    Built fresh per call because scoring writes back onto the candidate, and
    a board is not allowed to see one the previous board already marked up.
    """
    return [
        candidate("the ocean is salty", external_id="a", rate=1, age_hours=3),
        candidate("why the fastest sprinter cannot beat a horse",
                  external_id="b", rate=3000, age_hours=1),
    ]


def test_the_gate_keeps_a_board_on_subject_when_something_else_outscores_it(
        offline, monkeypatch):
    """The whole point of it. Under `prefer` the off-niche candidate wins the
    slot on velocity, which is the behaviour board #15 was full of; under
    `gate` being on the account's subject is a precondition it cannot outbid.

    A ledger each, because the first board would otherwise score its own
    winner down as already-seen on the second and prove nothing.
    """
    install_source(monkeypatch, "reddit", _two_candidates())
    prefer = topics.find_topics("ocean", sources=("reddit",), limit=1,
                                niche_mode="prefer",
                                ledger=topics.TopicLedger(offline / "a.json"))
    install_source(monkeypatch, "reddit", _two_candidates())
    gate = topics.find_topics("ocean", sources=("reddit",), limit=1,
                              niche_mode="gate",
                              ledger=topics.TopicLedger(offline / "b.json"))

    assert prefer["candidates"][0]["topic"].startswith("why the fastest")
    assert gate["candidates"][0]["topic"] == "the ocean is salty"
    assert any(entry["reason"] == "off the niche"
               for entry in gate["rejected"])


def test_the_gate_backfills_rather_than_handing_back_an_empty_board(
        offline, monkeypatch):
    """A gate that can empty the board turns a quiet day into no board at all.
    The honest answer is the off-niche pile, and a warning saying so."""
    install_source(monkeypatch, "reddit", [
        candidate("Braves at Twins on Sunday", external_id="a",
                  rate=300, age_hours=2),
        candidate("a new pasta shape from Bologna", external_id="b",
                  rate=200, age_hours=2),
    ])
    board = topics.find_topics("ocean", sources=("reddit",), limit=2,
                               niche_mode="gate")

    assert len(board["candidates"]) == 2
    assert any("matched the niche" in warning for warning in board["warnings"])


def test_niche_mode_off_does_not_judge_the_subject_at_all(offline, monkeypatch):
    install_source(monkeypatch, "reddit", [
        candidate("a new pasta shape from Bologna", external_id="a",
                  rate=200, age_hours=2),
    ])
    board = topics.find_topics("ocean", sources=("reddit",), niche_mode="off")
    parts = board["candidates"][0]["components"]

    assert "off the niche" not in parts and "niche match" not in parts
    assert board["niche_mode"] == "off"


def test_the_gate_spends_no_probe_requests_on_what_it_is_about_to_drop(
        offline, monkeypatch):
    """It runs before the probe, which costs a request per candidate."""
    provider = FakeProvider("fake", results=stock("ocean-waves"))
    monkeypatch.setattr(topics, "_ready_providers",
                        lambda session, warnings: [provider])
    install_source(monkeypatch, "reddit", _two_candidates())
    topics.find_topics("ocean", sources=("reddit",), limit=1,
                       niche_mode="gate", sourceability="probe")

    assert [query for query, _, _ in provider.searches] == ["ocean salty"]


def test_one_source_failing_is_a_warning_not_a_failure(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=100, age_hours=2)])
    install_source(monkeypatch, "trends",
                   error=topics.SourceError("429 rate limited", source="trends"))
    board = topics.find_topics("", sources=("reddit", "trends"))

    assert board["candidates"], "the working source still produced a board"
    assert any("429" in w for w in board["warnings"])


def test_every_source_failing_raises(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   error=topics.SourceError("no", source="reddit"))
    with pytest.raises(topics.TopicsError):
        topics.find_topics("", sources=("reddit",))


def test_an_unknown_source_name_is_a_warning(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=50, age_hours=2)])
    board = topics.find_topics("", sources=("reddit", "nonsense"))
    assert any("unknown source name" in w for w in board["warnings"])


def test_a_topic_already_made_never_comes_back(offline, monkeypatch):
    item = candidate("why the ocean is salty", rate=300, age_hours=2)
    ledger = topics.TopicLedger(offline / "usage.json")
    ledger.mark_used(item.key, item.topic)
    ledger.save()

    install_source(monkeypatch, "reddit", [
        item, candidate("how volcanoes form", external_id="v1",
                        rate=100, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))

    assert [c["topic"] for c in board["candidates"]] == ["how volcanoes form"]
    assert board["rejected"][0]["reason"] == "already made into a reel"


def test_a_topic_the_fact_check_keeps_refusing_leaves_the_board(offline,
                                                                monkeypatch):
    """Five Pro calls went on one line before the board could remember this."""
    monkeypatch.setattr(config, "TOPIC_MAX_REFUSALS", 2)
    item = candidate("what the sound of the deep ocean actually is",
                     rate=300, age_hours=2)
    ledger = topics.TopicLedger(offline / "usage.json")
    ledger.mark_refused(item.key, item.topic)
    ledger.mark_refused(item.key, item.topic)
    ledger.save()

    install_source(monkeypatch, "reddit", [
        item, candidate("how volcanoes form", external_id="v1",
                        rate=100, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))

    assert [c["topic"] for c in board["candidates"]] == ["how volcanoes form"]
    assert "refused" in board["rejected"][0]["reason"]


def test_one_refusal_lets_a_topic_back_on_the_board_carrying_the_cost(
        offline, monkeypatch):
    """A refusal is evidence about a draft as much as about the topic."""
    monkeypatch.setattr(config, "TOPIC_MAX_REFUSALS", 2)
    item = candidate("what the sound of the deep ocean actually is",
                     rate=300, age_hours=2)
    ledger = topics.TopicLedger(offline / "usage.json")
    ledger.mark_refused(item.key, item.topic)
    ledger.save()

    install_source(monkeypatch, "reddit", [item])
    board = topics.find_topics("", sources=("reddit",))

    assert [c["topic"] for c in board["candidates"]] == [item.topic]
    listed = board["candidates"][0]["components"]
    assert any("refused" in name for name in listed), listed


def test_a_named_subject_with_no_footage_is_dropped(offline, monkeypatch):
    monkeypatch.setattr(
        topics, "_ready_providers",
        lambda session, warnings: [FakeProvider("pexels", stock("a-man-in-a-studio"))],
    )
    install_source(monkeypatch, "reddit", [
        candidate("Trevon Diggs signs with Seattle", external_id="n1",
                  rate=300, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))

    assert board["candidates"] == []
    assert board["rejected"][0]["reason"] == "a named subject with no footage"


def test_an_unnamed_topic_with_no_footage_survives_with_a_penalty(offline, monkeypatch):
    """The provider knew a synonym this does not - "bioluminescence" returns
    jellyfish - so a zero-hit probe cannot drop a topic on its own."""
    monkeypatch.setattr(
        topics, "_ready_providers",
        lambda session, warnings: [FakeProvider(
            "pexels", stock("mesmerizing-underwater-jellyfish-dance"))],
    )
    install_source(monkeypatch, "reddit", [
        candidate("bioluminescence in the deep sea", external_id="b1",
                  rate=300, age_hours=2),
    ])
    board = topics.find_topics("", sources=("reddit",))

    assert len(board["candidates"]) == 1
    assert board["candidates"][0]["components"]["footage"] == topics.SOURCEABLE_FLOOR


def test_no_provider_configured_is_a_warning_and_no_probe(offline, monkeypatch):
    # The real _ready_providers, against an empty provider list - the warning
    # is the point of the test, so it cannot come from the fixture's stub.
    monkeypatch.undo()
    monkeypatch.setattr(topics, "_new_session", lambda: FakeSession())
    monkeypatch.setattr(config, "TOPIC_USAGE_FILE", offline / "usage.json")
    monkeypatch.setattr(config, "TOPIC_PHRASE_MODE", "heuristic")
    # The undo above dropped the fixture's pin too, and the footage pass
    # defaults to `llm` - without this the test asks Gemini for real and the
    # provider list it is about is never consulted.
    monkeypatch.setattr(config, "TOPIC_SOURCEABILITY", "probe")
    monkeypatch.setattr(config, "VISUAL_PROVIDERS", ())
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    board = topics.find_topics("", sources=("reddit",))

    assert board["stats"]["probed"] == 0
    assert any("no stock provider" in w for w in board["warnings"])
    assert board["candidates"], "an unprobed board is still a board"


def test_the_probe_is_confined_to_the_shortlist(offline, monkeypatch):
    provider = FakeProvider("pexels", stock("salty-ocean-waves"))
    monkeypatch.setattr(topics, "_ready_providers",
                        lambda session, warnings: [provider])
    monkeypatch.setattr(config, "TOPIC_PROBE_LIMIT", 2)
    install_source(monkeypatch, "reddit", [
        candidate(f"why the ocean is salty number {n}", external_id=f"n{n}",
                  rate=300 - n, age_hours=2)
        for n in range(10)
    ])
    board = topics.find_topics("", sources=("reddit",), limit=2)

    assert len(provider.searches) == 2, "eight unprobed candidates cost nothing"
    assert board["stats"]["probed"] == 2


def test_a_board_records_itself_so_the_next_one_differs(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    first = topics.find_topics("", sources=("reddit",))
    second = topics.find_topics("", sources=("reddit",))

    assert "on a recent board" not in first["candidates"][0]["components"]
    assert "on a recent board" in second["candidates"][0]["components"]
    assert second["candidates"][0]["score"] < first["candidates"][0]["score"]


def test_the_board_is_capped_at_the_limit(offline, monkeypatch):
    install_source(monkeypatch, "reddit", [
        candidate(f"why the ocean is salty {n}", external_id=f"n{n}",
                  rate=100, age_hours=2)
        for n in range(20)
    ])
    board = topics.find_topics("", sources=("reddit",), limit=3)
    assert len(board["candidates"]) == 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_sources_status_needs_no_network(monkeypatch, capsys):
    # Pinned for the same reason as the window above - this asserts on what
    # the status table prints, not on which sources this machine has enabled.
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit", "trends"))
    assert topics.main(["--sources-status"]) == 0
    printed = capsys.readouterr().out
    assert "trends" in printed and "reddit" in printed


def test_the_cli_prints_the_next_command(offline, monkeypatch, capsys):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    assert topics.main(["--no-probe", "--no-phrases"]) == 0
    printed = capsys.readouterr().out
    assert 'python pipeline/script_gen.py "why the ocean is salty"' in printed


def test_the_next_command_carries_the_length_the_board_decided(
        offline, monkeypatch, capsys):
    """The length is an argument this board made and can defend. Leaving it out
    of the command hands the topic to step 2's fallback instead."""
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    assert topics.main(["--no-probe", "--no-phrases"]) == 0
    printed = capsys.readouterr().out
    seconds = topics.suggest_duration(candidate("why the ocean is salty"))["seconds"]
    assert f'script_gen.py "why the ocean is salty" -d {seconds}' in printed
    assert f"length: {seconds}s" in printed
    assert f"explainer +{topics.DURATION_EXPLAINER_BONUS:g}" in printed


def test_a_board_saved_before_lengths_existed_still_prints_a_command():
    """Both the board file and the ledger outlive a release, so every reader of
    one has to survive a board written by the version before it."""
    command = topics._script_gen_command({"topic": "salt"})
    assert command == '  python pipeline/script_gen.py "salt"'


def test_take_marks_the_choice_and_prints_its_command(offline, monkeypatch, capsys):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    assert topics.main(["--no-probe", "--no-phrases", "--take", "1"]) == 0
    printed = capsys.readouterr().out
    seconds = topics.suggest_duration(candidate("why the ocean is salty"))["seconds"]
    assert "taken: why the ocean is salty" in printed
    assert f'script_gen.py "why the ocean is salty" -d {seconds}' in printed
    assert topics.TopicLedger(offline / "usage.json").was_used("reddit:why-the-ocean-is-salty")


def test_take_out_of_range_is_an_error(offline, monkeypatch, capsys):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    assert topics.main(["--no-probe", "--no-phrases", "--take", "9"]) == 1


def test_json_output_is_the_board(offline, monkeypatch, capsys):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    assert topics.main(["--no-probe", "--no-phrases", "--json"]) == 0
    board = json.loads(capsys.readouterr().out)
    assert board["candidates"][0]["topic"] == "why the ocean is salty"


# ---------------------------------------------------------------------------
# The keyless sources
# ---------------------------------------------------------------------------

def atom(*entries) -> bytes:
    """A Reddit hot feed, in the Atom shape the live site actually serves."""
    body = ["<?xml version='1.0' encoding='UTF-8'?>",
            "<feed xmlns='http://www.w3.org/2005/Atom'>"]
    for entry in entries:
        body.append(
            "<entry>"
            f"<author><name>{entry.get('author', '/u/someone')}</name></author>"
            f"<category term='{entry.get('sub', 'askscience')}'/>"
            f"<id>t3_{entry.get('id', 'abc123')}</id>"
            f"<link href='{entry.get('url', 'https://reddit.com/r/x/comments/1/')}'/>"
            f"<published>{entry.get('published', '2026-08-20T09:00:00+00:00')}</published>"
            f"<title>{entry.get('title', 'A title')}</title>"
            "</entry>"
        )
    body.append("</feed>")
    return "".join(body).encode("utf-8")


def rss_session(*entries, status=200):
    body = atom(*entries)
    return FakeSession({"reddit.com": FakeResponse(status, body=body)})


def test_the_rss_source_needs_no_key_but_still_needs_subreddits(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ())
    ready, detail = topics.RedditRssSource.readiness()
    assert not ready and "TOPIC_SUBREDDITS" in detail

    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    ready, detail = topics.RedditRssSource.readiness()
    assert ready and "no key" in detail


def test_the_rss_source_reads_an_atom_entry_into_a_candidate(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    session = rss_session({
        "title": "TIL that the deep ocean is mostly unexplored",
        "id": "1kasawp", "sub": "todayilearned",
        "url": "https://www.reddit.com/r/todayilearned/comments/1kasawp/x/",
    })
    got = topics.RedditRssSource(session).fetch("ocean", 10)

    assert len(got) == 1
    only = got[0]
    # `clean_title` still strips the forum furniture off an rss title.
    assert only.topic == "the deep ocean is mostly unexplored"
    # `t3_` is dropped so the ledger key matches the one the .json api writes.
    assert only.external_id == "1kasawp"
    assert only.key == "redditrss:1kasawp"
    assert "r/todayilearned" in only.context and "top #1" in only.context
    assert only.url.endswith("/1kasawp/x/")


def test_the_rss_source_asks_for_the_listing_it_was_configured_with(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "top")
    monkeypatch.setattr(config, "TOPIC_REDDIT_WINDOW", "month")
    session = rss_session({
        "title": "TIL that the deep ocean is mostly unexplored",
        "id": "1kasawp", "sub": "askscience",
        "url": "https://www.reddit.com/r/askscience/comments/1kasawp/x/",
    })
    got = topics.RedditRssSource(session).fetch("ocean", 10)

    assert "/top/.rss" in session.calls[0]["url"]
    assert session.calls[0]["params"]["t"] == "month"
    # The rank is quoted as a claim about a listing, so it has to say which.
    assert "top #1" in got[0].context
    assert got[0].rate_unit == "top rank"


def test_hot_sends_no_window_because_it_has_none(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    monkeypatch.setattr(config, "TOPIC_REDDIT_LISTING", "hot")
    session = rss_session({
        "title": "TIL that the deep ocean is mostly unexplored",
        "id": "1kasawp", "sub": "askscience",
        "url": "https://www.reddit.com/r/askscience/comments/1kasawp/x/",
    })
    topics.RedditRssSource(session).fetch("ocean", 10)

    assert "/hot/.rss" in session.calls[0]["url"]
    assert "t" not in session.calls[0]["params"]


def test_a_pinned_mod_post_is_not_a_candidate(monkeypatch):
    """The feed has no `stickied` flag, so the author is what gives it away."""
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    session = rss_session(
        {"title": "Joint Subreddit Statement", "author": "/u/AskScienceModerator"},
        {"title": "Weekly thread", "author": "/u/AutoModerator"},
        {"title": "Why is the ocean salty", "author": "/u/a_person", "id": "keep"},
    )
    got = topics.RedditRssSource(session).fetch("ocean", 10)
    assert [c.external_id for c in got] == ["keep"]


def test_the_hot_rank_stands_in_for_the_missing_score(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    session = rss_session(*[{"title": f"Question number {i}", "id": f"id{i}"}
                           for i in range(5)])
    got = topics.RedditRssSource(session).fetch("ocean", 10)

    rates = [c.rate for c in got]
    assert rates == sorted(rates, reverse=True)         # hot order is kept
    assert rates[0] == topics.RedditRssSource.TOP_RATE
    assert rates[-1] == pytest.approx(topics.RedditRssSource.FLOOR_RATE)
    # Reach is left at zero rather than filled with a number nobody measured.
    assert all(c.reach == 0.0 for c in got)


def test_the_rank_spread_survives_being_logged(monkeypatch):
    """A linear ramp would come out flat once the scorer takes log1p of it."""
    top = topics.RedditRssSource._rate_for_rank(0, 25)
    last = topics.RedditRssSource._rate_for_rank(24, 25)
    scored = []
    for rate in (top, last):
        scored.append(topics.score_candidate(
            candidate("a question about the deep ocean floor",
                      source="redditrss", rate=rate, age_hours=30.0)
        )["components"].get("velocity", 0.0))
    assert scored[0] - scored[1] > 0.8


def test_the_board_reports_what_fetched_what_even_after_the_probe(
        offline, monkeypatch):
    """The llm probe used to overwrite the source reports with its own ratings.

    `find_topics` binds `reports` to the per-source fetch summaries and returns
    them as `sources`; the sourceability pass rebound the same name to its
    {index: rating} dict, so a board scouted with TOPIC_SOURCEABILITY=llm came
    back with `sources` full of probe results and `_print_board` crashed
    iterating ints on its way to printing a board it had already built.
    """
    install_source(monkeypatch, "reddit", [
        candidate("why the ocean is salty", external_id="t1",
                  rate=300, age_hours=2),
    ])
    monkeypatch.setattr(
        topics, "classify_sourceability",
        lambda shortlist: ({i: {"rating": 3, "hits": 3, "sample": 3,
                                "query": c.topic, "provider": "model",
                                "checked": True, "by_provider": {},
                                "notes": []}
                            for i, c in enumerate(shortlist)}, []),
    )

    board = topics.find_topics("ocean", sources=("reddit",),
                               sourceability="llm", phrases="heuristic",
                               niche_mode="off")

    assert isinstance(board["sources"], list)
    assert board["sources"][0]["name"] == "reddit"
    assert board["sources"][0]["ready"] is True
    assert board["sources"][0]["fetched"] == 1
    # The probe still ran and still reached the candidate.
    assert board["stats"]["probed"] == 1
    topics._print_board(board)          # used to raise TypeError


def test_the_subreddits_go_up_in_no_more_feeds_than_configured(monkeypatch):
    """Anonymous reads are rate limited hard, so the feed count is bounded.

    Bounded, not one: one combined feed is ranked by absolute score, and the
    largest subreddit in the list takes every place in it. TOPIC_REDDIT_FEEDS
    is what buys the small ones a listing of their own - see
    `RedditRssSource._groups`.
    """
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS",
                        ("todayilearned", "askscience", "science",
                         "oceans", "geology", "marinebiology"))
    monkeypatch.setattr(config, "TOPIC_REDDIT_FEEDS", 3)
    monkeypatch.setattr(topics.RedditRssSource, "FEED_GAP_SECONDS", 0.0)
    session = rss_session({"title": "Why is the ocean salty"})
    topics.RedditRssSource(session).fetch("ocean", 25)

    assert len(session.calls) <= config.TOPIC_REDDIT_FEEDS
    # Every subreddit is actually read - the point of splitting at all.
    asked = " ".join(call["url"] for call in session.calls)
    for sub in config.TOPIC_SUBREDDITS:
        assert sub in asked
    # And the groups share TOPIC_PER_SOURCE rather than each taking it.
    assert sum(call["params"]["limit"] for call in session.calls) <= 25 + len(session.calls)


def test_one_feed_still_means_one_request(monkeypatch):
    """TOPIC_REDDIT_FEEDS=1 is exactly the single combined URL it replaced."""
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS",
                        ("todayilearned", "askscience", "science"))
    monkeypatch.setattr(config, "TOPIC_REDDIT_FEEDS", 1)
    session = rss_session({"title": "Why is the ocean salty"})
    topics.RedditRssSource(session).fetch("ocean", 25)

    assert len(session.calls) == 1
    assert "todayilearned+askscience+science" in session.calls[0]["url"]


def test_a_small_subreddit_is_ranked_against_its_own_group(monkeypatch):
    """Rank is per feed, so the top of a small sub is not buried by a big one."""
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("todayilearned", "oceans"))
    monkeypatch.setattr(config, "TOPIC_REDDIT_FEEDS", 2)
    monkeypatch.setattr(topics.RedditRssSource, "FEED_GAP_SECONDS", 0.0)
    session = rss_session({"title": "Why is the ocean salty"})
    found = topics.RedditRssSource(session).fetch("ocean", 25)

    # Both feeds returned a #1, and a #1 gets TOP_RATE wherever it came from.
    assert len(session.calls) == 2
    assert len(found) >= 1
    assert max(c.rate for c in found) == topics.RedditRssSource.TOP_RATE


def test_the_group_that_goes_first_rotates_by_day(monkeypatch):
    """Whichever feed is first survives a rate limit, so it must not be fixed."""
    groups = [("a",), ("b",), ("c",)]
    first = {topics.RedditRssSource._rotate(groups, day=d)[0][0]
             for d in range(3)}
    assert first == {"a", "b", "c"}

    # Same day, same order: two runs in an afternoon scout the same feeds.
    assert (topics.RedditRssSource._rotate(groups, day=7)
            == topics.RedditRssSource._rotate(groups, day=7))
    # And nothing is dropped or duplicated by the rotation.
    assert sorted(topics.RedditRssSource._rotate(groups, day=2)) == sorted(groups)
    assert topics.RedditRssSource._rotate([]) == []


def test_the_feeds_are_spaced_rather_than_fired_together(monkeypatch):
    """The rate limit is per minute, so the gap is what avoids it at all.

    Without this the three feeds go up back to back and are refused outright -
    which is how splitting the list made the scout worse before it made it
    better.
    """
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS",
                        ("askscience", "oceans", "geology"))
    monkeypatch.setattr(config, "TOPIC_REDDIT_FEEDS", 3)
    slept: list[float] = []
    monkeypatch.setattr(topics.RedditRssSource, "FEED_GAP_SECONDS", 4.0)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    session = rss_session({"title": "Why is the ocean salty"})
    topics.RedditRssSource(session).fetch("ocean", 25)

    # One gap between each pair of feeds, and none before the first.
    assert len(session.calls) == 3
    assert slept == [4.0, 4.0]


def test_one_failed_feed_does_not_lose_the_others(monkeypatch):
    """A 429 on the third group is some subreddits lost, not the source."""
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience", "oceans"))
    monkeypatch.setattr(config, "TOPIC_REDDIT_FEEDS", 2)
    monkeypatch.setattr(topics.RedditRssSource, "FEED_GAP_SECONDS", 0.0)
    monkeypatch.setattr(topics.RedditRssSource, "RETRY_WAITS", ())
    session = rss_session({"title": "Why is the ocean salty"})

    calls = {"n": 0}
    original = session.get

    def flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429, body=b"")
        return original(url, **kwargs)

    session.get = flaky
    source = topics.RedditRssSource(session)
    found = source.fetch("ocean", 25)

    assert found                        # the second group still came back
    assert source.notes                 # and the loss is on the record
    assert "429" in source.notes[0] or "rate limited" in source.notes[0]


def test_a_rate_limited_feed_is_retried_and_then_explained(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    monkeypatch.setattr(topics.RedditRssSource, "RETRY_WAITS", (0.0, 0.0))
    session = rss_session(status=429)

    with pytest.raises(topics.SourceError) as caught:
        topics.RedditRssSource(session).fetch("ocean", 10)

    assert "rate limited" in str(caught.value)
    assert len(session.calls) == 3          # the first try plus both retries


def test_a_feed_that_is_not_xml_is_one_bad_source_not_a_crash(monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SUBREDDITS", ("askscience",))
    session = FakeSession({"reddit.com": FakeResponse(200, body=b"<html>nope")})
    with pytest.raises(topics.SourceError):
        topics.RedditRssSource(session).fetch("ocean", 10)


# --- wikipedia -------------------------------------------------------------

def wiki_day(*pairs):
    return {"items": [{"articles": [{"article": name, "views": views}
                                    for name, views in pairs]}]}


def wiki_session(today, week_ago):
    """Routes the two day-URLs the source asks for by their date fragment."""
    import re as _re
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    latest = (_dt.now(_tz.utc).date()
              - _td(days=topics.WikipediaSource.LAG_DAYS))
    before = latest - _td(days=topics.WikipediaSource.BASELINE_DAYS)
    fmt = "{0.year}/{0.month:02d}/{0.day:02d}".format
    routes = {fmt(latest): FakeResponse(200, payload=today)}
    if week_ago is None:
        routes[fmt(before)] = FakeResponse(404, body=b"no such day")
    else:
        routes[fmt(before)] = FakeResponse(200, payload=week_ago)
    return FakeSession(routes)


def test_wikipedia_keeps_only_the_articles_that_climbed():
    session = wiki_session(
        wiki_day(("Bioluminescence", 9000), ("Steady_Article", 5000)),
        wiki_day(("Bioluminescence", 1000), ("Steady_Article", 5000)),
    )
    got = topics.WikipediaSource(session).fetch("ocean", 10)
    assert [c.external_id for c in got] == ["Bioluminescence"]
    assert "x9.0 vs last week" in got[0].context


def test_an_article_new_to_the_top_list_is_the_strongest_signal():
    session = wiki_session(
        wiki_day(("Mariana_Trench", 8000)),
        wiki_day(("Something_Else", 8000)),
    )
    got = topics.WikipediaSource(session).fetch("ocean", 10)
    assert got[0].context == "new to the top list"
    assert got[0].topic == "Mariana Trench"          # underscores become spaces


def test_the_front_door_and_the_namespaces_are_not_articles():
    session = wiki_session(
        wiki_day(("Main_Page", 900000), ("Special:Search", 800000),
                 ("Wikipedia:Featured", 700000), ("Deep_sea", 6000)),
        wiki_day(("Deep_sea", 1000)),
    )
    got = topics.WikipediaSource(session).fetch("ocean", 10)
    assert [c.external_id for c in got] == ["Deep_sea"]


def test_the_baseline_is_the_same_weekday_a_week_back():
    """Readership is weekly - a Sunday against a Saturday is not a spike."""
    session = wiki_session(wiki_day(("Deep_sea", 6000)), wiki_day(("Deep_sea", 1000)))
    topics.WikipediaSource(session).fetch("ocean", 10)

    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    latest = _dt.now(_tz.utc).date() - _td(days=topics.WikipediaSource.LAG_DAYS)
    before = latest - _td(days=7)
    assert before.weekday() == latest.weekday()
    assert len(session.calls) == 2


def test_a_missing_baseline_day_thins_the_source_rather_than_failing_it():
    session = wiki_session(wiki_day(("Deep_sea", 6000), ("Kelp", 5000)), None)
    got = topics.WikipediaSource(session).fetch("ocean", 10)
    # With nothing to compare against, everything is passed through unfiltered
    # rather than every article being called a new entrant.
    assert [c.external_id for c in got] == ["Deep_sea", "Kelp"]


def test_wikipedia_is_not_judged_against_the_breaking_news_window(monkeypatch):
    """Its freshest possible candidate is already a day old on arrival."""
    monkeypatch.setattr(config, "TOPIC_MAX_AGE_HOURS", 12.0)
    assert topics.max_age_for("wikipedia") > 24.0


def test_an_empty_day_from_the_api_is_a_source_error():
    session = wiki_session(wiki_day(), wiki_day(("x", 1)))
    with pytest.raises(topics.SourceError):
        topics.WikipediaSource(session).fetch("ocean", 10)


# ---------------------------------------------------------------------------
# The saved board
#
# Reading a board and choosing from it are two commands. Without a cache they
# are two different boards - measured on a live run, an anonymous Reddit read
# was rate limited on the second pass and the source it led with dropped out
# of the choice entirely.
# ---------------------------------------------------------------------------

def test_a_board_is_stamped_with_when_it_was_scouted(offline, monkeypatch):
    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    board = topics.find_topics(sourceability="off", phrases="heuristic")
    assert topics._parse_iso8601(board["scouted_at"]) is not None


def test_a_saved_board_comes_back(tmp_path):
    path = tmp_path / "last_board.json"
    board = {"scouted_at": topics.datetime.now(topics.timezone.utc).isoformat(),
             "candidates": [{"key": "redditrss:abc", "topic": "why oceans are salty"}]}
    assert topics.save_board(board, path) == []

    loaded, age = topics.load_board(path)
    assert loaded["candidates"][0]["key"] == "redditrss:abc"
    assert age < 1.0


def test_a_board_past_its_ttl_is_not_offered(tmp_path):
    """What was climbing two hours ago has probably crested."""
    old = topics.datetime.now(topics.timezone.utc) - topics.timedelta(hours=2)
    path = tmp_path / "last_board.json"
    topics.save_board({"scouted_at": old.isoformat(),
                       "candidates": [{"key": "x:1", "topic": "t"}]}, path)

    loaded, age = topics.load_board(path, max_age_minutes=60)
    assert loaded is None and age > 60


def test_an_unreadable_board_is_no_board_rather_than_an_error(tmp_path):
    path = tmp_path / "last_board.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert topics.load_board(path) == (None, 0.0)

    missing = tmp_path / "nothing_here.json"
    assert topics.load_board(missing) == (None, 0.0)


def test_a_board_without_a_stamp_is_not_trusted(tmp_path):
    path = tmp_path / "last_board.json"
    topics.save_board({"candidates": [{"key": "x:1", "topic": "t"}]}, path)
    assert topics.load_board(path)[0] is None


def test_take_acts_on_the_board_that_was_just_printed(offline, monkeypatch, capsys):
    """The whole point: one scout, then a choice made from it."""
    scouts = []
    real = topics.find_topics

    def counted(*args, **kwargs):
        scouts.append(1)
        return real(*args, **kwargs)

    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    monkeypatch.setattr(config, "TOPIC_BOARD_FILE", offline / "last_board.json")
    monkeypatch.setattr(topics, "find_topics", counted)

    assert topics.main(["--no-probe", "--no-phrases"]) == 0
    assert topics.main(["--no-probe", "--no-phrases", "--take", "1"]) == 0

    assert len(scouts) == 1                      # the second command reused it
    out = capsys.readouterr().out
    assert "taking from the board scouted" in out
    assert "taken: why the ocean is salty" in out


def test_rescout_asks_the_sources_again(offline, monkeypatch, capsys):
    scouts = []
    real = topics.find_topics

    def counted(*args, **kwargs):
        scouts.append(1)
        return real(*args, **kwargs)

    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    monkeypatch.setattr(config, "TOPIC_BOARD_FILE", offline / "last_board.json")
    monkeypatch.setattr(topics, "find_topics", counted)

    topics.main(["--no-probe", "--no-phrases"])
    topics.main(["--no-probe", "--no-phrases", "--take", "1", "--rescout"])
    assert len(scouts) == 2


def test_an_argument_that_changes_the_board_forces_a_fresh_scout(
        offline, monkeypatch, capsys):
    """--take with a different niche must not answer from the old board."""
    scouts = []
    real = topics.find_topics

    def counted(*args, **kwargs):
        scouts.append(1)
        return real(*args, **kwargs)

    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    monkeypatch.setattr(config, "TOPIC_BOARD_FILE", offline / "last_board.json")
    monkeypatch.setattr(topics, "find_topics", counted)

    topics.main(["--no-probe", "--no-phrases"])
    topics.main(["--no-probe", "--no-phrases", "--take", "1", "-n", "volcanoes"])
    assert len(scouts) == 2


def test_a_plain_scout_still_scouts_every_time(offline, monkeypatch, capsys):
    """The cache is only ever read for --take; looking means looking."""
    scouts = []
    real = topics.find_topics

    def counted(*args, **kwargs):
        scouts.append(1)
        return real(*args, **kwargs)

    install_source(monkeypatch, "reddit",
                   [candidate("why the ocean is salty", rate=300, age_hours=2)])
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("reddit",))
    monkeypatch.setattr(config, "TOPIC_BOARD_FILE", offline / "last_board.json")
    monkeypatch.setattr(topics, "find_topics", counted)

    topics.main(["--no-probe", "--no-phrases"])
    topics.main(["--no-probe", "--no-phrases"])
    assert len(scouts) == 2


# ---------------------------------------------------------------------------
# The evergreen backlog
# ---------------------------------------------------------------------------

def backlog(tmp_path, *lines) -> Path:
    """A backlog file holding `lines`, with comments and blanks mixed in."""
    path = tmp_path / "evergreen.txt"
    body = ["# a comment", ""] + list(lines) + ["", "   ", "# trailing note"]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def test_the_backlog_drops_comments_and_blank_lines(tmp_path):
    path = backlog(tmp_path, "why the ocean is salty", "how a cave forms")
    assert topics.EvergreenSource.read_file(path) == [
        "why the ocean is salty", "how a cave forms",
    ]


def test_the_backlog_drops_a_line_repeated(tmp_path):
    """The same line twice is an edit slip, not two topics."""
    path = backlog(tmp_path, "why the ocean is salty", "WHY THE OCEAN IS SALTY")
    assert topics.EvergreenSource.read_file(path) == ["why the ocean is salty"]


def test_a_missing_backlog_is_not_ready_rather_than_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE", tmp_path / "nope.txt")
    ready, detail = topics.EvergreenSource.readiness()
    assert ready is False
    assert "add one per line" in detail


def test_the_backlog_keys_on_the_line_not_its_position(tmp_path, monkeypatch):
    """Reordering the file must not re-open a topic already made."""
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty", "how a cave forms"))
    first = {c.key for c in topics.EvergreenSource(None).fetch("", 10)}

    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "how a cave forms", "why the ocean is salty"))
    assert {c.key for c in topics.EvergreenSource(None).fetch("", 10)} == first


def test_a_backlog_line_carries_no_velocity(tmp_path, monkeypatch):
    """No reach, no rate, no age - a backlog line is not a slow story."""
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty"))
    found = topics.EvergreenSource(None).fetch("", 10)
    assert [(c.rate, c.reach, c.age_hours) for c in found] == [(0.0, 0.0, 0.0)]

    scored = topics.score_candidate(found[0], on_niche=True)
    assert "velocity" not in scored["components"]
    assert "past its peak" not in scored["components"]


def test_the_backlog_pays_the_penalty_and_a_scouted_topic_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty"))
    line = topics.EvergreenSource(None).fetch("", 1)[0]
    assert topics.score_candidate(line, on_niche=True)["components"][
        "from the backlog"] == topics.EVERGREEN_PENALTY

    scouted = candidate("why the ocean is salty", rate=300, age_hours=2)
    assert "from the backlog" not in topics.score_candidate(
        scouted, on_niche=True)["components"]


def test_a_moving_scouted_topic_still_beats_the_backlog(tmp_path, monkeypatch):
    """The backlog is the floor under a thin board, not the ceiling over a good one."""
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty"))
    monkeypatch.setattr(config, "TOPIC_MAX_AGE_HOURS", 72.0)

    line = topics.EvergreenSource(None).fetch("", 1)[0]
    scouted = candidate("why the ocean is salty", rate=300, age_hours=2)
    assert (topics.score_candidate(scouted, on_niche=True)["score"]
            > topics.score_candidate(line, on_niche=True)["score"])


def test_the_backlog_window_advances_with_the_day(tmp_path, monkeypatch):
    """Longer than TOPIC_PER_SOURCE, so every line has to come round."""
    lines = [f"why thing number {n} happens" for n in range(10)]
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE", backlog(tmp_path, *lines))

    windows = set()
    for yday in (1, 2, 3):
        class FixedDate(topics.datetime):
            @classmethod
            def now(cls, tz=None):
                return topics.datetime(2026, 1, 1, tzinfo=tz).replace(
                    day=1) + topics.timedelta(days=yday - 1)
        monkeypatch.setattr(topics, "datetime", FixedDate)
        windows.add(tuple(c.topic for c in topics.EvergreenSource(None).fetch("", 3)))
    assert len(windows) == 3


def test_a_boarded_backlog_line_can_still_clear_the_floor(tmp_path, monkeypatch):
    """Board #41 skipped the day because it could not.

    SCOUTED_PENALTY and EVERGREEN_PENALTY together cost 4.3 against a backlog
    ceiling of 4.2, so the best possible line scored -0.1 and every daily run
    skipped. The recency penalty a backlog line pays is its own.
    """
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty"))
    line = topics.EvergreenSource(None).fetch("", 1)[0]

    components = topics.score_candidate(
        line, on_niche=True, seen=True)["components"]
    assert components["on a recent board"] == topics.EVERGREEN_SCOUTED_PENALTY
    assert components["from the backlog"] == topics.EVERGREEN_PENALTY

    # The ceiling EVERGREEN_PENALTY is priced against: 1.0 niche + 1.2 shape
    # + 1.5 footage + 0.5 curiosity. A line that reaches it must stay makeable
    # after both penalties, or the board deadlocks the way #41 did.
    ceiling = 1.0 + topics.EXPLAINER_BONUS + topics.SOURCEABLE_BONUS + 0.5
    assert (ceiling + topics.EVERGREEN_SCOUTED_PENALTY
            + topics.EVERGREEN_PENALTY) >= 1.0


def test_a_scouted_topic_still_pays_the_full_recency_penalty():
    seen = topics.score_candidate(
        candidate("why the ocean is salty", rate=300, age_hours=2),
        on_niche=True, seen=True)
    assert seen["components"]["on a recent board"] == topics.SCOUTED_PENALTY


def test_every_registered_source_is_known_to_the_self_check():
    """`python config.py` called the working evergreen source an unknown name.

    The registry is the list of sources that exist; TOPIC_SOURCE_KEYS is what
    the self-check consults to say what each one needs. A source missing from
    the second is reported as broken while running fine.
    """
    assert set(topics._SOURCES) <= set(config.TOPIC_SOURCE_KEYS)


def test_the_self_check_reports_the_backlog_as_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOPIC_SOURCES", ("evergreen",))
    monkeypatch.setattr(config, "TOPIC_EVERGREEN_FILE",
                        backlog(tmp_path, "why the ocean is salty"))
    row, = topics.source_status()
    assert row["known"] is True and row["ready"] is True
    assert "unknown" not in row["detail"].lower()



# ---------------------------------------------------------------------------
# Which subreddits share a feed
# ---------------------------------------------------------------------------

NINE = ["todayilearned", "askscience", "explainlikeimfive", "science",
        "geology", "EarthScience", "marinebiology", "oceans", "Awwducational"]
#: The four default subs large enough to take every place in a combined feed.
GIANTS = {"todayilearned", "askscience", "explainlikeimfive", "science"}


def _groups(subs, most):
    return topics.RedditRssSource._groups(subs, most)


def test_the_niche_subreddits_get_a_feed_with_no_giant_in_it():
    """The whole point of splitting at all. Nine into two has to go 4/5 and not
    5/4 - under 5/4 the fifth name was pushed in with the giants, and measured
    live on 2026-09-09 r/todayilearned took all twenty places while
    r/askscience, r/explainlikeimfive, r/science and r/geology returned
    nothing."""
    groups = _groups(NINE, 2)

    assert [len(group) for group in groups] == [4, 5]
    assert set(groups[0]) == GIANTS
    assert not (set(groups[1]) & GIANTS)
    # The one the old boundary sacrificed.
    assert "geology" in groups[1]


def test_the_spare_places_go_to_the_last_group():
    """Which is what keeps the head of the list packed together. The list is
    ordered biggest-first, so the giants share one feed rather than spilling
    into the group chosen for the niche."""
    assert [len(g) for g in _groups(list("abcdefghi"), 2)] == [4, 5]
    assert [len(g) for g in _groups(list("abcde"), 2)] == [2, 3]
    assert [len(g) for g in _groups(list("abcdefghij"), 3)] == [3, 3, 4]


def test_the_feed_budget_it_was_given_is_the_feed_budget_it_uses():
    """Chunking by a ceiling silently returned three feeds when four were asked
    for - spending fewer requests than allowed and reading fewer subreddits for
    no reason."""
    assert len(_groups(NINE, 4)) == 4
    assert len(_groups(NINE, 3)) == 3
    assert len(_groups(NINE, 2)) == 2


def test_the_split_never_loses_or_repeats_a_subreddit():
    for most in range(1, len(NINE) + 3):
        groups = _groups(NINE, most)
        flat = [name for group in groups for name in group]
        assert sorted(flat) == sorted(NINE)
        assert all(group for group in groups)


def test_two_runs_scout_the_same_feeds():
    """No state and no clock: `--rescout` is not a different board.

    Rotating membership by the day was tried and measured worse - it moves the
    largest subreddit into the niche group on most days. `_rotate` still varies
    which feed is read *first*, which is a different question.
    """
    assert _groups(NINE, 2) == _groups(NINE, 2)
    assert topics.RedditRssSource._groups(NINE, 2, day=1) == _groups(NINE, 2)


def test_a_feed_each_is_left_alone():
    assert _groups(["a", "b"], 5) == [("a",), ("b",)]


def test_one_feed_is_still_the_whole_list():
    """At TOPIC_REDDIT_FEEDS=1 this is exactly the old single-feed call."""
    groups = _groups(NINE, 1)
    assert len(groups) == 1
    assert list(groups[0]) == NINE
