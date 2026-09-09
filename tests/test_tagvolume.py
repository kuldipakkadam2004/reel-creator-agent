"""Tests for pipeline/tagvolume.py.

Everything here is offline - the estimator needs no key and no network, and the
provider path is exercised with a stub rather than a real data source.

What is worth covering is the judgement, not the arithmetic: that a tag built
from words people use comes out usable, that a coinage comes out dead, that a
measured count beats a guess, and that a bad provider cannot take the step down
with it.

    python -m pytest tests/test_tagvolume.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import tagvolume  # noqa: E402


# ---------------------------------------------------------------------------
# Reading a tag as words
# ---------------------------------------------------------------------------

def test_camel_case_is_split_on_its_capitals():
    assert tagvolume.segment("DeepSeaFish") == ["deep", "sea", "fish"]
    assert tagvolume.segment("#OceanLife") == ["ocean", "life"]


def test_a_flat_tag_is_split_against_the_vocabulary():
    assert tagvolume.segment("deepseafish") == ["deep", "sea", "fish"]


def test_a_split_that_leaves_a_non_word_behind_is_refused():
    # "angler" + "fish", never "angle" + "rfish": taking the longest known head
    # and keeping the remainder turns an ordinary tag into a coinage.
    assert tagvolume.segment("anglerfish") == ["angler", "fish"]


def test_a_word_recognised_whole_is_not_taken_apart():
    # photo + graphy would price a real word as two, one of them nonsense.
    assert tagvolume.segment("photography") == ["photography"]


def test_a_coinage_is_left_whole_rather_than_forced_into_pieces():
    assert tagvolume.segment("piezophile") == ["piezophile"]


# ---------------------------------------------------------------------------
# Rarity
# ---------------------------------------------------------------------------

def test_words_are_ranked_from_everyday_to_unheard_of():
    assert tagvolume.rarity("ocean") == 0            # everyday
    assert tagvolume.rarity("habitat") == 1          # ordinary
    assert tagvolume.rarity("subduction") == 2       # subject vocabulary
    assert tagvolume.rarity("piezophile") == 3       # nobody types this


def test_a_suffix_coming_off_rescues_a_derived_word():
    assert tagvolume.rarity("oceanic") == 2          # -> ocean


def test_a_known_prefix_rescues_a_built_up_term():
    # Without this #Bioluminescence reads as a coinage and is thrown away,
    # when it is one of the larger tags in this subject.
    assert tagvolume.rarity("bioluminescence") == 2
    assert tagvolume.rarity("astrophotography") == 2


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", [
    "Piezophile", "Osmolyte", "Selenology", "MariusHills", "Luciferase",
])
def test_a_tag_nobody_could_search_comes_out_dead(tag):
    assert tagvolume.band_of(tagvolume.estimate(tag)) == "dead"


@pytest.mark.parametrize("tag", [
    "DeepSeaFish", "LavaTubes", "Anglerfish", "Bioluminescence",
    "MarineBiology", "LunarCaves", "Cuttlefish",
])
def test_a_tag_made_of_real_words_comes_out_usable(tag):
    assert tagvolume.band_of(tagvolume.estimate(tag)) in tagvolume.USABLE


def test_the_broadest_tags_are_called_saturated():
    assert tagvolume.band_of(tagvolume.estimate("Ocean")) == "saturated"
    assert tagvolume.band_of(tagvolume.estimate("Nature")) == "saturated"


def test_every_word_added_makes_the_tag_smaller():
    # The whole reason a long compound is long-tail.
    assert tagvolume.estimate("Ocean") > tagvolume.estimate("OceanLife")
    assert tagvolume.estimate("OceanLife") > tagvolume.estimate("DeepOceanLife")


def test_an_empty_tag_is_worth_nothing_rather_than_raising():
    assert tagvolume.estimate("") == 0
    assert tagvolume.estimate("#") == 0


def test_the_bands_follow_the_configured_thresholds(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_FLOOR", 1_000)
    monkeypatch.setattr(config, "TAG_VOLUME_NICHE_MAX", 10_000)
    monkeypatch.setattr(config, "TAG_VOLUME_MID_MAX", 100_000)
    monkeypatch.setattr(config, "TAG_VOLUME_BROAD_MAX", 1_000_000)

    assert tagvolume.band_of(999) == "dead"
    assert tagvolume.band_of(1_000) == "niche"
    assert tagvolume.band_of(10_001) == "mid"
    assert tagvolume.band_of(100_001) == "broad"
    assert tagvolume.band_of(1_000_001) == "saturated"


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

def write_cache(path: Path, tags: dict) -> Path:
    path.write_text(json.dumps({"version": 1, "tags": tags}), encoding="utf-8")
    return path


def test_a_measured_count_beats_the_estimate(tmp_path):
    # #Piezophile really does have 400k posts? Then it is a good tag, whatever
    # the estimator thinks of the word.
    cache = write_cache(tmp_path / "vol.json",
                        {"piezophile": {"count": 400_000, "source": "measured"}})

    volumes, warnings = tagvolume.lookup(
        ["Piezophile"], cache=tagvolume.VolumeCache(cache)
    )

    assert volumes["Piezophile"].count == 400_000
    assert volumes["Piezophile"].source == "measured"
    assert volumes["Piezophile"].band == "mid"
    assert warnings == []


def test_a_hand_seeded_entry_never_expires(tmp_path):
    # No "at" stamp: somebody wrote a real figure down by hand, and a TTL has
    # no business throwing that away.
    cache = write_cache(tmp_path / "vol.json",
                        {"ocean": {"count": 123, "source": "measured"}})

    volumes, _ = tagvolume.lookup(["Ocean"],
                                  cache=tagvolume.VolumeCache(cache))

    assert volumes["Ocean"].count == 123


def test_a_stale_provider_answer_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_TTL_DAYS", 30)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    cache = write_cache(
        tmp_path / "vol.json",
        {"deepseafish": {"count": 7, "source": "provider", "at": long_ago}},
    )

    volumes, _ = tagvolume.lookup(["DeepSeaFish"],
                                  cache=tagvolume.VolumeCache(cache))

    assert volumes["DeepSeaFish"].source == "estimate"


def test_a_nonsense_cache_entry_falls_through_to_the_estimate(tmp_path):
    cache = write_cache(tmp_path / "vol.json",
                        {"deepseafish": {"count": "lots"}, "ocean": "no"})

    volumes, _ = tagvolume.lookup(["DeepSeaFish", "Ocean"],
                                  cache=tagvolume.VolumeCache(cache))

    assert volumes["DeepSeaFish"].source == "estimate"
    assert volumes["Ocean"].source == "estimate"


def test_an_unreadable_cache_file_is_not_fatal(tmp_path):
    broken = tmp_path / "vol.json"
    broken.write_text("{not json", encoding="utf-8")

    volumes, _ = tagvolume.lookup(["Ocean"],
                                  cache=tagvolume.VolumeCache(broken))

    assert volumes["Ocean"].source == "estimate"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def test_a_provider_answer_is_used_and_written_back(tmp_path):
    cache = tagvolume.VolumeCache(tmp_path / "vol.json")

    volumes, warnings = tagvolume.lookup(
        ["Piezophile"], cache=cache,
        provider=lambda tags: {"Piezophile": 250_000},
    )

    assert volumes["Piezophile"].source == "provider"
    assert volumes["Piezophile"].count == 250_000
    assert warnings == []
    # and the next run does not have to ask again
    saved = json.loads((tmp_path / "vol.json").read_text(encoding="utf-8"))
    assert saved["tags"]["piezophile"]["count"] == 250_000


def test_a_provider_that_skips_a_tag_leaves_it_to_the_estimator(tmp_path):
    volumes, _ = tagvolume.lookup(
        ["Ocean", "Piezophile"],
        cache=tagvolume.VolumeCache(tmp_path / "vol.json"),
        provider=lambda tags: {"Ocean": 40_000_000},
    )

    assert volumes["Ocean"].source == "provider"
    assert volumes["Piezophile"].source == "estimate"


def test_a_provider_that_throws_is_a_warning_not_a_failure(tmp_path):
    def broken(tags):
        raise RuntimeError("rate limited")

    volumes, warnings = tagvolume.lookup(
        ["DeepSeaFish"], cache=tagvolume.VolumeCache(tmp_path / "vol.json"),
        provider=broken,
    )

    assert volumes["DeepSeaFish"].source == "estimate"
    assert any("rate limited" in w for w in warnings)


def test_a_provider_returning_rubbish_is_ignored_per_tag(tmp_path):
    volumes, _ = tagvolume.lookup(
        ["Ocean", "DeepSeaFish"],
        cache=tagvolume.VolumeCache(tmp_path / "vol.json"),
        provider=lambda tags: {"Ocean": "millions", "DeepSeaFish": -5},
    )

    assert volumes["Ocean"].source == "estimate"
    assert volumes["DeepSeaFish"].source == "estimate"


def test_a_misconfigured_provider_path_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TAG_VOLUME_PROVIDER", "nosuchmodule.lookup")

    volumes, warnings = tagvolume.lookup(
        ["Ocean"], cache=tagvolume.VolumeCache(tmp_path / "vol.json")
    )

    assert volumes["Ocean"].source == "estimate"
    assert any("nosuchmodule" in w for w in warnings)


def test_a_provider_path_that_is_not_a_dotted_path_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_PROVIDER", "lookup")

    with pytest.raises(tagvolume.VolumeError):
        tagvolume.load_provider()


def test_no_provider_configured_means_no_provider():
    assert tagvolume.load_provider("") is None


# ---------------------------------------------------------------------------
# Writing a count down
#
# The cache was reachable only by hand-authoring its JSON, which is why it did
# not exist and every band in every post was an estimate. These cover the path
# that makes it reachable, and the one thing that matters about it: a seeded
# figure outranks the estimator and no TTL ever takes it back.
# ---------------------------------------------------------------------------

def test_a_count_is_read_the_way_instagram_prints_it():
    assert tagvolume.read_count("4.2M") == 4_200_000
    assert tagvolume.read_count("430K") == 430_000
    assert tagvolume.read_count("9,500") == 9_500
    assert tagvolume.read_count("12000") == 12_000
    # what you get if the whole label is pasted in rather than retyped
    assert tagvolume.read_count("1.2M posts") == 1_200_000


@pytest.mark.parametrize("raw", ["", "lots", "-5", "4.2Q"])
def test_a_count_that_is_not_a_count_is_refused(raw):
    # Better a refusal than a zero: a tag seeded at 0 reads as dead for ever.
    with pytest.raises(tagvolume.VolumeError):
        tagvolume.read_count(raw)


def test_a_seeded_entry_carries_no_timestamp(tmp_path):
    path = tmp_path / "vol.json"
    cache = tagvolume.VolumeCache(path)
    cache.seed("MarineBiology", 4_200_000)
    assert cache.save() == []

    entry = json.loads(path.read_text(encoding="utf-8"))["tags"]["marinebiology"]
    assert entry == {"count": 4_200_000, "source": "measured"}
    assert "at" not in entry             # a TTL must never reach this


def test_a_seeded_count_moves_the_tag_into_the_band_it_belongs_in(tmp_path):
    # The estimator prices a tag by how ordinary its words are, so a common
    # term of art reads far smaller than it is: #MarineBiology estimates into
    # the low hundreds of thousands, which is the one band a small account is
    # told it can rank in. This is what seeding is for.
    cache = tagvolume.VolumeCache(tmp_path / "vol.json")
    before, _ = tagvolume.lookup(["MarineBiology"], cache=cache)
    assert before["MarineBiology"].source == "estimate"

    cache.seed("MarineBiology", 4_200_000)
    after, _ = tagvolume.lookup(["MarineBiology"], cache=cache)

    assert after["MarineBiology"].source == "measured"
    assert after["MarineBiology"].band != before["MarineBiology"].band


def test_the_set_flag_writes_counts_and_leaves_everything_else_estimated(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_FILE", tmp_path / "vol.json")

    assert tagvolume.main(["--set", "marinebiology=4.2M", "#trees=9,500"]) == 0

    volumes, _ = tagvolume.lookup(["marinebiology", "trees", "botany"])
    assert volumes["marinebiology"].count == 4_200_000
    assert volumes["trees"].count == 9_500          # the leading # is not part
    assert volumes["botany"].source == "estimate"   # untouched, still a guess


@pytest.mark.parametrize("pair", ["marinebiology", "=42", "marinebiology=lots"])
def test_a_malformed_pair_writes_nothing_at_all(pair, tmp_path, monkeypatch):
    # Half a seeding run is worse than none: the operator would have no way of
    # knowing which of the tags they typed had landed.
    path = tmp_path / "vol.json"
    monkeypatch.setattr(config, "TAG_VOLUME_FILE", path)

    assert tagvolume.main(["--set", pair]) == 2
    assert not path.exists()

    # and the same when a good pair precedes the bad one - the cache is saved
    # once, after every pair has parsed, so a run either lands or it does not
    assert tagvolume.main(["--set", "oceanlife=430K", pair]) == 2
    assert not path.exists()


def test_the_bare_usage_line_mentions_both_modes(capsys):
    assert tagvolume.main([]) == 2
    printed = capsys.readouterr().err
    assert "--set" in printed and "<tag> [tag ...]" in printed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# The seeding worklist
# ---------------------------------------------------------------------------

def _post(directory: Path, day: str, run: str, *, tags=(), rejected=()) -> None:
    run_dir = directory / day / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "post.json").write_text(json.dumps({
        "hashtags": ["#" + t for t in tags],
        "hashtag_rejected": [{"tag": t, "reason": "below the volume floor",
                              "count": n, "source": "estimate"}
                             for t, n in rejected],
    }), encoding="utf-8")


@pytest.fixture()
def output_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "TAG_VOLUME_FILE", tmp_path / "volumes.json")
    (tmp_path / "out").mkdir()
    return tmp_path / "out"


def test_the_worklist_ranks_what_was_thrown_out_first(output_tree):
    """Following "seed real counts" used to mean knowing which tags to seed."""
    _post(output_tree, "2026-09-08", "a", tags=("ecology", "planetearth"),
          rejected=(("mycorrhiza", 900), ("suzannesimard", 120)))
    _post(output_tree, "2026-09-09", "b", tags=("ecology",),
          rejected=(("mycorrhiza", 900),))

    rows, total = tagvolume.worklist()

    assert total == 4      # ecology is one tag, used twice
    assert rows[0]["tag"] == "mycorrhiza"
    assert rows[0]["rejected"] == 2
    assert rows[0]["count"] == 900
    # Then whatever the account leans on most.
    assert [r["tag"] for r in rows[2:]] == ["ecology", "planetearth"]
    assert rows[2]["used"] == 2


def test_a_tag_already_seeded_by_hand_is_off_the_worklist(output_tree):
    _post(output_tree, "2026-09-08", "a", tags=("ecology", "planetearth"))
    cache = tagvolume.VolumeCache()
    cache.seed("ecology", 4_200_000)
    cache.save()

    rows, total = tagvolume.worklist()

    assert total == 2
    assert [r["tag"] for r in rows] == ["planetearth"]


def test_a_banned_tag_is_not_something_to_go_and_measure(output_tree):
    run_dir = output_tree / "2026-09-08" / "a"
    run_dir.mkdir(parents=True)
    (run_dir / "post.json").write_text(json.dumps({
        "hashtags": ["#ecology"],
        "hashtag_rejected": [{"tag": "sponsored", "reason": "banned"}],
    }), encoding="utf-8")

    rows, _ = tagvolume.worklist()

    assert [r["tag"] for r in rows] == ["ecology"]


def test_the_worklist_says_so_when_there_is_nothing_to_read(output_tree, capsys):
    assert tagvolume.print_worklist() == 0
    assert "make a reel first" in capsys.readouterr().out
