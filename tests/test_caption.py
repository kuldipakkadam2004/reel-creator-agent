"""Tests for pipeline/caption.py.

Everything here is offline: Gemini is never called, and the generator runs in
`heuristic` mode or against a stubbed draft. What is covered is the part that
decides things - how a hook is cut to the visible limit, how tags are
normalised and mixed, what a licence-required credit does to the caption, and
what happens when the copy runs past 2200 characters.

The model's own writing is checked by running it:

    python pipeline/caption.py -r output/<date>/<run id>

    python -m pytest tests/test_caption.py -v
    python tests/test_caption.py          # works without pytest installed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import caption, tagvolume  # noqa: E402
from pipeline.caption import Draft  # noqa: E402


SCRIPT = (
    "Imagine living in total darkness. That is the deep ocean. Sunlight "
    "vanishes after about 650 feet. So how do creatures see? They make their "
    "own light. It is called bioluminescence. About 80 percent of deep-sea "
    "animals glow. Some use it to hunt."
)


def make_credit(required: bool, provider: str = "pexels", **overrides) -> dict:
    credit = {
        "provider": provider,
        "creator": "A Person",
        "license": "CC BY 4.0" if required else "Pexels License",
        "attribution": f'"video" by A Person via {provider}',
        "attribution_required": required,
        "kind": "video",
    }
    credit.update(overrides)
    return credit


def stub_draft(monkeypatch, draft: Draft) -> None:
    """Answer the model with a fixed draft, so no key and no network is used."""
    monkeypatch.setattr(caption, "llm_draft",
                        lambda *args, **kwargs: (draft, []))


# ---------------------------------------------------------------------------
# The hook - the only part Instagram shows before "... more"
# ---------------------------------------------------------------------------

def test_a_hook_that_fits_is_left_alone():
    hook, warnings = caption.fit_hook("Most of the ocean has never been seen.", 125)

    assert hook == "Most of the ocean has never been seen."
    assert warnings == []


def test_a_long_hook_is_cut_at_a_sentence_not_mid_thought():
    text = ("Sunlight stops at 650 feet. Below that the ocean makes its own "
            "light, and it has been doing it for millions of years.")

    hook, warnings = caption.fit_hook(text, 60)

    assert hook == "Sunlight stops at 650 feet."
    assert warnings and "sentence" in warnings[0]


def test_a_single_long_sentence_is_cut_at_a_clause_boundary():
    text = ("The ocean is dark and cold, and it goes on for miles beneath the "
            "waves where nothing ever sees the sun.")

    hook, _ = caption.fit_hook(text, 60)

    assert hook == "The ocean is dark and cold"
    assert len(hook) <= 60


def test_a_hook_is_never_left_ending_on_a_dangling_connector():
    text = "Ninety percent of deep-sea animals glow in the dark, and"

    hook, _ = caption.fit_hook(text, 125)

    assert hook == "Ninety percent of deep-sea animals glow in the dark"


def test_a_finished_sentence_keeps_its_last_word():
    # "for." only looks like a dangling connector until you see the full stop.
    hook, _ = caption.fit_hook("That is what the light is for.", 125)

    assert hook == "That is what the light is for."


def test_a_hook_with_no_break_at_all_is_cut_on_a_word_and_flagged():
    text = "a" * 40 + " " + "b" * 40

    hook, warnings = caption.fit_hook(text, 50)

    assert hook == "a" * 40
    assert warnings and "word boundary" in warnings[0]


def test_a_hook_that_only_repeats_the_narration_is_reported(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(
        hook="Imagine living in total darkness.",
        body=["Something else entirely."],
        hashtags={"mid": ["ocean"]},
    ))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False)

    assert any("repeats the narration" in w for w in post["warnings"])


# ---------------------------------------------------------------------------
# Hashtags
# ---------------------------------------------------------------------------

def test_tags_are_stripped_of_anything_instagram_would_refuse():
    assert caption.normalise_tag("#Deep Sea!") == "DeepSea"
    assert caption.normalise_tag("  marine-biology  ") == "marinebiology"
    assert caption.normalise_tag("123") is None          # a number is not a tag
    assert caption.normalise_tag("a") is None
    assert caption.normalise_tag("x" * 40) is None       # nobody searches that


def test_the_mix_is_split_across_tiers_and_survives_a_changed_count():
    assert caption.tag_quota(8) == {"broad": 2, "mid": 4, "niche": 2}
    assert sum(caption.tag_quota(12).values()) == 12
    assert sum(caption.tag_quota(3).values()) == 3
    assert caption.tag_quota(0) == {"broad": 0, "mid": 0, "niche": 0}


def test_tags_are_drawn_from_every_tier(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_ENABLED", False)
    tiers = {
        "broad": ["ocean", "science", "nature"],
        "mid": ["bioluminescence", "deepsea", "anglerfish", "marinebiology"],
        "niche": ["abyssalzone", "luciferase", "midnightzone"],
    }

    tags, tier_of, _ = caption.select_hashtags(tiers, count=8)

    assert len(tags) == 8
    assert set(tier_of.values()) == {"broad", "mid", "niche"}
    assert list(tier_of.values()).count("mid") == 4


def test_always_on_tags_come_first_and_are_not_dropped():
    tiers = {"broad": ["ocean"], "mid": ["deepsea", "glow"], "niche": ["abyss"]}

    tags, tier_of, _ = caption.select_hashtags(
        tiers, count=3, always=("#MyBrand",)
    )

    assert tags[0] == "MyBrand"
    assert tier_of["MyBrand"] == "always"
    assert len(tags) == 3


def test_banned_tags_and_banned_words_never_make_it_into_the_set():
    tiers = {"broad": ["follow4follow", "ocean"], "mid": ["cryptoocean"],
             "niche": ["abyss"]}

    tags, _, warnings = caption.select_hashtags(
        tiers, count=4, banned_tags=("follow4follow",), banned_words=("crypto",)
    )

    assert "follow4follow" not in tags
    assert "cryptoocean" not in tags
    assert any("banned" in w for w in warnings)


def test_a_tag_is_only_used_once_however_many_tiers_offer_it():
    tiers = {"broad": ["ocean"], "mid": ["Ocean", "OCEAN"], "niche": ["ocean"]}

    tags, _, _ = caption.select_hashtags(tiers, count=8)

    assert len(tags) == 1


def test_a_recently_used_tag_gives_way_to_a_fresh_one():
    tiers = {"broad": [], "mid": ["lastweek", "thisweek"], "niche": []}

    tags, _, _ = caption.select_hashtags(tiers, count=1, recent=("lastweek",))

    assert tags == ["thisweek"]


def test_a_recent_tag_is_still_used_when_nothing_fresher_exists():
    tiers = {"broad": [], "mid": ["lastweek"], "niche": []}

    tags, _, warnings = caption.select_hashtags(
        tiers, count=1, recent=("lastweek",)
    )

    assert tags == ["lastweek"]              # relevance beats novelty
    assert any("not varying" in w for w in warnings)


def test_running_short_of_tags_is_reported_rather_than_padded():
    tags, _, warnings = caption.select_hashtags(
        {"broad": ["ocean"], "mid": [], "niche": []}, count=8
    )

    assert tags == ["ocean"]
    assert any("only 1 of 8" in w for w in warnings)


# ---------------------------------------------------------------------------
# Volume banding
#
# The generator's tiers are a guess about words. These cover what happens once
# every tag is priced: see pipeline/tagvolume.py and tests/test_tagvolume.py.
# ---------------------------------------------------------------------------

def vol(tag: str, count: int, source: str = "measured"):
    return tagvolume.Volume(tag, count, source, tagvolume.band_of(count))


def test_a_tag_too_small_to_be_found_is_dropped():
    tiers = {"broad": ["OceanLife"], "mid": ["DeepSeaFish"],
             "niche": ["Piezophile"]}
    volumes = {"OceanLife": vol("OceanLife", 4_000_000),
               "DeepSeaFish": vol("DeepSeaFish", 200_000),
               "Piezophile": vol("Piezophile", 180)}

    tags, _, warnings = caption.select_hashtags(tiers, count=3,
                                                volumes=volumes)

    assert "Piezophile" not in tags
    assert any("too few posts" in w for w in warnings)


def test_a_tag_is_placed_by_what_it_measures_not_by_what_it_was_called():
    # The model filed a four-million-post tag under "niche". It is not niche.
    tiers = {"broad": [], "mid": [], "niche": ["Bioluminescence", "LavaTubes"]}
    volumes = {"Bioluminescence": vol("Bioluminescence", 4_000_000),
               "LavaTubes": vol("LavaTubes", 60_000)}

    _, tier_of, _ = caption.select_hashtags(tiers, count=2, volumes=volumes)

    assert tier_of["Bioluminescence"] == "broad"
    assert tier_of["LavaTubes"] == "niche"


def test_only_so_many_of_the_hugest_tags_get_in(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_MAX_SATURATED", 1)
    tiers = {"broad": ["Ocean", "Nature", "Science"], "mid": ["DeepSeaFish"],
             "niche": ["LavaTubes"]}
    volumes = {"Ocean": vol("Ocean", 40_000_000),
               "Nature": vol("Nature", 90_000_000),
               "Science": vol("Science", 30_000_000),
               "DeepSeaFish": vol("DeepSeaFish", 200_000),
               "LavaTubes": vol("LavaTubes", 60_000)}

    tags, tier_of, _ = caption.select_hashtags(tiers, count=5,
                                               volumes=volumes)

    huge = [t for t in tags if tier_of[t] == "saturated"]
    assert len(huge) == 1
    # and the slots it did not take went to tags that can actually be ranked in
    assert "DeepSeaFish" in tags and "LavaTubes" in tags


def test_an_always_on_tag_is_never_touched_by_the_band_filter():
    # POST_HASHTAGS_ALWAYS is a series or brand tag. It is small by definition
    # and that is not a mistake to be corrected.
    tiers = {"broad": ["OceanLife"], "mid": [], "niche": []}
    volumes = {"OceanLife": vol("OceanLife", 4_000_000)}

    tags, tier_of, _ = caption.select_hashtags(
        tiers, count=2, always=("MyTinySeriesTag",), volumes=volumes
    )

    assert tags[0] == "MyTinySeriesTag"
    assert tier_of["MyTinySeriesTag"] == "always"


def test_the_biggest_of_the_dead_come_back_before_the_block_runs_short(
        monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_FLOOR", 5_000)
    monkeypatch.setattr(config, "TAG_VOLUME_RELAX", 0.2)   # down to 1,000
    tiers = {"broad": ["OceanLife"], "mid": ["Nearly", "Hopeless"], "niche": []}
    volumes = {"OceanLife": vol("OceanLife", 4_000_000),
               "Nearly": vol("Nearly", 3_000),             # dead, but close
               "Hopeless": vol("Hopeless", 12)}            # dead and staying so

    tags, _, warnings = caption.select_hashtags(tiers, count=3,
                                                volumes=volumes)

    assert "Nearly" in tags
    assert "Hopeless" not in tags
    assert any("below the volume floor" in w for w in warnings)


def test_nothing_is_put_back_when_relaxing_is_switched_off(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_RELAX", 1.0)
    tiers = {"broad": ["OceanLife"], "mid": ["Nearly"], "niche": []}
    volumes = {"OceanLife": vol("OceanLife", 4_000_000),
               "Nearly": vol("Nearly", 3_000)}

    tags, _, warnings = caption.select_hashtags(tiers, count=3,
                                                volumes=volumes)

    assert tags == ["OceanLife"]                 # short, and said so
    assert any("only 1 of 3" in w for w in warnings)


def test_a_tag_with_no_reading_keeps_the_tier_it_came_in_with():
    # No measurement is not evidence against a tag.
    tiers = {"broad": [], "mid": ["Unmeasured"], "niche": []}

    tags, tier_of, _ = caption.select_hashtags(tiers, count=1, volumes={})

    assert tags == ["Unmeasured"]
    assert tier_of["Unmeasured"] == "mid"


def test_banding_off_leaves_the_generator_s_tiers_alone(monkeypatch):
    monkeypatch.setattr(config, "TAG_VOLUME_ENABLED", False)
    tiers = {"broad": [], "mid": [], "niche": ["Piezophile"]}

    tags, tier_of, warnings = caption.select_hashtags(tiers, count=1)

    assert tags == ["Piezophile"]
    assert tier_of["Piezophile"] == "niche"
    assert not any("too few posts" in w for w in warnings)


def test_the_post_records_what_every_tag_measured_at(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(
        hook="Hook.", body=["Body."],
        hashtags={"broad": ["OceanLife"], "mid": ["DeepSeaFish"],
                  "niche": ["Piezophile"]},
    ))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False,
                                    hashtag_count=3)

    readings = post["hashtag_volumes"]
    assert readings                                  # one per surviving tag
    assert set(readings) == set(post["hashtags"])
    assert all(r["band"] in tagvolume.USABLE for r in readings.values())
    assert all(r["source"] == "estimate" for r in readings.values())
    assert "#Piezophile" not in post["hashtags"]


# ---------------------------------------------------------------------------
# The ledger behind the "not the same block every post" rule
# ---------------------------------------------------------------------------

def test_the_ledger_reports_what_recent_runs_used(tmp_path):
    ledger = caption.HashtagLedger(tmp_path / "usage.json")
    ledger.record("run-1", ["ocean", "glow"])
    ledger.record("run-2", ["kelp"])
    ledger.save()

    assert caption.HashtagLedger(tmp_path / "usage.json").recent(5) == [
        "ocean", "glow", "kelp"
    ]


def test_a_rerun_does_not_treat_its_own_tags_as_recently_used(tmp_path):
    ledger = caption.HashtagLedger(tmp_path / "usage.json")
    ledger.record("run-1", ["ocean"])
    ledger.record("run-2", ["kelp"])

    assert ledger.recent(5, exclude="run-1") == ["kelp"]


def test_history_can_be_switched_off(tmp_path):
    ledger = caption.HashtagLedger(tmp_path / "usage.json")
    ledger.record("run-1", ["ocean"])

    assert ledger.recent(0) == []


# ---------------------------------------------------------------------------
# Attribution carried through from step 4
# ---------------------------------------------------------------------------

def test_a_licence_that_demands_credit_gets_it_in_the_caption():
    credits, _ = caption.build_credits(
        [make_credit(True, "openverse",
                     attribution='"Kelp" by Ada (CC BY 4.0) - https://x.test')],
        "required", header="Footage:", maximum=6,
    )

    block = credits.block()
    assert "Footage:" in block
    assert '"Kelp" by Ada (CC BY 4.0) - https://x.test' in block


def test_stock_that_needs_no_credit_adds_nothing_by_default():
    credits, _ = caption.build_credits(
        [make_credit(False), make_credit(False, "pixabay")],
        "required", header="Footage:", maximum=6,
    )

    assert credits.block() == ""


def test_crediting_everything_collapses_the_optional_ones_into_one_line():
    credits, _ = caption.build_credits(
        [make_credit(False), make_credit(False, "pixabay")],
        "all", header="Footage:", maximum=6,
    )

    assert credits.block() == "Footage: via Pexels, Pixabay"


def test_attribution_off_means_off():
    credits, _ = caption.build_credits(
        [make_credit(True)], "off", header="Footage:", maximum=6,
    )

    assert credits.block() == ""


def test_more_required_credits_than_the_cap_are_kept_and_flagged():
    given = [make_credit(True, attribution=f'"clip {n}" by Someone (CC BY 4.0)')
             for n in range(5)]

    credits, warnings = caption.build_credits(
        given, "required", header="Footage:", maximum=2,
    )

    assert len(credits.required) == 5           # dropping one breaches the licence
    assert any("POST_ATTRIBUTION_MAX" in w for w in warnings)


def test_a_credit_is_rebuilt_when_step_4_left_no_ready_made_line():
    line = caption._credit_line({
        "creator": "Ada", "provider": "openverse", "license": "CC BY 4.0",
        "source_url": "https://x.test", "kind": "image",
    })

    assert line == '"image" by Ada via openverse (CC BY 4.0) - https://x.test'


# ---------------------------------------------------------------------------
# Composition and the 2200-character limit
# ---------------------------------------------------------------------------

def build(draft: Draft, **kwargs):
    settings = {
        "tags": ["ocean"],
        "credits": caption.Credits(),
        "must_include": [],
        "placement": "caption",
        "limit": 2200,
    }
    settings.update(kwargs)
    return caption.compose(draft, **settings)


def test_the_caption_reads_hook_body_cta_then_tags():
    text, _, _ = build(Draft(hook="Hook.", body=["Body."], cta="CTA?"))

    assert text == "Hook.\n\nBody.\n\nCTA?\n\n#ocean"


def test_tags_stay_out_of_the_caption_when_they_belong_in_a_comment():
    text, parts, _ = build(Draft(hook="Hook.", body=["Body."]),
                           placement="comment")

    assert "#ocean" not in text
    assert parts["hashtags"] == ["ocean"]


def test_the_courtesy_credit_is_the_first_thing_given_up_on_length():
    credits = caption.Credits(header="Footage:",
                              required=['"Kelp" by Ada (CC BY 4.0)'],
                              courtesy="via Pexels")
    draft = Draft(hook="Hook.", body=["x" * 200, "y" * 200], cta="CTA?")

    text, _, warnings = build(draft, credits=credits, limit=460)

    assert "via Pexels" not in text
    assert '"Kelp" by Ada (CC BY 4.0)' in text        # the licence still holds
    assert any("courtesy credit" in w for w in warnings)


def test_body_paragraphs_go_before_the_cta_does():
    draft = Draft(hook="Hook.", body=["x" * 100, "y" * 100], cta="CTA?")

    text, parts, _ = build(draft, limit=130)

    assert len(parts["body"]) == 1
    assert "CTA?" in text


def test_the_hook_and_a_required_credit_are_never_traded_away():
    credits = caption.Credits(header="Footage:", required=["z" * 300])
    draft = Draft(hook="Hook.", body=["x" * 100], cta="CTA?")

    text, parts, _ = build(draft, credits=credits, limit=340)

    assert text.startswith("Hook.")
    assert "z" * 300 in text
    assert parts["cta"] == ""


def test_a_caption_that_cannot_fit_at_all_is_cut_and_says_so():
    credits = caption.Credits(header="Footage:", required=["z" * 300])

    text, _, warnings = build(Draft(hook="Hook."), credits=credits, limit=100)

    assert len(text) <= 100
    assert any("cut" in w for w in warnings)


# ---------------------------------------------------------------------------
# Emoji, banned words, must-include
# ---------------------------------------------------------------------------

def test_the_emoji_budget_is_spent_on_the_hook_first():
    fields, warnings = caption.apply_emoji_budget(
        ["Hook \U0001f30a", "Body \U0001f420", "CTA \U0001f419"],
        allowed=True, maximum=1,
    )

    assert fields[0] == "Hook \U0001f30a"
    assert "\U0001f420" not in fields[1] and "\U0001f419" not in fields[2]
    assert warnings


def test_emoji_can_be_switched_off_entirely():
    fields, _ = caption.apply_emoji_budget(
        ["Hook \U0001f30a"], allowed=False, maximum=3
    )

    assert fields == ["Hook"]


def test_banned_words_are_caught_after_generation_not_just_asked_about():
    draft = Draft(hook="Clean.", body=["This is a scam."], cta="")

    assert caption.banned_hits(draft, ("scam",)) == ["scam"]
    assert caption.banned_hits(draft, ("scams",)) == []     # whole words only


def test_a_paragraph_using_a_banned_word_is_dropped_but_a_hook_is_reported():
    draft = Draft(hook="A scam.", body=["Fine.", "Also a scam."], cta="Scam?")

    scrubbed, warnings = caption.scrub_banned(draft, ("scam",))

    assert scrubbed.body == ["Fine."]
    assert scrubbed.cta == ""
    assert scrubbed.hook == "A scam."                        # not mangled
    assert any("fix it by hand" in w.lower() for w in warnings)


def test_a_must_include_phrase_is_appended_when_the_copy_leaves_it_out(
        monkeypatch, tmp_path):
    monkeypatch.setattr(config, "POST_MUST_INCLUDE", ("Follow @DeepDive",))
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False)

    assert "Follow @DeepDive" in post["caption"]
    assert any("must-include" in w for w in post["warnings"])


def test_a_must_include_phrase_already_written_in_is_not_repeated(
        monkeypatch, tmp_path):
    monkeypatch.setattr(config, "POST_MUST_INCLUDE", ("Follow @DeepDive",))
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Follow @DeepDive for more."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False)

    assert post["caption"].count("Follow @DeepDive") == 1


# ---------------------------------------------------------------------------
# What the module can be handed, and what it leaves behind
# ---------------------------------------------------------------------------

def test_a_script_dict_a_file_a_directory_and_a_string_all_work(tmp_path):
    script_json = tmp_path / "ocean.json"
    script_json.write_text(json.dumps({"topic": "the ocean", "script": SCRIPT}),
                           encoding="utf-8")
    (tmp_path / "ocean.txt").write_text(SCRIPT, encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "assembly.json").write_text(
        json.dumps({"assembly_version": 1, "run_id": "r-1", "topic": "the ocean",
                    "script": SCRIPT, "attributions": [make_credit(False)]}),
        encoding="utf-8",
    )

    assert caption.resolve_source({"topic": "t", "script": SCRIPT}).script == SCRIPT
    assert caption.resolve_source(script_json).topic == "the ocean"
    assert caption.resolve_source(tmp_path / "ocean.txt").topic == "the ocean"
    assert caption.resolve_source(SCRIPT).script == SCRIPT

    from_run = caption.resolve_source(run)
    assert from_run.run_id == "r-1"
    assert from_run.run_dir == run
    assert len(from_run.credits) == 1


def test_a_directory_with_nothing_to_caption_says_so(tmp_path):
    with pytest.raises(caption.CaptionError, match="assembly.json"):
        caption.resolve_source(tmp_path)


def test_an_empty_script_is_refused():
    with pytest.raises(caption.CaptionError, match="nothing to caption"):
        caption.resolve_source({"script": "   "})


def test_the_post_lands_in_the_run_directory_beside_the_video(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."], cta="CTA?",
                                  hashtags={"mid": ["ocean", "deepsea"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, run_id="r-1")

    text_path = tmp_path / caption.POST_TEXT_NAME
    assert text_path.read_text(encoding="utf-8").rstrip("\n") == post["caption"]
    assert json.loads((tmp_path / caption.POST_JSON_NAME).read_text(
        encoding="utf-8"))["run_id"] == "r-1"
    assert not (tmp_path / caption.POST_COMMENT_NAME).exists()


def test_the_tag_block_gets_its_own_file_when_it_belongs_in_a_comment(
        monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, run_id="r-1",
                                    placement="comment")

    assert "#ocean" not in (tmp_path / caption.POST_TEXT_NAME).read_text(
        encoding="utf-8")
    assert (tmp_path / caption.POST_COMMENT_NAME).read_text(
        encoding="utf-8").strip() == post["first_comment"]


def test_a_dry_run_writes_nothing(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False)

    assert post["post_text_path"] is None
    assert list(tmp_path.iterdir()) == []


def test_a_failed_model_call_still_produces_a_caption(monkeypatch, tmp_path):
    monkeypatch.setattr(caption, "llm_draft",
                        lambda *a, **k: (None, ["Gemini fell over"]))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption({"topic": "the deep ocean", "script": SCRIPT},
                                    run_dir=tmp_path, write=False)

    assert post["mode"] == "heuristic"
    assert post["caption"]
    assert post["hashtags"]
    assert any("keyword extraction" in w for w in post["warnings"])


def test_the_hard_limits_are_the_platform_s_not_ours(monkeypatch, tmp_path):
    # About Instagram's cap of 30, not about volume banding - tag0..tag39 are
    # coinages and the band filter is quite right to refuse every one of them.
    monkeypatch.setattr(config, "TAG_VOLUME_ENABLED", False)
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": [f"tag{n}" for n in range(40)]}))

    post = caption.generate_caption(SCRIPT, run_dir=tmp_path, write=False,
                                    hashtag_count=40)

    assert len(post["hashtags"]) == config.POST_HASHTAG_LIMIT
    assert post["chars"]["caption"] <= config.POST_MAX_CHARS
    assert any("at most 30" in w for w in post["warnings"])


def test_the_heuristic_fallback_keeps_contractions_out_of_the_tags():
    # "It's" used to survive as the tag #its, which says nothing about anything.
    assert "its" not in caption.keywords("It's dark. It's cold. It's the deep.")


def test_long_tail_tags_are_phrases_that_actually_occur_together():
    found = caption.phrases("The deep ocean is in total darkness. Total darkness.")

    assert "totaldarkness" in found
    assert "deepocean" in found


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Alt text for step 5's cover frame
# ---------------------------------------------------------------------------

def make_cover(**overrides) -> dict:
    """Step 5's cover record, shaped exactly like the real one."""
    cover = {
        "path": "/nowhere/cover.jpg",
        "filename": "cover.jpg",
        "written": True,
        "source": "hook_card",
        "mode": "auto",
        "reason": "the reel opens on a hook card",
        "time": 0.84,
        "source_time": 2.52,
        "shot_index": 0,
        "beat_index": 0,
        "asset_index": 0,
        "file": "beat00_0_pexels_1.mp4",
        "kind": "video",
        "hook_card_text": "Snailfish: extreme pressure",
        "depicts": "medusa blue ocean",
        "depicts_source": "source_url",
        "safe_aspect": "4:5",
        "card_in_safe_band": True,
        "score": 6.1,
        "width": 1080,
        "height": 1920,
        "bytes": 97035,
    }
    cover.update(overrides)
    return cover


def alt_for(cover: dict, limit: int = 100, **kwargs) -> str:
    return caption.build_alt_text(cover, limit=limit, **kwargs)[0]


def test_alt_text_says_what_the_picture_is_and_what_it_says():
    text = alt_for(make_cover())
    assert "Medusa blue ocean" in text
    assert "Snailfish: extreme pressure" in text


def test_the_text_burned_into_the_cover_is_quoted_exactly():
    """A screen reader user gets the headline from nowhere else."""
    text = alt_for(make_cover(hook_card_text='Snailfish: "extreme" pressure'))
    assert 'Snailfish: "extreme" pressure' in text


def test_a_footage_cover_is_described_without_inventing_a_headline():
    text = alt_for(make_cover(hook_card_text="", source="footage"))
    assert text == "Medusa blue ocean."
    assert "Text on the image" not in text


def test_a_cover_with_nothing_but_a_headline_still_reports_the_headline():
    text = alt_for(make_cover(depicts=""))
    assert text == 'Text on the image: "Snailfish: extreme pressure".'


def test_no_cover_means_no_alt_text_rather_than_a_guess():
    text, report, warnings = caption.build_alt_text({}, limit=100)
    assert text == ""
    assert report["source"] == "none"
    assert warnings == []


def test_a_cover_that_describes_nothing_is_reported():
    text, _, warnings = caption.build_alt_text(
        make_cover(depicts="", hook_card_text=""), limit=100
    )
    assert text == ""
    assert any("cannot reach" in w for w in warnings)


def test_the_description_and_the_headline_stay_separate_sentences():
    """There is no join that reads correctly for both casings it arrives in.

    A provider slug is lower case ("medusa blue ocean"); the model returns a
    capitalised sentence. "... over A jellyfish drifts" is the result of
    gluing them together.
    """
    literal = alt_for(make_cover())
    rewritten = alt_for(make_cover(), mode="llm",
                        rewritten="A jellyfish drifts in dark blue water")
    for text in (literal, rewritten):
        assert " over " not in text
        assert text.endswith('.')
    assert rewritten.startswith("A jellyfish drifts in dark blue water.")


def test_the_model_rephrases_the_picture_but_never_the_quoted_headline():
    text, report, _ = caption.build_alt_text(
        make_cover(), limit=100, mode="llm",
        rewritten="A jellyfish drifts in dark blue water",
    )
    assert "medusa" not in text.lower()
    assert "Snailfish: extreme pressure" in text
    assert report["source"] == "llm"


def test_an_empty_rewrite_falls_back_to_step_fives_own_words():
    text, report, _ = caption.build_alt_text(
        make_cover(), limit=100, mode="llm", rewritten="",
    )
    assert "Medusa blue ocean" in text
    assert report["source"] == "source_url"


def test_alt_text_is_held_to_the_limit():
    text = alt_for(make_cover(depicts="a " + "very " * 40 + "long description"))
    assert len(text) <= 100


def test_the_headline_survives_the_trim_and_the_description_gives_way():
    cover = make_cover(depicts="a " + "very " * 40 + "long description")
    text, _, warnings = caption.build_alt_text(cover, limit=100)
    assert "Snailfish: extreme pressure" in text
    assert len(text) <= 100
    assert any("kept ahead of" in w for w in warnings)


def test_a_trim_never_cuts_a_word_in_half():
    text = alt_for(make_cover(depicts="jellyfish " * 30))
    assert not text.replace(".", "").endswith("jellyf")


def test_a_headline_too_long_for_the_field_is_cut_inside_its_quotes():
    """Trimming the sentence instead says there are words without saying which."""
    text = alt_for(make_cover(hook_card_text="word " * 40, depicts="blue ocean"))
    assert text.startswith('Text on the image: "Word')
    assert len(text) <= 100


def test_the_first_letter_is_raised_without_flattening_the_rest():
    # .capitalize() would turn a provider's own "Blue Ocean Medusa" into
    # "Blue ocean medusa", which reads as though it had been retyped badly.
    assert caption._sentence_case("blue Ocean Medusa") == "Blue Ocean Medusa"


# --- how it reaches post.json ----------------------------------------------

def test_step_five_s_cover_is_read_out_of_the_assembly():
    resolved = caption.resolve_source({
        "script": SCRIPT, "topic": "the deep ocean",
        "assembly_version": 2, "cover": make_cover(),
    })
    assert resolved.cover["depicts"] == "medusa blue ocean"


def test_an_assembly_from_before_covers_existed_reads_as_no_cover():
    resolved = caption.resolve_source({"script": SCRIPT, "assembly_version": 1})
    assert resolved.cover == {}


def test_the_post_carries_the_alt_text_and_its_own_file(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption(
        {"script": SCRIPT, "topic": "the deep ocean", "cover": make_cover()},
        run_dir=tmp_path, run_id="r-1",
    )

    assert "Snailfish: extreme pressure" in post["alt_text"]
    assert post["alt"]["chars"] == len(post["alt_text"])
    assert post["alt"]["from_cover"] == "cover.jpg"
    assert (tmp_path / caption.POST_ALT_NAME).read_text(
        encoding="utf-8").strip() == post["alt_text"]


def test_alt_text_stays_out_of_the_caption_itself(monkeypatch, tmp_path):
    """It goes in a different Instagram field, so post.txt must not carry it."""
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption(
        {"script": SCRIPT, "cover": make_cover()}, run_dir=tmp_path, run_id="r-1",
    )

    assert post["alt_text"] not in post["caption"]
    assert "Text on the image" not in (
        tmp_path / caption.POST_TEXT_NAME).read_text(encoding="utf-8")


def test_a_run_with_no_cover_says_so_instead_of_describing_nothing(
        monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption({"script": SCRIPT}, run_dir=tmp_path,
                                    write=False)

    assert post["alt_text"] == ""
    assert post["alt"]["mode"] == "off"
    assert any("no cover frame" in w for w in post["warnings"])


def test_alt_text_can_be_switched_off(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")

    post = caption.generate_caption(
        {"script": SCRIPT, "cover": make_cover()},
        run_dir=tmp_path, run_id="r-1", alt=False,
    )

    assert post["alt_text"] == ""
    assert not (tmp_path / caption.POST_ALT_NAME).exists()


def test_a_stale_alt_file_from_an_earlier_cover_is_cleared(monkeypatch, tmp_path):
    """It would otherwise be pasted against a cover it does not describe."""
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))
    monkeypatch.setattr(config, "POST_HASHTAG_USAGE_FILE", tmp_path / "usage.json")
    stale = tmp_path / caption.POST_ALT_NAME
    stale.write_text("An older cover entirely.\n", encoding="utf-8")

    caption.generate_caption({"script": SCRIPT}, run_dir=tmp_path, run_id="r-1")

    assert not stale.exists()


def test_a_dry_run_writes_no_alt_file(monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption(
        {"script": SCRIPT, "cover": make_cover()}, run_dir=tmp_path, write=False,
    )

    assert post["alt_text"]
    assert post["post_alt_path"] is None
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_alt_mode_is_reported_and_treated_as_literal(
        monkeypatch, tmp_path):
    stub_draft(monkeypatch, Draft(hook="Hook.", body=["Body."],
                                  hashtags={"mid": ["ocean"]}))

    post = caption.generate_caption(
        {"script": SCRIPT, "cover": make_cover()}, run_dir=tmp_path,
        write=False, alt="sideways",
    )

    assert post["alt"]["mode"] == "literal"
    assert any("unknown POST_ALT_MODE" in w for w in post["warnings"])


def test_the_alt_field_is_only_asked_of_the_model_when_it_is_wanted():
    """A literal run must not pay for the extra instruction in the prompt."""
    plain = caption.build_prompt(SCRIPT, topic="t", niche="n", hook_max=125,
                                 count=8, tone="plain", cta="", rules=[])
    asking = caption.build_prompt(SCRIPT, topic="t", niche="n", hook_max=125,
                                  count=8, tone="plain", cta="", rules=[],
                                  alt_depicts="medusa blue ocean")
    assert '"alt"' not in plain
    assert '"alt"' in asking
    assert "medusa blue ocean" in asking


def test_the_model_is_told_not_to_add_what_it_cannot_see():
    asking = caption.build_prompt(SCRIPT, topic="t", niche="n", hook_max=125,
                                  count=8, tone="plain", cta="", rules=[],
                                  alt_depicts="medusa blue ocean")
    assert "Rewrite ONLY those words" in asking
    assert "do not guess" in asking
