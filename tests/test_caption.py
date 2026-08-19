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
from pipeline import caption  # noqa: E402
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


def test_tags_are_drawn_from_every_tier():
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
