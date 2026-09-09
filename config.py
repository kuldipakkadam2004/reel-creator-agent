"""Central configuration for the Reel Creator Agent.

Single source of truth for:
  * project paths (all derived from this file's location — never absolute)
  * secrets loaded from `.env` via python-dotenv
  * per-step required-key groups, so each module can validate only what it needs

Nothing here performs network calls or side effects beyond creating the
output/asset directories.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths — everything is resolved relative to this file so the project can be
# moved to a server without touching any code.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

ASSETS_DIR = BASE_DIR / "assets"
VISUALS_DIR = ASSETS_DIR / "visuals"
FONTS_DIR = ASSETS_DIR / "fonts"

#: Drop background tracks in here. Step 5 picks one per reel and rotates
#: between them. Empty is fine - it just means no music bed.
MUSIC_DIR = ASSETS_DIR / "music"

#: Drop one-shot effects in here, named for what they are: whoosh*.wav rides a
#: cut, impact*.wav lands on the open. Empty is fine, and is how a fresh clone
#: arrives - it just means no sound design.
SFX_DIR = ASSETS_DIR / "sfx"

#: Step 4 keeps downloaded media in a content-addressed cache and gives every
#: run its own directory of hardlinks into it, so re-runs cost no bytes.
VISUALS_CACHE_DIR = VISUALS_DIR / "cache"
VISUALS_RUNS_DIR = VISUALS_DIR / "runs"

#: Cross-run ledger of which stock asset has been used where, so the same clip
#: does not open every video.
VISUALS_USAGE_FILE = VISUALS_DIR / "usage.json"

OUTPUT_DIR = BASE_DIR / "output"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
AUDIO_DIR = OUTPUT_DIR / "audio"
#: Step 0 keeps its ledger of scouted topics here, so the same candidate
#: does not head every board and one already made is not made twice.
TOPICS_DIR = OUTPUT_DIR / "topics"
#: Step-1 scaffolding, kept for anything that wants a flat drop box. Step 5
#: writes finished reels to OUTPUT_DIR/<date>/<run id>/ instead — see
#: video_run_dir() below — so each reel keeps its captions and attribution
#: together for steps 6 and 7.
VIDEO_DIR = OUTPUT_DIR / "video"
CAPTIONS_DIR = OUTPUT_DIR / "captions"
#: Step 8 writes one log per run in here - the entire console output of that
#: run. It is the only trace an unattended 5pm job leaves behind, so it is a
#: managed directory rather than something the runner creates when it feels
#: like it.
LOGS_DIR = OUTPUT_DIR / "logs"

#: Drafts the claim check refused, and the report that refused them. A refused
#: script is the only artefact of a failed run that cannot be reconstructed -
#: step 2 raises and the caller drops it - and three separate attempts to fix
#: step 2's refusal rate were argued from the run log alone, which quotes the
#: finding the draft *started* with and never what the repairs came back with.
#: Kept here so the fourth attempt can read the evidence instead of the log.
REFUSED_DIR = SCRIPTS_DIR / "refused"

_MANAGED_DIRS = (
    ASSETS_DIR,
    VISUALS_DIR,
    VISUALS_CACHE_DIR,
    VISUALS_RUNS_DIR,
    FONTS_DIR,
    MUSIC_DIR,
    SFX_DIR,
    OUTPUT_DIR,
    SCRIPTS_DIR,
    REFUSED_DIR,
    AUDIO_DIR,
    TOPICS_DIR,
    VIDEO_DIR,
    CAPTIONS_DIR,
    LOGS_DIR,
)

load_dotenv(ENV_FILE)


class MissingConfigError(RuntimeError):
    """Raised when a required environment variable is absent or empty."""


# ---------------------------------------------------------------------------
# Env access
# ---------------------------------------------------------------------------

def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an env var. Blank strings count as unset."""
    value = os.getenv(name, "")
    value = value.strip() if value else ""
    if not value:
        if required:
            raise MissingConfigError(
                f"Missing required environment variable {name!r}. "
                f"Add it to {ENV_FILE.name} (see .env.example)."
            )
        return default
    return value


def require(*names: str) -> dict[str, str]:
    """Assert every named var is set; return them as a dict.

    Reports all missing keys at once instead of failing on the first.
    """
    missing = [n for n in names if not (os.getenv(n) or "").strip()]
    if missing:
        raise MissingConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + f". Add them to {ENV_FILE.name} (see .env.example)."
        )
    return {n: os.environ[n].strip() for n in names}


def get_env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` when unset."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a whole number, got {raw!r}."
        ) from exc


def get_env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to `default` when unset."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a number, got {raw!r}."
        ) from exc


_TRUTHY = frozenset({"1", "true", "yes", "on", "y"})
_FALSY = frozenset({"0", "false", "no", "off", "n"})


def get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/true/yes/on and 0/false/no/off."""
    raw = get_env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise MissingConfigError(
        f"{name} must be a boolean (true/false), got {raw!r}."
    )


def get_env_list(name: str, default: str, *, lower: bool = True) -> tuple[str, ...]:
    """Read a comma-separated env var as a tuple of stripped items.

    Lowercased by default - anything matched against provider names or hashtags
    wants that. Pass lower=False where the case is part of the meaning, such as
    a phrase that gets printed into a caption verbatim.
    """
    raw = get_env(name, default) or ""
    items = (item.strip() for item in raw.split(","))
    return tuple((item.lower() if lower else item) for item in items if item)


def ensure_dirs() -> None:
    """Create the output/asset directories if they don't exist."""
    for directory in _MANAGED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-step settings. Values are read at import time; required-ness is enforced
# by each module calling `require(*config.REQUIRED_<STEP>)` when it actually
# runs, so importing config never explodes on a half-filled .env.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 0 — topic scouting
#
# Where the topic comes from when you have not brought one. Step 0 gathers
# candidates from a few public sources, scores them here in code and prints a
# ranked shortlist for you to pick from. It never picks for you, for exactly
# the reason step 2 scores its own hooks instead of asking the model to rank
# them: a ranking you cannot read back is a ranking you cannot fix.
#
# The part that is specific to *this* pipeline: a topic is only worth making
# if step 4 can find footage for it. "Trending" is mostly proper nouns - a
# fixture, an election, a product launch - that no stock library holds a single
# clip of, and a reel about one of those comes out as generic filler laid over
# words that needed specific pictures. So a shortlisted candidate is probed
# against the configured stock providers before it is ranked.
# ---------------------------------------------------------------------------

#: Sources to gather from, in the order they are asked. An unconfigured source
#: is skipped with a warning rather than failing the scout.
#:   reddit    - hot posts in your own subreddits. The strongest of the lot: a
#:               post title is a hook a human wrote and upvotes already tested.
#:               Needs a free "script" app; Reddit no longer serves the .json
#:               endpoints anonymously.
#:   redditrss - the same subreddits over Reddit's public Atom feeds, which do
#:               still serve without a key. Keeps the titles, loses the scores
#:               and comment counts, and stands in for velocity with the post's
#:               position in the hot listing. Use it when there is no key.
#:   youtube   - videos published in the last month, ranked by views per day, so
#:               what comes back has already worked in short form.
#:   trends    - Google's daily trending searches RSS. Keyless, and general news
#:               rather than niche, so it scores down against a set niche.
#:   wikipedia - articles whose readership climbed against the same weekday last
#:               week. Keyless, official, and the only free source that measures
#:               a rise rather than a level. General interest, like trends.
#:   evergreen - the hand-written backlog in TOPIC_EVERGREEN_FILE. No request,
#:               no motion, always on the niche. The floor under a thin board
#:               rather than a competitor for the top of a good one - see
#:               EVERGREEN_PENALTY in pipeline/topics.py.
TOPIC_SOURCES = get_env_list("TOPIC_SOURCES", "redditrss,youtube,trends")

#: What the account is about. Steers the youtube search and is what niche
#: relevance is scored against. Falls back to POST_NICHE (step 6) so a project
#: that has already declared its subject area does not declare it twice. Read
#: through get_env rather than the constant because POST_NICHE is defined
#: further down this file, in step order.
TOPIC_NICHE = get_env("TOPIC_NICHE") or get_env("POST_NICHE", "") or ""

#: What a candidate off the account's subject is worth.
#:   gate   - it is not shown while there is anything on-niche to show. The
#:            board backfills from the off-niche pile only when the on-niche
#:            one runs out, and says so in a warning
#:   prefer - it is scored down by NICHE_MISS_PENALTY and left to compete
#:   off    - the niche is not scored at all
#:
#: `prefer` was the original behaviour and measurement did not support it.
#: Velocity is the largest positive on the board and a national trend feed
#: always has something moving faster than anything in one subject, so a miss
#: priced at a few points is a miss that still wins: on board #15 of an
#: ocean-and-earth-science account, eight of ten candidates were off-niche and
#: the top three were a hockey fixture, a Bollywood sequel and a wrestler.
#: Being on the subject the account is about is not a preference to be traded
#: against motion - it is the precondition, so it is a gate.
TOPIC_NICHE_MODE = get_env("TOPIC_NICHE_MODE", "gate")

#: How "on the niche" is decided in the first place.
#:   llm   - one Gemini call reads the whole shortlist and answers on/off for
#:           each. Falls back to `terms` on any failure
#:   terms - a word from TOPIC_NICHE appears in the topic
#:
#: `terms` is free and it is not good enough to gate on, which is a thing
#: worth knowing before turning the call off. Measured on board #16 of an
#: "ocean, earth science and nature" account, exact-word matching placed
#: *none* of 59 candidates on the niche - including "NASA has found evidence
#: of caves beneath the Moon", which is squarely what that account is for and
#: shares not one word with its description. Whether a subject belongs on a
#: channel is a question about meaning, and shared words are a poor proxy for
#: it in both directions: the same board scored a hockey World Cup on-niche.
#:
#: Note what is and is not delegated. The model is asked a closed question
#: about subject matter with the subject supplied; it is not shown the scores,
#: does not rank anything and does not choose. The ranking stays in code, for
#: the reason at the top of pipeline/topics.py.
TOPIC_NICHE_MATCH = get_env("TOPIC_NICHE_MATCH", "llm")

#: Subreddits to read, without the r/. No default: whose feed this is depends
#: entirely on the account, and guessing would hand back candidates from a
#: niche that is not yours. Blank means both reddit sources report not-ready.
TOPIC_SUBREDDITS = get_env_list("TOPIC_SUBREDDITS", "")

#: Which Reddit listing to read, and over what window when it is `top`.
#: Any listing Reddit serves - hot, top, new, rising - and for `top` any of
#: hour, day, week, month, year, all.
#:
#: `top` over a week by default, because this pipeline is not a news desk. A
#: 40-second explainer cut from stock footage costs a Gemini call, an
#: ElevenLabs render and a full encode, and it arrives long after a breaking
#: story has been covered by everyone who only had to talk to a camera. What
#: the format is good at is material that was as good last month as it is
#: today - and `top` is the same crowd's verdict as `hot`, taken after the
#: churn cleared rather than during it.
TOPIC_REDDIT_LISTING = get_env("TOPIC_REDDIT_LISTING", "top")
TOPIC_REDDIT_WINDOW = get_env("TOPIC_REDDIT_WINDOW", "week")

#: How many separate feed requests the subreddit list is read over. The
#: subreddits are split into this many combined `a+b+c` feeds, each given its
#: own share of TOPIC_PER_SOURCE.
#:
#: Not 1, which is what a single combined feed is. Reddit ranks a combined
#: listing by absolute score, and a subreddit list picked for a niche mixes
#: default subs with 38M members and specialist ones with 90k. Measured on
#: board #31: nine subreddits over one feed returned forty entries from two of
#: them - 33 from r/todayilearned and 2 from r/science - and all 33 were then
#: rejected off the niche, because that sub's top of the month is film trivia.
#: The seven subreddits chosen *for* the niche returned nothing, and the board
#: fell back to the evergreen backlog, as it had every day before it.
#:
#: Not high, either, and 2 is measured rather than chosen. From a cold address
#: on 2026-09-09: three feeds 4s apart lost two of them to a 429, and three
#: feeds 20s apart still lost the third. Two feeds 20s apart both returned
#: 200, which is what the anonymous RSS endpoint actually allows - see
#: topics.RedditRssSource.FEED_GAP_SECONDS. Above the ceiling the extra feeds
#: are retried, then dropped with a warning, which costs candidates rather
#: than gaining them.
#:
#: Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET and TOPIC_SOURCES=reddit to be
#: rid of the ceiling: the authenticated API allows 60 requests a minute.
TOPIC_REDDIT_FEEDS = get_env_int("TOPIC_REDDIT_FEEDS", 2)

#: Search terms for the youtube source. Blank falls back to TOPIC_NICHE.
TOPIC_KEYWORDS = get_env_list("TOPIC_KEYWORDS", "")

#: How many candidates the board shows, and how deep each source is read.
#: Fetching wide and showing few is the point - the scoring needs something to
#: choose between.
TOPIC_CANDIDATES = get_env_int("TOPIC_CANDIDATES", 8)
TOPIC_PER_SOURCE = get_env_int("TOPIC_PER_SOURCE", 25)

#: Anything older than this is scored down as past its peak. A search trend
#: that crested two days ago is a topic you would be arriving late to, and
#: being late on a trend is worse than being evergreen: the same competition,
#: none of the window.
TOPIC_MAX_AGE_HOURS = get_env_float("TOPIC_MAX_AGE_HOURS", 72.0)

#: How far back the youtube source looks. Longer than the reddit window on
#: purpose - it is measuring what has proven itself, not what is breaking.
TOPIC_YOUTUBE_WINDOW_DAYS = get_env_int("TOPIC_YOUTUBE_WINDOW_DAYS", 30)

#: Two-letter region for the Google Trends feed.
TOPIC_TRENDS_GEO = get_env("TOPIC_TRENDS_GEO", "US")

#: Which Wikipedia the pageviews source reads. A language edition, not a
#: country - "de.wikipedia" is German readers wherever they are.
TOPIC_WIKI_PROJECT = get_env("TOPIC_WIKI_PROJECT", "en.wikipedia")

#: Check that step 4 could actually find footage before recommending a topic.
#:   llm   - ask the model whether a stock library plausibly holds shots of
#:           this. The default, because the keyword probe below was measured
#:           not to separate good topics from bad ones at any threshold.
#:   probe - search the ready stock providers for the topic's keywords and read
#:           how many of the results are genuinely about it
#:   off   - do not ask. Faster, and the board will hold topics that cannot be
#:           made.
#:
#: Why `probe` is no longer the default, measured 2026-09-02 over 439 results
#: from six known-good topics and five known-bad ones (nonsense phrases plus
#: real fixtures out of the ledger - a Seahawks trade, a State Farm dividend,
#: an LA mayoral result):
#:
#:   - Nonsense scores like fact. "why a purple hexagon dreams of accounting"
#:     came back 20/20 on Pixabay and 7/20 on Pexels; "how a spider builds a
#:     web across a gap" came back 20/20. Across the whole evergreen backlog
#:     the effective count averaged 19.82 of 20, median 20, 56 of 62 lines
#:     scoring exactly 20 - a component that is the same number for every
#:     candidate does no ranking at all.
#:   - No threshold recovers it. Requiring two query words, or scaling the bar
#:     by how many words the result describes itself with (8, 10, 12), was
#:     tried: good and bad overlap under every one. Per-result term hits
#:     average 0.86 for good topics against 0.60 for bad on Pexels, 1.40
#:     against 1.22 on Pixabay.
#:   - The reason is the one PROBE_MIN_TERM_HITS already gives in
#:     pipeline/topics.py: probe_query strips a topic to three keywords, so
#:     "State Farm dividend" honestly retrieves farm footage. The results are
#:     right and the query was the wrong question. That is a fault no amount of
#:     reading search results can fix, so the question is asked of something
#:     that can read the topic instead.
#:
#: The cost of the fix is one extra Gemini call per board, batched across the
#: shortlist exactly like the niche pass. Set `probe` to go back to keywords,
#: or `off` to stop paying for the question at all - note that `off` also
#: retires the board's one hard filter, which matters more now that `trends`
#: is not usually in TOPIC_SOURCES.
TOPIC_SOURCEABILITY = get_env("TOPIC_SOURCEABILITY", "llm")

#: How many results the probe reads, and how many of them have to actually name
#: the topic before its footage counts as comfortable.
#:
#: TOPIC_SOURCEABILITY=probe only. Under the default `llm` the model rates each
#: topic 0-3 and SOURCEABILITY_COMPONENTS in pipeline/topics.py prices the
#: rungs, so neither of these is read.
#:
#: Note what this deliberately does *not* do: ask a provider how many results
#: it holds. Pexels answers about 2,500 for a string of random letters - stock
#: APIs fall back to a generic pool rather than returning nothing, so the total
#: is the same large number whatever is searched for and carries no information
#: at all. The only honest question is whether the results that came back are
#: about the thing that was asked for - which is the right question, and which
#: reading the results still turned out not to answer. See TOPIC_SOURCEABILITY.
TOPIC_PROBE_SAMPLE = get_env_int("TOPIC_PROBE_SAMPLE", 8)
TOPIC_SOURCEABLE_HITS = get_env_int("TOPIC_SOURCEABLE_HITS", 3)

#: How many candidates get probed. The probe costs requests, so it runs on the
#: shortlist the rest of the scoring has already produced, not on everything
#: that was fetched.
TOPIC_PROBE_LIMIT = get_env_int("TOPIC_PROBE_LIMIT", 16)

#: How the winning titles are turned into topic phrases for step 2.
#:   llm       - one Gemini call rewrites the shortlist. It rewrites only; the
#:               ranking was settled in code before the model was asked
#:   heuristic - strip the forum furniture (TIL, ELI5, [OC]) locally, no call
TOPIC_PHRASE_MODE = get_env("TOPIC_PHRASE_MODE", "llm")

#: Topics scouted in the last N boards are scored down so the same candidate
#: does not head every board; one already made into a reel is dropped. 0
#: disables the ledger.
TOPIC_HISTORY_RUNS = get_env_int("TOPIC_HISTORY_RUNS", 20)

#: Cross-run ledger backing the setting above.
TOPIC_USAGE_FILE = TOPICS_DIR / "usage.json"

#: Refusals a topic is allowed before the board stops offering it. A script the
#: fact-check refuses costs a ~150s Pro call and produces nothing, and until
#: this existed the board had no memory of it at all: "what the sound of the
#: deep ocean actually is" was chosen and refused five times over 26 boards.
#:
#: Two rather than one, because a refusal is evidence about a *draft* as much
#: as about the topic - that same line passed on the sixth attempt, and the
#: repair rounds it goes through now did not exist for the first four. The
#: first refusal is priced (topics.REFUSED_PENALTY) rather than banned; the
#: second ends it. 0 turns the whole thing off and refusals stop being
#: recorded.
TOPIC_MAX_REFUSALS = get_env_int("TOPIC_MAX_REFUSALS", 2)

#: The evergreen backlog: on-niche topics written by hand, read by the
#: `evergreen` source as the floor under a thin board.
#:
#: Measured, and this is the whole reason the source exists. On board #24 a
#: month of nine subreddits yielded 40 candidates, 32 of them off the niche,
#: and exactly one clearing RUN_TOPIC_MIN_SCORE. On-niche *and* explainer
#: shaped *and* moving is genuinely about one topic every other day, and this
#: pipeline runs daily. The subs that are reliably on-niche (thalassophobia,
#: natureismetal, earthporn) are photo subs whose titles - "step into the
#: darkness" - are captions, not topics a 40-second explainer can be cut from.
#: Scouting harder does not fix that; it is a supply problem, and a backlog is
#: what a supply problem takes.
TOPIC_EVERGREEN_FILE = Path(
    get_env("TOPIC_EVERGREEN_FILE") or ASSETS_DIR / "topics" / "evergreen.txt"
)

#: The last board printed, kept so `--take N` can act on the board you just
#: read rather than scouting a second time. Reading a board and choosing from
#: it are two commands, and without this they are two different boards - the
#: scout is not deterministic, and a source that was rate limited on the
#: second run silently drops out of the choice.
TOPIC_BOARD_FILE = TOPICS_DIR / "last_board.json"

#: How long a saved board may be acted on. Past this it is stale enough that
#: what was climbing has probably crested, so `--take` scouts afresh.
TOPIC_BOARD_TTL_MINUTES = get_env_int("TOPIC_BOARD_TTL_MINUTES", 60)

#: Words that disqualify a candidate outright. Empty by default - the shape
#: scoring already pushes down the fixtures and product launches this would
#: catch. Use it for subjects this account does not cover.
TOPIC_BANNED_WORDS = get_env_list("TOPIC_BANNED_WORDS", "")

# --- source credentials -----------------------------------------------------

#: In practice Reddit needs these. It used to serve the public .json endpoints
#: to anything with a User-Agent; measured against the live site, an
#: unauthenticated read now answers 403 with a block page. The anonymous path
#: is still attempted and still works from some clients, but a free "script"
#: app at https://www.reddit.com/prefs/apps is what makes the source usable.
REDDIT_CLIENT_ID = get_env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_env("REDDIT_CLIENT_SECRET")

#: Reddit asks for a descriptive User-Agent and throttles generic ones.
REDDIT_USER_AGENT = get_env(
    "REDDIT_USER_AGENT", "python:reel-creator-agent:0.1 (topic scouting)"
)

#: YouTube Data API v3. The free quota is 10,000 units/day and a search costs
#: 100, so about 100 searches a day - far more than this spends.
YOUTUBE_API_KEY = get_env("YOUTUBE_API_KEY")

#: Env var(s) each source needs, so `python config.py` can report on step 0
#: without importing the pipeline - the same trick VISUAL_PROVIDER_KEYS plays.
#: Unlike that one these are needed *together*, and for the sources named in
#: TOPIC_SOURCE_OPTIONAL_KEYS they are an upgrade rather than a requirement.
TOPIC_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "reddit": ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"),
    "redditrss": (),
    "youtube": ("YOUTUBE_API_KEY",),
    "trends": (),
    "wikipedia": (),
    "evergreen": (),
}

#: Sources that run without their keys, more slowly or more thinly.
TOPIC_SOURCE_OPTIONAL_KEYS = frozenset({"reddit"})

#: Nothing is unconditionally required: trends needs no key and reddit falls
#: back to anonymous, so a bare .env still returns a board.
REQUIRED_TOPICS: tuple[str, ...] = ()


# Step 2 — script generation
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
#: Pro rather than Flash. Four calls write the narration - the draft, then the
#: hook, structure and claim passes rewriting it - and each of those passes
#: optimises one narrow thing: opening punch, shape, factual survival. None of
#: them optimises for voice, so what shipped was Flash's rewrite of Flash's
#: rewrite of Flash's draft and it read like a metronome. At a few hundred
#: words once a day the difference in cost is not worth measuring.
#:
#: Read by more than step 2: step 0's topic scouting, step 4's query writing
#: and relevance screening, and the caption. The two screening calls in
#: `visuals.py` already branch on this name - they drop their zero thinking
#: budget, which only the flash family accepts - so they follow it correctly,
#: just more slowly than they were tuned for.
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-pro")
REQUIRED_SCRIPT_GEN = ("GEMINI_API_KEY",)

# --- the length -------------------------------------------------------------
# How long a reel should be is a decision, not a flag with a number in it.
# Completion is what the ranking runs on, so a 30-second reel watched to the
# end beats a 45-second one abandoned at 15 - and a single fact stretched to 45
# seconds is padding the viewer leaves during. Step 0 makes the call per topic
# from the shape signals it already has (`topics.suggest_duration`); these are
# the bounds it decides inside.

#: Used when nothing has an opinion: `script_gen.py` with no -d, and any topic
#: that reaches step 2 without coming through a board.
SCRIPT_DEFAULT_SECONDS = get_env_int("SCRIPT_DEFAULT_SECONDS", 30)

#: The range a suggestion may land in. Below the floor there is no room for a
#: hook, a middle and a payoff; above the ceiling the drop-off costs more than
#: the extra content pays.
#:
#: These bound the *suggestion*, not the flag. `-d 90` still works and is still
#: validated against script_gen's own MIN/MAX_DURATION_SECONDS, which are the
#: much wider range of what the module will write at all.
SCRIPT_MIN_SECONDS = get_env_int("SCRIPT_MIN_SECONDS", 20)
SCRIPT_MAX_SECONDS = get_env_int("SCRIPT_MAX_SECONDS", 45)

#: Suggestions are rounded to this. 31 seconds is not a more considered answer
#: than 30, and printing it as one invites the number to be read as a
#: measurement rather than a judgement.
SCRIPT_DURATION_STEP_SECONDS = get_env_int("SCRIPT_DURATION_STEP_SECONDS", 5)

# --- the hook ---------------------------------------------------------------
# A reel is decided inside its first second and a half, and the whole of that
# decision rests on one sentence. The scriptwriter prompt already asks for a
# hook, but asking is not checking: HOOK_MODE turns on a second pass that
# writes several openings and scores them here in code, the same way step 6
# enforces its own limits after the model has had its say.

#: llm - one extra Gemini call writes HOOK_VARIANTS openings, and the best
#:       scoring one replaces the script's first sentence
#: off - keep whatever the scriptwriter opened with
HOOK_MODE = get_env("HOOK_MODE", "llm")

#: How many openings to write before choosing. They all come back from the one
#: call so more costs nothing, but past a handful they stop being different.
HOOK_VARIANTS = get_env_int("HOOK_VARIANTS", 5)

#: A spoken hook longer than this is not a hook, it is a first paragraph. Over
#: the ceiling a candidate is scored down rather than thrown away, so a run
#: where every candidate is long still gets the shortest of them.
HOOK_MAX_WORDS = get_env_int("HOOK_MAX_WORDS", 14)

#: The on-screen version, burned over the opening by step 5. Much shorter than
#: the spoken hook because it is read at a glance rather than listened to.
HOOK_CARD_MAX_WORDS = get_env_int("HOOK_CARD_MAX_WORDS", 6)

# --- the shape --------------------------------------------------------------
# The hook pass buys the first second and a half. Everything after it used to
# be whatever the scriptwriter returned in one pass, unchecked — and a reel is
# decided at the top but lost in the middle. STRUCTURE_MODE scores the shape of
# the whole script the same way HOOK_MODE scores its opening: in code, on parts
# that can be read back, keeping the incumbent unless a rewrite actually beats
# it.

#: llm - the script's shape is scored, and a poorly shaped one buys exactly one
#:       rewrite of its body from a second Gemini call
#: off - write it once and take what comes back
STRUCTURE_MODE = get_env("STRUCTURE_MODE", "llm")

#: The score at or above which no rewrite is bought. It is set at the score of
#: a script that closes its loop and does nothing else, which makes the line it
#: draws a single sentence long: a script whose ending pays off its opening is
#: never rewritten for polish, and one whose ending does not is always
#: rewritten, whatever else it has going for it. Raise it towards 3.7 (the
#: ceiling, where every component fires) to spend a call on nearly every reel;
#: drop it below 0 to keep the scoring as a report and never act on it.
STRUCTURE_MIN_SCORE = get_env_float("STRUCTURE_MIN_SCORE", 2.0)

# --- what the viewer keeps --------------------------------------------------
# The three passes above buy attention: the hook wins the first second, the
# shape holds the next thirty, the claim check keeps them honest. None of them
# asks what the viewer is left holding when it ends, and the account says that
# is where it fails — across the 7 reels posted to 2026-09-08 every one has 0
# saves and 0 shares, with watch-through between 0.38 and 0.68. Retention is
# not the problem. Nothing is worth keeping.
#
# Measured over the 29 scripts in output/scripts/ on 2026-09-09: 1 of 29 ever
# tells the viewer where they will meet the thing themselves, 3 of 29 give the
# phenomenon a name they could repeat, and 23 of 29 do neither and score
# nothing. The endings are good writing and land nothing transferable: "It's
# the tilt.", "The ocean has no drain.", "The green just got out of the way."
#
# That 1-in-29 is the same shape as the hook pass's second-person finding: a
# virtue the scorer would pay for and the prompt never asked for. So this ships
# as both — PROMPT_TEMPLATE asks for a take-away, and this pass checks, because
# asking is not checking.

#: llm - the script is scored for what it hands the viewer, and one that hands
#:       them nothing buys exactly one rewrite of its body
#: off - score it and report, act on nothing
TAKEAWAY_MODE = get_env("TAKEAWAY_MODE", "llm")

#: The score at or above which no rewrite is bought. Set at the weight of the
#: transfer line alone (2.0), so one sentence putting the viewer in front of
#: the thing clears it outright, and so does a name plus the viewer in the
#: close (1.2 + 0.8). Anything less buys the rewrite.
#:
#: On the corpus this threshold was set against, 28 of the 29 scripts would buy
#: a call — only the noise-cancelling reel clears it, at 2.8. That is not the
#: threshold being wrong, it is the measurement above: they were written to a
#: prompt that never asked for a take-away. Judge the firing rate on scripts
#: written after this shipped, not before it, and if it is still near 100% then
#: the prompt is being ignored and that is the thing to fix — not this number.
TAKEAWAY_MIN_SCORE = get_env_float("TAKEAWAY_MIN_SCORE", 2.0)

# --- the claims -------------------------------------------------------------
# The two passes above ask whether the script is shaped like a reel. Neither
# asks whether it is true, and nothing else in the pipeline did either. The
# glacier reel opened on ice "moving at your walking speed" and said, eleven
# sentences later, "a few centimetres a day"; the Oakland Zoo reel narrated a
# crimson mountain lion that does not exist, gave it a real cougar's name, and
# closed on "Come see her". Both are well shaped. Both scored fine.

#: llm - a Gemini call looks for self-contradictions and false claims. What it
#:       finds is checked back against the script, a faulty script buys one
#:       repair, and what survives that is refused - which step 8 treats as
#:       this topic being a dead end and falls through to the next candidate.
#: off - publish whatever the scriptwriter asserts
VERIFY_MODE = get_env("VERIFY_MODE", "llm")

#: How many times step 2 may buy a rewrite for a script the claim check
#: refused, before giving the topic up.
#:
#: Was effectively 1, and 1 was measured too few. On 2026-09-02 two runs threw
#: away six topics in a row, each on a single failed repair: geyser, deep-ocean
#: sound, plankton, deep-sea dark, coral, sea ice. The findings behind them are
#: specific and mechanical - "all that water flashes to steam" should say some
#: of it, "ninety-nine percent gone. Total darkness" should say effectively
#: dark - which is exactly the kind of fault a second attempt lands.
#:
#: The economics are lopsided. A round here is two calls, a repair and its
#: re-check. Falling through to the next candidate costs a whole generation -
#: the script, the best-of-N hook pass, the structure pass and two checks -
#: and then usually fails the same way, because the fault is in how a 40-second
#: narration is written and not in which topic it is about.
#:
#: Rounds stop early only when a rewrite leaves *more* blocking findings than
#: it went in with - that one is going backwards, and the next round would
#: start from a worse script. A rewrite that holds level is re-sampled instead:
#: `revise_claims` is a sampled call, so asked twice for the same qualifier it
#: can miss once and land once. Requiring strict improvement was measured worse
#: than useless on 2026-09-03 - a script with one blocking finding can never
#: leave "fewer than one" without leaving none, so the single-finding case,
#: which is the common one, bought a single attempt whatever this was set to.
VERIFY_REPAIR_ROUNDS = get_env_int("VERIFY_REPAIR_ROUNDS", 2)

# Step 3 — text-to-speech
ELEVENLABS_API_KEY = get_env("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = get_env("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# --- delivery ---------------------------------------------------------------
# These were ElevenLabs' own defaults, hard-coded in pipeline/tts.py, which is
# a neutral read for a format that is not neutral. They are knobs now for two
# reasons: a narrator is worth trying variants of, and step 9 can only run a
# trial on something it can set.

#: Lower = more emotional range and more variation between takes. Higher =
#: flatter and more repeatable. 0.4-0.6 suits narration; 0.4 is the expressive
#: end of that, which is the end this format lives at.
TTS_STABILITY = get_env_float("TTS_STABILITY", 0.4)

#: How hard the model tries to match the original voice's timbre. Pushed too
#: high on a noisy source voice, it reproduces the artefacts too.
TTS_SIMILARITY_BOOST = get_env_float("TTS_SIMILARITY_BOOST", 0.75)

#: Style exaggeration. Costs latency and hurts stability, so this is a nudge
#: off zero rather than a setting to reach for.
TTS_STYLE = get_env_float("TTS_STYLE", 0.25)

#: Slight clarity/likeness bump. Free, so on.
TTS_SPEAKER_BOOST = get_env_bool("TTS_SPEAKER_BOOST", True)

#: Playback rate. 1.0 is a natural read and the API accepts 0.7-1.2.
#:
#: This is the one number in the delivery block that is a judgement about the
#: format rather than a measurement of this pipeline: short-form narration
#: generally sits a little above a natural read, and 1.08 is the middle of
#: that. Nothing in output/ can confirm it, because every reel so far was made
#: at 1.0 - so it is worth running as a PERFORMANCE_EXPLORE arm rather than
#: taken on trust.
#:
#: SCRIPT_WORDS_PER_MINUTE below is multiplied by this, so changing it keeps
#: the reels the length they were asked for.
TTS_SPEED = get_env_float("TTS_SPEED", 1.08)

#: How fast the narration is spoken at TTS_SPEED = 1.0, in words per minute.
#: Step 2 turns a target length into a word count with it, and step 0 turns it
#: back the other way.
#:
#: Measured over the thirteen reels on disk - script word count against the
#: rendered narration's real duration - at 153.2 wpm mean and 152.7 median
#: against the 150 assumed here, so the assumption is sound and stays put. The
#: per-reel spread is 135-173, which is content and not calibration: a script
#: full of figures is spoken slower than one without.
SCRIPT_WORDS_PER_MINUTE = get_env_int("SCRIPT_WORDS_PER_MINUTE", 150)

#: "Roger" - an ElevenLabs default voice, verified callable on a free-tier API
#: key. Used only when ELEVENLABS_VOICE_ID is absent from .env, so a fresh
#: clone still speaks. Note: Voice Library voices (and some default ones, e.g.
#: Rachel and Aria) return 402 on free plans - a paid plan lifts that.
_FALLBACK_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

#: Voice used by pipeline/tts.py when no voice_id is passed explicitly.
DEFAULT_VOICE_ID = get_env("ELEVENLABS_VOICE_ID", _FALLBACK_VOICE_ID)

#: Kept as an alias so callers can tell "configured" from "fell back".
ELEVENLABS_VOICE_ID = get_env("ELEVENLABS_VOICE_ID")

#: The voice has a working default, so only the key is genuinely required.
REQUIRED_TTS = ("ELEVENLABS_API_KEY",)

# Step 5 — the frame (9:16 vertical). Declared up here, ahead of step 4,
# because the visual resolution floor derives from the size of the finished
# frame. The rest of step 5's settings are further down.
VIDEO_WIDTH = get_env_int("VIDEO_WIDTH", 1080)
VIDEO_HEIGHT = get_env_int("VIDEO_HEIGHT", 1920)
VIDEO_FPS = get_env_int("VIDEO_FPS", 30)

# --- Step 4 — visual sourcing ----------------------------------------------

#: Providers to query, in preference order. An unconfigured provider is skipped
#: at run time rather than raising, so a missing key degrades the run instead of
#: killing it.
#:   pexels    - video + photo, free key, strongest vertical library
#:   pixabay   - video + photo, free key
#:   openverse - photo only, NO KEY NEEDED (200 req/day anon). Last-resort net.
#:   youtube   - video, Creative Commons uploads only. Off by default: it needs
#:               yt-dlp installed, and its searches cost 100 quota units each
#:               out of the same daily 10,000 the step 0 topic scout spends.
#:   nasa      - photo only, NO KEY NEEDED, public domain. Off by default
#:               because it only has anything to say about space and earth
#:               science - on a matching account it is the best source here.
VISUAL_PROVIDERS = get_env_list("VISUAL_PROVIDERS", "pexels,pixabay,openverse")

#: STOCK_API_KEY was the step-1 placeholder; still honoured as the Pexels key so
#: an .env written before the provider was chosen keeps working.
STOCK_API_KEY = get_env("STOCK_API_KEY")
PEXELS_API_KEY = get_env("PEXELS_API_KEY", STOCK_API_KEY)
PIXABAY_API_KEY = get_env("PIXABAY_API_KEY")

#: Env var(s) each provider can take its key from. Empty tuple = keyless.
#: pipeline/visuals.py reads this to decide which providers are usable.
VISUAL_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "pexels": ("PEXELS_API_KEY", "STOCK_API_KEY"),
    "pixabay": ("PIXABAY_API_KEY",),
    "openverse": (),
    "youtube": ("YOUTUBE_API_KEY",),
    "nasa": (),
}

#: Resolution floor, measured AFTER a centre crop to the 9:16 frame: a 4K
#: landscape clip passes (2160px tall leaves 1215px of width), a 1920x1080 one
#: does not (1080px tall leaves only 608px). Set below VIDEO_* to accept
#: footage that will be upscaled slightly.
VISUAL_MIN_WIDTH = get_env_int("VISUAL_MIN_WIDTH", VIDEO_WIDTH)
VISUAL_MIN_HEIGHT = get_env_int("VISUAL_MIN_HEIGHT", VIDEO_HEIGHT)


def _parse_provider_floors(raw: str | None) -> dict[str, tuple[int, int]]:
    """Parse `youtube:608x1080,nasa:900x1600` into {name: (width, height)}.

    A malformed pair is dropped rather than raised on: a typo should cost that
    one provider its override, not kill every run.
    """
    out: dict[str, tuple[int, int]] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip().lower()
        if not pair or ":" not in pair:
            continue
        name, _, size = pair.partition(":")
        width, _, height = size.partition("x")
        try:
            out[name.strip()] = (int(width), int(height))
        except ValueError:
            continue
    return out


#: Per-provider overrides of the floor above, as `name:WIDTHxHEIGHT` pairs.
#:
#: Providers do not all draw from the same shaped library, so one number cannot
#: serve them all. Pexels and Pixabay are portrait-first and clear 1080x1920
#: comfortably. YouTube is a 16:9 library, where clearing it means being
#: genuinely 4K - a slice small enough to make the provider barely worth
#: enabling. Hence the default below: 608x1080 is exactly what a 1920x1080
#: source leaves after the 9:16 crop, so ordinary 1080p uploads qualify and are
#: upscaled ~78% in the final frame. That is a real quality cost, and it is
#: confined to the one provider that needs it - dropping VISUAL_MIN_WIDTH
#: instead would have let weaker stock back in everywhere.
VISUAL_PROVIDER_FLOORS = _parse_provider_floors(
    get_env("VISUAL_PROVIDER_FLOORS", "youtube:608x1080")
)

#: How many assets to source per visual beat. >1 gives step 5 alternates to cut
#: between without going back to the network.
VISUAL_ASSETS_PER_BEAT = get_env_int("VISUAL_ASSETS_PER_BEAT", 2)

#: Source this multiple of the narration length so step 5 has slack to pace
#: cuts. 1.6 = 60% more footage than there is narration.
VISUAL_DURATION_HEADROOM = get_env_float("VISUAL_DURATION_HEADROOM", 1.6)

#: Stills are allowed as filler; VISUAL_IMAGE_DURATION is how long one is
#: assumed to hold the screen when counting coverage (step 5 decides for real).
VISUAL_ALLOW_IMAGES = get_env_bool("VISUAL_ALLOW_IMAGES", True)
VISUAL_IMAGE_DURATION = get_env_float("VISUAL_IMAGE_DURATION", 4.0)

#: Landscape sources are accepted when they survive the crop-aware floor above.
#: Set false to take only portrait/square originals.
VISUAL_ALLOW_LANDSCAPE = get_env_bool("VISUAL_ALLOW_LANDSCAPE", True)

#: Skip an asset used within this many previous runs, so the same stock clip
#: does not open every video. 0 disables the cooldown.
VISUAL_REUSE_COOLDOWN_RUNS = get_env_int("VISUAL_REUSE_COOLDOWN_RUNS", 3)

#: How search queries are written for each beat.
#:   llm       - one Gemini call turns the beats into visual search queries
#:               (falls back to `heuristic` if the call fails)
#:   heuristic - keyword extraction only, no API call
VISUAL_QUERY_MODE = get_env("VISUAL_QUERY_MODE", "llm")

#: Whether a clip is checked against the beat it will play under before it is
#: allowed on screen. A stock library answers every query with something, and
#: ranks that something on how well it is shot rather than on what it shows -
#: so an unscreened run answers "Hadza hunter gatherers" with a black bear,
#: scores it well for being a sharp vertical 4K clip, and ships it.
#:   llm     - one Gemini call screens every beat's shortlist at once
#:             (falls back to `lexical` if the call fails)
#:   lexical - keep only clips whose own words overlap the beat's
#:   off     - take whatever ranked highest on picture quality alone
VISUAL_RELEVANCE_MODE = get_env("VISUAL_RELEVANCE_MODE", "llm")

#: Floor for the lexical check, as the share of a beat's search terms that must
#: appear in a clip's own description. Deliberately low, because the lexical
#: check is the fallback and not the intended path: a provider that describes a
#: bat as a "grey-headed flying fox" shares no word with "brown bat" and is
#: still exactly the right shot. Set high enough to drop a clip that matches
#: neither the query nor the narration, and no higher.
VISUAL_RELEVANCE_FLOOR = get_env_float("VISUAL_RELEVANCE_FLOOR", 0.15)

#: How many of a beat's ranked candidates are put in front of the screener.
#: Every one costs prompt tokens, and the ranking below the top few is mostly
#: noise, so this buys variety to choose from without paying for the tail.
#: It is also the depth of judged footage a beat has: what sits below the
#: shortlist is unjudged, and is reached only after the neutral shot. Raise it
#: for a run that keeps warning it placed unscreened clips.
VISUAL_RELEVANCE_SHORTLIST = get_env_int("VISUAL_RELEVANCE_SHORTLIST", 6)

#: Two clips can be different files, clear every filter, and still read as one
#: shot held too long - stock libraries sell whole shoots a frame apart. The
#: sleep reel spent four of its last six shots on two men in two beds, and its
#: back half stopped moving. A run therefore prefers footage it has not already
#: got something like, in two senses.

#: How much two clips' own descriptions may share before the second is treated
#: as the same shot again ("man feeling sleepy" against "man sleeping").
#: Measured only against what this run has already taken.
VISUAL_SIMILAR_OVERLAP = get_env_float("VISUAL_SIMILAR_OVERLAP", 0.45)

#: How many clips one creator may contribute to a single run. A stock creator's
#: clips are usually one shoot, one model and one room, which is how two files
#: sharing no word in their descriptions - "relaxing rest tired indoors" and
#: "video of a man sleeping" - turn out to be the same man on the same bed.
#: 0 disables the cap.
VISUAL_MAX_PER_CREATOR = get_env_int("VISUAL_MAX_PER_CREATOR", 1)

#: Nothing is unconditionally required: openverse needs no key, so a bare .env
#: still sources visuals. Provider readiness is reported per provider instead.
REQUIRED_VISUALS: tuple[str, ...] = ()

# --- Step 5 — video assembly -----------------------------------------------
# The frame itself is VIDEO_WIDTH/HEIGHT/FPS above (declared early because the
# step-4 resolution floor derives from it). Everything below shapes how the
# footage is paced and how the burned-in captions look — change the look here,
# never in pipeline/assemble.py.

#: ffmpeg/ffprobe locations. Empty = resolve at run time: $PATH first, then the
#: binary shipped by the `imageio-ffmpeg` wheel, so a machine with no system
#: ffmpeg still assembles. ffprobe is optional; without it durations are read
#: from ffmpeg's own stream banner.
FFMPEG_BINARY = get_env("FFMPEG_BINARY")
FFPROBE_BINARY = get_env("FFPROBE_BINARY")

#: Finished reels land in OUTPUT_DIR/<date>/<run id>/ — one directory per reel,
#: holding the .mp4, its captions and the attribution carried over from step 4,
#: so steps 6 and 7 have a single place to read from and add to.
VIDEO_RUN_DATE_FORMAT = get_env("VIDEO_RUN_DATE_FORMAT", "%Y-%m-%d")
VIDEO_FILENAME = get_env("VIDEO_FILENAME", "reel.mp4")

# --- encoding ---------------------------------------------------------------
# Instagram re-encodes whatever it is given, so this targets "clean input to a
# transcoder": high profile H.264, yuv420p, faststart. CRF over a bitrate
# target — the source is stock footage of wildly varying complexity.

VIDEO_CODEC = get_env("VIDEO_CODEC", "libx264")
VIDEO_CRF = get_env_int("VIDEO_CRF", 20)
VIDEO_PRESET = get_env("VIDEO_PRESET", "medium")
VIDEO_PIXEL_FORMAT = get_env("VIDEO_PIXEL_FORMAT", "yuv420p")
AUDIO_CODEC = get_env("AUDIO_CODEC", "aac")
AUDIO_BITRATE = get_env("AUDIO_BITRATE", "192k")
AUDIO_SAMPLE_RATE = get_env_int("AUDIO_SAMPLE_RATE", 44100)

# --- loudness ---------------------------------------------------------------
# Instagram normalises playback to roughly -14 LUFS. ElevenLabs renders at
# whatever level the chosen voice happens to sit at, so without this two reels
# cut a week apart land at audibly different volumes - and a quiet one is
# turned up by the platform along with its noise floor. loudnorm fixes the
# perceived loudness of the narration instead of guessing at a gain.

#: Off leaves the narration exactly as ElevenLabs rendered it.
AUDIO_NORMALIZE = get_env_bool("AUDIO_NORMALIZE", True)

#: Integrated loudness target, LUFS. Instagram's own working point. This is a
#: single-pass loudnorm - it keeps step 5 to one encode, and lands within a
#: couple of LU of the target rather than exactly on it. Measured on a real
#: narration: -24.4 LUFS in, -15.6 LUFS out.
AUDIO_LOUDNESS_TARGET = get_env_float("AUDIO_LOUDNESS_TARGET", -14.0)

#: True-peak ceiling, dBTP. -1.5 leaves headroom for the lossy re-encode
#: Instagram puts the file through, which can push samples above the peak.
AUDIO_LOUDNESS_PEAK = get_env_float("AUDIO_LOUDNESS_PEAK", -1.5)

#: Loudness range, LU. How much quiet-to-loud variation to preserve; a single
#: narrator reading to camera does not have much, so this rarely bites.
AUDIO_LOUDNESS_RANGE = get_env_float("AUDIO_LOUDNESS_RANGE", 11.0)

# --- pacing -----------------------------------------------------------------
# Step 4 decided *which* footage belongs to which stretch of narration; these
# decide how often the picture cuts inside one of those beats.

#: Roughly how long one shot should hold. A beat gets ceil(beat/target) shots,
#: capped by how many assets step 4 actually sourced for it.
SHOT_TARGET_SECONDS = get_env_float("SHOT_TARGET_SECONDS", 3.2)

#: Hard floor and ceiling on a single shot. The floor stops a short beat being
#: chopped into flashes; the ceiling forces a cut when a beat drags.
#:
#: The floor gives way to the pacing curve below rather than fighting it: where
#: the curve asks for shots shorter than SHOT_MIN_SECONDS, the curve wins. It
#: is a floor on the arithmetic, not on the edit.
SHOT_MIN_SECONDS = get_env_float("SHOT_MIN_SECONDS", 1.6)
SHOT_MAX_SECONDS = get_env_float("SHOT_MAX_SECONDS", 6.0)

#: How shot length moves across the reel.
#:   curve - fast through the opening, easing out to SHOT_TARGET_SECONDS, then
#:           tightening again over the close
#:   flat  - SHOT_TARGET_SECONDS everywhere, whatever the position
#:
#: One rate for the whole reel is the safe choice and the wrong one. Cuts buy
#: attention at the start, where there is none yet to lose, and cost
#: comprehension in the middle, where the viewer is actually following
#: something. The curve spends them where they are worth it.
SHOT_PACING = get_env("SHOT_PACING", "curve")

#: Shot length at t=0: short enough that the picture changes before the viewer
#: has finished deciding whether to stay.
SHOT_HOOK_SECONDS = get_env_float("SHOT_HOOK_SECONDS", 1.4)

#: How long the opening takes to ease out to the steady SHOT_TARGET_SECONDS.
SHOT_PACING_RAMP_SECONDS = get_env_float("SHOT_PACING_RAMP_SECONDS", 8.0)

#: The close. Shot length is pulled back down to SHOT_OUTRO_SECONDS over the
#: last SHOT_OUTRO_WINDOW_SECONDS, so the reel does not run out of energy on
#: the line that has to land. A window of 0 switches the tightening off.
SHOT_OUTRO_SECONDS = get_env_float("SHOT_OUTRO_SECONDS", 2.0)
SHOT_OUTRO_WINDOW_SECONDS = get_env_float("SHOT_OUTRO_WINDOW_SECONDS", 6.0)

#: How far a single shot may overrun what the curve asked for at that point,
#: as a multiple of it. Past this the beat splits into another shot.
#:
#: SHOT_MAX_SECONDS is an absolute ceiling at 6.0s and the curve asks for 3.2s,
#: so the absolute one never binds and shot length was decided entirely by the
#: rounding in `_shot_count`: a window between 1.0 and 1.49 times the target
#: rounds down to one shot, and holds for up to 1.5 times the target. Measured
#: across the thirteen reels on disk, that is exactly where the long shots came
#: from - the longest shot in the whole set is 4.81s against a 3.2s target,
#: which is 1.50x, and 28 of the 32 shots over 3.5s were beats that held one
#: clip while a *second already downloaded clip for that same beat* went
#: unused. The footage was there; the arithmetic threw it away.
#:
#: At 1.25 those 34 long shots become 5, the longest drops from 4.77s to 3.99s,
#: and nothing lands under 1.2s - the reel gains 32 cuts and pays for none of
#: them. Below about 1.15 the return flattens and the cutting starts fighting
#: the curve. 0 switches the rule off and restores the rounding.
#:
#: It can never invent footage: `_shot_count` still caps the result at twice
#: the assets step 4 actually sourced for the beat.
SHOT_MAX_OVERRUN = get_env_float("SHOT_MAX_OVERRUN", 1.25)

#: Cut on a word boundary instead of wherever the arithmetic landed.
#:
#: Step 5 already knows when every word is spoken - it aligns the narration to
#: write the captions - so a cut can be nudged onto the gap between two words
#: rather than falling in the middle of one. A cut mid-word reads as a mistake;
#: the same cut 200ms later reads as an edit. Costs nothing: the timings are
#: already on hand, and the alignment now runs before the edit is planned.
SHOT_SNAP_TO_WORDS = get_env_bool("SHOT_SNAP_TO_WORDS", True)

#: How far a cut may be moved to find that gap. Far enough to reach the next
#: pause, near enough that the pacing curve still decides the rhythm.
SHOT_SNAP_MAX_SECONDS = get_env_float("SHOT_SNAP_MAX_SECONDS", 0.35)

#: Cut the opening of the reel on the music's beat instead of on the words.
#:
#: A cut that lands on a beat is not heard as a cut at all - it is heard as the
#: track doing it, which is the whole trick behind an edit that feels scored
#: rather than assembled. It only works where the bed is actually audible, so
#: it is confined to the opening: MUSIC_FADE_IN_SECONDS has just finished, the
#: narration has not yet built up, and the cutting is at its fastest.
#:
#: It needs a BPM, and nothing here can measure one. Write it beside the track,
#: in the same .txt sidecar the credit line lives in:
#:
#:     "Autumn prelude" by zero-project (CC BY 3.0)
#:     bpm: 92
#:     offset: 0.21
#:
#: `bpm` is the tempo; `offset` is where the first downbeat sits inside the
#: file, in seconds, for a track that does not start exactly on one. A sidecar
#: without a bpm line is read exactly as before - a credit and nothing else -
#: so this changes nothing until a tempo is written down. MUSIC_BPM below is
#: the fallback when a track has no sidecar of its own.
SHOT_SNAP_TO_BEAT = get_env_bool("SHOT_SNAP_TO_BEAT", True)

#: How much of the reel's opening the beat grid governs. Past this the words
#: take the cuts back - by then the narration is carrying the reel and the bed
#: has settled under it, and cutting to a tempo nobody is listening for buys
#: nothing.
SHOT_BEAT_WINDOW_SECONDS = get_env_float("SHOT_BEAT_WINDOW_SECONDS", 8.0)

#: How far a cut may be moved to reach a beat. Wider than SHOT_SNAP_MAX_SECONDS
#: on purpose: beats are 0.67s apart at 90 BPM and a full second at 60, so a
#: 0.35s reach would miss the grid outright on a slow track.
SHOT_BEAT_MAX_SECONDS = get_env_float("SHOT_BEAT_MAX_SECONDS", 0.5)

#: When a clip is much longer than the slot it fills, skip this fraction of it
#: before cutting in — stock footage often opens on a fade or a static frame.
SHOT_SKIP_HEAD_RATIO = get_env_float("SHOT_SKIP_HEAD_RATIO", 0.1)

#: How a shot meets the next one. "cut" is a straight cut (the format's norm);
#: "dip" fades each shot up from and down to black over TRANSITION_SECONDS.
TRANSITION = get_env("TRANSITION", "cut")
TRANSITION_SECONDS = get_env_float("TRANSITION_SECONDS", 0.22)

#: Stills get a slow push-in so they do not read as a frozen video. 1.0 = off.
IMAGE_ZOOM = get_env_float("IMAGE_ZOOM", 1.12)

# --- motion -----------------------------------------------------------------
# Stills have always moved. Clips never did: they were scaled, centre-cropped
# and left alone, so a reel cut entirely from stock video — which is every reel
# this has made — was a sequence of static frames from footage that happened to
# have things moving inside it. The difference between an edit and a slideshow
# is that the camera is doing something, and a slow push costs one filter.
#
# IMAGE_ZOOM above still governs stills; this governs clips. They are separate
# because they are different problems: a still has no motion at all without
# help, while a clip only needs enough to read as a camera move rather than a
# locked-off frame.

#: alternate - shots push in and pull back in turn, so a run of them does not
#:             read as one mechanical move repeated
#: push      - every shot pushes in
#: off       - clips are left static, as they were before this existed
SHOT_MOTION = get_env("SHOT_MOTION", "alternate")

#: How far a shot travels over its own length, as a zoom factor. Small on
#: purpose: at 1.08 over three seconds the move is under 3% a second, which
#: reads as a camera rather than as an effect. Past ~1.15 on a long shot it
#: starts to look like a zoom, and the crop eats real resolution.
SHOT_MOTION_ZOOM = get_env_float("SHOT_MOTION_ZOOM", 1.08)

#: The opening shot, which is on screen for the second and a half in which the
#: viewer decides. It gets a harder push than the rest and always pushes in
#: rather than out: a pull-back on the first shot moves away from the subject
#: at exactly the moment the reel is asking to be watched.
SHOT_MOTION_HOOK_ZOOM = get_env_float("SHOT_MOTION_HOOK_ZOOM", 1.18)

#: Top and tail of the finished reel, in seconds.
#:
#: The open is deliberately 0. A reel is judged in its first half-second, and
#: that is exactly the stretch a fade spends showing black - the hook is being
#: spoken over a picture nobody can see yet, and the viewer decides while it is
#: still dark. Feeds also autoplay straight into the first frame, so a fade
#: reads as a loading artefact rather than as an edit. Set it to 0.3-0.5 if a
#: particular reel wants a soft open; the tail keeps its fade either way,
#: because by then the viewer has already stayed.
VIDEO_FADE_IN_SECONDS = get_env_float("VIDEO_FADE_IN_SECONDS", 0.0)
VIDEO_FADE_OUT_SECONDS = get_env_float("VIDEO_FADE_OUT_SECONDS", 0.6)
AUDIO_FADE_OUT_SECONDS = get_env_float("AUDIO_FADE_OUT_SECONDS", 0.35)

#: Picture held after the narration stops, so the last word is not cut off by
#: the video ending on the same frame.
VIDEO_TAIL_SECONDS = get_env_float("VIDEO_TAIL_SECONDS", 0.5)

# --- music bed --------------------------------------------------------------
# A reel without music reads as a document; a reel whose music sits on top of
# the voice reads as an advert. What follows keeps the bed present and out of
# the way. Drop tracks into assets/music/ - nothing is downloaded, and whatever
# you put there is yours to have licensed.

#: Permission, not a requirement: an empty assets/music/ simply means no bed.
MUSIC_ENABLED = get_env_bool("MUSIC_ENABLED", True)

#: Pin one track instead of rotating. A bare filename is looked up inside
#: MUSIC_DIR; anything else is treated as a path.
MUSIC_FILE = get_env("MUSIC_FILE")

#: What counts as a track in assets/music/.
MUSIC_EXTENSIONS = get_env_list(
    "MUSIC_EXTENSIONS", ".mp3,.m4a,.aac,.wav,.flac,.ogg,.opus"
)

#: Skip a track used within this many previous runs, so a small library does
#: not put the same bed under every reel. 0 disables the rotation.
MUSIC_REUSE_COOLDOWN_RUNS = get_env_int("MUSIC_REUSE_COOLDOWN_RUNS", 3)

#: Cross-run ledger backing the setting above.
MUSIC_USAGE_FILE = MUSIC_DIR / "usage.json"

#: Integrated loudness the bed is normalised to, LUFS - the one knob that
#: decides how loud the music is. Set against AUDIO_LOUDNESS_TARGET above:
#: -30 sits about 16 LU under a -14 LUFS narration, which is audible without
#: competing. Normalising means a quiet track and a mastered one land in the
#: same place, instead of the level depending on what you happened to drop in.
MUSIC_LOUDNESS_TARGET = get_env_float("MUSIC_LOUDNESS_TARGET", -30.0)

#: Duck the bed under the narration instead of holding it at one level. This is
#: what makes the music breathe: it lifts in the gaps between sentences and
#: pulls back the moment the voice returns. Off holds a flat level throughout.
MUSIC_DUCKING = get_env_bool("MUSIC_DUCKING", True)

#: The compressor behind the ducking. These are expert knobs - MUSIC_DUCKING
#: above is the one worth touching, and MUSIC_LOUDNESS_TARGET decides how loud
#: the bed is in the first place.
#:
#: Threshold is a linear amplitude: 0.03 is about -30 dBFS, which measured as
#: the best of the values tried. It has to sit between speech and the pauses -
#: lower is not better, because then the pauses are compressed too and the bed
#: stops lifting between lines at all. Measured on a real narration: -30 dBFS
#: gave 5.1 dB of lift in the gaps, -40 dBFS only 3.9 dB.
MUSIC_DUCK_THRESHOLD = get_env_float("MUSIC_DUCK_THRESHOLD", 0.03)

#: How hard the bed is pushed once it is over the threshold. The reduction is
#: capped by how far speech sits above the threshold - about 5 dB for a
#: loudness-normalised narration - so this saturates: 8 and 20 measured the
#: same 5.1 dB. Left at 8 because there is nothing to gain past it.
MUSIC_DUCK_RATIO = get_env_float("MUSIC_DUCK_RATIO", 8.0)

#: Milliseconds. The attack has to beat the first syllable, or the bed is still
#: loud under the word that matters; the release is slow enough that the music
#: does not pump between words inside a sentence.
MUSIC_DUCK_ATTACK_MS = get_env_float("MUSIC_DUCK_ATTACK_MS", 15.0)
MUSIC_DUCK_RELEASE_MS = get_env_float("MUSIC_DUCK_RELEASE_MS", 350.0)

#: Music starts before the first word and outlives the last, so the reel does
#: not open and close on an abrupt edit.
MUSIC_FADE_IN_SECONDS = get_env_float("MUSIC_FADE_IN_SECONDS", 1.0)
MUSIC_FADE_OUT_SECONDS = get_env_float("MUSIC_FADE_OUT_SECONDS", 1.5)

#: Fallback tempo for a track with no `bpm:` line in its sidecar, in BPM, and
#: where its first downbeat sits inside the file, in seconds. A track's own
#: sidecar always wins. 0 means "no grid", which is what a library with no
#: tempos written down gets - see SHOT_SNAP_TO_BEAT above.
#:
#: Anything outside 30-300 BPM is treated as a typo and dropped with a warning:
#: a stray `bpm: 9000` would otherwise put a beat every 7ms and make the grid
#: mean nothing at all.
MUSIC_BPM = get_env_float("MUSIC_BPM", 0.0)
MUSIC_BEAT_OFFSET = get_env_float("MUSIC_BEAT_OFFSET", 0.0)
MUSIC_BPM_MIN = get_env_float("MUSIC_BPM_MIN", 30.0)
MUSIC_BPM_MAX = get_env_float("MUSIC_BPM_MAX", 300.0)

# --- sound design -----------------------------------------------------------
# Cuts that are heard as well as seen. A whoosh across a cut is the cheapest
# production value there is: it costs one small file and it turns a change of
# picture into a deliberate edit.
#
# One-shots come from assets/sfx/ and nowhere else - nothing is downloaded and
# nothing is generated, exactly like the music bed. The role is read off the
# front of the filename, so whoosh.wav, whoosh-02.wav and whoosh_soft.mp3 are
# all whooshes and step 5 rotates between them. An empty assets/sfx/ means no
# effects at all, which is how a fresh clone arrives and what every reel cut
# before this feature sounded like.

SFX_ENABLED = get_env_bool("SFX_ENABLED", True)

#: What counts as a one-shot in assets/sfx/.
SFX_EXTENSIONS = get_env_list(
    "SFX_EXTENSIONS", ".wav,.mp3,.m4a,.aac,.flac,.ogg,.opus"
)

#: Level the one-shots are held to, dBFS peak.
#:
#: Peak rather than LUFS, and deliberately: these are 200ms transients, and
#: integrated loudness is not defined over a window that short - EBU R128
#: measures in 400ms blocks and would hand back nothing usable. Peak is also
#: the right measure for the job, being what a sound editor levels a one-shot
#: by. Measured per file at plan time, so a hot whoosh and a quiet one land in
#: the same place instead of the mix depending on what you happened to
#: download. -14 dBFS sits under a -14 LUFS narration without fighting it.
SFX_PEAK_TARGET = get_env_float("SFX_PEAK_TARGET", -14.0)

#: A whoosh across each cut.
SFX_ON_CUTS = get_env_bool("SFX_ON_CUTS", True)

#: Placed this early, so the transient lands ON the cut rather than after it.
#: A whoosh that arrives once the picture has already changed is heard as a
#: separate noise instead of as the cut itself.
SFX_LEAD_SECONDS = get_env_float("SFX_LEAD_SECONDS", 0.08)

#: Cuts closer together than this get one whoosh between them, not two. The
#: pacing curve opens on ~1.4s shots and an effect on every one of them stops
#: being punctuation and becomes noise.
SFX_MIN_GAP_SECONDS = get_env_float("SFX_MIN_GAP_SECONDS", 1.6)

#: An impact on the first frame, under the hook. The one effect with no cut to
#: ride: nothing has happened yet, which is the point of it.
SFX_IMPACT_AT_OPEN = get_env_bool("SFX_IMPACT_AT_OPEN", True)

# --- captions ---------------------------------------------------------------
# Burned into the picture with libass. These are the *on-screen* captions; the
# Instagram caption text is step 6 and unrelated.

CAPTIONS_ENABLED = get_env_bool("CAPTIONS_ENABLED", True)

#: Where the word timings come from:
#:   auto       - elevenlabs, then whisper, then estimate (first one available)
#:   elevenlabs - /v1/forced-alignment: the audio plus the known script, so the
#:                words are exact rather than transcribed. Needs the step-3 key.
#:   whisper    - faster-whisper locally, word_timestamps=True. No key, but the
#:                model has to be downloaded once.
#:   estimate   - no network, no model: words are spread across step 4's beat
#:                boundaries (already measured against the .mp3) by syllable
#:                weight. Always available, least accurate.
CAPTION_ALIGNMENT = get_env("CAPTION_ALIGNMENT", "auto")
CAPTION_WHISPER_MODEL = get_env("CAPTION_WHISPER_MODEL", "base")

#: Font for the burned-in captions. CAPTION_FONT_FILE wins if set; otherwise
#: the first font dropped into assets/fonts/ is used; otherwise CAPTION_FONT_NAME
#: is looked up among the installed system fonts. A file inside the project is
#: the portable choice — the reel then looks the same on any machine.
CAPTION_FONT_FILE = get_env("CAPTION_FONT_FILE")
CAPTION_FONT_NAME = get_env("CAPTION_FONT_NAME", "Arial Black")

#: Sized for a 1080x1920 frame and scaled automatically if VIDEO_HEIGHT changes.
CAPTION_FONT_SIZE = get_env_int("CAPTION_FONT_SIZE", 96)

#: One cue is at most this many words / characters — short bursts read better on
#: a phone than full sentences, and keep the highlight moving.
CAPTION_MAX_WORDS = get_env_int("CAPTION_MAX_WORDS", 3)
CAPTION_MAX_CHARS = get_env_int("CAPTION_MAX_CHARS", 24)

#: A cue is held at least this long even if the words were spoken faster.
CAPTION_MIN_CUE_SECONDS = get_env_float("CAPTION_MIN_CUE_SECONDS", 0.45)

#: "highlight" tints the word currently being spoken; "plain" shows the whole
#: cue in one colour.
CAPTION_STYLE = get_env("CAPTION_STYLE", "highlight")
CAPTION_UPPERCASE = get_env_bool("CAPTION_UPPERCASE", True)

#: #RRGGBB. Converted to ASS's &HBBGGRR by pipeline/assemble.py.
CAPTION_PRIMARY_COLOR = get_env("CAPTION_PRIMARY_COLOR", "#FFFFFF")
CAPTION_HIGHLIGHT_COLOR = get_env("CAPTION_HIGHLIGHT_COLOR", "#FFE14D")
CAPTION_OUTLINE_COLOR = get_env("CAPTION_OUTLINE_COLOR", "#000000")

CAPTION_OUTLINE_WIDTH = get_env_float("CAPTION_OUTLINE_WIDTH", 6.0)
CAPTION_SHADOW_DEPTH = get_env_float("CAPTION_SHADOW_DEPTH", 3.0)

#: Distance from the bottom of the frame, in 1080x1920 pixels. ~420 keeps the
#: text clear of Instagram's own caption/button overlay along the bottom.
CAPTION_MARGIN_V = get_env_int("CAPTION_MARGIN_V", 420)

#: Left/right margin, so long cues wrap instead of running to the frame edge.
CAPTION_MARGIN_H = get_env_int("CAPTION_MARGIN_H", 90)

# --- caption pop ------------------------------------------------------------
# The highlight already keeps up with the voice. What it does not do is arrive:
# a cue cuts in at full size and sits there, so three words replacing three
# other words in the same place at the same size reads as a slideshow. A cue
# that lands a little oversized and settles is read as a new thing being said.
#
# Both knobs below are drawn by libass from the same subtitle file as
# everything else - there is no extra pass and no measurable encode cost.

#: Scale the cue starts at, as a percentage, before settling to 100 over
#: CAPTION_POP_MS. 112 is a nudge you feel rather than watch; past about 125 it
#: starts reading as a bounce, which is a different and much cheaper effect.
#: Off restores cues to cutting in at full size, which is what every reel made
#: before this setting existed looks like. It does not turn off the payoff
#: colour below - that is a separate look, disabled separately.
CAPTION_POP = get_env_bool("CAPTION_POP", True)
CAPTION_POP_SCALE = get_env_int("CAPTION_POP_SCALE", 112)

#: How long the settle takes. Short enough to be over before the first word is
#: finished - a pop still running under the second word is an animation, and
#: the eye follows it instead of reading.
CAPTION_POP_MS = get_env_int("CAPTION_POP_MS", 90)

#: The payoff word - the number the sentence was built to deliver - in its own
#: colour, and with a harder pop of its own as it is spoken.
#:
#: Step 2 is told to "write numbers and symbols the way they should be said
#: aloud", so the narration says "ninety-five percent" and the captions carry
#: almost no digits at all. The detector therefore reads number *words*; see
#: _PAYOFF_NUMBER_WORDS in pipeline/assemble.py.
#:
#: Highlight style only. "plain" means one colour for the whole cue, and a
#: second colour in it would make that setting a lie. Set the colour equal to
#: CAPTION_HIGHLIGHT_COLOR to keep the pop and drop the distinction.
CAPTION_PAYOFF_COLOR = get_env("CAPTION_PAYOFF_COLOR", "#4DF0FF")
CAPTION_PAYOFF_SCALE = get_env_int("CAPTION_PAYOFF_SCALE", 124)

# --- the hook card ----------------------------------------------------------
# The hook in big type, high in the frame, over the opening of the reel while
# the narration is saying it.
#
# This is the muted-viewer path. Feeds autoplay without sound, and the running
# captions below are a transcript arriving three words at a time - fine once
# someone is watching, useless for deciding whether to. The card is the whole
# proposition, readable in one glance, and gone before it starts covering the
# video it sits on.

HOOK_CARD_ENABLED = get_env_bool("HOOK_CARD_ENABLED", True)

#: How long it holds. Long enough to read twice at a scroll.
HOOK_CARD_SECONDS = get_env_float("HOOK_CARD_SECONDS", 2.2)

#: Fade at each end, milliseconds. A card that snaps on reads as a glitch.
HOOK_CARD_FADE_MS = get_env_int("HOOK_CARD_FADE_MS", 140)

#: Bigger than the running captions: it is the headline, they are the
#: transcript. Shrunk automatically when the words would run past the frame.
HOOK_CARD_FONT_SIZE = get_env_int("HOOK_CARD_FONT_SIZE", 112)

#: Distance from the TOP of the frame, in 1080x1920 pixels - the card is
#: top-anchored, unlike the captions. High enough to clear the captions along
#: the bottom, low enough to clear the account bar Instagram draws over the top.
HOOK_CARD_MARGIN_V = get_env_int("HOOK_CARD_MARGIN_V", 320)

HOOK_CARD_COLOR = get_env("HOOK_CARD_COLOR", "#FFFFFF")
HOOK_CARD_OUTLINE_COLOR = get_env("HOOK_CARD_OUTLINE_COLOR", "#000000")
HOOK_CARD_OUTLINE_WIDTH = get_env_float("HOOK_CARD_OUTLINE_WIDTH", 7.0)
HOOK_CARD_UPPERCASE = get_env_bool("HOOK_CARD_UPPERCASE", True)

# --- the cover frame -------------------------------------------------------
# The still Instagram shows in the grid, on the profile and in search. The reel
# earns the watch; the cover earns the tap that gets it played at all, and it
# is the only part of a reel that is judged with no motion and no sound.
#
# It is NOT a frame grabbed out of reel.mp4. Every frame of the finished file
# has a running caption burned across it - three words out of the middle of a
# sentence - and that is not a poster. The cover is rebuilt from the source
# asset with the edit's own crop, and the hook card is re-drawn over it.

COVER_ENABLED = get_env_bool("COVER_ENABLED", True)

#: Written into the run directory beside reel.mp4. JPEG because Instagram
#: re-encodes whatever it is handed, so the alpha and the extra megabytes a PNG
#: would buy are thrown away on upload.
COVER_FILENAME = get_env("COVER_FILENAME", "cover.jpg")

#: Where the cover comes from:
#:   auto      - the hook card when the reel has one and it clears the
#:               COVER_SAFE_ASPECT crop, else the best-scoring footage frame
#:   hook_card - always the card, even where the crop will bite
#:   footage   - always footage, even when there is a card
#: `auto` prefers the card for the same reason the card exists at all: the grid
#: is a browse surface, and a line of type says what a reel is about where a
#: beautiful shot of open water only says that it is blue.
COVER_SOURCE = get_env("COVER_SOURCE", "auto")

#: ffmpeg's -q:v for the JPEG, 2 (near-lossless) to 31. The cover is one frame
#: and is re-encoded on upload anyway, so there is nothing to be saved by
#: starting it off soft.
COVER_JPEG_QUALITY = get_env_int("COVER_JPEG_QUALITY", 2)

#: Only shots opening inside this window are candidates. The cover should show
#: what the reel promises, which lives at the top; a frame from the payoff at
#: 0:48 is a picture of the answer to a question the browser has not been asked.
COVER_SEARCH_SECONDS = get_env_float("COVER_SEARCH_SECONDS", 15.0)

#: The tightest crop the cover is displayed at, as W:H. Instagram centre-crops
#: covers differently in the grid, the profile and search, and has changed which
#: it uses more than once - so this is a knob rather than a constant. It is used
#: to check the hook card survives the crop: at the 1080x1920 default the card
#: sits 320px down, which clears 4:5 (from 285px) and 3:4 (from 240px) but is
#: clipped by 1:1 (from 420px). A card that fails the check makes `auto` fall
#: back to footage and says so.
COVER_SAFE_ASPECT = get_env("COVER_SAFE_ASPECT", "4:5")

#: Nothing is required: captions fall back to `estimate` with no key at all, and
#: ffmpeg comes from the imageio-ffmpeg wheel if the system has none.
REQUIRED_ASSEMBLE: tuple[str, ...] = ()


def video_run_dir(run_id: str, when=None) -> Path:
    """Directory for one finished reel: output/<date>/<run id>/.

    Dated so a day's output stays together, and keyed by step 4's run id so the
    video sits next to the visuals it was cut from. Not created here — the
    caller makes it, so a dry run leaves no empty directories behind.
    """
    from datetime import date as _date

    stamp = (when or _date.today()).strftime(VIDEO_RUN_DATE_FORMAT)
    return OUTPUT_DIR / stamp / run_id


# --- Step 6 — Instagram post text (caption + hashtags) ---------------------
# The copy that goes in Instagram's caption field. Nothing here touches the
# burned-in on-screen captions above - those are CAPTION_* / step 5. Change how
# every post reads from here, never inside pipeline/caption.py.

#: llm       - Gemini writes the caption and tags from the script (falls back
#:             to `heuristic` by itself if the call fails)
#: heuristic - lines lifted from the script + keyword tags, no API call
POST_MODE = get_env("POST_MODE", "llm")

#: Instagram's hard caption limit - hashtags, line breaks and all.
POST_MAX_CHARS = get_env_int("POST_MAX_CHARS", 2200)

#: Only the opening of a caption survives before Instagram cuts it off with
#: "... more", so the first line has to read as a finished thought on its own.
#: 125 is where the feed truncates; the hook is built to fit inside it and is
#: never left mid-sentence.
POST_HOOK_MAX_CHARS = get_env_int("POST_HOOK_MAX_CHARS", 125)

#: The voice of the post copy. One knob, every caption.
POST_TONE = get_env(
    "POST_TONE",
    "curious and plainspoken - short lines, concrete detail, "
    "no hype stacking and no engagement bait",
)

#: What the closing line should do. Blank omits the CTA altogether.
POST_CTA = get_env(
    "POST_CTA",
    "ask one question the post genuinely wants answered in the replies",
)

#: The account's subject area, used to steer the broad end of the hashtag mix
#: when the script alone is too specific to place. Blank = infer from the script.
POST_NICHE = get_env("POST_NICHE", "")

#: Emoji read as native on Instagram, but a caption studded with them reads as
#: a brand account. A couple, at most.
POST_ALLOW_EMOJI = get_env_bool("POST_ALLOW_EMOJI", True)
POST_MAX_EMOJI = get_env_int("POST_MAX_EMOJI", 2)

#: How many hashtags to land on. Instagram allows 30 and Instagram's own advice
#: is 3-5; a wall of 30 reads as spam and dilutes what the post is classified
#: as. 8 is the default: enough surface for the recommender to place the reel,
#: well short of looking desperate.
POST_HASHTAG_COUNT = get_env_int("POST_HASHTAG_COUNT", 8)

#: Platform hard cap, not a preference. Instagram rejects captions above this.
POST_HASHTAG_LIMIT = 30

#: caption - hashtags close the caption (what most accounts do)
#: comment - kept out of the caption text and handed to step 7 separately, to
#:           be posted as the first comment
POST_HASHTAG_PLACEMENT = get_env("POST_HASHTAG_PLACEMENT", "caption")

#: Tags added to every post whatever the topic - a brand or series tag. These
#: are kept even when they push the generated ones out.
POST_HASHTAGS_ALWAYS = get_env_list("POST_HASHTAGS_ALWAYS", "")

#: Tags never to use. The default list is engagement bait that Instagram
#: quietly limits reach on - it is not a moderation list.
POST_HASHTAGS_BANNED = get_env_list(
    "POST_HASHTAGS_BANNED",
    "follow4follow,followforfollow,like4like,likeforlike,f4f,l4l,"
    "tagsforlikes,followback,followme",
)

#: Tags used by the last N runs are passed to the generator as "recently used"
#: so consecutive posts do not carry an identical block. Relevance still wins -
#: a tag that genuinely belongs is allowed through. 0 disables the ledger.
POST_HASHTAG_HISTORY_RUNS = get_env_int("POST_HASHTAG_HISTORY_RUNS", 5)

#: Cross-run ledger backing the setting above.
POST_HASHTAG_USAGE_FILE = CAPTIONS_DIR / "hashtag_usage.json"

# --- Hashtag volume banding ------------------------------------------------
# The generator's "broad/mid/niche" tiers describe how the words feel, not how
# many posts the tag carries. #Piezophile is niche in the vocabulary sense and
# empty in the only sense that matters. These bands hold every tag to a post
# count instead, so the long-tail slot lands somewhere searchable rather than
# somewhere nobody has ever been. See pipeline/tagvolume.py.

#: Off puts the tiers back exactly as the generator returned them.
TAG_VOLUME_ENABLED = get_env_bool("TAG_VOLUME_ENABLED", True)

#: Below this a tag is "dead" and is dropped: too few posts to have browsers,
#: so it is not reach, only characters.
TAG_VOLUME_FLOOR = get_env_int("TAG_VOLUME_FLOOR", 5_000)

#: Band ceilings. niche = small enough for a small account to rank in;
#: mid = the band that actually converts; broad = places the post in a
#: subject area. Anything above the last is "saturated".
TAG_VOLUME_NICHE_MAX = get_env_int("TAG_VOLUME_NICHE_MAX", 150_000)
TAG_VOLUME_MID_MAX = get_env_int("TAG_VOLUME_MID_MAX", 2_000_000)
TAG_VOLUME_BROAD_MAX = get_env_int("TAG_VOLUME_BROAD_MAX", 25_000_000)

#: Saturated tags are not banned - one signals the subject to the recommender.
#: More than that and the post is competing where it cannot place.
TAG_VOLUME_MAX_SATURATED = get_env_int("TAG_VOLUME_MAX_SATURATED", 1)

#: Rather than post a short tag block, the floor is relaxed by this factor
#: (in stages) until the count is met. Every relaxation is warned about.
#: 1.0 disables it - a short block is then preferred to a weak one.
TAG_VOLUME_RELAX = get_env_float("TAG_VOLUME_RELAX", 0.2)

#: Known post counts, and where provider answers are cached. Hand-seeded
#: entries (no "at" timestamp) are authoritative and never expire.
TAG_VOLUME_FILE = CAPTIONS_DIR / "hashtag_volume.json"

#: How long a provider's answer stays good. 0 keeps them forever.
TAG_VOLUME_TTL_DAYS = get_env_int("TAG_VOLUME_TTL_DAYS", 30)

#: Dotted path to a callable taking a list of tags and returning
#: {tag: post count}, for whoever has a real data source. Blank uses the
#: offline estimator. There is no built-in Instagram provider: the Graph API
#: hashtag node carries no post count and allows 30 lookups per 7 days.
TAG_VOLUME_PROVIDER = get_env("TAG_VOLUME_PROVIDER", "")

#: Words that must never appear in the copy. Checked after generation, not just
#: asked for in the prompt.
POST_BANNED_WORDS = get_env_list("POST_BANNED_WORDS", "")

#: Phrases every caption must contain - a handle, a disclaimer. Appended
#: verbatim (case intact) if the generated copy does not already carry them.
POST_MUST_INCLUDE = get_env_list("POST_MUST_INCLUDE", "", lower=False)

#: What to do with step 4's per-asset credits, carried into the run directory
#: by step 5:
#:   required - credit only what the licence demands (CC-BY and friends).
#:              Pexels and Pixabay do not, so most runs add nothing.
#:   all      - also add a courtesy "via Pexels" line for the rest
#:   off      - never touch the caption
POST_ATTRIBUTION = get_env("POST_ATTRIBUTION", "required")
#: Heads the credit block. "Credits:" rather than "Footage:" because step 5
#: can put a music credit in there as well, and a bed credited under a heading
#: that says footage is a small lie in a published caption.
POST_ATTRIBUTION_HEADER = get_env("POST_ATTRIBUTION_HEADER", "Credits:")

#: Ceiling on individual credit lines, so a 15-shot reel does not bury the
#: copy. Credits a licence actually requires are never dropped to meet it - if
#: there are more of them than this, they all go in and a warning is raised
#: instead, because dropping one would breach the licence.
POST_ATTRIBUTION_MAX = get_env_int("POST_ATTRIBUTION_MAX", 6)

# --- alt text --------------------------------------------------------------
# The description of the cover image that goes in Instagram's "Write alt text"
# field. Two jobs, and they do not conflict: a screen reader reads it out, and
# Instagram reads it for what the picture is of. It is a description of an
# IMAGE, not a second caption - short, literal, no hashtags, no selling.

POST_ALT_ENABLED = get_env_bool("POST_ALT_ENABLED", True)

#: literal - assembled from what step 5 recorded about the cover: the hook card
#:           text burned into it, plus the provider's own words for the clip
#: llm     - the same facts, rephrased into a sentence by the caption call
#:           (no extra request - it is one more field in the JSON it already
#:           returns), falling back to `literal` if that fails
#:
#: `literal` is the default because the model cannot see the picture. Asked to
#: describe one, it writes something plausible instead - and step 4's own
#: `image_prompt` shows how far that lands from the truth: it asked for "a
#: snailfish in the deep ocean" and the clip that got used is a jellyfish in
#: bright blue water. Alt text is read by people who cannot check it.
POST_ALT_MODE = get_env("POST_ALT_MODE", "literal")

#: Instagram's alt text field. 100 characters is what the app has accepted for
#: years, and like every other limit here it is enforced in code rather than
#: asked for in a prompt.
POST_ALT_MAX_CHARS = get_env_int("POST_ALT_MAX_CHARS", 100)

#: Nothing is unconditionally required: `llm` reuses step 2's GEMINI_API_KEY and
#: degrades to `heuristic` without it.
REQUIRED_POST: tuple[str, ...] = ()


# Step 7 — Telegram delivery
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")
REQUIRED_TELEGRAM = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


# ---------------------------------------------------------------------------
# Step 8 — the end-to-end run
#
# One call that goes from "no topic" to a finished reel: step 0 chooses the
# subject, steps 2-6 make the thing. It adds no plumbing — every step already
# accepts the previous step's result dict — so what it actually supplies is the
# two things a human was supplying until now: which candidate off the board to
# make, and what to do when a step fails at 5pm with nobody watching.
#
# Both of those are decisions, so both are settings here rather than judgement
# buried in the runner.
# ---------------------------------------------------------------------------

#: Refuse to start when today already has a finished reel. This is the money
#: guard: every run spends Gemini calls and ElevenLabs characters, and the way
#: that bill runs away is not one expensive run, it is a scheduler that fired
#: twice because the machine woke up late and the catch-up ran on top of the
#: real one. `--force` overrides it for a deliberate second reel.
RUN_ONCE_PER_DAY = get_env_bool("RUN_ONCE_PER_DAY", True)

#: Read the numbers on the reels already out there at the start of every run.
#:
#: Step 9 can only compare readings taken between PERFORMANCE_MIN_AGE_HOURS and
#: 48h after posting, and `--fetch` was run by hand. Read off the ledger on
#: 2026-09-09: of the 7 posted reels, 5 had been read only past 48h and 1 only
#: at 5.6h - so 6 of 7 sat outside the window they had to be inside, and the
#: 5.6h reading gave the worst reel in the ledger a watch_ratio of 2.38, the
#: highest score in it. That is not a fetch anyone forgot to run: a 24-hour
#: window per reel cannot be hit by hand on any schedule a person keeps.
#:
#: A job that fires once every 24 hours lands inside any 24-hour span, so one
#: daily read guarantees every reel is asked about at least once while its
#: answer still counts. That is the whole argument for doing it here - step 8
#: is already the thing the scheduler runs every day.
RUN_FETCH_INSIGHTS = get_env_bool("RUN_FETCH_INSIGHTS", True)

#: How many topics off the board a single run may try before giving up on the
#: day. The second attempt exists for one failure in particular: step 4 coming
#: back with no footage, which does not mean the pipeline is broken - it means
#: *this topic* cannot be shot, and the board has ranked alternatives sitting
#: right there. Only failures that are about the topic fall through; a bad key
#: or a dead ffmpeg stops the run, because the next topic would fail the same
#: way and pay for the privilege. Set to 1 to never retry.
RUN_TOPIC_ATTEMPTS = get_env_int("RUN_TOPIC_ATTEMPTS", 2)

#: Score a candidate must reach before an unattended run will make it. 0 takes
#: the best of whatever the board holds, which is the right default: step 0
#: already gates the niche and probes for footage, so the top of the board has
#: cleared the checks that matter. Raise it if you would rather publish nothing
#: on a thin day than publish the best of a weak board - the run then ends
#: "skipped", not "failed", because nothing went wrong.
RUN_TOPIC_MIN_SCORE = get_env_float("RUN_TOPIC_MIN_SCORE", 0.0)

#: One line per run: when it started, what it made, what it cost, how it ended.
#: Append-only, and the file the once-per-day guard reads. This is what you
#: look at after a week of unattended runs to see which days produced nothing.
RUN_LEDGER_FILE = OUTPUT_DIR / "runs.json"

#: Runs kept in the ledger. Old entries are dropped from the front - it is a
#: record to glance at, not an archive, and the run directories are the real
#: history.
RUN_LEDGER_KEEP = get_env_int("RUN_LEDGER_KEEP", 90)

#: Days of per-run logs to keep in output/logs/. Pruned at the start of each
#: run, so an unattended job cannot slowly fill a disk with its own output.
#: 0 keeps them forever.
RUN_LOG_KEEP_DAYS = get_env_int("RUN_LOG_KEEP_DAYS", 30)

#: Both keys are required *before* the run starts rather than when each step
#: reaches them. A scheduled run that scouts topics for two minutes and then
#: dies on a missing ELEVENLABS_API_KEY has spent Gemini calls to discover
#: something knowable at second zero.
REQUIRED_RUN = REQUIRED_SCRIPT_GEN + REQUIRED_TTS


# --- step 9: how it did -----------------------------------------------------
# `pipeline/perform.py` records what a reel did once it was posted, and reads
# it back. Nothing here changes what gets made unless PERFORMANCE_FEEDBACK is
# turned on, and that is off until there are enough reels to argue with.

#: One entry per reel: what it was, and what it did. The report reads this file
#: alone, so pruning a run directory does not cost you its history.
PERFORMANCE_FILE = OUTPUT_DIR / "performance.json"

#: The number a reel is judged on. Watch time is the default because it is the
#: one metric that does not simply climb with the follower count: an account
#: that doubles its reach doubles every reel's views whatever was in them, and
#: a ledger ranking on views would spend a year discovering that later reels do
#: better than earlier ones.
#:
#: `watch_ratio` rather than `watch_through`, and the difference is the whole
#: reason this default moved. Instagram's average watch time counts replays, so
#: a looped reel reports more watch time than it has reel - 89.5s of watching
#: on a 37.6s reel, measured. `watch_through` is a share of the reel and is
#: therefore capped at 1.0, which turned that into a perfect score for the
#: worst-performing reel in the ledger; `watch_ratio` is the same division left
#: uncapped, so 2.38 stays 2.38 and the ranking keeps working above the cap.
#: Below 1.0 they are the same number. Both are stored on every reel.
#: Never `saves` or `shares`, however much they are the metrics that matter:
#: `ratios` divides by the median of the nearest reels and `local_baseline`
#: refuses a median of zero, so a metric that is zero on most reels reads as an
#: empty table rather than as an answer. `was_kept` reads those two as a rate
#: instead, in the report's SAVED OR SENT table.
PERFORMANCE_METRIC = get_env("PERFORMANCE_METRIC", "watch_ratio")

#: How many reels either side of one being scored make up its baseline. Every
#: figure in the report is a ratio to the median of a reel's own neighbours in
#: time, which is what takes the account's growth curve out of the comparison.
#: Neighbours on both sides, not only earlier ones - a trailing baseline leaves
#: the first reels of an account with nothing to be measured against, and that
#: is the half of the history where the curve is steepest.
PERFORMANCE_BASELINE_RUNS = get_env_int("PERFORMANCE_BASELINE_RUNS", 10)

#: Reels a segment needs on *both* sides of a question before the report reads
#: it out. At one reel a day this is the setting that decides whether the thing
#: is honest: eight-versus-eight is already a thin comparison, and three reels
#: that happened to do well will otherwise print as a finding.
PERFORMANCE_MIN_SAMPLES = get_env_int("PERFORMANCE_MIN_SAMPLES", 8)

#: Hours after posting within which a measurement is comparable with the rest.
#: A reel read at twelve hours and one read at a week are not the same
#: measurement, and averaging them measures when you happened to look. 0 turns
#: the window off and compares everything.
PERFORMANCE_WINDOW_HOURS = get_env_int("PERFORMANCE_WINDOW_HOURS", 48)

#: The other end of that window, and the one that was missing. A reel read six
#: hours after it went out has not finished happening: measured live, the first
#: reel to be read this way entered the ledger at three views and would have
#: been scored against reels read a day later as though its hook had failed.
#: A reading younger than this is kept and shown, and is not compared with
#: anything. 0 compares everything, however green.
PERFORMANCE_MIN_AGE_HOURS = get_env_int("PERFORMANCE_MIN_AGE_HOURS", 24)

#: How old a *reading* has to be before the provider stops re-reading the reel.
#: Numbers were taken once and never again, so whatever the reel happened to
#: have at the moment somebody ran --fetch is what it was judged on forever.
#: Under this age a posted reel is asked about again and the newer answer wins;
#: the older ones are kept in `measurements` rather than overwritten, because
#: how a reel got to its numbers is worth more than the number.
#: Sits inside PERFORMANCE_WINDOW_HOURS on purpose - re-reading past the window
#: would spend calls producing readings the report then refuses to compare.
PERFORMANCE_SETTLE_HOURS = get_env_int("PERFORMANCE_SETTLE_HOURS", 36)

#: Whether the ledger may put a "measured" component on step 0's board. Off,
#: because the honest use of this for a long while is a report a person reads.
#: Turned on, it adds one more named component that argues its own case in the
#: printed board like every other one - it never edits a weight in the code.
PERFORMANCE_FEEDBACK = get_env_bool("PERFORMANCE_FEEDBACK", False)

#: The most that component may ever be worth, either way. Deliberately small
#: beside the hand-set weights of 1 to 3: measured evidence at this sample size
#: is a nudge to a ranking, not a replacement for one.
PERFORMANCE_MAX_ADJUST = get_env_float("PERFORMANCE_MAX_ADJUST", 1.0)

#: Reels the ledger must hold before feedback does anything at all, whatever
#: PERFORMANCE_FEEDBACK says. A loop that starts steering after four reels is
#: not learning, it is amplifying the first thing that happened.
PERFORMANCE_MIN_REELS = get_env_int("PERFORMANCE_MIN_REELS", 30)

#: One thing to vary deliberately from reel to reel, so there is ever anything
#: to compare: "off", "hook" (run the hook pass on alternate reels) or
#: "duration" (a step shorter, then a step longer than step 0 suggested).
#: Without a trial the pipeline makes the same choice every day and the ledger
#: fills up with 365 reels that cannot answer whether it was the right one.
#: Arms alternate rather than randomise - see perform.choose_arm.
PERFORMANCE_EXPLORE = get_env("PERFORMANCE_EXPLORE", "off")

#: Dotted path to something that can fetch the numbers - `mypkg.mod.fetch`,
#: handed the waiting run ids and expected to return {run_id: {metric: value}}.
#: Empty, and the numbers are typed in - which stays the right answer for the
#: first few weeks, because the one column that matters most early on is the
#: reels you decided not to post and no API has it. `pipeline.instagram.fetch`
#: is the provider in this repo; it needs the two keys below.
PERFORMANCE_PROVIDER = get_env("PERFORMANCE_PROVIDER", "")


# --- step 9: reading the numbers off Instagram ------------------------------
# Only used when PERFORMANCE_PROVIDER points at pipeline.instagram.fetch.
# Nothing in this pipeline posts anything: these read insights and nothing else.

#: A long-lived Instagram Graph API token. Sixty days, then it has to be
#: refreshed - `python pipeline/instagram.py --refresh` does it and prints the
#: replacement to paste back in here.
INSTAGRAM_ACCESS_TOKEN = get_env("INSTAGRAM_ACCESS_TOKEN")

#: The Instagram *professional* account's numeric user id, not the @handle.
#: `python pipeline/instagram.py --check` prints it once the token works.
INSTAGRAM_USER_ID = get_env("INSTAGRAM_USER_ID")

REQUIRED_INSTAGRAM = ("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID")

#: Which of Meta's two hosts answers for the token, or empty to work it out
#: from the token itself. They are not interchangeable: a token minted through
#: Instagram Login (those all begin "IG") is only parseable by
#: graph.instagram.com, and one from the Facebook Login flow ("EAA...") only by
#: graph.facebook.com. Sending either to the other host is answered with
#: "Cannot parse access token", which reads exactly like an expired token and
#: is not one - hence the default is to follow the token rather than ask.
INSTAGRAM_API_HOST = get_env("INSTAGRAM_API_HOST", "")

#: Graph API version, pinned rather than left to float onto Meta's newest.
#: The metric names are exactly what churns between versions, so an
#: unversioned call is a provider that breaks on somebody else's release day.
INSTAGRAM_API_VERSION = get_env("INSTAGRAM_API_VERSION", "v21.0")

#: How many recent posts to read when looking for the reel a run made. The
#: match is by caption first, so this only has to be deep enough to still hold
#: a post from the day the reel was made.
INSTAGRAM_MEDIA_LIMIT = get_env_int("INSTAGRAM_MEDIA_LIMIT", 50)

#: How far from a reel's own creation time a post may be and still be taken as
#: that reel when the caption did not settle it. Wide, because a reel made on
#: Tuesday is often posted on Wednesday - but not so wide that two reels made
#: in the same week become interchangeable.
INSTAGRAM_MATCH_HOURS = get_env_int("INSTAGRAM_MATCH_HOURS", 96)


# --- scheduling -------------------------------------------------------------
# `pipeline/schedule.py` registers step 8 with the OS scheduler. Nothing here
# runs anything by itself - these are the values that get written into the task
# definition when you install it, and re-reading them changes nothing until you
# install again.

#: When the daily run fires, 24-hour local time.
SCHEDULE_TIME = get_env("SCHEDULE_TIME", "17:00")

#: Task name in Windows Task Scheduler. Also what --uninstall looks for.
SCHEDULE_TASK_NAME = get_env("SCHEDULE_TASK_NAME", "ReelCreatorAgent")

#: Run as soon as the machine is available again after a missed start. This is
#: the setting that makes a *local* daily job work at all: a desktop is asleep
#: or off at 5pm more often than not, and without it those days silently
#: produce nothing. The once-per-day guard is what stops the catch-up run
#: doubling up with a real one.
SCHEDULE_CATCH_UP = get_env_bool("SCHEDULE_CATCH_UP", True)

#: Wake the machine from sleep to run. Off: waking a sleeping computer every
#: evening to make a video nobody asked for yet is more than a reel is worth,
#: and SCHEDULE_CATCH_UP already covers the day.
SCHEDULE_WAKE = get_env_bool("SCHEDULE_WAKE", False)

#: If the run exits non-zero, try again after this many minutes, this many
#: times. Aimed at the failure a retry actually fixes - a provider having a bad
#: ten minutes, or no network because the laptop has not found the wifi yet.
SCHEDULE_RETRY_MINUTES = get_env_int("SCHEDULE_RETRY_MINUTES", 30)
SCHEDULE_RETRY_COUNT = get_env_int("SCHEDULE_RETRY_COUNT", 2)

#: Kill a run that has been going this long. A reel is minutes of work; an hour
#: in, something is hung, and a hung run holds the once-per-day slot.
SCHEDULE_TIME_LIMIT_MINUTES = get_env_int("SCHEDULE_TIME_LIMIT_MINUTES", 120)


ensure_dirs()


# ---------------------------------------------------------------------------
# Self-check: `python config.py`
# ---------------------------------------------------------------------------

def _mask(value: str | None) -> str:
    if not value:
        return "(not set)"
    return f"{value[:4]}...{value[-2:]} ({len(value)} chars)" if len(value) > 8 else "set"


def _report() -> None:
    print(f"BASE_DIR         : {BASE_DIR}")
    print(f".env present     : {ENV_FILE.exists()}")
    print("\nDirectories")
    for directory in _MANAGED_DIRS:
        rel = directory.relative_to(BASE_DIR)
        print(f"  {'ok ' if directory.is_dir() else 'MISSING'} {rel}")

    groups = {
        "Step 0 topics": ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                          "YOUTUBE_API_KEY"),
        "Step 2 script gen": REQUIRED_SCRIPT_GEN,
        "Step 3 tts": REQUIRED_TTS,
        "Step 4 visuals": ("PEXELS_API_KEY", "PIXABAY_API_KEY",
                           "STOCK_API_KEY", "YOUTUBE_API_KEY"),
        "Step 6 post text": ("GEMINI_API_KEY",),
        "Step 7 telegram": REQUIRED_TELEGRAM,
        "Step 9 instagram": REQUIRED_INSTAGRAM,
    }
    print("\nSecrets")
    for label, keys in groups.items():
        for key in keys:
            print(f"  {label:<18} {key:<22} {_mask(get_env(key))}")

    print("\nStep 0 topic sources (in the order they are asked)")
    for name in TOPIC_SOURCES:
        needed = TOPIC_SOURCE_KEYS.get(name)
        if needed is None:
            state = "unknown source name"
        elif not needed:
            state = "ready (no key required)"
        else:
            missing = [k for k in needed if not get_env(k)]
            if not missing:
                state = "ready"
            elif name in TOPIC_SOURCE_OPTIONAL_KEYS:
                state = ("will try anonymously - set " + " + ".join(missing)
                         + ", which is what Reddit now needs in practice")
            else:
                state = "needs " + " + ".join(missing)
        if name in ("reddit", "redditrss") and not TOPIC_SUBREDDITS:
            state = "needs TOPIC_SUBREDDITS"
        print(f"  {name:<12} {state}")

    print("\nStep 4 providers (in preference order)")
    for name in VISUAL_PROVIDERS:
        needed = VISUAL_PROVIDER_KEYS.get(name)
        if needed is None:
            state = "unknown provider name"
        elif not needed:
            state = "ready (no key required)"
        else:
            have = next((k for k in needed if get_env(k)), None)
            state = (f"ready (via {have})" if have
                     else "needs " + " or ".join(needed))
        print(f"  {name:<12} {state}")

    print("\nNon-secret settings")
    for key, value in (
        ("TOPIC_NICHE", TOPIC_NICHE or "(not set - inferred per source)"),
        ("TOPIC_NICHE_MODE",
         f"{TOPIC_NICHE_MODE}, decided by "
         + ("one Gemini call over the shortlist"
            if TOPIC_NICHE_MATCH == "llm"
            else "words from TOPIC_NICHE (free, much stricter)")),
        ("TOPIC_SUBREDDITS", ", ".join(TOPIC_SUBREDDITS) or "(none set)"),
        ("TOPIC_REDDIT_FEEDS",
         f"{TOPIC_REDDIT_FEEDS} feed(s), so "
         f"~{-(-len(TOPIC_SUBREDDITS) // max(1, TOPIC_REDDIT_FEEDS))} "
         f"subreddit(s) ranked against each other at a time"
         if TOPIC_SUBREDDITS else f"{TOPIC_REDDIT_FEEDS}"),
        ("TOPIC_REDDIT_LISTING",
         f"{TOPIC_REDDIT_LISTING} of the {TOPIC_REDDIT_WINDOW}"
         if TOPIC_REDDIT_LISTING == "top" else TOPIC_REDDIT_LISTING),
        ("TOPIC_CANDIDATES", f"{TOPIC_CANDIDATES} shown, "
                             f"{TOPIC_PER_SOURCE} read per source"),
        ("TOPIC_SOURCEABILITY",
         f"asked of the model, top {TOPIC_PROBE_LIMIT}"
         if TOPIC_SOURCEABILITY == "llm"
         else f"probe, {TOPIC_SOURCEABLE_HITS} of {TOPIC_PROBE_SAMPLE} results "
              f"naming it, top {TOPIC_PROBE_LIMIT} (measured not to separate "
              f"good topics from bad - see config.py)"
         if TOPIC_SOURCEABILITY == "probe"
         else "off (the board may hold topics step 4 cannot shoot)"),
        ("TOPIC_PHRASE_MODE", TOPIC_PHRASE_MODE),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("SCRIPT_LENGTH", f"{SCRIPT_DEFAULT_SECONDS}s default, step 0 suggests "
                          f"{SCRIPT_MIN_SECONDS}-{SCRIPT_MAX_SECONDS}s "
                          f"in steps of {SCRIPT_DURATION_STEP_SECONDS}"),
        ("ELEVENLABS_MODEL", ELEVENLABS_MODEL),
        ("DEFAULT_VOICE_ID", f"{DEFAULT_VOICE_ID}"
                             f"{'' if ELEVENLABS_VOICE_ID else '  (fallback)'}"),
        ("VIDEO", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps"),
        ("VISUAL_QUERY_MODE", VISUAL_QUERY_MODE),
        ("VISUAL_VARIETY",
         f"<= {VISUAL_SIMILAR_OVERLAP} description overlap"
         + (f", <= {VISUAL_MAX_PER_CREATOR} clip(s) per creator"
            if VISUAL_MAX_PER_CREATOR else ", any number per creator")),
        ("VISUAL_RELEVANCE",
         f"{VISUAL_RELEVANCE_MODE}, top {VISUAL_RELEVANCE_SHORTLIST} per beat"
         + (f", lexical floor {VISUAL_RELEVANCE_FLOOR}"
            if VISUAL_RELEVANCE_MODE != "off" else "")),
        ("VISUAL_MIN_RESOLUTION",
         f"{VISUAL_MIN_WIDTH}x{VISUAL_MIN_HEIGHT} (after 9:16 crop)"),
        ("VISUAL_ASSETS_PER_BEAT", VISUAL_ASSETS_PER_BEAT),
        ("VISUAL_DURATION_HEADROOM",
         f"{VISUAL_DURATION_HEADROOM}x narration"),
        ("VISUAL_REUSE_COOLDOWN_RUNS", VISUAL_REUSE_COOLDOWN_RUNS),
        ("SHOT_TARGET_SECONDS", f"{SHOT_TARGET_SECONDS}s per shot "
                                f"({SHOT_MIN_SECONDS}-{SHOT_MAX_SECONDS}s)"
                                + (f", never past {SHOT_MAX_OVERRUN:g}x the curve"
                                   if SHOT_MAX_OVERRUN > 0 else "")),
        ("SHOT_PACING", (f"curve: {SHOT_HOOK_SECONDS:g}s opening -> "
                         f"{SHOT_TARGET_SECONDS:g}s over "
                         f"{SHOT_PACING_RAMP_SECONDS:g}s -> "
                         f"{SHOT_OUTRO_SECONDS:g}s over the last "
                         f"{SHOT_OUTRO_WINDOW_SECONDS:g}s")
                        if SHOT_PACING == "curve"
                        else f"flat {SHOT_TARGET_SECONDS:g}s throughout"),
        ("SHOT_SNAP_TO_WORDS",
         f"on, up to {SHOT_SNAP_MAX_SECONDS:g}s of nudge"
         if SHOT_SNAP_TO_WORDS else "off (cuts land on the arithmetic)"),
        ("SHOT_SNAP_TO_BEAT",
         (f"on, first {SHOT_BEAT_WINDOW_SECONDS:g}s, "
          f"up to {SHOT_BEAT_MAX_SECONDS:g}s of nudge"
          + (f", {MUSIC_BPM:g} BPM fallback" if MUSIC_BPM > 0
             else " (needs a bpm: line in a track's .txt)"))
         if SHOT_SNAP_TO_BEAT else "off (the words take every cut)"),
        ("TRANSITION", f"{TRANSITION} ({TRANSITION_SECONDS}s)"),
        ("SHOT_MOTION", f"{SHOT_MOTION}, {SHOT_MOTION_ZOOM:g}x "
                        f"({SHOT_MOTION_HOOK_ZOOM:g}x on the open)"
                        if SHOT_MOTION not in ("", "off", "none") else "off"),
        ("AUDIO_NORMALIZE",
         f"{AUDIO_LOUDNESS_TARGET:g} LUFS, peak {AUDIO_LOUDNESS_PEAK:g} dBTP"
         if AUDIO_NORMALIZE else "off (narration left as rendered)"),
        ("CAPTION_ALIGNMENT", CAPTION_ALIGNMENT),
        ("CAPTION_STYLE", f"{CAPTION_STYLE}, {CAPTION_FONT_NAME} "
                          f"{CAPTION_FONT_SIZE}px, "
                          f"<={CAPTION_MAX_WORDS} words/cue"),
        ("CAPTION_FONT_FILE", CAPTION_FONT_FILE or "(auto-resolved)"),
        ("CAPTION_POP",
         (f"{CAPTION_POP_SCALE}% settling over {CAPTION_POP_MS}ms, "
          f"payoff words {CAPTION_PAYOFF_SCALE}% in {CAPTION_PAYOFF_COLOR}")
         if CAPTION_POP else "off (cues cut in at full size)"),
        ("HOOK_MODE", f"{HOOK_MODE}"
                      + (f", best of {HOOK_VARIANTS} openings, "
                         f"<={HOOK_MAX_WORDS} words"
                         if HOOK_MODE == "llm" else "")),
        ("TTS_DELIVERY", f"speed {TTS_SPEED:g}, stability {TTS_STABILITY:g}, "
                         f"style {TTS_STYLE:g} "
                         f"(-> {round(SCRIPT_WORDS_PER_MINUTE * TTS_SPEED)} wpm)"),
        ("STRUCTURE_MODE", f"{STRUCTURE_MODE}"
                           + (f", rewritten below {STRUCTURE_MIN_SCORE:g}"
                              if STRUCTURE_MODE == "llm" else "")),
        ("TAKEAWAY_MODE", f"{TAKEAWAY_MODE}"
                           + (f", rewritten below {TAKEAWAY_MIN_SCORE:g}"
                              if TAKEAWAY_MODE == "llm" else "")),
        ("VERIFY_MODE", f"{VERIFY_MODE}"
                        + (f", false claims and self-contradictions refuse "
                           f"the script, up to {VERIFY_REPAIR_ROUNDS} repair "
                           f"round(s) to save it"
                           if VERIFY_MODE == "llm" else "")),
        ("HOOK_CARD", (f"{HOOK_CARD_SECONDS:g}s, {HOOK_CARD_FONT_SIZE}px, "
                       f"<={HOOK_CARD_MAX_WORDS} words")
                      if HOOK_CARD_ENABLED else "off"),
        ("COVER", (f"{COVER_SOURCE}, from the first {COVER_SEARCH_SECONDS:g}s, "
                   f"safe at {COVER_SAFE_ASPECT} -> {COVER_FILENAME}")
                  if COVER_ENABLED else "off"),
        ("VIDEO_RUN_DIR", f"output/<{VIDEO_RUN_DATE_FORMAT}>/<run id>/"
                          f"{VIDEO_FILENAME}"),
        ("MUSIC", (f"{MUSIC_LOUDNESS_TARGET:g} LUFS"
                   + (", ducked" if MUSIC_DUCKING else ", flat")
                   + (f", pinned to {MUSIC_FILE}" if MUSIC_FILE
                      else ", rotating over assets/music/")
                   + (f", {MUSIC_BPM:g} BPM" if MUSIC_BPM > 0 else ""))
                  if MUSIC_ENABLED else "off"),
        ("SFX", (f"{SFX_PEAK_TARGET:g} dBFS peak"
                 + (", whoosh on cuts" if SFX_ON_CUTS else "")
                 + (", impact on the open" if SFX_IMPACT_AT_OPEN else "")
                 + f", from assets/sfx/")
                if SFX_ENABLED else "off"),
        ("POST_MODE", POST_MODE),
        ("POST_HASHTAGS", f"{POST_HASHTAG_COUNT} tags in the "
                          f"{POST_HASHTAG_PLACEMENT}"
                          + (f", always {' '.join('#' + t for t in POST_HASHTAGS_ALWAYS)}"
                             if POST_HASHTAGS_ALWAYS else "")),
        ("TAG_VOLUME", (f"banded, {TAG_VOLUME_FLOOR:,}+ posts, "
                        f"<= {TAG_VOLUME_MAX_SATURATED} saturated"
                        + (f", via {TAG_VOLUME_PROVIDER}"
                           if TAG_VOLUME_PROVIDER else ", estimated"))
                       if TAG_VOLUME_ENABLED
                       else "off - the generator's own tiers"),
        ("POST_LIMITS", f"{POST_MAX_CHARS} chars, hook <= {POST_HOOK_MAX_CHARS}"),
        ("POST_ATTRIBUTION", POST_ATTRIBUTION),
        ("POST_ALT", f"{POST_ALT_MODE}, <= {POST_ALT_MAX_CHARS} chars"
                     if POST_ALT_ENABLED else "off"),
        ("RUN_GUARD", "one reel per day" if RUN_ONCE_PER_DAY
                      else "off - every run makes another reel"),
        ("RUN_INSIGHTS", "read at the start of every run"
                         if RUN_FETCH_INSIGHTS
                         else "off - numbers are read by hand"),
        ("RUN_TOPIC_PICK",
         f"top of the board, up to {RUN_TOPIC_ATTEMPTS} topic(s) per run"
         + (f", score >= {RUN_TOPIC_MIN_SCORE:g}" if RUN_TOPIC_MIN_SCORE
            else "")),
        ("RUN_LOGS", f"output/logs/, kept {RUN_LOG_KEEP_DAYS} days"
                     if RUN_LOG_KEEP_DAYS else "output/logs/, kept forever"),
        ("PERFORMANCE", f"on {PERFORMANCE_METRIC}, n>={PERFORMANCE_MIN_SAMPLES} "
                        f"to read a segment"
                        + (f", via {PERFORMANCE_PROVIDER}"
                           if PERFORMANCE_PROVIDER else ", recorded by hand")
                        + (f"  (feeding back, <= {PERFORMANCE_MAX_ADJUST:g} "
                           f"after {PERFORMANCE_MIN_REELS} reels)"
                           if PERFORMANCE_FEEDBACK
                           else "  (report only - nothing feeds back)")),
        ("PERFORMANCE_EXPLORE",
         f"{PERFORMANCE_EXPLORE}, alternating per reel"
         if PERFORMANCE_EXPLORE not in ("", "off", "none")
         else "off - every reel makes the same choices, so nothing "
              "can be compared"),
        ("SCHEDULE", f"{SCHEDULE_TIME} daily, as \"{SCHEDULE_TASK_NAME}\""
                     + (", catching up a missed day" if SCHEDULE_CATCH_UP
                        else ", skipping a missed day")
                     + (", waking the machine" if SCHEDULE_WAKE else "")
                     + "  (install it: python pipeline/schedule.py install)"),
    ):
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    _report()
