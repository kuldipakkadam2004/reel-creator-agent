"""Step 0 - topic scouting.

Where the topic comes from when you have not brought one. Gathers candidates
from public sources, scores them here in code, and prints a ranked shortlist
with the exact `script_gen.py` command for whichever one you pick.

It does not pick for you, and that is deliberate. The decision that matters is
which topic gets made, and a model asked to rank its own candidates will
explain why all eight are excellent - the same reason step 2 scores its own
hooks in code and step 6 enforces its own character limits after the call.
Everything here can be read back: every candidate prints the parts its score
was built from.

What happens, in order:

  1. each configured source is read - reddit's top posts of the week (needs a
     free script app), youtube's last month by views (needs a key), google's
     trending searches (needs neither) - and an unready source is skipped with
     a warning rather than failing the scout;
  2. titles are cleaned into topic phrases and deduplicated across sources;
  3. everything is scored on velocity, niche fit, curiosity, shape and age,
     and anything off the account's subject is held off the board while there
     is something on it to show;
  4. the top TOPIC_PROBE_LIMIT are *checked for footage* - the check that
     makes this specific to this pipeline, and the subject of the long comment
     further down. By default the model is asked whether a stock library could
     show the topic; TOPIC_SOURCEABILITY=probe reads stock search results
     instead, which was measured not to separate good topics from bad;
  5. the survivors are re-ranked, the winners' titles are polished into topic
     phrases, and the board is recorded on the ledger so the same candidate
     does not head every board.

Nothing here writes a script, spends an ElevenLabs credit or downloads a byte
of media. The output is a list of strings and the command to use one.

Standalone use:
    python pipeline/topics.py
    python pipeline/topics.py --niche "ocean science" --source reddit
    python pipeline/topics.py --sources-status
    python pipeline/topics.py --take 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

# Allow both `python pipeline/topics.py` and `python -m pipeline.topics`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import perform, script_gen, visuals  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables. Which sources, which subreddits, the keys and the probe threshold
# all live in config.py - what follows only shapes the local scoring, the same
# split step 4 makes.
# ---------------------------------------------------------------------------

#: (connect, read) seconds. These are all small JSON/XML bodies.
FETCH_TIMEOUT = (10, 20)

#: Sent on every request that is not Reddit's (which insists on its own).
USER_AGENT = "reel-creator-agent/0.1 (topic scouting)"

# --- velocity ---------------------------------------------------------------
# The one signal that actually says "trending". It is deliberately a *rate*
# rather than a total: a post with 8,000 upvotes accumulated over a week is a
# good post, not a moving one, and the reel would arrive after the wave.
#
# It used to be the largest single positive on the board, and that was a
# mistake worth writing down. Every source here is ranked by motion before this
# code sees it, so velocity does not separate a good candidate from a bad one -
# it separates the front of a feed from the back of it, which the feed had
# already done. Priced at 3.0 against 1.5 for being off the account's subject
# entirely, the arithmetic said a fast enough wrestling result beat a slower
# ocean one, and on board #15 that is exactly what it said. Motion is now worth
# less than subject and about what shape is worth.

#: Ceiling the velocity component pays out at.
VELOCITY_WEIGHT = 1.8

#: What counts as "clearly moving", in each source's own unit. Raw numbers are
#: not comparable across sources - 300 upvotes an hour and 300 youtube views a
#: day are not the same event - so each is normalised against its own
#: reference before the scores meet, and only then can they be ranked together.
SOURCE_REFERENCE_RATE = {
    "reddit": 300.0,        # upvotes per hour
    "redditrss": 300.0,     # stand-in rate from the listing rank - see below
    "youtube": 20_000.0,    # views per day
    "trends": 50_000.0,     # approximate searches
    "wikipedia": 150_000.0,  # views per day, and already filtered to a rise
}

#: A candidate that is both very fresh and already moving is on the way up
#: rather than at its peak, which is the half of the wave worth catching.
RISING_BONUS = 0.8
RISING_MAX_AGE_HOURS = 12.0
RISING_RATE_FRACTION = 0.5

#: Comments per upvote, capped. A post people argued about has an angle in it;
#: one people only upvoted is a picture.
DISCUSSION_WEIGHT = 0.6

# --- shape ------------------------------------------------------------------

#: Niche fit. The miss is a real penalty rather than a zero because a board
#: that wanders out of the account's subject is worse than a short board.
#:
#: Under the default TOPIC_NICHE_MODE=gate the penalty barely matters - an
#: off-niche candidate is held off the board entirely while there is anything
#: on-niche to show, so what this really does is order the backfill on a thin
#: day. It is priced to also stand on its own under `prefer`, where it is the
#: only thing holding a national trend feed back.
NICHE_MATCH_WEIGHT = 2.0
NICHE_MISS_PENALTY = -3.0

#: Explainer phrasing - "why", "how", "what happens when". These are the
#: topics that can be told in 35 seconds and shot with library footage. The
#: format's whole engine is a question a viewer did not know they had.
EXPLAINER_BONUS = 1.2

# --- curiosity --------------------------------------------------------------
# What makes someone stop, as against what makes something popular. Everything
# else in this file scores whether a topic is *available*: moving, on-subject,
# shootable, the right shape and length. None of that says the reel is worth
# watching, and a board can be full of perfectly available boring topics.
#
# What a short-form explainer actually trades on is a gap - a claim that sounds
# wrong until it is explained, a number nobody would have guessed, one end of a
# range. Those leave marks in the wording and the marks are cheap to read.
#
# Two caveats, because the rest of this file earns its numbers by measurement
# and this one does not yet. First, these weights are a prior: nothing here has
# been checked against how a finished reel performed, because the pipeline
# cannot see that - posting is manual and no step reads the numbers back. The
# honest fix is a performance ledger, and when there is one these are the first
# weights it should overwrite. Second, a signal in a *title* is weaker evidence
# than the same signal in a hook, because the title was written to win a feed
# this pipeline is not posting to. Both are why the ceiling here sits below
# what being on the niche is worth.

#: Per distinct signal. Three signals exist, so the most this can add is 1.5 -
#: under NICHE_MATCH_WEIGHT on purpose, per the caveats above.
CURIOSITY_SIGNAL_WEIGHT = 0.5

#: Hype in the source's own title, applied once however much of it there is.
#: The opposite of a curiosity gap: a headline that has to tell you it is
#: amazing is one with nothing specific to open on, and step 2 scores the same
#: words down in a hook for the same reason.
HYPE_PENALTY = -0.6

#: Proper nouns, per name, and the most they can cost.
#:
#: Note this is the *opposite* sign to step 2's hook scoring, which pays for
#: naming something - and both are right. A hook names something because a
#: specific image stops a scroll. A topic that is mostly names is a fixture,
#: an election or a product launch: it decays in hours and no stock library
#: holds a frame of it. The real arbiter is the sourceability probe below;
#: this is the cheap prior that runs first and decides who is worth probing.
PROPER_NOUN_PENALTY = -0.8
PROPER_NOUN_FLOOR = -2.4

#: What counts as a Title Case headline rather than a sentence with names in
#: it. Words of four letters or more only, because Title Case leaves "of" and
#: "the" lower case and counting them would hide the pattern.
TITLE_CASE_RATIO = 0.6
TITLE_CASE_MIN_WORDS = 3

#: A title so short it is a fragment, or so long it is a paragraph.
TITLE_SHORT_PENALTY = -1.0
TITLE_LONG_PENALTY = -0.8
TITLE_MIN_WORDS = 4
TITLE_MAX_WORDS = 16

#: Past the source's own age window. Arriving late to a trend is worse than
#: being evergreen: the same competition, none of the window. A source with no
#: window at all - see below - never pays it, which is the point of reading one.
STALE_PENALTY = -2.0

#: How far back each Reddit `top` window reaches, in hours. 0 means no window,
#: which `max_age_for` reads as "nothing from here can be late": `top/all` is a
#: standing best-of, and a mechanism that was true four years ago is not stale,
#: it is simply still true. That is the whole case for reading `top` over `hot`
#: on an account whose subject does not expire.
_REDDIT_WINDOW_HOURS = {
    "hour": 1.0, "day": 24.0, "week": 168.0,
    "month": 720.0, "year": 8760.0, "all": 0.0,
}

#: Listings Reddit serves under /r/<subs>/. Only `top` takes a window.
_REDDIT_LISTINGS = frozenset({"hot", "top", "new", "rising", "best"})

#: A word from TOPIC_BANNED_WORDS. Large enough to sink a candidate outright
#: without needing a second mechanism to do it.
BANNED_WORD_PENALTY = -6.0

#: Shown on a recent board and not taken. Not a ban - if it is still the best
#: thing available it still wins - but a board that repeats itself is a board
#: worth running once.
SCOUTED_PENALTY = -2.5

#: What a topic off the evergreen backlog gives up for not being news.
#:
#: The backlog is the *floor* under a thin board, not the ceiling over a good
#: one, and without a penalty it would be the ceiling: a hand-written line is
#: on-niche by construction, explainer-shaped by construction and carries no
#: proper nouns, so it banks its niche and shape components before a scouted
#: candidate has finished paying for its own headline.
#:
#: The arithmetic, because the first number tried here was wrong and the way
#: it was wrong is worth keeping. A backlog line shares no *word* with a niche
#: like "ocean, earth science and nature", so it is scored through
#: NICHE_VERDICT_SHARE and collects 1.0, not the full NICHE_MATCH_WEIGHT of
#: 2.0. Its ceiling is therefore 1.0 niche + 1.2 shape + 1.5 footage + at most
#: 0.5 curiosity = 4.2, not the 5.2 a quick reading suggests, and a penalty
#: priced off the wrong ceiling left every line scraping RUN_TOPIC_MIN_SCORE.
#:
#: At -1.8 a clean line with footage lands at 1.9, or 2.4 carrying curiosity -
#: clear of a 1.0 floor, and under the 2.60 that board #24's best real find
#: scored. That is the intended order: a moving on-niche story wins, the
#: backlog beats filler, and a backlog line the probe cannot shoot (footage
#: +0.167, total 0.57) still fails the floor rather than shipping.
#:
#: Like the curiosity weights above, this is a prior and not a measurement.
#: There is no performance ledger yet to say whether an evergreen reel does
#: worse than a topical one; when there is, this is among the first numbers it
#: should overwrite.
EVERGREEN_PENALTY = -1.8

#: SCOUTED_PENALTY's replacement for a backlog line, because the full one and
#: EVERGREEN_PENALTY together cost more than a backlog line can ever score.
#:
#: Measured on board #41 (2026-09-01): the daily run skipped with "the best of
#: 8 scored -0.1", and all 8 were backlog lines carrying both penalties. The
#: ceiling documented on EVERGREEN_PENALTY above is 1.0 niche + 1.2 shape + 1.5
#: footage + 0.5 curiosity = 4.2, so -2.5 and -1.8 together put the *best
#: possible* backlog line at -0.1 against a RUN_TOPIC_MIN_SCORE of 1.0. The
#: board's top score being exactly -0.1 is that ceiling, not a coincidence.
#:
#: It could not recover on its own either. All 45 evergreen entries in the
#: ledger had been boarded inside TOPIC_HISTORY_RUNS, and the lines that score
#: best are the ones shown most often, so the good ones stayed the poisoned
#: ones. Left alone the scheduled run skips every day, forever.
#:
#: The two numbers were priced independently and never against each other:
#: EVERGREEN_PENALTY's own arithmetic ("lands at 1.9, or 2.4 carrying
#: curiosity - clear of a 1.0 floor") assumes nothing else is subtracted.
#:
#: Not zero, because a board that opens on the same line every day is still
#: worth varying, and `used` already drops a taken line outright. At -0.6 a
#: clean line with footage lands at 1.3 and one carrying curiosity at 1.8, both
#: clear of the floor, while a line the probe cannot shoot (footage +0.167)
#: lands at -0.03 and still fails it - which is the trade EVERGREEN_PENALTY
#: documents wanting.
EVERGREEN_SCOUTED_PENALTY = -0.6

#: A topic whose script the fact-check refused, and which the board offered
#: again anyway.
#:
#: Measured across the logs to 2026-09-08: "what the sound of the deep ocean
#: actually is" was chosen and refused **five times** - the phrase pass rewords
#: it every run, so the log shows three spellings, but the ledger key is stable
#: and it sat on 26 boards before a draft finally survived. Every one of those
#: five costs a ~150s Pro call, and nothing recorded that the last one failed:
#: a topic is marked `used` only once money has gone on it (run.py), which a
#: refused draft never reaches, so the whole cost of a refusal was a single
#: EVERGREEN_SCOUTED_PENALTY of -0.6 for having been on a board.
#:
#: Priced against the ceiling documented on EVERGREEN_PENALTY, not in the
#: abstract - the mistake board #41 was built out of. A clean backlog line
#: lands at 1.3 and one carrying curiosity at 1.8, so -1.2 puts a
#: once-refused line under a RUN_TOPIC_MIN_SCORE of 1.0 unless it is carrying
#: enough else to be worth the second call. That is the intent: a refusal is
#: evidence about the topic, not proof - the deep-ocean line did eventually
#: pass - so it costs a retry rather than banning one.
#:
#: Charged **once**, however many times a topic has been refused, so it cannot
#: stack its way into another deadlock. What ends a topic is the count instead:
#: at TOPIC_MAX_REFUSALS it is dropped from the board outright, the same
#: treatment `used` gets, where a filter cannot interact with the arithmetic.
REFUSED_PENALTY = -1.2

# --- sourceability ----------------------------------------------------------
# The check that makes step 0 belong to *this* pipeline rather than being a
# generic trend reader.
#
# Step 4 sources from stock libraries. Half of what is trending anywhere is
# proper nouns that no stock library holds a single frame of, and a reel about
# one of those does not fail - it comes out as generic filler laid over words
# that needed specific pictures, which is the expensive kind of wrong: it costs
# a Gemini call, an ElevenLabs render and a full encode before anyone can see
# that it went wrong.
#
# The obvious implementation is to ask the provider how many results it has and
# call a small number unshootable. That does not work, and it is worth writing
# down why: Pexels reports about 2,500 results for `zzqxwv`. Stock APIs do not
# return nothing - they fall back to a generic pool, so the total is the same
# large number whatever is searched for. Measured, the counts for "ocean",
# "shawver" and a random string were 8000, 2526 and 2560.
#
# So the probe reads the results instead. `provider.search` already returns the
# titles and source URLs, and Pexels names each clip in its own link even when
# the alt text is empty - "serene-ocean-surface-with-gentle-waves". How many of
# the returned clips name what was asked for is a real measurement.
#
# It is a strong positive and a weak negative, and the scoring reflects that.
# Results that name the topic mean it is definitely shootable. Results that do
# not may still be shootable through a synonym the provider understood and this
# does not: "bioluminescence" returns jellyfish footage, which is exactly right
# and matches not one letter of the query. That asymmetry is why a failed probe
# scores down instead of dropping - see the drop rule in `find_topics`.

#: Score at no matching results and at TOPIC_SOURCEABLE_HITS, interpolated
#: between and flat above. Continuous on purpose - a step would put a cliff
#: between two candidates one result apart. The floor is well short of the
#: banned-word penalty because a miss here is evidence, not proof.
SOURCEABLE_FLOOR = -2.5
SOURCEABLE_BONUS = 1.5

#: Words in the query the probe sends. The provider is being asked whether the
#: subject exists in its library at all, so a short query is the right one -
#: a five-word query returns nothing for topics that are perfectly shootable.
PROBE_QUERY_WORDS = 3

#: Words no stock library has anything to show for.
#:
#: They survive step 4's stopword list because they are content words, and they
#: beat the real subject to a slot in the query because `_keywords` scores on
#: length: "actually" is eight characters and "ocean" is five. Measured on a
#: real board, "how deep the ocean actually goes" probed as 'deep ocean
#: actually' and found nothing, and "why the sky turns red at sunset" probed as
#: 'sky turns sunset' and found one result in eight. Dropping them promotes the
#: next real word rather than shortening the query.
_UNFILMABLE = frozenset({
    "actually", "really", "happens", "happen", "happened", "nothing",
    "something", "anything", "everything", "enough", "thing", "things",
    "reason", "reasons", "process", "point", "part", "parts", "kind", "sort",
    "builds", "build", "built", "decides", "decide", "decided", "becomes",
    "become", "makes", "make", "made", "goes", "going", "gets", "getting",
    "turns", "turned", "before", "after", "during", "while", "always",
    "never", "often", "still", "even", "just", "also", "very", "quite",
})

#: Shortest prefix two words must share to be treated as the same word. Long
#: enough that "ocean" and "octopus" stay apart, short enough that "salty"
#: matches "salt" and "volcano" matches "volcanoes". Below it, words have to
#: match exactly.
STEM_LENGTH = 4

#: How many of the query's words a result has to name to count as being about
#: the topic. One - and the alternatives were measured, not assumed.
#:
#: Requiring two looks obviously better: "ted cruz american" matched all eight
#: results on a one-word rule, every one a stock flag, because "american" is
#: in half the library. But measured across six known-good topics and four
#: known-bad ones, requiring two scored zero for FIVE OF THE SIX GOOD ONES -
#: 'lava ocean water' 0/8, 'ocean salty' 0/8, 'raindrop lands' 0/8 - because a
#: library describes a clip in its own words and rarely in two of yours at
#: once. Taking the minimum across terms was worse still: zero everywhere.
#:
#: One word separates them: good topics scored 2,6,3,7,3,7 and bad ones
#: 1,6,0,1. The single overlap is 'seattle seahawks trevon', where "seattle"
#: is genuinely filmable and simply is not what the video is about - a
#: distinction no amount of reading search results can recover, because the
#: results are right and the query was the wrong question.
#:
#: So the probe is left deliberately permissive. A wrong pass costs a place on
#: a board a human is reading anyway, with the footage count printed beside it;
#: a wrong drop silently removes a topic that could have been made.
PROBE_MIN_TERM_HITS = 1
PROBE_MIN_TERM_HITS_FROM = 2

# --- length -----------------------------------------------------------------
# How long the reel should be, in seconds off config.SCRIPT_DEFAULT_SECONDS.
#
# These are adjustments to a base rather than a scale of their own, because
# there is no natural zero: every topic gets a length, and the question is only
# whether this one earns more or less than the default. The signals are the
# ones the board already computed for scoring - nothing here is a new
# measurement, which is the argument for making the call in step 0 at all
# instead of asking a second model how long to write.

#: "why", "how", "what happens when" - a mechanism, and a mechanism is setup,
#: turn and payoff. That is three beats where a fact is one, and the reel has
#: to hold all three or the payoff lands on nobody.
DURATION_EXPLAINER_BONUS = 8.0

#: A named subject is usually a fixture - a match, a launch, a ruling. It has
#: one thing to say, it decays in hours, and the extra seconds go to restating
#: it. Applied once however many names there are: unlike the score's own
#: penalty this is not cumulative, because two names do not make a topic half
#: as long again as one.
DURATION_FIXTURE_PENALTY = -6.0

#: An argument people are having has more than one side to put, and the second
#: side is what the middle of the reel is for. Scaled by the same
#: comments-to-reach ratio the score uses.
DURATION_DISCUSSION_BONUS = 4.0

#: What the footage probe is worth, interpolated between these across the same
#: 0..TOPIC_SOURCEABLE_HITS range as `sourceability_component`.
#:
#: Deliberately asymmetric. Thin footage is a real argument against length -
#: fewer distinct clips over more seconds is the same shots coming round
#: again, which reads as padding whatever the narration is doing. Plenty of
#: footage is not the mirror of that: it removes the objection rather than
#: making a case, so it pays a third as much as thinness costs.
#:
#: It is also why this is an adjustment and not a cap. A probe that found
#: nothing may only mean the library knew a synonym this does not -
#: `bioluminescence` returns jellyfish - so the miss shortens a reel it cannot
#: be trusted to veto. The conclusive case already has its own hard filter in
#: `find_topics`.
DURATION_FOOTAGE_THIN = -6.0
DURATION_FOOTAGE_RICH = 2.0

# --- the niche pass ---------------------------------------------------------

#: Gemini settings for the on/off-niche classification. Temperature 0: this is
#: a closed question with a right answer, not a piece of writing.
NICHE_TEMPERATURE = 0.0
NICHE_MAX_TOKENS = 4096
NICHE_THINKING_BUDGET = 0

#: What a model verdict is worth on its own, as a share of NICHE_MATCH_WEIGHT,
#: when the topic shares no word with the niche description.
#:
#: The two inputs answer different halves of the question and are kept apart on
#: purpose. The verdict decides *membership* - whether the account covers this
#: at all, which is about meaning. Shared words decide *strength* - how squarely
#: it sits, which is about wording. A candidate the model placed on-niche with
#: nothing in common still pays this much, because between the two the verdict
#: is the more reliable and the wording is the part that was already wrong.
NICHE_VERDICT_SHARE = 0.5

# --- the sourceability pass -------------------------------------------------

#: Gemini settings for the footage question. Temperature 0 for the same reason
#: the niche pass uses it: a closed question with a right answer.
SOURCEABILITY_TEMPERATURE = 0.0
SOURCEABILITY_MAX_TOKENS = 4096
SOURCEABILITY_THINKING_BUDGET = 0

#: The scale the model answers on, and what each rung means. Kept small and
#: named in the prompt, because a model asked for "0 to 100" invents precision
#: it does not have and clusters on round numbers.
#:
#:   0 - nothing depicting this exists: a person, a fixture, a named event, a
#:       product. This is the rung the hard filter reads.
#:   1 - only loosely related or generic imagery
#:   2 - the subject can be shown with common footage, but not exactly
#:   3 - a library plainly holds shots of this
SOURCEABILITY_SCALE = 3

#: What each rung is worth, named rather than interpolated.
#:
#: The rungs are not evenly spaced in meaning, so spacing them evenly in score
#: was measured to be wrong. Running the 62-line evergreen backlog through the
#: pass on the straight line `sourceability_component` draws - rung 2 landing
#: at +0.17, a drop of 1.33 from rung 3 - put ten lines under
#: RUN_TOPIC_MIN_SCORE that the keyword probe had passed, and they were lines
#: like "what happens inside a volcano before it erupts", "why earthquakes
#: happen where two plates meet" and "how a sinkhole opens without warning".
#: The model was right to call those a 2 - nobody has filmed inside a volcano
#: - and the reels are perfectly makeable from footage that stands in. The
#: rung was reading correctly and being priced as a near-failure.
#:
#: So rung 2 sits just under the bonus: it is a topic that will be shot with
#: stand-ins, which is what half this account's back catalogue already is. The
#: cliff belongs between 1 and 0, where "filler" becomes "there is nothing".
SOURCEABILITY_COMPONENTS = {
    0: SOURCEABLE_FLOOR,
    1: -0.7,
    2: 0.9,
    3: SOURCEABLE_BONUS,
}


def sourceability_rung(rating: int) -> float:
    """What a model rating is worth. Clamped, because the model can answer 7."""
    rating = min(SOURCEABILITY_SCALE, max(0, int(rating)))
    return SOURCEABILITY_COMPONENTS[rating]


# --- the phrase pass --------------------------------------------------------

#: Gemini settings for rewriting titles into topic phrases. Low temperature:
#: these are labels, not prose.
PHRASE_TEMPERATURE = 0.3
PHRASE_MAX_TOKENS = 1024
#: Output budget per headline, used when the batch is bigger than the fixed
#: floor above covers. One {"index": 0, "topic": "...", "explains": true}
#: object runs about 31 tokens; 56 leaves room for a long subject without
#: risking a truncated array.
PHRASE_TOKENS_PER_PHRASE = 56
PHRASE_THINKING_BUDGET = 0

#: How much of a rewrite has to have come from the original before it counts
#: as a rewrite rather than a new topic.
#:
#: Measured on the rewrite, not on the original, and the direction matters. A
#: long headline - "We've Explored Only 5% of the Ocean... What's Hiding in
#: the Other 95%?" - cannot keep most of its own words in a six-word topic, so
#: scoring recall threw away good rewrites. What actually signals drift is a
#: rewrite made of words that were never there.
PHRASE_MIN_OVERLAP = 0.5

#: How many boards the ledger keeps. Well past TOPIC_HISTORY_RUNS so raising
#: that setting does not need a fresh ledger.
LEDGER_KEEP_BOARDS = 200

#: Searches the youtube source may spend on one scout. Each costs 100 quota
#: units of a 10,000/day allowance, so this is not the binding constraint -
#: it is here so a long TOPIC_KEYWORDS cannot quietly become 40 searches.
MAX_YOUTUBE_SEARCHES = 3

#: Forum furniture, stripped from the front of a title before anything reads
#: it. Reddit titles carry a lot of it and none of it is the topic.
_TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"til(?:\s+that)?|eli5|psa|oc|update|discussion|question|help|serious|"
    r"image|video|article|news|breaking|opinion|meta|megathread|"
    r"\[[^\]]*\]|\([^)]*\)"
    r")\s*[:\-–—]?\s*",
    flags=re.IGNORECASE,
)

#: Trailing furniture: credit tags, resolution stamps, source attributions.
_TITLE_SUFFIX_RE = re.compile(
    r"\s*(?:\[[^\]]*\]|\([^)]*\)|\|\s*\S.*|-\s*(?:youtube|reddit|bbc|cnn)\b.*)\s*$",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")

#: Creator tags, which every YouTube short carries a tail of.
_HASHTAG_RE = re.compile(r"(?:(?<=\s)|^)#\w+")

#: Openings that make a topic an explainer rather than an event.
_EXPLAINER_OPENERS = (
    "why", "how", "what", "when", "where", "who",
)
_EXPLAINER_PHRASES = (
    "the reason", "the science", "turns out", "actually", "really works",
    "what happens", "the truth about", "the story of", "inside",
)

#: Words that sit a claim against something else - the shape of "true, and
#: yet". Matched as substrings, so they can be phrases.
_CURIOSITY_CONTRADICTION = (
    "despite", "even though", "even when", "but still", "actually",
    "turns out", "myth", "misconception", "wrong about", "nobody knows",
    "no one knows", "still don't know", "still dont know", "paradox",
    "counterintuitive", "counter-intuitive", "isn't", "isnt", "aren't",
    "arent", "doesn't", "doesnt", "don't", "dont", "can't", "cant",
    "cannot", "won't", "wont", "shouldn't", "shouldnt", "hasn't", "hasnt",
    "never", " not ",
)

#: One end of a range. An extreme is a place a viewer can be taken, and it is
#: the difference between "the ocean is deep" and "the deepest point there is".
#: Matched as whole words - "epic" inside "epicentre" is not a superlative,
#: and on an earth-science account that would fire constantly.
_CURIOSITY_EXTREME = frozenset({
    "most", "least", "largest", "biggest", "smallest", "deepest", "highest",
    "oldest", "longest", "fastest", "slowest", "rarest", "strongest",
    "only", "first", "last", "hidden", "unexplained", "unsolved", "extinct",
})

#: Scale. A number in a title promises something specific, and these units
#: carry a sense of size even without one.
_CURIOSITY_SCALE = frozenset({
    "million", "billion", "trillion", "thousand", "hundred", "percent",
    "times", "years", "decades", "centuries", "miles", "kilometres",
    "kilometers", "metres", "meters", "tons", "tonnes", "degrees",
})

#: Hype, matched as whole words. Kept deliberately in step with step 2's
#: `_HOOK_VAGUE`: the same words are worthless in a hook and in a title, and
#: two lists that drifted apart would be two different opinions about that.
_CURIOSITY_HYPE = frozenset({
    "amazing", "incredible", "unbelievable", "mindblowing", "insane",
    "crazy", "shocking", "epic", "stunning", "wild", "jaw-dropping",
    "mind-blowing", "gamechanger", "revolutionary", "ultimate",
})

_CURIOSITY_DIGIT_RE = re.compile(r"\d")

#: Words too common to carry a subject, stripped out of the niche vocabulary
#: before anything is scored against it.
#:
#: The niche used to be read from TOPIC_NICHE *and* TOPIC_KEYWORDS together,
#: which quietly conflated two different things: the niche says what the
#: account is about, the keywords are query strings for the YouTube search. A
#: keyword of "natural world" donated the bare token "world" to the subject
#: vocabulary, and on board #15 that is what scored "2026 Men's FIH Hockey
#: World Cup" a full niche match on an ocean-science account while "the false
#: positive paradox" was marked off-niche. Keywords are now a fallback used
#: only when no niche is set at all, and this list catches the rest.
_GENERIC_NICHE_TERMS = frozenset({
    "world", "work", "works", "working", "thing", "things", "stuff",
    "explained", "explain", "explainer", "explaining", "fact", "facts",
    "video", "videos", "shorts", "channel", "content", "guide", "tips",
    "best", "top", "new", "interesting", "amazing", "story", "stories",
    "about", "day", "days", "time", "times", "way", "ways",
})

#: Capitalised words that are not the kind of proper noun that hurts. A place
#: or a body in the solar system is shot by every stock library on earth; a
#: footballer is shot by none of them.
_HARMLESS_PROPER_NOUNS = frozenset({
    "earth", "moon", "sun", "mars", "venus", "jupiter", "saturn", "mercury",
    "neptune", "uranus", "pluto", "antarctica", "arctic", "atlantic",
    "pacific", "indian", "sahara", "amazon", "himalayas", "everest", "nile",
    "africa", "asia", "europe", "america", "australia", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "i",
})


class TopicsError(RuntimeError):
    """Raised when the scout cannot run at all - never for one bad source."""


class SourceError(TopicsError):
    """One source failed. Caught by the scout and turned into a warning."""

    def __init__(self, message: str, *, source: str = ""):
        super().__init__(message)
        self.source = source


@dataclass
class Candidate:
    """One topic under consideration, and everything known about it."""

    source: str
    external_id: str
    #: Exactly as the source gave it. Kept so a board can be audited.
    title: str
    #: The cleaned phrase - what step 2 would be handed.
    topic: str = ""
    url: str = ""
    context: str = ""
    #: Reach in the source's own unit (upvotes, views, searches) and the rate
    #: that reach is arriving at, which is what velocity scores.
    reach: float = 0.0
    rate: float = 0.0
    rate_unit: str = ""
    comments: int = 0
    age_hours: float = 0.0
    score: float = 0.0
    components: dict = dataclass_field(default_factory=dict)
    sourceability: dict = dataclass_field(default_factory=dict)
    #: True while `topic` is a cleaned headline the phrase pass has yet to
    #: rewrite - so the wording is not what would ship, and the parts of the
    #: score that read wording cannot be charged against it yet. Cleared by
    #: `polish_phrases` for every candidate it is handed, rewritten or not:
    #: after that pass the headline standing *is* the final wording.
    provisional: bool = False
    #: Whether the headline names anything a video could explain, as the
    #: phrase pass judged it. None when nothing asked - the pass is off, it
    #: failed, or this candidate never reached it - and None is kept, never
    #: dropped. See TOPIC_REQUIRE_EXPLAINABLE.
    explains: bool | None = None
    #: How long the reel about this should be, and the parts that decided it.
    #: Filled in after the phrase pass, because the phrasing is what says
    #: whether this is a mechanism or a fact.
    duration: int = 0
    duration_components: dict = dataclass_field(default_factory=dict)
    notes: list = dataclass_field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable ledger identity. Survives a retitle by the source."""
        return f"{self.source}:{self.external_id}"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "source": self.source,
            "topic": self.topic,
            "title": self.title,
            "url": self.url,
            "context": self.context,
            "reach": round(self.reach, 1),
            "rate": round(self.rate, 2),
            "rate_unit": self.rate_unit,
            "comments": self.comments,
            "age_hours": round(self.age_hours, 1),
            "score": self.score,
            "components": self.components,
            "sourceability": self.sourceability,
            "explains": self.explains,
            "duration": self.duration,
            "duration_components": self.duration_components,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def clean_title(raw: str) -> str:
    """Turn a source's title into something step 2 could be handed.

    Strips the furniture (TIL, ELI5, [OC], trailing credits), unwraps quotes
    and normalises whitespace. Deliberately conservative: this runs on every
    fetched item before anything is scored, so it may not change the meaning -
    the model pass later on is where a title becomes a good topic phrase.
    """
    text = unicodedata.normalize("NFKC", raw or "").strip()
    if not text:
        return ""

    # Step 2's scrubber, because these end up as a topic that step 2 is handed
    # and step 6 writes a caption from. A thumbnail-bait emoji is not part of
    # what the video is about.
    text = script_gen._EMOJI_RE.sub(" ", text)
    # Step 2's ranges do not cover every emoji a title can carry, and the ones
    # they miss reached a cp1252 console as "?". Anything Unicode files as a
    # symbol, or that lives outside the basic plane, is not part of a topic.
    text = "".join(
        ch for ch in text
        if ord(ch) <= 0xFFFF and unicodedata.category(ch) not in ("So", "Cs", "Co")
    )
    # Creator tags: "... Save Coastlines #shorts #nature". They are metadata
    # about the video, never about the subject.
    text = _HASHTAG_RE.sub(" ", text)

    previous = None
    while previous != text:            # "[OC] TIL that ..." needs two passes
        previous = text
        text = _TITLE_PREFIX_RE.sub("", text, count=1).strip()
    text = _TITLE_SUFFIX_RE.sub("", text).strip()

    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()

    text = re.sub(r"\s+", " ", text)
    # A title that opened "TIL that the ocean ..." is left starting on "that".
    text = re.sub(r"^that\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" .,:;–—-")


def _terms(text: str) -> set[str]:
    """Content words, lowercased. Step 4's stopword list, so the two agree."""
    return {
        word.lower() for word in _WORD_RE.findall(text or "")
        if len(word) > 2 and word.lower() not in visuals._STOPWORDS
    }


def slugify(text: str, limit: int = 60) -> str:
    """Comparison key for deduplicating the same story from two sources."""
    normalised = unicodedata.normalize("NFKD", text or "")
    ascii_only = normalised.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug[:limit].strip("-")


def _is_title_case(words: list[str]) -> bool:
    """True when a headline capitalises words as a house style.

    YouTube titles are written like this - "What If All Oceans Vanished" - and
    against them a capital letter says nothing about whether a word names
    anything. Measured on the first live board, every YouTube candidate took
    the full proper-noun penalty and not one of them contained a name.
    """
    substantial = [w for w in words[1:] if len(w) > 3]
    if len(substantial) < TITLE_CASE_MIN_WORDS:
        return False
    capitalised = sum(1 for w in substantial if w[:1].isupper())
    return capitalised / len(substantial) >= TITLE_CASE_RATIO


def _proper_nouns(text: str) -> list[str]:
    """Capitalised words past the first that name something specific.

    Empty for a Title Case headline: the signal this reads is capitalisation,
    and where capitalisation is a house style there is no signal to read. That
    also holds back the drop rule in `find_topics`, which is right - it exists
    to catch a named subject with no footage, and a title that may not contain
    a name at all is not evidence of one.
    """
    words = _WORD_RE.findall(text or "")
    if _is_title_case(words):
        return []
    out = []
    for word in words[1:]:
        if not word[:1].isupper():
            continue
        if word.lower() in _HARMLESS_PROPER_NOUNS or word.isupper():
            continue        # acronyms are usually the subject, not a name
        out.append(word)
    return out


def max_age_for(source: str) -> float:
    """How old is too old, for this source.

    TOPIC_MAX_AGE_HOURS is a breaking-news window and it is the wrong ruler for
    YouTube, which is deliberately not a breaking-news source: it is asked what
    has *already worked* over the last month, ranked by views per day. Measured
    on the first live board, every YouTube candidate was twelve days old and
    every one of them took the stale penalty - the source was being punished
    for answering the question it was asked. Its window is its own.
    """
    if source == YouTubeSource.name:
        return max(1.0, config.TOPIC_YOUTUBE_WINDOW_DAYS * 24.0)

    if source in (RedditSource.name, RedditRssSource.name):
        # The same argument, for the same reason. `top/week` returns posts up
        # to a week old *by design*, so judging them against a 72-hour news
        # window scores down the six-day-old post that won the week - the
        # source punished for answering the question it was asked. The other
        # listings are current by construction and keep the news window.
        listing, window = reddit_listing()
        return (_REDDIT_WINDOW_HOURS[window] if listing == "top"
                else config.TOPIC_MAX_AGE_HOURS)
    if source == WikipediaSource.name:
        # The pageviews API only publishes a day once it is complete, so the
        # freshest candidate this source can offer is already a day old and a
        # breaking-news ruler would mark all of them stale on arrival.
        return max(config.TOPIC_MAX_AGE_HOURS,
                   (WikipediaSource.LAG_DAYS + 2) * 24.0)
    return config.TOPIC_MAX_AGE_HOURS


def _is_explainer(topic: str) -> bool:
    lowered = (topic or "").lower()
    first = lowered.split(" ", 1)[0] if lowered else ""
    return (first in _EXPLAINER_OPENERS
            or any(phrase in lowered for phrase in _EXPLAINER_PHRASES))


def _niche_fit(topic: str, niche_terms: set[str] | None,
               verdict: bool | None = None) -> tuple[bool | None, float]:
    """(on the niche?, how strongly) for one topic.

    `None` for the first when there is nothing to judge against - no niche set
    and no verdict - which is the case where the board simply does not score
    the subject at all.

    See NICHE_VERDICT_SHARE for why a verdict and a word overlap are read as
    answers to two different questions rather than as two votes on one.
    """
    overlap = 0
    if niche_terms:
        # _same_word, not set intersection: a niche of "ocean" has to match a
        # topic about "oceans", and exact comparison silently did not.
        overlap = sum(
            1 for term in _terms(topic)
            if any(_same_word(term, want) for want in niche_terms)
        )

    if verdict is None:
        if not niche_terms:
            return None, 0.0
        verdict = bool(overlap)

    if not verdict:
        return False, 0.0
    return True, max(NICHE_VERDICT_SHARE, min(1.0, overlap / 2.0))


def niche_vocabulary(niche: str, keywords=None) -> set[str]:
    """The subject words a candidate's topic is judged against.

    The niche alone, minus the words too common to mean anything. Keywords are
    a *fallback* for an account that set TOPIC_KEYWORDS and no TOPIC_NICHE, not
    an addition - see `_GENERIC_NICHE_TERMS` for what mixing them cost.
    """
    terms = _terms(niche)
    if not terms:
        terms = {term for keyword in (keywords or ()) for term in _terms(keyword)}
    return {term for term in terms if term not in _GENERIC_NICHE_TERMS}


def curiosity_signals(topic: str) -> list[str]:
    """Which curiosity signals a topic's wording carries, in a stable order.

    Returned as names rather than a number so the board can print what fired -
    "curiosity (contradiction, scale)" is an argument a reader can disagree
    with, where "+1.0" is one they can only accept.
    """
    lowered = (topic or "").lower()
    words = {word.strip("-'’") for word in _WORD_RE.findall(lowered)}
    found: list[str] = []

    if any(phrase in lowered for phrase in _CURIOSITY_CONTRADICTION):
        found.append("contradiction")
    if words & _CURIOSITY_EXTREME:
        found.append("extreme")
    if _CURIOSITY_DIGIT_RE.search(lowered) or (words & _CURIOSITY_SCALE):
        found.append("scale")
    return found


def curiosity_component(topic: str) -> tuple[float, list[str]]:
    """Curiosity score for a topic, and the signals that produced it.

    Linear in the number of *distinct* signals, not in how often each one
    fires. A title carrying three separate reasons to keep watching is a
    better bet than one saying "never" three times.
    """
    signals = curiosity_signals(topic)
    return round(CURIOSITY_SIGNAL_WEIGHT * len(signals), 3), signals


def hype_words(topic: str) -> list[str]:
    """Hype adjectives in a title, deduplicated and in the order they appear."""
    out: list[str] = []
    for word in _WORD_RE.findall((topic or "").lower()):
        word = word.strip("-'’")
        if word in _CURIOSITY_HYPE and word not in out:
            out.append(word)
    return out


def reddit_listing() -> tuple[str, str]:
    """(listing, window) for the reddit sources, normalised.

    Anything Reddit does not serve falls back rather than 404s the source: a
    typo in an .env should cost a default, not the whole board.
    """
    listing = (config.TOPIC_REDDIT_LISTING or "").strip().lower()
    if listing not in _REDDIT_LISTINGS:
        listing = "top"
    window = (config.TOPIC_REDDIT_WINDOW or "").strip().lower()
    if window not in _REDDIT_WINDOW_HOURS:
        window = "week"
    return listing, window


# ---------------------------------------------------------------------------
# HTTP
#
# JSON goes through step 4's `_get_json` rather than a second copy of it: the
# status mapping, the timeouts and the "provider rejected the key" case are
# already written and already tested, and a source that fails here is handled
# the same way a provider that fails there is - skipped, with a warning.
# ---------------------------------------------------------------------------

def _new_session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get_json(session, url: str, *, source: str, params=None, headers=None) -> dict:
    """GET a JSON body, mapping every failure onto SourceError."""
    try:
        return visuals._get_json(
            session, url, provider=source, params=params, headers=headers
        )
    except visuals.ProviderError as exc:
        raise SourceError(str(exc), source=source) from exc


def _get_bytes(session, url: str, *, source: str, params=None) -> bytes:
    """GET a body as bytes. The trends feed is RSS, so it cannot go through above.

    Bytes rather than text on purpose: `requests` guesses latin-1 for an XML
    body served without a charset, which turns every accented name in the feed
    into mojibake. The XML declaration says what the encoding really is, so the
    parser is the right thing to decode it.
    """
    import requests

    try:
        response = session.get(url, params=params, timeout=FETCH_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise SourceError(
            f"{source} timed out after {FETCH_TIMEOUT[1]}s", source=source
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise SourceError(
            f"could not reach {source}: {type(exc).__name__}: {exc}", source=source
        ) from exc

    if response.status_code >= 400:
        raise SourceError(
            f"{source} failed [{response.status_code}]: "
            f"{(response.text or '').strip()[:200]}",
            source=source,
        )
    return response.content or b""


# ---------------------------------------------------------------------------
# Sources
#
# Each one maps its own JSON onto Candidate and nothing else: no scoring, no
# filtering, no deduplication. Adding a source means writing one `fetch` and
# adding it to _SOURCES.
# ---------------------------------------------------------------------------

class Source:
    """Base class. Subclasses fill in `name`, `key_env`, `fetch`."""

    name = ""
    key_env: tuple[str, ...] = ()
    #: True when the source runs without its keys, more thinly.
    keys_optional = False
    #: The unit `Candidate.rate` is measured in, for the printed board.
    rate_unit = ""
    #: False where `Candidate.topic` is a cleaned headline rather than a topic
    #: phrase, which is every source that reads someone else's titles. The
    #: backlog is the exception: its lines are written as topics already.
    wording_is_final = False

    def __init__(self, session):
        self.session = session
        self.requests = 0
        #: What went wrong without stopping the fetch. A source that reads
        #: several feeds can lose one and still return a board, and the
        #: difference between "fetched 40" and "fetched 40 of the 60 asked for,
        #: because a third of the subreddits were rate limited" is not visible
        #: anywhere else - `fetch` returning candidates is the only signal the
        #: scout otherwise gets.
        self.notes: list[str] = []

    @classmethod
    def missing_keys(cls) -> list[str]:
        return [name for name in cls.key_env if not config.get_env(name)]

    @classmethod
    def overrides(cls, **supplied) -> dict:
        """Which of the caller's overrides this source takes. Base takes none.

        Here rather than in `find_topics` so that what a source can be handed
        is declared by the source, not by a chain of name comparisons in the
        scout - which is what makes adding one a matter of writing `fetch`.
        """
        return {}

    @classmethod
    def readiness(cls, **supplied) -> tuple[bool, str]:
        """(ready, why). Ready sources still explain what they are running on.

        Takes the same overrides as `overrides()`, because whether a source can
        run may depend on them - a subreddit list passed on the command line
        makes reddit ready even with nothing in .env, and a readiness check
        that could not see it would gate the source before the override was
        ever applied.
        """
        missing = cls.missing_keys()
        if not missing:
            return True, "ready"
        if cls.keys_optional:
            return True, ("will try anonymously - set " + " + ".join(missing)
                          + " if it is refused")
        return False, "needs " + " + ".join(missing)

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        raise NotImplementedError


class RedditSource(Source):
    """Hot posts from your own subreddits.

    The strongest of the three, and not because the numbers are better. A
    reddit title is a hook a human wrote and several thousand people have
    already voted on, so what comes back is a *framing* rather than a noun -
    and the comment count says whether there is an argument in it. Google
    Trends can tell you that people searched for something; it cannot tell you
    which sentence made them care.

    It is also the one source here that in practice needs credentials. Reddit
    used to serve the public .json endpoints to anything with a User-Agent;
    measured against the live site, an unauthenticated read now comes back 403
    with an HTML block page. The anonymous path is still attempted, because it
    costs one request and still works from some clients, but a free "script"
    app at https://www.reddit.com/prefs/apps is what makes this source usable.
    """

    name = "reddit"
    key_env = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
    #: Anonymous reads still work often enough to be worth attempting - see
    #: `_explain` - but they are no longer something to rely on.
    keys_optional = True
    rate_unit = "upvotes/hr"

    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_URL = "https://oauth.reddit.com/r/{subs}/{listing}"
    PUBLIC_URL = "https://www.reddit.com/r/{subs}/{listing}.json"

    def __init__(self, session, subreddits=None):
        super().__init__(session)
        self.subreddits = tuple(subreddits or config.TOPIC_SUBREDDITS)

    @classmethod
    def overrides(cls, **supplied) -> dict:
        subreddits = supplied.get("subreddits")
        return {"subreddits": subreddits} if subreddits else {}

    @classmethod
    def readiness(cls, **supplied) -> tuple[bool, str]:
        """Keys are optional here; a subreddit list is not."""
        if not (supplied.get("subreddits") or config.TOPIC_SUBREDDITS):
            return False, "needs TOPIC_SUBREDDITS"
        return super().readiness(**supplied)

    def _token(self) -> str | None:
        """Client-credentials token, or None to fall back to anonymous.

        A failure here is not fatal: the public .json endpoints serve the same
        listings without a key, only harder rate limited.
        """
        import requests

        client_id = config.get_env("REDDIT_CLIENT_ID")
        secret = config.get_env("REDDIT_CLIENT_SECRET")
        if not (client_id and secret):
            return None
        try:
            response = self.session.post(
                self.TOKEN_URL,
                auth=(client_id, secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": config.REDDIT_USER_AGENT},
                timeout=FETCH_TIMEOUT,
            )
            self.requests += 1
            if response.status_code >= 400:
                return None
            return (response.json() or {}).get("access_token") or None
        except (requests.exceptions.RequestException, ValueError):
            return None

    @classmethod
    def _explain(cls, exc: SourceError, *, authenticated: bool) -> SourceError:
        """Turn Reddit's refusal into something worth reading.

        A blocked anonymous read answers 403 with several kilobytes of
        stylesheet, and the first 200 characters of that is what would
        otherwise land in the warning line - which says nothing about what to
        do next. The status is the whole of the information here.
        """
        message = str(exc)
        if not any(code in message for code in ("[401", "[403", "[429")):
            return exc
        if authenticated:
            return SourceError(
                "Reddit rejected REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET, or "
                "rate limited this app. Check the credentials are for a "
                "'script' app and that REDDIT_USER_AGENT names it.",
                source=cls.name,
            )
        return SourceError(
            "Reddit refused an unauthenticated read - it no longer serves the "
            "public .json endpoints to scripts. Set REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET from a free 'script' app at "
            "https://www.reddit.com/prefs/apps",
            source=cls.name,
        )

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        if not self.subreddits:
            raise SourceError(
                "no subreddits configured - set TOPIC_SUBREDDITS in .env "
                "(whose feed this is depends on the account, so there is no "
                "sensible default)",
                source=self.name,
            )

        subs = "+".join(self.subreddits)
        listing, window = reddit_listing()
        token = self._token()
        if token:
            url = self.OAUTH_URL.format(subs=subs, listing=listing)
            headers = {"Authorization": f"bearer {token}",
                       "User-Agent": config.REDDIT_USER_AGENT}
        else:
            url = self.PUBLIC_URL.format(subs=subs, listing=listing)
            headers = {"User-Agent": config.REDDIT_USER_AGENT}

        params = {"limit": min(max(limit, 1), 100), "raw_json": 1}
        if listing == "top":
            params["t"] = window

        try:
            payload = _get_json(
                self.session, url, source=self.name, headers=headers,
                params=params,
            )
        except SourceError as exc:
            raise self._explain(exc, authenticated=bool(token)) from exc
        self.requests += 1

        now = datetime.now(timezone.utc)
        out: list[Candidate] = []
        for child in ((payload.get("data") or {}).get("children") or []):
            post = child.get("data") or {}
            # Stickies are the subreddit's own furniture, not what is moving.
            if post.get("stickied") or post.get("over_18"):
                continue
            title = clean_title(post.get("title") or "")
            if not title:
                continue
            created = float(post.get("created_utc") or 0)
            age = max(
                (now - datetime.fromtimestamp(created, timezone.utc)).total_seconds()
                / 3600.0,
                0.25,
            ) if created else 24.0
            upvotes = float(post.get("score") or 0)
            permalink = post.get("permalink") or ""
            out.append(Candidate(
                source=self.name,
                external_id=str(post.get("id") or slugify(title)),
                title=post.get("title") or "",
                topic=title,
                url=f"https://www.reddit.com{permalink}" if permalink else "",
                context=f"r/{post.get('subreddit') or '?'} {listing}",
                reach=upvotes,
                rate=upvotes / age,
                rate_unit=self.rate_unit,
                comments=int(post.get("num_comments") or 0),
                age_hours=age,
            ))
        return out


class YouTubeSource(Source):
    """Videos published inside the last window, ranked by views per day.

    Not the `videos?chart=mostPopular` trending endpoint, on purpose: that is
    music, film trailers and celebrity, whatever niche you point it at. A
    keyword search bounded by `publishedAfter` and ordered by view count asks a
    better question - what has already worked in short form, in this subject,
    recently - and views per day is the answer, because raw views only says
    which video has been up longest.
    """

    name = "youtube"
    key_env = ("YOUTUBE_API_KEY",)
    rate_unit = "views/day"

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(self, session, keywords=None, window_days=None):
        super().__init__(session)
        self.keywords = tuple(keywords or config.TOPIC_KEYWORDS)
        self.window_days = int(
            config.TOPIC_YOUTUBE_WINDOW_DAYS if window_days is None else window_days
        )

    @classmethod
    def overrides(cls, **supplied) -> dict:
        keywords = supplied.get("keywords")
        return {"keywords": keywords} if keywords else {}

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        key = config.get_env("YOUTUBE_API_KEY")
        if not key:
            raise SourceError("needs YOUTUBE_API_KEY", source=self.name)

        queries = [q for q in (self.keywords or ((niche,) if niche else ())) if q]
        if not queries:
            raise SourceError(
                "nothing to search for - set TOPIC_KEYWORDS or TOPIC_NICHE",
                source=self.name,
            )
        queries = queries[:MAX_YOUTUBE_SEARCHES]

        published_after = (
            datetime.now(timezone.utc) - timedelta(days=self.window_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        found: dict[str, dict] = {}
        for query in queries:
            payload = _get_json(
                self.session, self.SEARCH_URL, source=self.name,
                params={
                    "key": key,
                    "q": query,
                    "part": "snippet",
                    "type": "video",
                    "order": "viewCount",
                    "publishedAfter": published_after,
                    "maxResults": min(max(limit, 1), 50),
                },
            )
            self.requests += 1
            for item in payload.get("items") or []:
                video_id = ((item.get("id") or {}).get("videoId") or "").strip()
                if video_id and video_id not in found:
                    found[video_id] = item.get("snippet") or {}

        if not found:
            return []

        # One extra request buys the view counts for every id at once - and
        # without them the "order by views" above is the only ranking there is,
        # which cannot tell a video that is climbing from one that has arrived.
        stats = self._statistics(key, list(found))

        now = datetime.now(timezone.utc)
        out: list[Candidate] = []
        for video_id, snippet in found.items():
            title = clean_title(snippet.get("title") or "")
            if not title:
                continue
            published = _parse_iso8601(snippet.get("publishedAt") or "")
            age_hours = max((now - published).total_seconds() / 3600.0, 1.0) \
                if published else float(self.window_days * 24)
            views = float(stats.get(video_id, 0))
            out.append(Candidate(
                source=self.name,
                external_id=video_id,
                title=snippet.get("title") or "",
                topic=title,
                url=f"https://www.youtube.com/watch?v={video_id}",
                context=(snippet.get("channelTitle") or "")[:40],
                reach=views,
                rate=views / max(age_hours / 24.0, 0.5),
                rate_unit=self.rate_unit,
                age_hours=age_hours,
            ))
        return out

    def _statistics(self, key: str, video_ids: list[str]) -> dict[str, int]:
        """View counts, 50 ids per request. Missing counts are 0, not fatal."""
        out: dict[str, int] = {}
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start:start + 50]
            payload = _get_json(
                self.session, self.VIDEOS_URL, source=self.name,
                params={"key": key, "id": ",".join(batch), "part": "statistics"},
            )
            self.requests += 1
            for item in payload.get("items") or []:
                views = (item.get("statistics") or {}).get("viewCount")
                if item.get("id") and views is not None:
                    out[item["id"]] = int(views)
        return out


class TrendsSource(Source):
    """Google's daily trending searches, over the official RSS feed.

    Not pytrends: it is an unofficial client scraping an internal endpoint and
    it breaks every few months, which is not a dependency worth putting under a
    pipeline. The RSS is published, stable and needs no key.

    Read it for what it is. Search trends measure what people type when they
    already want an answer, and a reel wins on what stops someone who was not
    looking for anything - so this feed is general news, national in scope, and
    scores down hard against a set niche. It earns its place by occasionally
    surfacing the one story in your subject that everybody is asking about on
    the day you happen to run the scout.
    """

    name = "trends"
    key_env = ()
    rate_unit = "searches"

    FEED_URL = "https://trends.google.com/trending/rss"

    def __init__(self, session, geo=None):
        super().__init__(session)
        self.geo = (geo or config.TOPIC_TRENDS_GEO or "US").upper()

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        body = _get_bytes(
            self.session, self.FEED_URL, source=self.name, params={"geo": self.geo}
        )
        self.requests += 1
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise SourceError(
                f"the trends feed was not valid XML: {exc}", source=self.name
            ) from exc

        now = datetime.now(timezone.utc)
        out: list[Candidate] = []
        for item in root.iter():
            if _local_name(item.tag) != "item":
                continue
            fields = {_local_name(child.tag): child for child in item}
            title = clean_title((fields.get("title") is not None
                                 and fields["title"].text) or "")
            if not title:
                continue

            traffic = _approx_traffic(_text_of(fields.get("approx_traffic")))
            published = _parse_rfc822(_text_of(fields.get("pubDate")))
            age = max((now - published).total_seconds() / 3600.0, 0.25) \
                if published else 12.0

            # The headline the trend came from says what it is *about*, which
            # a two-word search term on its own never does.
            headline = ""
            for child in item:
                if _local_name(child.tag) == "news_item":
                    for grandchild in child:
                        if _local_name(grandchild.tag) == "news_item_title":
                            headline = clean_title(grandchild.text or "")
                            break
                if headline:
                    break

            out.append(Candidate(
                source=self.name,
                external_id=slugify(title) or sha1(title.encode()).hexdigest()[:10],
                title=title,
                topic=headline or title,
                url="",
                context=f"{self.geo} trending",
                reach=traffic,
                rate=traffic,
                rate_unit=self.rate_unit,
                age_hours=age,
            ))
            if len(out) >= limit:
                break
        return out


class RedditRssSource(Source):
    """Posts from your own subreddits, over Reddit's public Atom feeds.

    The keyless way in, and as of this writing the only one left. Reddit's
    `.json` endpoints now answer an unauthenticated read with 403 and a page
    of stylesheet (measured against the live site, not assumed); the `.rss`
    feeds beside them still serve 200 to the same client. So when there is no
    script app to be had, this is how the subreddits stay readable.

    What it keeps is the half that matters and cannot be got anywhere else: a
    post title is a hook a human wrote and a crowd already voted on. What it
    loses is the votes themselves - the Atom feed carries title, id, link,
    author, category and timestamp, and no score or comment count. That costs
    the two largest scoring inputs, `velocity` and `discussion`.

    Velocity is partly recovered rather than faked. Whichever listing is
    requested is itself an ordering by score - `hot` weights it by age, `top`
    settles it over a window - so a post's *position* carries the signal its
    missing score would have. `_rate_for_rank` turns that position back into a
    rate, which is a proxy, is documented as one, and is why `reach` stays 0
    instead of being filled with a number nobody measured. The board prints
    "top #3", not an upvote count that was never in the feed.

    Which listing is TOPIC_REDDIT_LISTING's to say, and it defaults to `top`
    over a week rather than `hot`. `hot` is what is moving in the next few
    hours, which is the right question for a news desk and the wrong one for a
    pipeline whose reel lands hours later and whose subject does not expire.
    """

    name = "redditrss"
    key_env = ()
    #: Replaced per instance with the configured listing - see `__init__`.
    rate_unit = "top rank"

    FEED_URL = "https://www.reddit.com/r/{subs}/{listing}/.rss"

    #: Rate assigned to the top of the listing and to the bottom of it. The
    #: gap between them is what makes rank discriminate at all: the scorer
    #: normalises log1p(rate) against the source reference, so a narrow band
    #: would collapse every post onto the same velocity score.
    TOP_RATE = 300.0
    FLOOR_RATE = 5.0

    #: Reddit rate limits anonymous feed reads hard - the third request is
    #: refused even 20s after the second - so the subreddits go up as combined
    #: `a+b+c` URLs, at most TOPIC_REDDIT_FEEDS of them, and a 429 is retried
    #: rather than lost.
    #:
    #: These waits are the recovery, not the prevention: once a limit has
    #: started, neither is long enough to clear it. FEED_GAP_SECONDS is what
    #: keeps the requests under it in the first place.
    RETRY_WAITS = (2.0, 5.0)

    #: Seconds left between one group's feed and the next. The limit is per
    #: minute and per address, so spacing the requests is the remedy for it;
    #: retrying after a 429 is what happens when the spacing was not enough.
    #:
    #: Measured 2026-09-09 from a cold address: three feeds 4s apart lost two
    #: of the three, and the two RETRY_WAITS were not long enough to clear a
    #: limit once it had started. At 20s the first two feeds returned 200 and
    #: the third was still refused - which is the measurement behind
    #: TOPIC_REDDIT_FEEDS defaulting to 2 rather than 3.
    #:
    #: The scout runs once a day and step 0 already takes over a minute, so
    #: the seconds this adds are not worth optimising away.
    FEED_GAP_SECONDS = 20.0

    #: The feed gives no `stickied` flag, so the mod posts pinned to the top
    #: of every subreddit arrive looking like candidates. They are recognised
    #: by their author instead - and they are why an age cutoff alone is not
    #: enough: a sticky can be a year old and still sit at hot #1.
    _MOD_AUTHOR_RE = re.compile(r"(moderator|modteam|automoderator)", re.IGNORECASE)

    def __init__(self, session, subreddits=None):
        super().__init__(session)
        self.subreddits = tuple(subreddits or config.TOPIC_SUBREDDITS)
        self.listing, self.window = reddit_listing()
        #: Shadows the class attribute so a board says which listing the rank
        #: it quotes came from - "top #3" and "hot #3" are not the same claim
        #: about a post.
        self.rate_unit = f"{self.listing} rank"

    @classmethod
    def overrides(cls, **supplied) -> dict:
        subreddits = supplied.get("subreddits")
        return {"subreddits": subreddits} if subreddits else {}

    @classmethod
    def readiness(cls, **supplied) -> tuple[bool, str]:
        """No keys to want. A subreddit list is still not optional."""
        if not (supplied.get("subreddits") or config.TOPIC_SUBREDDITS):
            return False, "needs TOPIC_SUBREDDITS"
        return True, "ready (no key required)"

    @staticmethod
    def _groups(subreddits, most: int,
                day: int | None = None) -> list[tuple[str, ...]]:
        """Split the subreddit list into at most `most` combined feeds.

        One combined `a+b+c` feed is one request, which is what the rate limit
        wants - but Reddit ranks a combined listing by absolute score, and
        subreddit sizes here span three orders of magnitude. Measured on board
        #31: nine subreddits went up as one feed, and the forty entries that
        came back held two subreddits. r/todayilearned (38M members) took 33 of
        them and r/science the other 2; r/oceans, r/marinebiology,
        r/EarthScience, r/geology, r/askscience, r/explainlikeimfive and
        r/Awwducational returned nothing at all. Every one of the 33 was then
        rejected off the niche, because r/todayilearned's top of the month is
        film trivia and sports anecdotes. The board fell back to the backlog,
        as it does every day.

        So the subreddits that were chosen *for* the niche were never actually
        read. Splitting the list into a few feeds gives each group its own
        forty places, and a small subreddit competes against its own size
        rather than against the largest default sub on the site. `rank` is per
        group for the same reason - position among comparable subreddits is
        the honest reading of `_rate_for_rank`.

        **List order decides who shares a feed, so the biggest subreddits go
        first.** The remainder lands on the *last* group rather than the first,
        which is what keeps them together: nine subreddits into two feeds split
        4/5 and not 5/4, so the four large defaults take one feed and the five
        chosen for the niche take the other. Under 5/4 the fifth name was
        pushed in with the giants and never read - measured live on 2026-09-09,
        r/todayilearned took all twenty places in that group and r/askscience,
        r/explainlikeimfive, r/science and r/geology returned nothing.

        Rotating membership by the day was tried here and **measured worse**.
        It rescues the fifth name by moving the boundary, but it also moves the
        largest subreddit into the niche group on most days: on 2026-09-09 that
        put r/todayilearned in with r/EarthScience, r/marinebiology, r/oceans
        and r/Awwducational, and it took all twenty places from the four subs
        that were actually producing the board. Keeping the giants together
        every day beats sharing the damage out. `_rotate` still rotates which
        feed is read *first*, because that is a different question - it is
        about which group loses when the rate limit bites, not about who is in
        it. `day` is accepted and unused, so a caller that was passing it does
        not break.

        Balanced rather than one-per-request: the constraint being respected is
        the request count, so the list is divided into `most` groups and not
        into groups of `most`. At 1 this is exactly the old single-feed call.
        Exactly `most` groups, too - chunking by a ceiling silently returned
        three feeds when four were asked for, spending fewer requests than it
        was allowed and reading fewer subreddits for no reason.
        """
        subs = [str(name).strip() for name in subreddits if str(name).strip()]
        if not subs:
            return []
        most = max(1, int(most))
        if most >= len(subs):
            return [(name,) for name in subs]

        base, spare = divmod(len(subs), most)
        groups: list[tuple[str, ...]] = []
        at = 0
        for index in range(most):
            # The spare places go to the last `spare` groups, so the head of
            # the list - where the big subreddits are - stays as tightly packed
            # as it can be.
            size = base + (1 if index >= most - spare else 0)
            groups.append(tuple(subs[at:at + size]))
            at += size
        return groups

    @staticmethod
    def _rotate(groups: list, day: int | None = None) -> list:
        """The groups, reordered so a different one goes first each day.

        The first feed is the one that gets the benefit of the rate limit: a
        later one that is refused is simply not read that day. In a fixed
        order that is always the *same* subreddits missing, which is the bug
        this class already had in a different shape - measured on board #62,
        where two of three feeds were refused and the board came back holding
        nothing but the group that happened to be first.

        Rotating by the day makes the loss fall on everyone in turn, so a week
        of partial failures still reads every subreddit several times instead
        of reading a third of them seven times. Derived from the date rather
        than stored: there is no state to keep, and two runs on one day should
        scout the same feeds.
        """
        if not groups:
            return []
        if day is None:
            day = datetime.now(timezone.utc).toordinal()
        start = day % len(groups)
        return list(groups[start:]) + list(groups[:start])

    @classmethod
    def _rate_for_rank(cls, rank: int, total: int) -> float:
        """Position in the listing -> a stand-in rate.

        Geometric rather than linear between TOP_RATE and FLOOR_RATE, because
        the scorer takes log1p of this: a linear ramp would come out nearly
        flat once logged, which is the thing the spread exists to avoid.
        """
        if total <= 1:
            return cls.TOP_RATE
        share = min(max(rank, 0), total - 1) / (total - 1)
        return cls.TOP_RATE * (cls.FLOOR_RATE / cls.TOP_RATE) ** share

    @classmethod
    def _explain(cls, exc: SourceError) -> SourceError:
        """Say what a 429 means here, since it is the failure to expect."""
        if "[429" in str(exc):
            return SourceError(
                "Reddit rate limited the anonymous feed read. It allows only a "
                "few per minute from one address, and the feeds are already "
                "spaced FEED_GAP_SECONDS apart. Lower TOPIC_REDDIT_FEEDS to "
                "read fewer subreddit groups per run, or set REDDIT_CLIENT_ID "
                "+ REDDIT_CLIENT_SECRET and TOPIC_SOURCES=reddit - the "
                "authenticated API allows 60 requests a minute and does not "
                "have this limit.",
                source=cls.name,
            )
        return exc

    def _fetch_feed(self, subs: str, limit: int) -> bytes:
        """GET the feed, giving a rate limit a second and third chance."""
        import time

        attempts = len(self.RETRY_WAITS) + 1
        last: SourceError | None = None
        for attempt in range(attempts):
            try:
                params = {"limit": min(max(limit, 1), 100)}
                if self.listing == "top":
                    params["t"] = self.window
                body = _get_bytes(
                    self.session,
                    self.FEED_URL.format(subs=subs, listing=self.listing),
                    source=self.name, params=params,
                )
                self.requests += 1
                return body
            except SourceError as exc:
                last = exc
                self.requests += 1
                if "[429" not in str(exc) or attempt == attempts - 1:
                    break
                time.sleep(self.RETRY_WAITS[attempt])
        raise self._explain(last if last else SourceError(
            "no response from the reddit feed", source=self.name
        ))

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        if not self.subreddits:
            raise SourceError(
                "no subreddits configured - set TOPIC_SUBREDDITS in .env "
                "(whose feed this is depends on the account, so there is no "
                "sensible default)",
                source=self.name,
            )

        groups = self._groups(self.subreddits, config.TOPIC_REDDIT_FEEDS)
        # The groups share `limit` rather than each taking it, so raising
        # TOPIC_REDDIT_FEEDS splits the same board between more subreddits
        # instead of quietly fetching a multiple of what TOPIC_PER_SOURCE asked
        # for.
        share = max(1, math.ceil(limit / len(groups)))
        groups = self._rotate(groups)

        out: list[Candidate] = []
        seen: set[str] = set()
        failures: list[SourceError] = []
        for index, group in enumerate(groups):
            if index and self.FEED_GAP_SECONDS > 0:
                import time
                time.sleep(self.FEED_GAP_SECONDS)
            try:
                body = self._fetch_feed("+".join(group), share)
            except SourceError as exc:
                # One group lost is some subreddits lost, not the source. A 429
                # on the third feed should not throw away the two that came
                # back - that would make raising TOPIC_REDDIT_FEEDS strictly
                # more fragile, which is the opposite of what it is for.
                failures.append(exc)
                continue
            for candidate in self._parse(body, share):
                if candidate.external_id in seen:
                    continue        # a crosspost sitting in two of the groups
                seen.add(candidate.external_id)
                out.append(candidate)

        if failures and not out:
            raise failures[-1]
        if failures:
            self.notes.append(
                f"{len(failures)} of {len(groups)} subreddit feed(s) failed - "
                f"{failures[-1]}"
            )

        kept = out[:limit]
        self.notes.extend(self._starved(kept))
        return kept

    def _starved(self, found: list[Candidate]) -> list[str]:
        """A note naming the subreddits that returned nothing, if any.

        A combined feed is ranked by absolute score, so a subreddit that
        shares its group with a much larger one is not read thinly - it is
        not read at all, and says nothing about it. That is exactly how nine
        subreddits went up as one feed and seven of them went unread for
        weeks: `fetched 40` looked like a healthy number and no other line
        contradicted it.

        Measured live on 2026-09-09 at TOPIC_REDDIT_FEEDS=2: of the group
        holding todayilearned, askscience, explainlikeimfive, science and
        geology, r/todayilearned took all twenty places and the other four
        returned nothing. A note rather than a warning - a starved feed is
        worth seeing on the board, not worth stopping a scout for.
        """
        if not found:
            return []
        # Case-insensitively: the feed spells a subreddit the way its mods
        # do (r/EarthScience) and TOPIC_SUBREDDITS however it was typed.
        seen = {
            candidate.context.split()[0].removeprefix("r/").lower()
            for candidate in found if candidate.context
        }
        missing = [name for name in self.subreddits
                   if str(name).strip().lower() not in seen]
        if not missing:
            return []
        return [f"{len(missing)} of {len(self.subreddits)} subreddit(s) "
                f"returned nothing - {', '.join(missing)}. A bigger "
                f"subreddit in the same feed outranks them; give them a "
                f"feed of their own by raising TOPIC_REDDIT_FEEDS, or "
                f"reorder TOPIC_SUBREDDITS so they group together."]

    def _parse(self, body: bytes, limit: int) -> list[Candidate]:
        """One feed's XML into candidates. `rank` is within this feed."""
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise SourceError(
                f"the reddit feed was not valid XML: {exc}", source=self.name
            ) from exc

        entries = [e for e in root.iter() if _local_name(e.tag) == "entry"]
        now = datetime.now(timezone.utc)
        out: list[Candidate] = []

        for rank, entry in enumerate(entries):
            fields = {_local_name(child.tag): child for child in entry}

            author = ""
            node = fields.get("author")
            if node is not None:
                for child in node:
                    if _local_name(child.tag) == "name":
                        author = (child.text or "").strip()
                        break
            if self._MOD_AUTHOR_RE.search(author):
                continue            # the subreddit's own furniture, pinned

            title = clean_title(_text_of(fields.get("title")))
            if not title:
                continue

            published = _parse_iso8601(_text_of(fields.get("published")))
            age = max((now - published).total_seconds() / 3600.0, 0.25) \
                if published else 24.0

            # `<id>` is `t3_<id36>`; the bare id36 is what the .json API calls
            # the post, so a ledger written by either source agrees with the
            # other and a topic already made into a reel stays suppressed.
            raw_id = _text_of(fields.get("id"))
            external_id = raw_id.split("_", 1)[-1] if raw_id else slugify(title)

            subreddit = ""
            category = fields.get("category")
            if category is not None:
                subreddit = (category.get("term") or "").strip()

            link = fields.get("link")
            url = (link.get("href") or "").strip() if link is not None else ""

            out.append(Candidate(
                source=self.name,
                external_id=external_id,
                title=_text_of(fields.get("title")),
                topic=title,
                url=url,
                context=f"r/{subreddit or '?'} {self.listing} #{rank + 1}",
                reach=0.0,          # not in the feed - see the class docstring
                rate=self._rate_for_rank(rank, len(entries)),
                rate_unit=self.rate_unit,
                age_hours=age,
            ))
            if len(out) >= limit:
                break
        return out


class WikipediaSource(Source):
    """Articles people are suddenly reading, from Wikimedia's pageviews API.

    Official, keyless, and the one free source here that measures a *rise*
    rather than a level. It asks for the most-read articles of a day and again
    for the same weekday a week earlier, then keeps only the articles whose
    views actually climbed between the two. Two requests, and every candidate
    that survives is by construction trending rather than merely popular.

    Same weekday, not yesterday, because readership is strongly weekly -
    comparing a Sunday against a Saturday would report a spike that is only
    the weekend. An article with no baseline at all is new to the top list,
    which is the strongest form of the signal this source can see.

    Read it the way TrendsSource is read: general interest, national, and it
    will score down against a set niche most days. It earns its place on the
    day something in your subject is what the internet is looking up.
    """

    name = "wikipedia"
    key_env = ()
    rate_unit = "views/day"

    TOP_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
               "{project}/all-access/{year}/{month}/{day}")

    #: The pageviews API publishes a day once it is complete, which lands
    #: roughly a day late. Asking for today returns 404, so the newest day
    #: worth asking for is this many back.
    LAG_DAYS = 1
    BASELINE_DAYS = 7

    #: Below this the article is being read at its usual rate, which is not a
    #: trend. 1.4 keeps a genuine climb and drops the daily noise.
    MIN_SPIKE = 1.4
    #: What an article absent from the baseline list is treated as having
    #: risen by. Capped rather than infinite: it left the top list at some
    #: unknown position, so the true ratio is unknown too.
    NEW_ENTRANT_SPIKE = 3.0

    #: Namespaced pages, the front door, and the search box. None of them are
    #: articles and all of them outrank everything that is.
    _SKIP_PREFIX = ("Special:", "Wikipedia:", "Portal:", "Help:", "Category:",
                    "Template:", "File:", "Talk:", "User:", "Draft:")
    _SKIP_EXACT = frozenset({"Main_Page", "-"})

    def __init__(self, session, project=None):
        super().__init__(session)
        self.project = project or config.TOPIC_WIKI_PROJECT or "en.wikipedia"

    def _top_for(self, day) -> dict[str, float]:
        """{article: views} for one complete day."""
        url = self.TOP_URL.format(
            project=self.project, year=day.year,
            month=f"{day.month:02d}", day=f"{day.day:02d}",
        )
        payload = _get_json(self.session, url, source=self.name)
        self.requests += 1
        out: dict[str, float] = {}
        for item in payload.get("items") or []:
            for article in item.get("articles") or []:
                name = (article.get("article") or "").strip()
                if name:
                    out[name] = float(article.get("views") or 0)
        return out

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        today = datetime.now(timezone.utc).date()
        latest = today - timedelta(days=self.LAG_DAYS)
        current = self._top_for(latest)
        if not current:
            raise SourceError(
                f"the pageviews API returned no articles for {latest}",
                source=self.name,
            )

        try:
            baseline = self._top_for(latest - timedelta(days=self.BASELINE_DAYS))
        except SourceError:
            # One missing day should thin the source, not fail it: with no
            # baseline every article would read as a new entrant, which is
            # wrong, so the spike filter is skipped for this run instead.
            baseline = {}

        # The day the API reported ended at midnight UTC; age is measured from
        # there so the staleness rule is given a real number rather than a 0.
        day_end = datetime(latest.year, latest.month, latest.day,
                           23, 59, 59, tzinfo=timezone.utc)
        age = max((datetime.now(timezone.utc) - day_end).total_seconds() / 3600.0,
                  1.0)

        ranked = sorted(current.items(), key=lambda kv: kv[1], reverse=True)
        out: list[Candidate] = []
        for article, views in ranked:
            if article in self._SKIP_EXACT or article.startswith(self._SKIP_PREFIX):
                continue

            was = baseline.get(article)
            if baseline:
                spike = (self.NEW_ENTRANT_SPIKE if was is None
                         else views / max(was, 1.0))
                if spike < self.MIN_SPIKE:
                    continue
            else:
                spike = 1.0

            title = clean_title(article.replace("_", " "))
            if not title:
                continue

            out.append(Candidate(
                source=self.name,
                external_id=article,
                title=article.replace("_", " "),
                topic=title,
                url=f"https://{self.project}.org/wiki/{article}",
                context=("new to the top list" if was is None and baseline
                         else f"x{spike:.1f} vs last week"),
                reach=views,
                rate=views,
                rate_unit=self.rate_unit,
                age_hours=age,
            ))
            if len(out) >= limit:
                break
        return out


class EvergreenSource(Source):
    """The hand-written backlog, read as a source like any other.

    Its lines are written as topic phrases, so `wording_is_final` is True and
    the phrase pass leaves them alone.

    A source that makes no request and measures no motion, which is the point.
    Every other source here answers "what is moving?", and on a narrow niche
    that question has an empty answer most days - see TOPIC_EVERGREEN_FILE in
    config.py for the board that made the case. This one answers "what is
    always worth explaining?", which is the question this format is actually
    good at: step 0's own reasoning for reading `top` over `hot` is that a
    mechanism true four years ago is not stale, it is still true.

    Being a Source rather than a special case in `find_topics` is deliberate.
    A backlog line goes through the same niche gate, the same footage probe,
    the same ledger and the same scorer as a Reddit post, and is beaten by a
    better candidate on the same board rather than by a rule about where it
    came from. The only thing it gets that nothing else does is
    EVERGREEN_PENALTY, and that prints on the board like every other component.

    Rotation is by day rather than by shuffle: the file is read from an offset
    that advances once a day and wraps, so every line reaches a board within a
    cycle even when the file is longer than TOPIC_PER_SOURCE, and two runs on
    the same day see the same window. What stops a line heading every board is
    the ledger - SCOUTED_PENALTY for one shown recently, dropped outright once
    taken - which is the same thing that stops a Reddit post doing it.
    """

    name = "evergreen"
    key_env = ()
    rate_unit = ""
    wording_is_final = True

    @classmethod
    def overrides(cls, **supplied) -> dict:
        return {}

    @classmethod
    def readiness(cls, **supplied) -> tuple[bool, str]:
        path = config.TOPIC_EVERGREEN_FILE
        try:
            entries = cls.read_file(path)
        except OSError as exc:
            return False, f"could not read {path}: {exc}"
        if not entries:
            return False, f"no topics in {path} - add one per line"
        return True, f"ready ({len(entries)} in the backlog)"

    @staticmethod
    def read_file(path) -> list[str]:
        """The backlog, in file order. Blank lines and `#` comments dropped.

        Deduplicated on the way out: the same line twice in the file is an
        edit slip, and it would otherwise be two ledger entries for one topic.
        """
        path = Path(path)
        if not path.exists():
            return []
        seen: set[str] = set()
        out: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fold = line.lower()
            if fold in seen:
                continue
            seen.add(fold)
            out.append(line)
        return out

    def __init__(self, session, path=None):
        super().__init__(session)
        self.path = Path(path or config.TOPIC_EVERGREEN_FILE)

    def fetch(self, niche: str, limit: int) -> list[Candidate]:
        try:
            entries = self.read_file(self.path)
        except OSError as exc:
            raise SourceError(
                f"could not read the backlog at {self.path}: {exc}",
                source=self.name,
            ) from exc
        if not entries:
            raise SourceError(
                f"the backlog at {self.path} has no topics in it",
                source=self.name,
            )

        # Day-of-year offset: advances once a day, wraps, and is the same for
        # every run on one day. `--take` re-scouting an hour later has to see
        # the window it printed from.
        start = datetime.now(timezone.utc).timetuple().tm_yday % len(entries)
        ordered = entries[start:] + entries[:start]

        out: list[Candidate] = []
        for line in ordered[:max(1, limit)]:
            out.append(Candidate(
                source=self.name,
                # Slug, not position: reordering the file must not re-open a
                # topic the ledger has already recorded as made.
                external_id=slugify(line),
                title=line,
                topic=line,
                url="",
                context="evergreen backlog",
                # No reach, no rate and no age. There is no reference rate for
                # this source in SOURCE_REFERENCE_RATE either, so velocity is
                # not scored rather than being scored as zero - a backlog line
                # is not a slow story, it is not a story.
                age_hours=0.0,
            ))
        return out


#: Every source this module knows how to read, by config name.
_SOURCES: dict[str, type[Source]] = {
    RedditSource.name: RedditSource,
    RedditRssSource.name: RedditRssSource,
    YouTubeSource.name: YouTubeSource,
    TrendsSource.name: TrendsSource,
    WikipediaSource.name: WikipediaSource,
    EvergreenSource.name: EvergreenSource,
}


def source_status() -> list[dict]:
    """What each configured source needs, and whether it has it."""
    status = []
    for name in config.TOPIC_SOURCES:
        cls = _SOURCES.get(name)
        if cls is None:
            status.append({"name": name, "known": False, "ready": False,
                           "detail": "unknown source name"})
            continue
        ready, detail = cls.readiness()
        status.append({"name": name, "known": True, "ready": ready,
                       "detail": detail})
    return status


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------

def _local_name(tag: str) -> str:
    """Tag without its namespace.

    Matched on the local name rather than the full URI because Google has
    changed the trends namespace before and the feed shape did not change with
    it - a hard-coded URI would turn that into an empty board.
    """
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text_of(element) -> str:
    return (element.text or "").strip() if element is not None else ""


def _approx_traffic(raw: str) -> float:
    """'20,000+' -> 20000.0. Also handles '2K+' and '1.5M+'."""
    text = (raw or "").strip().replace(",", "").rstrip("+").strip()
    if not text:
        return 0.0
    multiplier = 1.0
    if text[-1:].upper() in ("K", "M", "B"):
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[text[-1].upper()]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def _parse_iso8601(raw: str):
    """RFC 3339 as YouTube writes it. None when it cannot be read."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_rfc822(raw: str):
    """RFC 822 as RSS writes it. None when it cannot be read."""
    from email.utils import parsedate_to_datetime

    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scoring
#
# The same shape as step 2's hook scoring, for the same reason: every number
# below is printed with the candidate it belongs to, so a board that ranked
# something oddly can be read rather than argued with. Nothing here is a hard
# filter except an empty stock library - a thin board still has to return
# something, but a topic with no footage anywhere cannot become a reel at all.
# ---------------------------------------------------------------------------

def score_candidate(candidate: Candidate, *, niche_terms: set[str] | None = None,
                    banned: tuple[str, ...] = (), max_age_hours: float | None = None,
                    seen: bool = False, on_niche: bool | None = None,
                    refused: int = 0, curiosity: bool | None = None) -> dict:
    """Score one candidate. Higher is better.

    Returns the score and the parts it was built from. Sourceability is not
    included - it is added by `apply_sourceability` after the probe, so the
    expensive check only runs on candidates the free checks already like.

    A candidate carrying `provisional` is scored on its subject alone. The
    three components that read the *form* of the wording - the opening word,
    the length and the proper nouns - are the three the phrase pass exists
    to fix, so charging them to a headline marks a candidate down for a
    shape it is about to lose. `find_topics` scores twice for this reason:
    once to pick a shortlist, and again once `polish_phrases` has settled
    what that shortlist actually says.

    `on_niche` is the niche pass's verdict where there is one. Left at None the
    subject is judged on shared words alone, which is what the standalone
    scorer and every caller without a Gemini key gets.

    `refused` is how many drafts for this topic the fact-check has thrown away.
    It is charged once whatever the count, and the count itself is a filter in
    `find_topics` rather than more arithmetic here.

    `curiosity` False stops the curiosity component being charged at all. It
    exists for the `curiosity` arm of PERFORMANCE_EXPLORE, and it turns a
    component off rather than reweighting it because the question the trial
    asks is whether `CURIOSITY_SIGNAL_WEIGHT` is earning its place - and any
    other number to compare it against would be a second prior invented to
    test the first. None leaves it charged, which is every other caller.
    """
    topic = candidate.topic or candidate.title
    lowered = topic.lower()
    words = _WORD_RE.findall(topic)
    parts: dict[str, float] = {}
    limit = max_age_for(candidate.source) if max_age_hours is None else max_age_hours

    if not words:
        return {"score": -99.0, "components": {"empty": -99.0}}

    for word in banned:
        if word and word in lowered:
            parts["banned word"] = BANNED_WORD_PENALTY
            break

    reference = SOURCE_REFERENCE_RATE.get(candidate.source, 0.0)
    if reference > 0 and candidate.rate > 0:
        share = min(1.0, math.log1p(candidate.rate) / math.log1p(reference))
        parts["velocity"] = round(VELOCITY_WEIGHT * share, 3)
        if (candidate.age_hours <= RISING_MAX_AGE_HOURS
                and candidate.rate >= reference * RISING_RATE_FRACTION):
            parts["still climbing"] = RISING_BONUS

    if candidate.comments and candidate.reach > 0:
        ratio = min(1.0, candidate.comments / max(candidate.reach, 1.0) * 10.0)
        if ratio > 0:
            parts["discussion"] = round(DISCUSSION_WEIGHT * ratio, 3)

    fits, strength = _niche_fit(topic, niche_terms, on_niche)
    if fits is True:
        parts["niche match"] = round(NICHE_MATCH_WEIGHT * strength, 3)
    elif fits is False:
        parts["off the niche"] = NICHE_MISS_PENALTY

    if curiosity is not False:
        worth, signals = curiosity_component(topic)
        if signals:
            # Named after what fired, so the printed board argues its own case.
            parts[f"curiosity ({', '.join(signals)})"] = worth

    if hype_words(topic):
        parts["hype in the title"] = HYPE_PENALTY

    # The three components that read the *form* of the wording rather than its
    # subject, and so cannot be charged while the wording is still a headline.
    # The phrase pass exists to turn "TIL that X, which is why Y" into "why Y":
    # it fixes exactly the opening word, the length and the proper nouns these
    # three measure, so scoring them on the headline marks a candidate down for
    # a shape it is about to lose. Measured on board #33: across seventeen
    # reddit rows these three summed to -2.24 a row against +0.49 for the
    # seventeen backlog lines, whose wording is final and explainer-shaped by
    # hand - a 2.7 point handicap on the source, not on the topic.
    #
    # Content is charged either way. Curiosity, hype, the niche match and the
    # banned words all read what the topic is *about*, and the phrase pass is
    # told to keep the subject, so those survive the rewrite.
    if not candidate.provisional:
        if _is_explainer(topic):
            parts["explainer shape"] = EXPLAINER_BONUS

        names = _proper_nouns(topic)
        if names:
            parts["proper nouns"] = round(
                max(PROPER_NOUN_FLOOR, PROPER_NOUN_PENALTY * len(names)), 3
            )

        if len(words) < TITLE_MIN_WORDS:
            parts["too short"] = TITLE_SHORT_PENALTY
        elif len(words) > TITLE_MAX_WORDS:
            parts["too long"] = TITLE_LONG_PENALTY

    if limit > 0 and candidate.age_hours > limit:
        parts["past its peak"] = STALE_PENALTY

    if candidate.source == EvergreenSource.name:
        if seen:
            parts["on a recent board"] = EVERGREEN_SCOUTED_PENALTY
        parts["from the backlog"] = EVERGREEN_PENALTY
    elif seen:
        parts["on a recent board"] = SCOUTED_PENALTY

    if refused > 0:
        # Named for what it is, so the board argues its own case: this topic
        # has already been written and thrown away, and that is the reason it
        # is below a line it would otherwise beat.
        parts[f"a draft was refused ({refused}x)" if refused > 1
              else "a draft was refused"] = REFUSED_PENALTY

    return {"score": round(sum(parts.values()), 3), "components": parts}


def sourceability_component(hits: int, needed: int | None = None) -> float:
    """Score for a topic whose stock search returned `hits` matching results.

    SOURCEABLE_FLOOR at none, SOURCEABLE_BONUS at the threshold, straight line
    between, flat above. Continuous because a step would put a cliff between
    two candidates one result apart.
    """
    threshold = max(1, config.TOPIC_SOURCEABLE_HITS if needed is None else needed)
    share = min(1.0, max(0, hits) / threshold)
    return round(SOURCEABLE_FLOOR + (SOURCEABLE_BONUS - SOURCEABLE_FLOOR) * share, 3)


def apply_sourceability(candidate: Candidate, needed: int | None = None) -> None:
    """Fold a completed check into a candidate's score, in place.

    A report carrying `component` priced itself - that is the model pass, whose
    rungs are named in SOURCEABILITY_COMPONENTS rather than interpolated. Any
    other report is a keyword probe and is read off its hit count.
    """
    probe = candidate.sourceability or {}
    if not probe.get("checked"):
        return
    if probe.get("component") is not None:
        component = round(float(probe["component"]), 3)
    else:
        component = sourceability_component(int(probe.get("hits") or 0), needed)
    candidate.components["footage"] = component
    candidate.score = round(sum(candidate.components.values()), 3)


def apply_measured(candidate: Candidate) -> None:
    """Fold what step 9 has measured about components like these into the score.

    This is the only weight on the board that was not set by hand, and it is
    deliberately shaped to look exactly like the ones that were: a named
    component, printed with the evidence behind it, small enough that it moves
    a ranking rather than deciding one. The alternative - a step 9 that rewrote
    EXPLAINER_BONUS in the source - would buy slightly better tuning at the
    cost of the property that makes this board worth reading, which is that
    every number on it can be argued with.

    Does nothing at all unless PERFORMANCE_FEEDBACK is on and the ledger holds
    PERFORMANCE_MIN_REELS reels. See pipeline/perform.py.
    """
    adjustment = perform.adjustment_for(candidate.components)
    if not adjustment["value"]:
        return
    # Named after what it measured, so the board argues its own case - the same
    # shape as the curiosity component above.
    candidate.components[f"measured ({adjustment['note']})"] = adjustment["value"]
    candidate.score = round(sum(candidate.components.values()), 3)


# ---------------------------------------------------------------------------
# How long it should be
# ---------------------------------------------------------------------------

def suggest_duration(candidate: Candidate) -> dict:
    """How long the reel about this topic should be, and why.

    Returns {"seconds", "components"} - the same shape `score_candidate`
    returns, and for the same reason: the number is printed with the parts it
    was built from, so a board can be read back instead of trusted.

    Length used to be a flag with a number in it. `-d 45` was applied to a
    mechanism worth forty seconds and a single fact worth twenty alike, and the
    second of those is where the reach went: completion is what the ranking
    runs on, so a fact stretched to 45 seconds is fifteen seconds of restating
    it that the viewer leaves during. A shorter reel watched through is the
    better outcome and the better signal.

    `components` holds the base and then signed adjustments to it, so the
    printed line reads as the argument it is: 30 seconds, plus eight because
    this is a mechanism, minus three because the footage is thin.

    The result is rounded to SCRIPT_DURATION_STEP_SECONDS and clamped into
    SCRIPT_MIN_SECONDS..SCRIPT_MAX_SECONDS - clamped last, so a floor that is
    not a multiple of the step is still honoured exactly as it was set.
    """
    topic = candidate.topic or candidate.title
    parts: dict[str, float] = {"base": float(config.SCRIPT_DEFAULT_SECONDS)}

    if _is_explainer(topic):
        parts["explainer"] = DURATION_EXPLAINER_BONUS

    if _proper_nouns(topic):
        parts["a named subject"] = DURATION_FIXTURE_PENALTY

    if candidate.comments and candidate.reach > 0:
        ratio = min(1.0, candidate.comments / max(candidate.reach, 1.0) * 10.0)
        if ratio > 0:
            # One decimal, not the score's three: these are seconds, and the
            # total is rounded to a five-second step regardless. Printing
            # +2.667 would offer a precision the answer does not have.
            parts["discussion"] = round(DURATION_DISCUSSION_BONUS * ratio, 1)

    probe = candidate.sourceability or {}
    if probe.get("checked"):
        # The model pass carries a rung and the keyword probe a hit count.
        # Read the rung where there is one: it is the finer of the two, and
        # `hits` under that mode is only ever zero or the threshold, which
        # would hand a topic rated "filler" the same seconds as one the
        # library plainly holds.
        if probe.get("rating") is not None:
            share = min(1.0, max(0, int(probe["rating"])) / SOURCEABILITY_SCALE)
        else:
            threshold = max(1, config.TOPIC_SOURCEABLE_HITS)
            share = min(1.0, max(0, int(probe.get("hits") or 0)) / threshold)
        parts["footage"] = round(
            DURATION_FOOTAGE_THIN
            + (DURATION_FOOTAGE_RICH - DURATION_FOOTAGE_THIN) * share, 1
        )

    step = max(1, config.SCRIPT_DURATION_STEP_SECONDS)
    seconds = int(round(sum(parts.values()) / step) * step)
    seconds = max(config.SCRIPT_MIN_SECONDS,
                  min(config.SCRIPT_MAX_SECONDS, seconds))
    return {"seconds": seconds, "components": parts}


def apply_duration(candidate: Candidate) -> None:
    """Fold a length suggestion into a candidate, in place."""
    result = suggest_duration(candidate)
    candidate.duration = result["seconds"]
    candidate.duration_components = result["components"]


# ---------------------------------------------------------------------------
# The sourceability probe
# ---------------------------------------------------------------------------

def probe_query(topic: str, words: int = PROBE_QUERY_WORDS) -> str:
    """The stock search a topic would turn into.

    Step 4's own keyword extractor, so the probe asks the same question the
    pipeline will later ask for real. Not step 4's *llm* query writer: that is
    one Gemini call per candidate to answer a question a keyword search already
    answers well enough to filter on.

    Possessives are trimmed, because they arrive attached to the noun that
    matters - a headline about "State Farm's payout" reduces to the keyword
    "farm's", which no stock library has ever tagged anything with. Abstract
    words are dropped for the same reason, and twice as many keywords are asked
    for as are wanted so that dropping one promotes a real word into its place
    rather than leaving the query a word short.
    """
    terms = []
    fallback = []
    for term in visuals._keywords(topic or "", limit=words * 2):
        term = re.sub(r"['’]s$", "", term).strip("-'’")
        if len(term) <= 2:
            continue
        fallback.append(term)
        if term.lower() not in _UNFILMABLE:
            terms.append(term)
        if len(terms) >= words:
            break
    chosen = terms or fallback[:words]
    return " ".join(chosen) or " ".join((topic or "").split()[:words])


def _same_word(one: str, other: str) -> bool:
    """True when two words are the same word. "salt" and "salty" are."""
    if len(one) < STEM_LENGTH or len(other) < STEM_LENGTH:
        return one == other
    return one.startswith(other) or other.startswith(one)


def _result_terms(candidate) -> set[str]:
    """The words a stock result describes itself with.

    The title where there is one, plus the slug out of the source URL. That
    second part is not a nicety: Pexels leaves `alt` empty on video, and the
    link - /video/serene-ocean-surface-with-gentle-waves-29547478/ - is the
    only description of a clip it gives away without a second request.
    """
    words = {w.lower() for w in _WORD_RE.findall(getattr(candidate, "title", "") or "")}
    path = urlsplit(getattr(candidate, "source_url", "") or "").path
    segments = [segment for segment in path.split("/") if segment]
    if segments:
        words |= {part for part in segments[-1].lower().split("-")
                  if part and not part.isdigit()}
    return {word for word in words if len(word) > 2}


def _result_matches(candidate, terms: set[str]) -> bool:
    """True when a stock result is about what was searched for.

    Counts how many of the query's own words the result names back, and holds
    a longer query to a higher bar - see PROBE_MIN_TERM_HITS.
    """
    described = _result_terms(candidate)
    landed = sum(
        1 for term in terms
        if any(_same_word(term, word) for word in described)
    )
    needed = PROBE_MIN_TERM_HITS if len(terms) >= PROBE_MIN_TERM_HITS_FROM else 1
    return landed >= needed


def probe_sourceability(topic: str, providers: list, *, sample: int | None = None,
                        needed: int | None = None) -> dict:
    """Ask the stock providers whether a topic can actually be shot.

    Searches for the topic's keywords and counts how many of the results come
    back describing what was asked for - see the long comment above for why the
    result count a provider reports cannot be used instead.

    Providers are asked in step 4's own preference order and the walk stops at
    the first one holding enough, so a topic Pexels can shoot never costs an
    Openverse request - which matters, because Openverse allows 200 anonymous
    requests a day and a wide board would eat them.

    Never raises. A provider that fails is recorded and skipped, and a probe
    where every provider failed comes back `checked: False`, which scores as
    nothing rather than as no footage.
    """
    size = max(1, config.TOPIC_PROBE_SAMPLE if sample is None else sample)
    threshold = max(1, config.TOPIC_SOURCEABLE_HITS if needed is None else needed)
    query = probe_query(topic)
    terms = _terms(query)
    report = {"query": query, "hits": 0, "sample": 0, "provider": "",
              "checked": False, "by_provider": {}, "notes": []}
    if not query or not terms or not providers:
        return report

    for provider in providers:
        kind = "video" if "video" in provider.kinds else "image"
        try:
            results = provider.search(query, kind, size)
        except (visuals.ProviderError, ValueError, TypeError) as exc:
            report["notes"].append(f"{provider.name}: {exc}")
            continue
        hits = sum(1 for result in results if _result_matches(result, terms))
        report["checked"] = True
        report["by_provider"][provider.name] = {"hits": hits,
                                                "sample": len(results)}
        if hits > report["hits"] or not report["provider"]:
            report["hits"] = max(hits, report["hits"])
            report["sample"] = len(results)
            report["provider"] = provider.name
        if hits >= threshold:
            break
    return report


_SOURCEABILITY_SYSTEM_INSTRUCTION = """\
You decide whether a stock footage library could illustrate a video topic.

You answer about pictures only - whether generic, unbranded library clips \
could show this - and never about whether the topic is interesting, popular, \
recent, true or well written.

You always return valid JSON and nothing else."""


_SOURCEABILITY_PROMPT_TEMPLATE = """\
For each topic below, rate how well an ordinary stock footage library \
(Pexels, Pixabay, Openverse) could illustrate a short explainer about it.

Rate 0 to 3:
3 - the library plainly holds shots of this. Nature, weather, landscapes, \
animals, water, cities, people doing everyday things.
2 - the subject can be carried by common footage without being shown exactly. \
Something abstract or microscopic that stands in well - an idea about time, \
pressure, sound, growth.
1 - only loosely related or generic imagery. The clips would be filler.
0 - nothing depicting this exists in such a library. A named person, a \
specific company, a sports fixture, an election result, a product launch, a \
one-off news event, or an invented phrase that means nothing.

Judge the subject, not the wording. A topic sharing no word with anything \
filmable can still be a 3 - "how a jellyfish survives without a brain" is \
jellyfish footage. A topic full of ordinary filmable words can still be a 0 - \
"State Farm's dividend payout" is not farm footage, it is a company report, \
and the farm clips a keyword search returns are the wrong pictures.

Topics:
{topics}

Return ONLY a JSON array, {count} objects long, in the same order:
[{{"index": 0, "footage": 3}}]"""


def _parse_sourceability_json(raw: str) -> dict[int, int]:
    """Read the ratings, tolerantly.

    Shaped like `_parse_niche_json`: a missing entry is simply absent, and an
    absent rating leaves the candidate unchecked rather than guessed at.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")

    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("model output was not a JSON array")

    out: dict[int, int] = {}
    for position, entry in enumerate(data):
        if isinstance(entry, bool):
            continue
        if isinstance(entry, (int, float)):
            out[position] = int(entry)
            continue
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        rating = entry.get("footage", entry.get("rating"))
        if isinstance(rating, str):
            rating = rating.strip()
            rating = int(rating) if rating.lstrip("-").isdigit() else None
        if isinstance(rating, bool) or rating is None:
            continue
        try:
            out[index] = int(rating)
        except (TypeError, ValueError):
            continue
    return out


def classify_sourceability(candidates: list[Candidate], *,
                           needed: int | None = None
                           ) -> tuple[dict[int, dict], list[str]]:
    """Ask the model whether each topic can be shot. Never raises.

    Returns a report per index in the same shape `probe_sourceability` returns,
    so `apply_sourceability` and the hard filter in `find_topics` cannot tell
    which mode produced it, and any failure path - no key, a refused call,
    unparseable JSON - returns no reports at all. An unreported candidate is
    left `checked: False`, which scores as nothing rather than as no footage.

    One call for the whole shortlist, like the niche pass: sixteen topics is a
    short list to read and sixteen requests would be the expensive way to ask
    the same question.
    """
    if not candidates:
        return {}, []

    listing = "\n".join(
        f"{index}. {candidate.topic or candidate.title}"
        for index, candidate in enumerate(candidates)
    )
    prompt = _SOURCEABILITY_PROMPT_TEMPLATE.format(
        topics=listing, count=len(candidates)
    )

    try:
        raw = script_gen.call_gemini(
            prompt,
            system_instruction=_SOURCEABILITY_SYSTEM_INSTRUCTION,
            temperature=SOURCEABILITY_TEMPERATURE,
            max_output_tokens=SOURCEABILITY_MAX_TOKENS,
            thinking_budget=SOURCEABILITY_THINKING_BUDGET,
            json_mode=True,
        )
        ratings = _parse_sourceability_json(raw)
    except (script_gen.ScriptGenerationError, config.MissingConfigError,
            ValueError) as exc:
        return {}, [f"the footage pass failed ({exc}). No candidate was "
                    f"checked for footage, so the board may hold topics step 4 "
                    f"cannot shoot. Set TOPIC_SOURCEABILITY=probe to fall back "
                    f"to the keyword probe."]

    reports: dict[int, dict] = {}
    for index, rating in ratings.items():
        if not 0 <= index < len(candidates):
            continue
        rating = min(SOURCEABILITY_SCALE, max(0, int(rating)))
        threshold = max(1, config.TOPIC_SOURCEABLE_HITS
                        if needed is None else needed)
        reports[index] = {
            "query": candidates[index].topic or candidates[index].title,
            # Carried so the hard filter reads a zero the same way whichever
            # mode ran. Only rung 0 is a zero - a 1 is thin, not absent.
            "hits": 0 if rating == 0 else threshold,
            "sample": threshold,
            "provider": "model",
            "checked": True,
            "rating": rating,
            "component": sourceability_rung(rating),
            "by_provider": {},
            "notes": [],
        }

    missing = len(candidates) - len(reports)
    warnings = ([f"the footage pass returned no rating for {missing} of "
                 f"{len(candidates)} candidates - those were left unchecked "
                 f"and scored as neither shootable nor not"] if missing else [])
    return reports, warnings


# ---------------------------------------------------------------------------
# Cross-board ledger
# ---------------------------------------------------------------------------

class TopicLedger:
    """What recent boards showed, and what was actually made.

    Two different facts with two different consequences. A topic *shown* on a
    recent board and not taken is scored down - it had its chance, and a board
    that opens with the same five candidates every morning is a board worth
    running once a week. A topic *taken* is dropped outright, because the reel
    exists and the second one would be the same reel.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.TOPIC_USAGE_FILE)
        self._data = visuals._read_json(
            self.path, {"version": 1, "boards": 0, "topics": {}}
        )
        self._data.setdefault("boards", 0)
        if not isinstance(self._data.get("topics"), dict):
            self._data["topics"] = {}

    @property
    def board_index(self) -> int:
        return int(self._data.get("boards") or 0)

    def entry(self, key: str) -> dict:
        entry = self._data["topics"].get(key)
        return entry if isinstance(entry, dict) else {}

    def was_used(self, key: str) -> bool:
        return bool(self.entry(key).get("used"))

    def seen_recently(self, key: str, *, within: int | None = None) -> bool:
        """Shown on one of the last `within` boards."""
        runs = config.TOPIC_HISTORY_RUNS if within is None else within
        if runs <= 0:
            return False
        last = int(self.entry(key).get("last_board") or 0)
        return bool(last) and (self.board_index - last) < runs

    def record_board(self, candidates: list[Candidate]) -> None:
        """Note that these candidates were shown. Does not persist by itself."""
        index = self.board_index + 1
        self._data["boards"] = index
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for candidate in candidates:
            entry = self._data["topics"].setdefault(
                candidate.key, {"used": False, "boards": 0}
            )
            entry["boards"] = int(entry.get("boards", 0)) + 1
            entry["last_board"] = index
            entry["last_seen"] = stamp
            entry["topic"] = candidate.topic
            entry["source"] = candidate.source
        self._prune()

    def mark_used(self, key: str, topic: str = "") -> None:
        entry = self._data["topics"].setdefault(key, {"boards": 0})
        entry["used"] = True
        entry["used_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if topic:
            entry["topic"] = topic

    def refusals(self, key: str) -> int:
        """How many times a script for this topic was refused outright."""
        try:
            return int(self.entry(key).get("refusals") or 0)
        except (TypeError, ValueError):
            return 0

    def was_refused_out(self, key: str) -> bool:
        """Refused enough times that the board should stop offering it.

        A filter rather than a penalty, deliberately. Board #41 deadlocked
        because two penalties priced independently stacked past a source's
        ceiling, and a count that cannot be added to a score cannot do that
        to a later board.
        """
        limit = config.TOPIC_MAX_REFUSALS
        return limit > 0 and self.refusals(key) >= limit

    def mark_refused(self, key: str, topic: str = "") -> None:
        """Note that a script for this topic was written and thrown away.

        The gap this closes: a topic is marked `used` at the first moment money
        has gone on it, which a refused draft never reaches, so the board's
        entire memory of a refusal was that the topic had been *shown*.
        """
        entry = self._data["topics"].setdefault(key, {"used": False, "boards": 0})
        entry["refusals"] = self.refusals(key) + 1
        entry["refused_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry["last_refusal_board"] = self.board_index
        if topic:
            entry["topic"] = topic

    def _prune(self) -> None:
        """Forget candidates last seen many boards ago, keeping what was made.

        A ledger that only grows is a ledger nobody ever reads. What was made
        into a reel is kept whatever its age - that entry is the record that
        stops a topic being made twice, and it never expires. A topic that cost
        a refused script is kept for the same reason: forgetting it is buying
        the refusal a second time, which is the whole thing the count exists to
        stop.
        """
        cutoff = self.board_index - LEDGER_KEEP_BOARDS
        if cutoff <= 0:
            return
        self._data["topics"] = {
            key: entry for key, entry in self._data["topics"].items()
            if entry.get("used") or entry.get("refusals")
            or int(entry.get("last_board") or 0) > cutoff
        }

    def save(self) -> list[str]:
        """Persist. A ledger that cannot be written is a warning, not a failure."""
        try:
            visuals._write_json(self.path, self._data)
        except OSError as exc:
            return [f"could not update the topic ledger at {self.path}: {exc}"]
        return []


# ---------------------------------------------------------------------------
# The phrase pass
#
# A reddit title is not a topic. "TIL that the Dead Sea is nine times saltier
# than the Atlantic because it has no outlet" is a fact with a preamble; step 2
# wants the subject. clean_title() has already taken the furniture off, and one
# Gemini call over the whole shortlist takes it the rest of the way.
#
# It rewrites and nothing else. The ranking was settled in code before the
# model was asked, the model is not shown the scores, and a rewrite that drifts
# off its original is discarded here rather than trusted - the same division of
# labour as the hook pass.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The niche pass
#
# Whether a headline is about what the account is about. Shared words cannot
# answer this and the measurement is in config.TOPIC_NICHE_MATCH: on board #16
# of an ocean-and-earth-science account, exact matching placed none of 59
# candidates on the niche, "caves beneath the Moon" among the misses, while an
# earlier board had scored a hockey World Cup a full match off the word
# "world". Both errors are the same error - a question about meaning answered
# by comparing spellings.
#
# This is the one place in step 0 where a model has a say in what reaches the
# board, so it is worth being exact about how much of one. It is asked a closed
# question about subject matter, with the subject supplied and the candidates
# in a fixed order. It is not shown the scores, does not rank, does not choose,
# and is not told how many places are on the board. Everything it returns is
# printed on the board it affected, and the interesting case - a candidate it
# kept that shares no word with the niche - is noted against that candidate so
# it can be argued with. The ranking stays in code.
# ---------------------------------------------------------------------------

_NICHE_SYSTEM_INSTRUCTION = """\
You decide whether a headline belongs on a particular video channel.

You answer about subject matter only - whether that channel covers this kind \
of thing - and never about whether the headline is interesting, popular, \
recent or well written.

You always return valid JSON and nothing else."""


_NICHE_PROMPT_TEMPLATE = """\
A short-video channel is about: {niche}

For each headline below, say whether that channel would cover its subject.

Judge the subject, not the wording. A headline that shares no word with the \
description can be squarely on it - a channel about "earth science" covers \
caves on the Moon - and one that repeats a word from it can be squarely off \
it: a channel about the "natural world" does not cover a World Cup.

Be lenient about wording and strict about the boundary. Where the subject is \
only loosely adjacent to the channel, answer false.

Headlines:
{headlines}

Return ONLY a JSON array, {count} objects long, in the same order:
[{{"index": 0, "on_topic": true}}]"""


def _parse_niche_json(raw: str) -> dict[int, bool]:
    """Read the verdicts, tolerantly.

    A missing entry is simply absent, and an absent verdict falls back to
    shared words rather than to a guess.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")

    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("model output was not a JSON array")

    out: dict[int, bool] = {}
    for position, entry in enumerate(data):
        if isinstance(entry, bool):
            out[position] = entry
            continue
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        verdict = entry.get("on_topic", entry.get("on_niche"))
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() in ("true", "yes", "y", "1")
        if verdict is not None:
            out[index] = bool(verdict)
    return out


def classify_niche(candidates: list[Candidate], niche: str, *,
                   mode: str | None = None) -> tuple[dict[int, bool], list[str]]:
    """Which candidates the account's subject covers, by index. Never raises.

    Returns the verdicts and any warnings. Every failure path - no key, a
    refused call, unparseable JSON - returns no verdicts, and the scorer falls
    back to shared words for every candidate rather than to a guess.
    """
    setting = (mode or config.TOPIC_NICHE_MATCH or "terms").strip().lower()
    if setting != "llm" or not candidates or not (niche or "").strip():
        return {}, []

    headlines = "\n".join(
        f"{index}. {candidate.topic or candidate.title}"
        for index, candidate in enumerate(candidates)
    )
    prompt = _NICHE_PROMPT_TEMPLATE.format(
        niche=niche.strip(), headlines=headlines, count=len(candidates)
    )

    try:
        raw = script_gen.call_gemini(
            prompt,
            system_instruction=_NICHE_SYSTEM_INSTRUCTION,
            temperature=NICHE_TEMPERATURE,
            max_output_tokens=NICHE_MAX_TOKENS,
            thinking_budget=NICHE_THINKING_BUDGET,
            json_mode=True,
        )
        verdicts = _parse_niche_json(raw)
    except (script_gen.ScriptGenerationError, config.MissingConfigError,
            ValueError) as exc:
        return {}, [f"the niche pass failed ({exc}). Fell back to matching "
                    f"words from TOPIC_NICHE, which is much stricter - expect "
                    f"a thinner board."]

    verdicts = {index: value for index, value in verdicts.items()
                if 0 <= index < len(candidates)}

    missing = len(candidates) - len(verdicts)
    warnings = ([f"the niche pass returned no verdict for {missing} of "
                 f"{len(candidates)} candidates - those fell back to matching "
                 f"words from TOPIC_NICHE"] if missing else [])
    return verdicts, warnings


_PHRASE_SYSTEM_INSTRUCTION = """\
You turn headlines into video topics.

A topic is the subject of a short explainer video, written as a phrase a \
person would type into a search box: 4 to 10 words, no punctuation at the end, \
no preamble, no clickbait, no colon-separated subtitle. It names what the \
video is about and nothing else.

You always return valid JSON and nothing else."""


_PHRASE_PROMPT_TEMPLATE = """\
Rewrite each headline below as a video topic, and say whether it gives a video \
anything to explain.

Rules:
- Keep the subject of the original. Do not invent a different one, and do not \
add a fact that is not in the headline.
- Drop forum wording ("TIL", "ELI5"), outlet names, and any reference to the \
post or video itself.
- Drop the verdict and keep the subject: "scientists finally explain why the \
ocean is salty" becomes "why the ocean is salty".
- 4 to 10 words. Lower case except for names. No trailing punctuation.
- Do not reorder them, do not score them, do not leave any out.

"explains" is a separate question, about the headline as written and not \
about your rewrite. Any headline can be turned into a question, so whether \
your topic reads as one is not the test. Ask whether the headline names \
something specific a video could explain: a fact, a finding, an event, a \
phenomenon, a process, or a particular creature, rock, landform or place \
with something about it that has a cause. It counts even when the post \
around it is personal or is a caption on a photo - "striped cliffs on the \
Oregon coast" names cliffs with stripes in them, and "my lecturer hates the \
word moraine" names a moraine.

Answer false when the headline names nothing a video could explain: a \
caption that points at a picture instead of naming anything in it ("my \
favourite spot on this river", "sunset at the lake last night", "look what \
I found on my hike"), the poster's own story or achievement ("my first time \
seeing the ocean"), an opinion or a matter of taste, a request to identify \
something it does not name ("what is this rock?"), or a meme. Give it a \
topic all the same.

Headlines:
{headlines}

Return ONLY a JSON array, {count} objects long, in the same order:
[{{"index": 0, "topic": "...", "explains": true}}]"""


def _parse_phrase_json(raw: str, expected: int) -> dict[int, str]:
    """Read the model's rewrites, tolerantly. Missing entries are simply absent."""
    return {index: entry["topic"]
            for index, entry in _parse_phrase_entries(raw, expected).items()
            if entry["topic"]}


def _parse_phrase_entries(raw: str, expected: int) -> dict[int, dict]:
    """Read each entry as {"topic": str, "explains": bool | None}.

    `explains` is None where the model left it out or answered with something
    that is not a yes or a no, and None keeps the candidate: a verdict that
    was never given must not drop a topic.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")

    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("model output was not a JSON array")

    out: dict[int, dict] = {}
    for position, entry in enumerate(data):
        if isinstance(entry, str):
            entry = {"index": position, "topic": entry}
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        if not 0 <= index < expected:
            continue
        topic = " ".join(str(entry.get("topic") or "").split()).strip(" .!?,;:")
        explains = entry.get("explains")
        if isinstance(explains, str):
            explains = {"true": True, "yes": True,
                        "false": False, "no": False}.get(explains.strip().lower())
        if not isinstance(explains, bool):
            explains = None
        if topic or explains is not None:
            out[index] = {"topic": topic, "explains": explains}
    return out


def _phrase_is_faithful(original: str, rewritten: str) -> bool:
    """True when a rewrite still describes the thing it was given.

    Checked here rather than asked for in the prompt, because a model that has
    drifted will not report that it drifted.

    The question is how much of the REWRITE came from the original, not how
    much of the original survived - see PHRASE_MIN_OVERLAP. A topic built
    entirely from words that were never in the headline is a topic nobody
    chose; a short topic drawn from a long headline is the whole point.
    """
    source_terms = _terms(original)
    new_terms = _terms(rewritten)
    if not source_terms or not new_terms:
        return True
    kept = sum(
        1 for term in new_terms
        if any(_same_word(term, word) for word in source_terms)
    )
    return kept / len(new_terms) >= PHRASE_MIN_OVERLAP


def polish_phrases(candidates: list[Candidate], *, mode: str | None = None) -> list[str]:
    """Rewrite each candidate's topic in place. Never raises.

    Returns warnings. Every failure path - no key, a refused call, unparseable
    JSON, a rewrite that wandered - leaves the heuristic phrase in place, which
    is a usable topic already.

    Every candidate handed in comes back with `provisional` cleared, on all of
    those paths. The flag means "a rewrite is still coming", not "a rewrite
    succeeded": once this pass has had its go the wording that stands is the
    wording that ships, and `score_candidate` is entitled to read its shape
    even when what stands is the headline it arrived with.
    """
    setting = (mode or config.TOPIC_PHRASE_MODE or "heuristic").strip().lower()
    if setting != "llm" or not candidates:
        # Before the return, so that switching the pass off does not leave the
        # board scored as though a rewrite were still coming. With no rewrite
        # coming, the cleaned headline is the final wording.
        for candidate in candidates:
            candidate.provisional = False
        return []

    headlines = "\n".join(
        f"{index}. {candidate.title or candidate.topic}"
        for index, candidate in enumerate(candidates)
    )
    prompt = _PHRASE_PROMPT_TEMPLATE.format(
        headlines=headlines, count=len(candidates)
    )

    try:
        raw = script_gen.call_gemini(
            prompt,
            system_instruction=_PHRASE_SYSTEM_INSTRUCTION,
            temperature=PHRASE_TEMPERATURE,
            # Scaled to the batch rather than fixed. This runs on the probe
            # shortlist now, not on the eight-row board the 1024 was sized
            # for, and a budget that truncates the array loses every rewrite
            # past the cut - which arrives here as absent entries and is then
            # kept as headlines, silently and with no warning.
            max_output_tokens=max(
                PHRASE_MAX_TOKENS, PHRASE_TOKENS_PER_PHRASE * len(candidates)
            ),
            thinking_budget=PHRASE_THINKING_BUDGET,
            json_mode=True,
        )
        entries = _parse_phrase_entries(raw, len(candidates))
    except (script_gen.ScriptGenerationError, config.MissingConfigError,
            ValueError) as exc:
        for candidate in candidates:
            candidate.provisional = False
        return [f"the phrase pass failed ({exc}). "
                f"Kept the cleaned headlines as topics."]

    warnings: list[str] = []
    drifted = 0
    for index, candidate in enumerate(candidates):
        # Whatever the model returned for it, this candidate has had its go.
        candidate.provisional = False
        entry = entries.get(index) or {}
        # Recorded before the drift check, and whether or not the rewrite is
        # kept: it is a verdict on the headline, which a discarded rewrite
        # does not change.
        candidate.explains = entry.get("explains")
        topic = entry.get("topic")
        if not topic:
            continue
        if not _phrase_is_faithful(candidate.title or candidate.topic, topic):
            drifted += 1
            candidate.notes.append(
                f"the phrase pass rewrote this as {topic!r}, which shares too "
                f"little with the original - kept the headline"
            )
            continue
        candidate.notes.append(f"headline: {candidate.title}")
        candidate.topic = topic

    if drifted:
        warnings.append(
            f"{drifted} of {len(candidates)} rewrites drifted off their "
            f"original and were discarded"
        )
    return warnings


def _drop_unexplainable(candidates: list[Candidate],
                        rejected: list[dict]) -> set[int]:
    """Move the candidates the phrase pass called empty to `rejected`, and
    return their `id()`s so the caller can take them off its own lists.

    By identity because `Candidate` is a dataclass, and `==` would compare
    every field. Only an explicit False drops a candidate. None means no
    verdict: the pass is off or failed, or never reached this one. None keeps
    the candidate, for the reason the niche pass falls back rather than
    guessing - a wrong drop silently removes a topic that could have been made.
    """
    if not config.TOPIC_REQUIRE_EXPLAINABLE:
        return set()
    gone: set[int] = set()
    for candidate in candidates:
        if candidate.explains is False:
            rejected.append({
                **candidate.as_dict(),
                "reason": "names nothing to explain - a caption or a post, "
                          "not a fact",
            })
            gone.add(id(candidate))
    return gone


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _dedupe(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """One entry per topic, keeping the highest-reach version of a duplicate.

    The same story reaches reddit and google trends on the same morning, and
    two slots on an eight-slot board is one slot wasted.
    """
    best: dict[str, Candidate] = {}
    order: list[str] = []
    dropped = 0
    for candidate in candidates:
        key = slugify(candidate.topic) or candidate.key
        existing = best.get(key)
        if existing is None:
            best[key] = candidate
            order.append(key)
            continue
        dropped += 1
        if candidate.reach > existing.reach:
            candidate.notes.append(f"also on {existing.source}")
            best[key] = candidate
        else:
            existing.notes.append(f"also on {candidate.source}")
    return [best[key] for key in order], dropped


def _ready_providers(session, warnings: list[str]) -> list:
    """Step 4's providers, in step 4's preference order, that have their keys."""
    providers = []
    for name in config.VISUAL_PROVIDERS:
        cls = visuals._PROVIDERS.get(name)
        if cls is None or not cls.is_configured():
            continue
        providers.append(cls(session))
    if not providers:
        warnings.append(
            "no stock provider is configured, so no candidate could be checked "
            "for footage - the board may contain topics step 4 cannot shoot"
        )
    return providers


def find_topics(niche: str | None = None, *, sources=None, limit: int | None = None,
                per_source: int | None = None, sourceability: str | None = None,
                phrases: str | None = None, niche_mode: str | None = None,
                niche_match: str | None = None, subreddits=None, keywords=None,
                curiosity: bool | None = None,
                ledger: TopicLedger | None = None) -> dict:
    """Scout topic candidates and return a ranked board.

    Args:
        niche: what the account is about. Defaults to config.TOPIC_NICHE.
        sources: override config.TOPIC_SOURCES.
        limit: how many candidates the board holds.
        per_source: how deep to read each source.
        sourceability: "llm", "probe" or "off". Defaults to
            config.TOPIC_SOURCEABILITY.
        phrases: "llm" or "heuristic". Defaults to config.TOPIC_PHRASE_MODE.
        niche_mode: "gate", "prefer" or "off". Defaults to
            config.TOPIC_NICHE_MODE.
        niche_match: "llm" or "terms" - how on-niche is decided. Defaults to
            config.TOPIC_NICHE_MATCH.
        subreddits / keywords: override the reddit and youtube inputs.
        curiosity: False stops the curiosity component being charged, which is
            the "off" arm of the `curiosity` trial. See `score_candidate`.
        ledger: an open TopicLedger, for a caller running several boards.

    Returns:
        {"niche", "candidates", "rejected", "sources", "stats", "warnings"} -
        where `candidates` is best first and each carries the parts its score
        was built from.

    Raises:
        TopicsError: every configured source failed, so there is no board.

    One source failing is a warning. A source that is not configured is a
    warning. The scout only raises when it has nothing at all to show.
    """
    niche = (niche if niche is not None else config.TOPIC_NICHE) or ""
    wanted = max(1, config.TOPIC_CANDIDATES if limit is None else limit)
    depth = max(1, config.TOPIC_PER_SOURCE if per_source is None else per_source)
    probe_mode = (sourceability or config.TOPIC_SOURCEABILITY or "off").lower()
    niche_mode = (niche_mode or config.TOPIC_NICHE_MODE or "gate").strip().lower()
    if niche_mode not in ("gate", "prefer", "off"):
        niche_mode = "gate"
    names = tuple(sources or config.TOPIC_SOURCES)

    warnings: list[str] = []
    reports: list[dict] = []
    gathered: list[Candidate] = []
    session = _new_session()
    ledger = ledger if ledger is not None else TopicLedger()

    # --- 1. read every source ---------------------------------------------
    for name in names:
        cls = _SOURCES.get(name)
        if cls is None:
            warnings.append(f"{name}: unknown source name - skipped")
            reports.append({"name": name, "ready": False, "fetched": 0,
                            "detail": "unknown source name"})
            continue

        ready, detail = cls.readiness(subreddits=subreddits, keywords=keywords)
        if not ready:
            warnings.append(f"{name}: {detail} - skipped")
            reports.append({"name": name, "ready": False, "fetched": 0,
                            "detail": detail})
            continue

        source = cls(session, **cls.overrides(subreddits=subreddits,
                                              keywords=keywords))

        try:
            found = source.fetch(niche, depth)
        except SourceError as exc:
            warnings.append(f"{name}: {exc}")
            reports.append({"name": name, "ready": True, "fetched": 0,
                            "detail": str(exc)})
            continue
        except Exception as exc:                       # a source may not be fatal
            warnings.append(f"{name}: unexpected failure - {type(exc).__name__}: {exc}")
            reports.append({"name": name, "ready": True, "fetched": 0,
                            "detail": f"{type(exc).__name__}: {exc}"})
            continue

        for candidate in found:
            # Set here rather than at six Candidate(...) sites: whether a
            # topic still needs rewriting is a fact about the source, and the
            # source class is the thing that knows it.
            candidate.provisional = not cls.wording_is_final
        gathered.extend(found)
        for note in getattr(source, "notes", ()):
            warnings.append(f"{name}: {note}")
        reports.append({"name": name, "ready": True, "fetched": len(found),
                        "requests": source.requests, "detail": detail,
                        "notes": list(getattr(source, "notes", ()))})

    if not gathered:
        raise TopicsError(
            "no source returned any candidates. "
            + (" ".join(warnings) if warnings else
               "Check TOPIC_SOURCES and the network.")
        )

    # --- 2. dedupe and score the free way ---------------------------------
    candidates, duplicates = _dedupe(gathered)
    niche_terms = (set() if niche_mode == "off" else
                   niche_vocabulary(niche, keywords or config.TOPIC_KEYWORDS))
    banned = tuple(config.TOPIC_BANNED_WORDS)

    rejected: list[dict] = []
    fresh: list[Candidate] = []
    for candidate in candidates:
        if ledger.was_used(candidate.key):
            rejected.append({**candidate.as_dict(),
                             "reason": "already made into a reel"})
            continue
        if ledger.was_refused_out(candidate.key):
            rejected.append({
                **candidate.as_dict(),
                "reason": f"the fact-check refused "
                          f"{ledger.refusals(candidate.key)} script(s) for this",
            })
            continue
        fresh.append(candidate)

    # The niche pass, before scoring because the subject is one of the things
    # scored. Run on what survived the ledger rather than on everything read,
    # so the call is not spent classifying reels that already exist.
    verdicts: dict[int, bool] = {}
    if niche_mode != "off":
        verdicts, niche_warnings = classify_niche(fresh, niche, mode=niche_match)
        warnings.extend(niche_warnings)
    # By key as well as by index: the re-score after the phrase pass works off
    # a shortlist that has been sorted twice by then, so the position a verdict
    # arrived at is no longer the position its candidate is in.
    verdict_by_key = {candidate.key: verdicts.get(index)
                      for index, candidate in enumerate(fresh)}

    scored: list[Candidate] = []
    for index, candidate in enumerate(fresh):
        verdict = verdicts.get(index)
        result = score_candidate(
            candidate, niche_terms=niche_terms, banned=banned,
            seen=ledger.seen_recently(candidate.key), on_niche=verdict,
            refused=ledger.refusals(candidate.key), curiosity=curiosity,
        )
        candidate.score = result["score"]
        candidate.components = result["components"]
        # The case worth auditing: kept on a judgement rather than on a word.
        if verdict and niche_terms and not any(
                _same_word(term, want)
                for term in _terms(candidate.topic) for want in niche_terms):
            candidate.notes.append(
                "the niche pass placed this on-niche - it shares no word "
                "with TOPIC_NICHE"
            )
        scored.append(candidate)

    scored.sort(key=lambda c: (-c.score, len(c.topic)))

    # --- 2b. the niche gate -------------------------------------------------
    # Whether a candidate is about the account's subject is a precondition, not
    # a preference to be traded against how fast it is moving - see
    # TOPIC_NICHE_MODE for the board that made the argument.
    #
    # It runs before the probe on purpose. The probe costs a request per
    # candidate, and there is no sense asking whether a topic this account will
    # not cover is shootable.
    #
    # And it backfills rather than truncating. A gate that can empty the board
    # turns a quiet day into no board at all; the honest answer on a quiet day
    # is the off-niche pile with a warning saying that is what you are reading.
    # Membership is all this decides - the order stays the score's to set, so a
    # board reads the same whether or not the probe ran.
    if niche_mode == "gate" and (niche_terms or verdicts):
        on_niche = [c for c in scored if "off the niche" not in c.components]
        off_niche = [c for c in scored if "off the niche" in c.components]
        short = max(0, wanted - len(on_niche))
        for candidate in off_niche[short:]:
            rejected.append({**candidate.as_dict(), "reason": "off the niche"})
        if short and off_niche:
            warnings.append(
                f"only {len(on_niche)} candidate(s) matched the niche, so "
                f"{min(short, len(off_niche))} off it filled the board out. "
                f"Widen TOPIC_NICHE, add a subreddit, or read a longer window "
                f"with TOPIC_REDDIT_WINDOW."
            )
        scored = sorted(on_niche + off_niche[:short],
                        key=lambda c: (-c.score, len(c.topic)))

    # --- 2c. settle the wording before anything reads it -------------------
    # The phrase pass used to run in step 4, on the finished board. That put it
    # after both of the passes that read wording, and each was reading a
    # headline it was about to rewrite:
    #
    #   * `score_candidate` charges the opening word, the length and the proper
    #     nouns. A reddit headline has none of the first and too much of the
    #     other two, so a live post was marked down for a shape the rewrite
    #     was about to give it. Measured on board #33, those three components
    #     came to -2.24 a row across seventeen reddit candidates against +0.49
    #     across seventeen backlog lines - a handicap on the source rather than
    #     on the topic, and the backlog won every board because of it.
    #   * the footage probe searches for what the topic says. It was asking the
    #     stock libraries for "Naked mole-rats can survive up to 18 minutes
    #     without oxygen by switching their metabolism to run on fructose
    #     instead of glucose, a trick otherwise only seen in plants" - a
    #     sentence no library has ever been asked for - instead of for "how
    #     naked mole-rats survive without oxygen".
    #
    # On the shortlist and not on everything read, for the reason the probe
    # gives below: it is one model call and its output is worth spending on
    # candidates the free checks already like. Sharing the probe's shortlist
    # rather than taking its own keeps the two passes talking about one set.
    shortlist = scored[:max(wanted, config.TOPIC_PROBE_LIMIT)]
    warnings.extend(polish_phrases(shortlist, mode=phrases))

    # Straight after the rewrite and before anything scores it, because the
    # rewrite is what hides the problem: it turns "My favorite place at this
    # river" into "what makes a river spot a favorite place", which then earns
    # the explainer-shape points that a caption could never have earned. See
    # TOPIC_REQUIRE_EXPLAINABLE.
    gone = _drop_unexplainable(shortlist, rejected)
    shortlist = [c for c in shortlist if id(c) not in gone]
    scored = [c for c in scored if id(c) not in gone]

    # Re-scored, because the wording moved under the score. This is the point
    # at which the form components are charged, and it is deliberately before
    # the probe: `apply_sourceability` and `apply_measured` add to
    # `components` and recompute from it, so a rebuild after either of them
    # would throw their work away.
    for candidate in shortlist:
        result = score_candidate(
            candidate, niche_terms=niche_terms, banned=banned,
            seen=ledger.seen_recently(candidate.key),
            on_niche=verdict_by_key.get(candidate.key),
            refused=ledger.refusals(candidate.key), curiosity=curiosity,
        )
        candidate.score = result["score"]
        candidate.components = result["components"]
    scored.sort(key=lambda c: (-c.score, len(c.topic)))

    # --- 3. probe the shortlist for footage --------------------------------
    # Only the shortlist: the probe costs a request per candidate, and there is
    # no point asking whether a topic that lost on every other count is
    # shootable.
    probed = 0
    if probe_mode == "llm" and shortlist:
        # One call for the shortlist rather than a request per candidate. The
        # keyword probe this replaced is still here and still selectable, but
        # it was measured not to separate good topics from bad at any
        # threshold - see TOPIC_SOURCEABILITY in config.py for the numbers.
        # Not `reports`: that name holds the per-source fetch reports this
        # function returns as `sources`, and rebinding it here replaced them
        # with the probe's {index: rating} dict. `_print_board` then iterated
        # a dict of ints looking for `entry["ready"]` and died - so every
        # scout with TOPIC_SOURCEABILITY=llm crashed on its way to printing
        # the board it had already built, and every last_board.json written
        # that way recorded the probe under "sources" instead of what fetched
        # what.
        rated, notes = classify_sourceability(shortlist)
        warnings.extend(notes)
        for index, candidate in enumerate(shortlist):
            report = rated.get(index)
            if not report:
                continue
            candidate.sourceability = report
            apply_sourceability(candidate)
            probed += 1
    elif probe_mode == "probe":
        providers = _ready_providers(session, warnings)
        if providers:
            for candidate in shortlist:
                candidate.sourceability = probe_sourceability(
                    candidate.topic, providers
                )
                apply_sourceability(candidate)
                probed += 1

    # The one hard filter on the board, and it takes two signals to trigger
    # because neither is sound alone. A probe that found nothing may only mean
    # the provider knew a synonym this does not, and under `llm` a zero is a
    # judgement that can be wrong in its own way. Proper nouns alone are often
    # fine - the Atlantic is proper and every library has it. Together they are
    # conclusive: a named subject that no stock library can show is a person, a
    # fixture or a product, and there is no footage of it to find.
    #
    # Kept conjunctive under `llm` even though the verdict is the stronger
    # signal, for the reason PROBE_MIN_TERM_HITS gives: a wrong pass costs a
    # place on a board a human is reading anyway, with the footage rating
    # printed beside it, and a wrong drop silently removes a topic that could
    # have been made.
    if probed:
        survivors = []
        for candidate in scored:
            probe = candidate.sourceability
            if (probe.get("checked") and not int(probe.get("hits") or 0)
                    and _proper_nouns(candidate.topic)):
                rejected.append({**candidate.as_dict(),
                                 "reason": "a named subject with no footage"})
                continue
            survivors.append(candidate)
        scored = survivors
        scored.sort(key=lambda c: (-c.score, len(c.topic)))

    # --- 3b. what step 9 has measured about boards like this one -----------
    # Last, so that a component added by the probe counts the same as one that
    # was there from the start, and applied to every candidate rather than only
    # the probed shortlist - a board where half the rows carry a measured
    # component and half do not is not a ranking.
    if config.PERFORMANCE_FEEDBACK:
        for candidate in scored:
            apply_measured(candidate)
        scored.sort(key=lambda c: (-c.score, len(c.topic)))

    board = scored[:wanted]

    # --- 4. catch any winner the shortlist missed --------------------------
    # Step 2c polishes the shortlist, but the sorts after it read every
    # candidate, so one that was never on the shortlist can still finish inside
    # `wanted` - the same way an unprobed candidate always could. Those arrive
    # here still provisional, and without this they would ship to step 2 as raw
    # headlines. Normally an empty list, and `polish_phrases` returns straight
    # away on one.
    stragglers = [candidate for candidate in board if candidate.provisional]
    warnings.extend(polish_phrases(stragglers, mode=phrases))
    # The same check as after 2c. A straggler that fails it leaves the board a
    # row short rather than pulling up another, which would arrive unpolished
    # and unchecked in its turn.
    gone = _drop_unexplainable(stragglers, rejected)
    board = [c for c in board if id(c) not in gone]

    # --- 5. decide how long each one should be -----------------------------
    # After the phrase pass and not before. The rewrite is what turns "TIL that
    # the ocean is salty because rivers carry minerals into it" into "why the
    # ocean is salty" - a fact into a mechanism, which is most of what the
    # length turns on. Scoring runs before the rewrite because it has to rank
    # candidates that are about to be discarded; this runs after because it
    # only concerns the ones that survived, and reads their final wording.
    for candidate in board:
        apply_duration(candidate)

    # --- 6. remember what was shown ----------------------------------------
    ledger.record_board(board)
    warnings.extend(ledger.save())

    return {
        "niche": niche,
        "niche_mode": niche_mode,
        "niche_verdicts": len(verdicts),
        "scouted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates": [candidate.as_dict() for candidate in board],
        "rejected": rejected,
        "sources": reports,
        "stats": {
            "fetched": len(gathered),
            "duplicates": duplicates,
            "considered": len(candidates),
            "probed": probed,
            "rejected": len(rejected),
            "shown": len(board),
            "board": ledger.board_index,
        },
        "warnings": warnings,
    }


def save_board(board: dict, path: Path | None = None) -> list[str]:
    """Write the board so the next command can act on this one.

    Never fatal: failing to cache a board must not lose the board itself, so a
    write error comes back as a warning to print rather than an exception.
    """
    target = path or config.TOPIC_BOARD_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(board, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    except OSError as exc:
        return [f"could not save the board to {target}: {exc}"]
    return []


def load_board(path: Path | None = None, *, max_age_minutes: float | None = None):
    """The last saved board, or None if there is none worth using.

    Returns (board, age_minutes). A board that is unreadable, unparseable, from
    a newer format or simply too old is treated the same way - as no board -
    because every one of those cases wants a fresh scout, not an error.
    """
    target = path or config.TOPIC_BOARD_FILE
    limit = (config.TOPIC_BOARD_TTL_MINUTES if max_age_minutes is None
             else max_age_minutes)
    try:
        board = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, 0.0
    if not isinstance(board, dict) or not board.get("candidates"):
        return None, 0.0

    stamped = _parse_iso8601(board.get("scouted_at") or "")
    if stamped is None:
        return None, 0.0
    age = (datetime.now(timezone.utc) - stamped).total_seconds() / 60.0
    if limit > 0 and age > limit:
        return None, age
    return board, max(age, 0.0)


def take_topic(key: str, topic: str = "", *, ledger: TopicLedger | None = None) -> list[str]:
    """Mark a topic as made, so no later board offers it again.

    Returns warnings. Called by `--take`, and by anything later that turns a
    board into a reel without a human in between.
    """
    ledger = ledger if ledger is not None else TopicLedger()
    ledger.mark_used(key, topic)
    return ledger.save()


def refuse_topic(key: str, topic: str = "",
                 *, ledger: TopicLedger | None = None) -> list[str]:
    """Note that a script for this topic was written and refused.

    The counterpart to `take_topic`, and the one that was missing. A refusal
    costs a Pro call and leaves nothing behind, so a board with no memory of it
    buys the same refusal again - five times over, for the topic that prompted
    this. Returns warnings.
    """
    if config.TOPIC_MAX_REFUSALS <= 0 or not key:
        return []
    ledger = ledger if ledger is not None else TopicLedger()
    ledger.mark_refused(key, topic)
    return ledger.save()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _script_gen_command(candidate: dict) -> str:
    """The exact step 2 command for a candidate, so nothing is retyped.

    The length rides along in `-d` because it is a decision this board made and
    can defend, not a default worth inheriting silently. A board written before
    lengths existed has none, and the command is still correct without it.
    """
    command = f'  python pipeline/script_gen.py "{candidate["topic"]}"'
    if candidate.get("duration"):
        command += f' -d {candidate["duration"]}'
    return command


def _format_duration(candidate: dict) -> str:
    """The suggested length and the argument for it, on one line."""
    parts = candidate.get("duration_components") or {}
    detail = ", ".join(
        f"{name} {value:g}" if name == "base" else f"{name} {value:+g}"
        for name, value in parts.items()
    )
    return f"length: {candidate['duration']}s   {detail}"


def _format_reach(candidate: dict) -> str:
    """The source's own numbers, in the source's own words."""
    reach, rate = candidate["reach"], candidate["rate"]
    unit = candidate["rate_unit"] or "per hour"
    bits = []
    source = candidate["source"]
    if source == "reddit":
        bits.append(f"{_compact(reach)} upvotes")
        if candidate["comments"]:
            bits.append(f"{_compact(candidate['comments'])} comments")
    elif source in ("youtube", "wikipedia"):
        bits.append(f"{_compact(reach)} views")
    elif reach:
        bits.append(f"~{_compact(reach)} searches")
    # The rss source has no reach to print and its rate is a proxy for one, so
    # printing either as a number would invent a measurement. Its position in
    # the listing is already in `context`, which is the honest version.
    if source == "redditrss":
        pass
    elif rate and abs(rate - reach) > 0.5:
        bits.append(f"{_compact(rate)} {unit}")
    bits.append(f"{candidate['age_hours']:.0f}h old")
    return ", ".join(bits)


def _compact(value: float) -> str:
    """1234 -> 1.2k. Boards are read at a glance, not audited."""
    number = float(value or 0)
    for threshold, suffix in ((1e9, "b"), (1e6, "m"), (1e3, "k")):
        if abs(number) >= threshold:
            return f"{number / threshold:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{number:.0f}"


def _print_board(board: dict) -> None:
    rule = "-" * 78
    stats = board["stats"]
    print(rule)
    mode = board.get("niche_mode") or ""
    judged = board.get("niche_verdicts") or 0
    detail = ", ".join(part for part in
                       (mode, f"{judged} judged by the niche pass"
                        if judged else "") if part)
    print(f"NICHE   : {board['niche'] or '(not set)'}"
          + (f"   ({detail})" if detail else ""))
    print("SOURCES : " + ", ".join(
        f"{entry['name']} {entry['fetched']}" if entry["ready"]
        else f"{entry['name']} -" for entry in board["sources"]
    ) + f"   ({stats['fetched']} fetched, {stats['considered']} after dedupe)")
    if stats["probed"]:
        how = (config.TOPIC_SOURCEABILITY or "").strip().lower()
        print(f"PROBE   : {stats['probed']} checked for footage, "
              + (f"rated 0-{SOURCEABILITY_SCALE} by the model"
                 if how == "llm" else
                 f"{config.TOPIC_SOURCEABLE_HITS} of "
                 f"{config.TOPIC_PROBE_SAMPLE} results naming it is comfortable"))
    print(f"BOARD   : #{stats['board']}, {stats['shown']} shown, "
          f"{stats['rejected']} rejected")
    print(rule)
    print("TOPICS  (scored here, not by the model - best first)")

    for position, candidate in enumerate(board["candidates"], start=1):
        print(f"{position:>2}. {candidate['score']:>6.2f}  {candidate['topic']}")
        where = f"{candidate['source']}"
        if candidate["context"]:
            where += f" {candidate['context']}"
        print(f"           {where}  {_format_reach(candidate)}")
        detail = ", ".join(f"{name} {value:+g}"
                           for name, value in candidate["components"].items())
        print(f"           {detail}")
        probe = candidate.get("sourceability") or {}
        if probe.get("checked"):
            print(f"           footage: {probe['hits']} of {probe['sample']} "
                  f"results name {probe['query']!r} "
                  f"on {probe['provider'] or '-'}")
        if candidate.get("duration"):
            print(f"           {_format_duration(candidate)}")
        for note in candidate["notes"]:
            print(f"           - {note}")

    print(rule)
    if board["rejected"]:
        reasons: dict[str, int] = {}
        for entry in board["rejected"]:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
        print("rejected:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3} x {reason}")
    if board["warnings"]:
        print("warnings:")
        for warning in board["warnings"]:
            print(f"    - {warning}")
    if board["candidates"]:
        # The exact next command, so the topic never has to be retyped.
        print("next:")
        print(_script_gen_command(board["candidates"][0]))
        print("  (or --take N to mark one made and print its command)")
    print(rule)


def _make_output_lossy() -> None:
    """Never lose a finished board to the console's codepage.

    Titles arrive from the sources with emoji and in every alphabet, and a
    Windows console encodes cp1252. Printing one character it cannot represent
    raises UnicodeEncodeError *after* every request has been spent - the board
    is gathered, probed, paid for and then thrown away on the last line, which
    is the one failure mode this whole module is written to avoid. A mangled
    character in a printed title costs nothing by comparison, and the JSON
    output and the ledger keep the real text either way.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass            # not a reconfigurable stream: nothing to do


def main(argv: list[str] | None = None) -> int:
    _make_output_lossy()
    parser = argparse.ArgumentParser(
        description="Scout topic candidates from public sources (step 0)."
    )
    parser.add_argument("-n", "--niche", default=None,
                        help="what the account is about (default: TOPIC_NICHE)")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="how many candidates to show")
    parser.add_argument("--per-source", type=int, default=None,
                        help="how deep to read each source")
    parser.add_argument("--source", action="append", dest="sources", default=None,
                        help="override TOPIC_SOURCES (repeatable)")
    parser.add_argument("--subreddit", action="append", dest="subreddits",
                        default=None, help="override TOPIC_SUBREDDITS (repeatable)")
    parser.add_argument("--keyword", action="append", dest="keywords", default=None,
                        help="override TOPIC_KEYWORDS for the youtube search")
    parser.add_argument("--niche-mode", default=None,
                        choices=("gate", "prefer", "off"),
                        help="what an off-niche candidate is worth: held off "
                             "the board (gate), scored down (prefer), or not "
                             "judged (off). Default: TOPIC_NICHE_MODE")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the stock-footage check (faster, and the "
                             "board may hold topics step 4 cannot shoot)")
    parser.add_argument("--niche-match", default=None,
                        choices=("llm", "terms"),
                        help="how on-niche is decided: one Gemini call over "
                             "the shortlist (llm) or a word from TOPIC_NICHE "
                             "appearing in the topic (terms, free and much "
                             "stricter). Default: TOPIC_NICHE_MATCH")
    parser.add_argument("--no-phrases", action="store_true",
                        help="make no Gemini calls at all - keep the cleaned "
                             "headlines as topics and decide the niche on "
                             "words. Overridden by an explicit --niche-match")
    parser.add_argument("--no-curiosity", action="store_true",
                        help="score the board without the curiosity component, "
                             "which is the 'off' arm of PERFORMANCE_EXPLORE="
                             "curiosity. Run it against an ordinary board with "
                             "--rescout to see whether the 0.5 changes who wins")
    parser.add_argument("--take", type=int, default=None, metavar="N",
                        help="mark candidate N as made, so no later board "
                             "offers it, and print its script_gen command")
    parser.add_argument("--rescout", action="store_true",
                        help="scout again even when --take could act on the "
                             "board already saved")
    parser.add_argument("--sources-status", action="store_true",
                        help="show which sources are configured, then exit")
    parser.add_argument("--json", action="store_true",
                        help="print the board as JSON instead of a summary")
    args = parser.parse_args(argv)

    if args.sources_status:
        for entry in source_status():
            mark = "ready" if entry["ready"] else "not ready"
            print(f"{entry['name']:<12} {mark:<10} {entry['detail']}")
        return 0

    # `--take N` on its own means "the one I just looked at". Scouting again
    # would choose from a different board: the sources move between runs, and
    # an anonymous Reddit read that was rate limited on the second pass drops
    # out of the running entirely. Any argument that would change what the
    # board contains forces a fresh scout, as does --rescout.
    shaping = (args.niche, args.niche_mode, args.niche_match, args.sources,
               args.limit, args.per_source, args.subreddits, args.keywords)
    board, age = (None, 0.0)
    if args.take is not None and not args.rescout and not any(shaping):
        board, age = load_board()

    if board is None:
        try:
            board = find_topics(
                args.niche,
                sources=args.sources,
                limit=args.limit,
                per_source=args.per_source,
                sourceability="off" if args.no_probe else None,
                phrases="heuristic" if args.no_phrases else None,
                niche_mode=args.niche_mode,
                # --no-phrases is what people reach for to get a board without
                # spending anything, so it has to cover both of step 0's calls
                # or it is a free mode that quietly is not one.
                niche_match=(args.niche_match
                             or ("terms" if args.no_phrases else None)),
                subreddits=args.subreddits,
                keywords=args.keywords,
                curiosity=False if args.no_curiosity else None,
            )
        except (ValueError, config.MissingConfigError, TopicsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for warning in save_board(board):
            print(f"note: {warning}", file=sys.stderr)
        reused = False
    else:
        reused = True

    if args.json:
        print(json.dumps(board, indent=2, ensure_ascii=False))
    else:
        _print_board(board)
        if reused:
            print(f"(taking from the board scouted {age:.0f} minute(s) ago - "
                  f"--rescout to look again)")

    if args.take is not None:
        if not 1 <= args.take <= len(board["candidates"]):
            print(f"error: --take must be between 1 and "
                  f"{len(board['candidates'])}", file=sys.stderr)
            return 1
        chosen = board["candidates"][args.take - 1]
        for warning in take_topic(chosen["key"], chosen["topic"]):
            print(f"note: {warning}", file=sys.stderr)
        print(f"taken: {chosen['topic']}")
        print(_script_gen_command(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
