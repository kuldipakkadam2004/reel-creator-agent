"""Step 4 - visual sourcing.

Turns a narration script (step 2) plus its rendered audio (step 3) into an
ordered set of downloaded video/photo assets for step 5 to assemble.

What happens, in order:

  1. the script is cut into *visual beats* - runs of sentences short enough to
     hold one shot - and each beat is given start/end timestamps scaled to the
     measured length of the narration .mp3, not to an estimate;
  2. every beat gets a search query (Gemini writes them by default, with a
     keyword-extraction fallback that needs no API call);
  3. the queries are run against the configured stock providers until enough
     usable candidates come back;
  4. candidates are filtered against a crop-aware resolution floor - the test is
     what survives a centre crop to 9:16, so a 4K landscape clip passes and a
     1080p one does not - then ranked and downloaded;
  5. the run is written out as a directory of media plus a manifest.json.

Downloads are content-addressed in assets/visuals/cache and hardlinked into the
per-run directory, so re-running a script costs no bytes and no API calls. A
usage ledger remembers which stock clips recent runs used, so the same footage
does not open every video.

Nothing here touches ffmpeg, captions or encoding - that is step 5.

Standalone use:
    python pipeline/visuals.py --script-file output/scripts/ocean.txt --audio output/audio/ocean.mp3
    python pipeline/visuals.py "the script text" --duration 42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
from urllib.parse import urlsplit

# Allow both `python pipeline/visuals.py` and `python -m pipeline.visuals`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables. Provider choice, API keys, the resolution floor, assets-per-beat and
# the duration headroom all live in config.py - what follows only shapes the
# local heuristics: how the script is cut up, how hard we retry, and so on.
# ---------------------------------------------------------------------------

#: Aspect ratio every candidate is judged against, taken from the step-5 frame.
TARGET_ASPECT = config.VIDEO_WIDTH / config.VIDEO_HEIGHT

#: Beat length window, in seconds. Under ~2.5s reads as a jump cut; over ~6s a
#: single shot starts to feel static in a reel. The maximum is a soft ceiling:
#: absorbing a too-short beat into a neighbour can exceed it by up to
#: BEAT_MIN_SECONDS, which is the better of the two compromises.
BEAT_MIN_SECONDS = 2.5
BEAT_TARGET_SECONDS = 4.0
BEAT_MAX_SECONDS = 6.5

#: Splitting changes the timing rate, which changes what counts as too long.
#: Two passes settle it for any realistic script; the third is paranoia.
_SPLIT_PASSES = 3

#: Dead air after a sentence, expressed in "characters worth of time" so it can
#: be folded straight into the proportional timing weights below.
SENTENCE_PAUSE_WEIGHT = 6

#: Ceiling on how much of one clip counts toward coverage. A 60s stock clip is
#: not 60s of usable variety, and pretending otherwise starves the run.
COVERAGE_CLIP_CAP_SECONDS = 12.0

#: Hard stop on assets per run, so a long script cannot run away with the disk.
MAX_TOTAL_ASSETS = 40

#: How many results to ask a provider for per query, and the hard cap on how
#: big a candidate pool is worth building for one beat.
SEARCH_PAGE_SIZE = 15
CANDIDATE_POOL_SIZE = 30

#: Searching stops once a beat has this many times more *usable* candidates
#: than it needs assets. The surplus absorbs dead links, duplicates and the
#: reuse cooldown; going wider than that only spends other providers' quota.
CANDIDATE_MARGIN = 4

#: (connect, read) seconds. Searches are small; downloads stream real media.
SEARCH_TIMEOUT = (10, 30)
DOWNLOAD_TIMEOUT = (10, 180)
DOWNLOAD_CHUNK_SIZE = 1 << 16

#: Refuse a single file bigger than this - a 300MB 4K master is not worth it.
MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024

#: One retry after a 429, then the provider is benched for the rest of the run.
RATE_LIMIT_BACKOFF_SECONDS = 2.0

#: How many downloads a single beat may attempt before giving up. Without a cap,
#: one beat can burn a hundred requests on a provider that refuses them all.
MAX_DOWNLOAD_ATTEMPTS_PER_BEAT = 8

#: Sent on every request. Openverse asks for a real UA; the rest tolerate it.
USER_AGENT = "reel-creator-agent/0.1 (visual sourcing)"

#: Gemini settings for query writing. Low temperature: these are labels, not
#: prose, and run-to-run churn would defeat the download cache.
QUERY_MODEL_TEMPERATURE = 0.35
QUERY_MODEL_MAX_TOKENS = 2048
QUERY_THINKING_BUDGET = 0

#: Words that never make a good stock-footage query on their own.
_STOPWORDS = frozenset("""
a about above after again against all almost also am an and another any are
around as at back be became because become been before being below best better
between both but by came can cannot come could did do does doing done down
during each either else even ever every few first for from further get gets
getting give go goes going got had has have having he her here hers herself him
himself his how however if in into is it its itself just keep know last least
less let like little long made make makes making many may me might more most
much must my myself never new next no nor not now of off often on once one only
or other others our ours out over own perhaps put quite rather really same say
says see seen seems she should since so some someone something still such take
than that the their theirs them themselves then there these they thing things
think this those though through thus to too under until up upon us use used
using very want was way we well were what when where whether which while who
whom why will with within without would yet you your yours
""".split())

#: Abbreviations that end in a full stop without ending a sentence.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "st", "mt", "vs", "etc", "e.g", "i.e",
    "jr", "sr", "inc", "ltd", "co", "approx", "fig", "vol",
})

#: Appended to image prompts so an image-generation provider - or a human
#: reading the manifest - knows which frame these assets are cut for.
IMAGE_PROMPT_STYLE = (
    "cinematic vertical 9:16 photograph, natural light, shallow depth of field, "
    "no text, no watermark"
)

#: Extension used when a download URL carries none.
_CONTENT_TYPE_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

#: Content types a download is allowed to return. Anything else (an HTML "file
#: removed" page, most often) is treated as a dead URL.
_ALLOWED_CONTENT_PREFIXES = ("video/", "image/", "application/octet-stream")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VisualsError(RuntimeError):
    """Base class for every failure surfaced by this module."""


class ProviderError(VisualsError):
    """A stock provider failed: network, bad response, or an API error."""

    def __init__(self, message: str, *, provider: str = "", fatal: bool = False):
        super().__init__(message)
        self.provider = provider
        #: True when retrying this provider during this run is pointless (bad
        #: key, rate limited) - the caller benches it instead of hammering it.
        self.fatal = fatal


class ProviderAuthError(ProviderError):
    """The provider rejected the key. Benched for the run."""

    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(message, provider=provider, fatal=True)


class ProviderRateLimitError(ProviderError):
    """The provider is rate limiting us. Benched for the run."""

    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(message, provider=provider, fatal=True)


class NoAssetsError(VisualsError):
    """Not one usable asset came back for the whole script."""


# ---------------------------------------------------------------------------
# Data carried between the stages
# ---------------------------------------------------------------------------

@dataclass
class Beat:
    """One visual beat: a slice of narration that holds a single shot."""

    index: int
    text: str
    start: float
    end: float
    query: str = ""
    alt_queries: tuple[str, ...] = ()
    image_prompt: str = ""
    query_source: str = "heuristic"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
            "query": self.query,
            "alt_queries": list(self.alt_queries),
            "image_prompt": self.image_prompt,
            "query_source": self.query_source,
        }


@dataclass(frozen=True)
class Candidate:
    """One search hit, before anything has been downloaded."""

    provider: str
    provider_id: str
    kind: str                        # "video" | "image"
    width: int
    height: int
    download_url: str
    source_url: str = ""
    creator: str = ""
    creator_url: str = ""
    license: str = ""
    title: str = ""
    duration: float | None = None    # videos only
    preview_url: str = ""

    @property
    def key(self) -> str:
        """Stable cross-run identity: `provider:id`."""
        return f"{self.provider}:{self.provider_id}"

    @property
    def attribution(self) -> str:
        """A credit line safe to paste into a post description."""
        who = self.creator or "unknown"
        what = self.title or self.kind
        licence = self.license or "see source"
        return f'"{what}" by {who} via {self.provider} ({licence}) - {self.source_url}'


# ---------------------------------------------------------------------------
# Beat segmentation
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])["\'”’)\]]*\s+')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;:—–])\s+')
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _ends_with_abbreviation(fragment: str) -> bool:
    """True if `fragment` ends in "Dr." or "U.S." rather than a real full stop."""
    stripped = fragment.rstrip()
    if not stripped.endswith("."):
        return False
    words = stripped.rstrip(".").split()
    if not words:
        return False
    last = words[-1].lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(text: str) -> list[str]:
    """Split narration into sentences, keeping abbreviations intact."""
    pieces = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _speech_weight(text: str) -> int:
    """Proxy for how long `text` takes to say: non-space characters, plus a pause.

    Character count tracks spoken time better than word count once word lengths
    vary, and the flat pause term stops a run of three-word sentences from being
    timed as though they were read without breath.
    """
    return len(re.sub(r"\s+", "", text)) + SENTENCE_PAUSE_WEIGHT


def _split_by_words(text: str, max_weight: float) -> list[str]:
    """Last-resort split of a clause that has no punctuation to break on.

    The text of a beat only feeds query building and timing, so cutting one
    long clause mid-phrase costs nothing a viewer can see - whereas leaving a
    nine-second static shot in a reel does.
    """
    words = text.split()
    if len(words) < 2:
        return [text]
    parts = int(_speech_weight(text) // max(max_weight, 1)) + 1
    size = max(1, -(-len(words) // parts))      # ceil division
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _split_long_sentence(sentence: str, max_weight: float) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at word ones."""
    if _speech_weight(sentence) <= max_weight:
        return [sentence]

    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]

    packed: list[str] = []
    buffer = ""
    for clause in clauses:
        candidate = f"{buffer} {clause}".strip()
        if buffer and _speech_weight(candidate) > max_weight:
            packed.append(buffer)
            buffer = clause
        else:
            buffer = candidate
    if buffer:
        packed.append(buffer)

    out: list[str] = []
    for piece in packed or [sentence]:
        if _speech_weight(piece) > max_weight:
            out.extend(_split_by_words(piece, max_weight))
        else:
            out.append(piece)
    return out


def _merge_short_groups(
    groups: list[list[str]], min_weight: float, max_weight: float
) -> list[list[str]]:
    """Fold any beat too short to register into whichever neighbour is smaller.

    Packing can leave a stub anywhere, not just at the end - a trailing
    "forever." from a clause split is its own group and would flash by in under
    two seconds.
    """
    while len(groups) > 1:
        weights = [sum(_speech_weight(u) for u in g) for g in groups]
        try:
            index = next(i for i, w in enumerate(weights) if w < min_weight)
        except StopIteration:
            break
        neighbours = [i for i in (index - 1, index + 1) if 0 <= i < len(groups)]
        # Prefer a neighbour that can absorb the stub without going over the
        # ceiling; if neither can, take the smaller one anyway - a beat a little
        # over length still reads better than one that flashes past.
        fits = [i for i in neighbours if weights[i] + weights[index] <= max_weight]
        target = min(fits or neighbours, key=lambda i: weights[i])

        if target < index:
            groups[target] = groups[target] + groups[index]
        else:
            groups[target] = groups[index] + groups[target]
        groups.pop(index)
    return groups


def segment_script(script: str, total_duration: float) -> list[Beat]:
    """Cut `script` into beats timed against a `total_duration`-second read.

    Timings are proportional to how much there is to say in each beat, and are
    rescaled so the last beat lands exactly on `total_duration`. When that
    duration was measured from the real .mp3, the beat boundaries line up with
    the narration and step 5 can treat them as placement hints.
    """
    if total_duration <= 0:
        raise ValueError(f"total_duration must be positive, got {total_duration}")

    sentences = split_sentences(script)
    if not sentences:
        raise ValueError("script contains no sentences to segment")

    # Splitting a sentence adds one pause term per fragment, so the weight total
    # grows every time something is cut - which changes how many seconds a unit
    # of weight is worth, which changes what counts as over-long. Iterate to a
    # fixed point: each pass splits at the current rate, and because splitting
    # only ever lowers the rate, a piece that fits one pass still fits the next.
    units = list(sentences)
    for _ in range(_SPLIT_PASSES):
        rate = total_duration / sum(_speech_weight(u) for u in units)
        split: list[str] = []
        for unit in units:
            split.extend(_split_long_sentence(unit, BEAT_MAX_SECONDS / rate))
        if split == units:
            break
        units = split

    seconds_per_weight = total_duration / sum(_speech_weight(u) for u in units)
    max_weight = BEAT_MAX_SECONDS / seconds_per_weight
    target_weight = BEAT_TARGET_SECONDS / seconds_per_weight
    min_weight = BEAT_MIN_SECONDS / seconds_per_weight

    # Pack units into beats, flushing once a beat is long enough to hold a shot.
    groups: list[list[str]] = []
    current: list[str] = []
    current_weight = 0.0
    for unit in units:
        weight = _speech_weight(unit)
        if current and current_weight + weight > max_weight:
            groups.append(current)
            current, current_weight = [unit], float(weight)
            continue
        current.append(unit)
        current_weight += weight
        if current_weight >= target_weight:
            groups.append(current)
            current, current_weight = [], 0.0
    if current:
        groups.append(current)

    groups = _merge_short_groups(groups, min_weight, max_weight)

    beats: list[Beat] = []
    elapsed = 0.0
    for index, group in enumerate(groups):
        span = sum(_speech_weight(u) for u in group) * seconds_per_weight
        start = elapsed
        elapsed = min(float(total_duration), elapsed + span)
        beats.append(Beat(index=index, text=" ".join(group), start=start, end=elapsed))

    beats[-1].end = float(total_duration)  # absorb accumulated rounding
    return beats


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _keywords(text: str, limit: int = 3) -> list[str]:
    """Pull the most search-worthy words out of a beat, in reading order.

    Scored on length, on repetition, and on looking like a proper noun;
    stopwords and very short tokens are dropped outright. Deliberately
    dependency-free - a POS tagger would do better and is not worth a large
    model download in this pipeline.
    """
    scores: dict[str, float] = {}
    order: list[str] = []
    for raw in _WORD_RE.findall(text):
        word = raw.lower()
        if word in _STOPWORDS or len(word) < 3 or word.isdigit():
            continue
        score = min(len(word), 12) / 12.0
        if raw[0].isupper() and raw != "I":
            score += 0.6
        if word not in scores:
            order.append(word)
        scores[word] = scores.get(word, score) + 0.15  # repeats matter
    ranked = sorted(order, key=lambda w: (-scores[w], order.index(w)))
    chosen = set(ranked[:limit])
    return [w for w in order if w in chosen]


def heuristic_queries(beats: list[Beat], topic: str) -> None:
    """Fill in query/alt_queries/image_prompt without calling any API."""
    topic_terms = _keywords(topic, limit=2) if topic else []
    topic_hint = " ".join(topic_terms)

    for beat in beats:
        terms = _keywords(beat.text, limit=3)
        primary = " ".join(terms) or topic_hint or "cinematic background"

        alts: list[str] = []
        if len(terms) > 1:
            alts.append(" ".join(terms[:2]))
        if topic_hint:
            alts.append(topic_hint)
        if terms and topic_terms and terms[0] != topic_terms[0]:
            alts.append(f"{terms[0]} {topic_terms[0]}")

        beat.query = primary
        beat.alt_queries = tuple(
            dict.fromkeys(a for a in alts if a and a != primary)
        )
        beat.image_prompt = f"{primary}, {IMAGE_PROMPT_STYLE}"
        beat.query_source = "heuristic"


_QUERY_SYSTEM_INSTRUCTION = """\
You turn narration into stock-footage search queries for vertical short-form \
video.

A good query names something a camera can actually point at: a place, an object, \
a person doing something, a material, a weather condition. It is 2 to 5 words, \
with no punctuation, no camera directions, no abstractions such as "success" or \
"innovation", and no words that only make sense as metaphor.

You always return valid JSON and nothing else."""


_QUERY_PROMPT_TEMPLATE = """\
Topic of the video: {topic}

Below are the numbered beats of the narration. For each beat, write the stock \
search query for the shot that should be on screen while those words are heard.

{beats}

Rules:
- The visual must match what THAT beat is saying, not the topic in general. \
Consecutive beats must not repeat the same query.
- "query" is the stock search: 2 to 5 concrete, literal, filmable words.
- "alt" is a second, broader query to fall back on if the first finds nothing.
- "image_prompt" is one sentence describing the same shot for an image \
generator, naming both the subject and the setting.

Return ONLY a JSON array, one object per beat, in beat order:
[{{"index": 0, "query": "...", "alt": "...", "image_prompt": "..."}}]"""


def _parse_query_json(raw: str, expected: int) -> dict[int, dict]:
    """Read the model's JSON array into {beat_index: entry}, tolerantly."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")

    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("model output was not a JSON array")

    parsed: dict[int, dict] = {}
    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        if not 0 <= index < expected:
            continue
        query = str(entry.get("query") or "").strip()
        if query:
            parsed[index] = {
                "query": query,
                "alt": str(entry.get("alt") or "").strip(),
                "image_prompt": str(entry.get("image_prompt") or "").strip(),
            }
    return parsed


def llm_queries(beats: list[Beat], topic: str) -> list[str]:
    """Ask Gemini for one query per beat. Returns warnings; never raises.

    Beats the model does not answer for keep whatever the heuristic pass gave
    them, so a partial response costs one beat rather than the run.
    """
    try:
        api_key = config.require(*config.REQUIRED_SCRIPT_GEN)["GEMINI_API_KEY"]
    except config.MissingConfigError as exc:
        return [f"query_mode=llm but {exc} Used keyword extraction instead."]

    listing = "\n".join(f'{b.index}. ({b.duration:.1f}s) "{b.text}"' for b in beats)
    prompt = _QUERY_PROMPT_TEMPLATE.format(topic=topic or "(not given)", beats=listing)

    try:
        from google import genai
        from google.genai import types

        kwargs = {
            "system_instruction": _QUERY_SYSTEM_INSTRUCTION,
            "temperature": QUERY_MODEL_TEMPERATURE,
            "max_output_tokens": QUERY_MODEL_MAX_TOKENS,
            "response_mime_type": "application/json",
        }
        # Only the 2.5 flash family accepts a zero thinking budget; pro rejects it.
        if "pro" not in config.GEMINI_MODEL.lower():
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=QUERY_THINKING_BUDGET
            )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**kwargs),
        )
        entries = _parse_query_json(response.text or "", len(beats))
    except Exception as exc:  # noqa: BLE001 - any failure falls back to keywords
        return [
            f"Gemini query generation failed ({type(exc).__name__}: {exc}). "
            "Used keyword extraction instead."
        ]

    if not entries:
        return ["Gemini returned no usable queries. Used keyword extraction instead."]

    warnings: list[str] = []
    for beat in beats:
        entry = entries.get(beat.index)
        if not entry:
            warnings.append(
                f"beat {beat.index}: no query from Gemini, kept keyword extraction"
            )
            continue
        alts = [a for a in (entry["alt"], beat.query) if a and a != entry["query"]]
        beat.query = entry["query"]
        beat.alt_queries = tuple(dict.fromkeys(alts))
        beat.image_prompt = (
            entry["image_prompt"] or f"{entry['query']}, {IMAGE_PROMPT_STYLE}"
        )
        beat.query_source = "llm"
    return warnings


def build_queries(beats: list[Beat], topic: str, mode: str) -> list[str]:
    """Populate every beat's query fields. Returns non-fatal warnings."""
    heuristic_queries(beats, topic)          # always run: it is the safety net
    if mode == "llm":
        return llm_queries(beats, topic)
    if mode != "heuristic":
        return [f"unknown VISUAL_QUERY_MODE {mode!r}; used keyword extraction"]
    return []


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _new_session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get_json(session, url: str, *, provider: str, params=None, headers=None) -> dict:
    """GET a JSON body, mapping transport and API failures onto ProviderError."""
    import requests

    try:
        response = session.get(
            url, params=params, headers=headers, timeout=SEARCH_TIMEOUT
        )
    except requests.exceptions.Timeout as exc:
        raise ProviderError(
            f"{provider} search timed out after {SEARCH_TIMEOUT[1]}s", provider=provider
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderError(
            f"could not reach {provider}: {type(exc).__name__}: {exc}", provider=provider
        ) from exc

    body = (response.text or "").strip()[:200]
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "")
        suffix = f", retry after {retry_after}s" if retry_after else ""
        raise ProviderRateLimitError(
            f"{provider} rate limited [429{suffix}]: {body}", provider=provider
        )
    if response.status_code in (401, 403):
        raise ProviderAuthError(
            f"{provider} rejected the API key [{response.status_code}]: {body}",
            provider=provider,
        )
    if response.status_code >= 400:
        raise ProviderError(
            f"{provider} search failed [{response.status_code}]: {body}",
            provider=provider,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(
            f"{provider} returned a non-JSON body: {body!r}", provider=provider
        ) from exc


# ---------------------------------------------------------------------------
# Providers
#
# Each provider maps its own JSON onto Candidate and nothing else: no filtering,
# no ranking, no downloading. Adding a provider (an image-generation API, say)
# means writing one `search` method and adding it to _PROVIDERS.
# ---------------------------------------------------------------------------

class Provider:
    """Base class. Subclasses fill in `name`, `kinds`, `key_env` and `search`."""

    name = ""
    kinds: tuple[str, ...] = ()
    key_env: tuple[str, ...] = ()
    attribution_required = False

    def __init__(self, session):
        self.session = session
        #: Set when the provider is benched mid-run (bad key, rate limited).
        self.disabled_reason = ""
        self.searches = 0

    @classmethod
    def api_key(cls) -> str | None:
        """First non-empty value among this provider's accepted env vars."""
        for name in cls.key_env:
            value = config.get_env(name)
            if value:
                return value
        return None

    @classmethod
    def is_configured(cls) -> bool:
        return not cls.key_env or bool(cls.api_key())

    @classmethod
    def missing_key_message(cls) -> str:
        return f"{cls.name}: set " + " or ".join(cls.key_env) + " in .env"

    def search(self, query: str, kind: str, limit: int) -> list[Candidate]:
        raise NotImplementedError


class PexelsProvider(Provider):
    """Pexels - video and photo, free key, the strongest vertical library."""

    name = "pexels"
    kinds = ("video", "image")
    key_env = ("PEXELS_API_KEY", "STOCK_API_KEY")
    attribution_required = False  # appreciated, not required
    VIDEO_URL = "https://api.pexels.com/videos/search"
    PHOTO_URL = "https://api.pexels.com/v1/search"
    LICENSE = "Pexels License"

    def search(self, query: str, kind: str, limit: int) -> list[Candidate]:
        headers = {"Authorization": self.api_key() or ""}
        params = {"query": query, "per_page": limit, "orientation": "portrait"}
        url = self.VIDEO_URL if kind == "video" else self.PHOTO_URL
        payload = _get_json(
            self.session, url, provider=self.name, params=params, headers=headers
        )
        self.searches += 1
        if kind == "video":
            return self._videos(payload)
        return self._photos(payload)

    @staticmethod
    def _pick_video_file(files: list[dict]) -> dict | None:
        """Smallest rendition that still clears the crop-aware floor.

        Pexels ships every clip at several sizes. Taking the largest wastes
        bandwidth; taking the smallest fails the floor. So: cheapest file that
        passes, and if none passes, the biggest one - which the filter will
        then reject with an honest resolution.
        """
        usable = [
            f for f in files
            if f.get("link") and int(f.get("width") or 0) and int(f.get("height") or 0)
        ]
        if not usable:
            return None
        passing = [
            f for f in usable
            if meets_resolution_floor(int(f["width"]), int(f["height"]))
        ]
        pool = passing or usable
        return min(pool, key=lambda f: int(f["width"]) * int(f["height"])) if passing \
            else max(pool, key=lambda f: int(f["width"]) * int(f["height"]))

    def _videos(self, payload: dict) -> list[Candidate]:
        out = []
        for item in payload.get("videos") or []:
            chosen = self._pick_video_file(item.get("video_files") or [])
            if not chosen:
                continue
            user = item.get("user") or {}
            out.append(Candidate(
                provider=self.name,
                provider_id=str(item.get("id")),
                kind="video",
                width=int(chosen["width"]),
                height=int(chosen["height"]),
                download_url=chosen["link"],
                source_url=item.get("url", ""),
                creator=user.get("name", ""),
                creator_url=user.get("url", ""),
                license=self.LICENSE,
                title=(item.get("alt") or "").strip(),
                duration=float(item.get("duration") or 0) or None,
                preview_url=item.get("image", ""),
            ))
        return out

    def _photos(self, payload: dict) -> list[Candidate]:
        out = []
        for item in payload.get("photos") or []:
            src = item.get("src") or {}
            link = src.get("original") or src.get("large2x") or src.get("large")
            if not link:
                continue
            out.append(Candidate(
                provider=self.name,
                provider_id=str(item.get("id")),
                kind="image",
                width=int(item.get("width") or 0),
                height=int(item.get("height") or 0),
                download_url=link,
                source_url=item.get("url", ""),
                creator=item.get("photographer", ""),
                creator_url=item.get("photographer_url", ""),
                license=self.LICENSE,
                title=(item.get("alt") or "").strip(),
                preview_url=src.get("medium", ""),
            ))
        return out


class PixabayProvider(Provider):
    """Pixabay - video and photo, free key."""

    name = "pixabay"
    kinds = ("video", "image")
    key_env = ("PIXABAY_API_KEY",)
    VIDEO_URL = "https://pixabay.com/api/videos/"
    PHOTO_URL = "https://pixabay.com/api/"
    LICENSE = "Pixabay Content License"

    #: Pixabay serves photos to unapproved keys at a longest side of 1280px,
    #: whatever the original measured. Scaling the reported dimensions down to
    #: that keeps the resolution floor honest about the file we would get.
    PHOTO_MAX_EDGE = 1280

    def search(self, query: str, kind: str, limit: int) -> list[Candidate]:
        params = {
            "key": self.api_key() or "",
            "q": query,
            "per_page": max(3, limit),   # the API rejects per_page < 3
            "safesearch": "true",
        }
        if kind == "video":
            payload = _get_json(
                self.session, self.VIDEO_URL, provider=self.name, params=params
            )
            self.searches += 1
            return self._videos(payload)

        params.update({"image_type": "photo", "orientation": "vertical"})
        payload = _get_json(
            self.session, self.PHOTO_URL, provider=self.name, params=params
        )
        self.searches += 1
        return self._photos(payload)

    def _videos(self, payload: dict) -> list[Candidate]:
        out = []
        for item in payload.get("hits") or []:
            streams = item.get("videos") or {}
            best = None
            for quality in ("large", "medium", "small", "tiny"):
                stream = streams.get(quality) or {}
                width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
                if stream.get("url") and width and height:
                    if best is None or width * height > best[1] * best[2]:
                        best = (stream["url"], width, height)
            if not best:
                continue
            url, width, height = best
            out.append(Candidate(
                provider=self.name,
                provider_id=str(item.get("id")),
                kind="video",
                width=width,
                height=height,
                download_url=url,
                source_url=item.get("pageURL", ""),
                creator=item.get("user", ""),
                license=self.LICENSE,
                title=(item.get("tags") or "").strip(),
                duration=float(item.get("duration") or 0) or None,
            ))
        return out

    def _photos(self, payload: dict) -> list[Candidate]:
        out = []
        for item in payload.get("hits") or []:
            link = item.get("largeImageURL")
            width = int(item.get("imageWidth") or 0)
            height = int(item.get("imageHeight") or 0)
            if not link or not width or not height:
                continue
            longest = max(width, height)
            if longest > self.PHOTO_MAX_EDGE:
                scale = self.PHOTO_MAX_EDGE / longest
                width, height = int(width * scale), int(height * scale)
            out.append(Candidate(
                provider=self.name,
                provider_id=str(item.get("id")),
                kind="image",
                width=width,
                height=height,
                download_url=link,
                source_url=item.get("pageURL", ""),
                creator=item.get("user", ""),
                license=self.LICENSE,
                title=(item.get("tags") or "").strip(),
                preview_url=item.get("previewURL", ""),
            ))
        return out


class OpenverseProvider(Provider):
    """Openverse - CC-licensed photos, no key at all (200 requests/day anon).

    The keyless safety net: a fresh clone with an empty .env still sources
    visuals. Quality is well below the stock providers and attribution is a
    licence condition, so it sits last in the default provider order.
    """

    name = "openverse"
    kinds = ("image",)
    key_env = ()
    attribution_required = True
    SEARCH_URL = "https://api.openverse.org/v1/images/"

    def search(self, query: str, kind: str, limit: int) -> list[Candidate]:
        if kind != "image":
            return []
        # No aspect_ratio filter on purpose: asking Openverse for "tall" throws
        # away the large landscape photographs that survive a 9:16 crop
        # comfortably, and backfills with tall-but-tiny scans that do not. The
        # crop-aware floor is the better judge of shape.
        params = {
            "q": query,
            "page_size": min(limit, 20),
            "size": "large",
            "license_type": "commercial,modification",
            "mature": "false",
        }
        payload = _get_json(
            self.session, self.SEARCH_URL, provider=self.name, params=params
        )
        self.searches += 1

        out = []
        for item in payload.get("results") or []:
            link = item.get("url")
            if not link:
                continue
            licence = item.get("license", "")
            version = item.get("license_version") or ""
            out.append(Candidate(
                provider=self.name,
                provider_id=str(item.get("id")),
                kind="image",
                width=int(item.get("width") or 0),
                height=int(item.get("height") or 0),
                download_url=link,
                source_url=item.get("foreign_landing_url", ""),
                creator=item.get("creator", ""),
                creator_url=item.get("creator_url", ""),
                license=f"CC {licence.upper()} {version}".strip(),
                title=(item.get("title") or "").strip(),
                preview_url=item.get("thumbnail", ""),
            ))
        return out


#: Every provider this module knows how to talk to, by config name.
_PROVIDERS: dict[str, type[Provider]] = {
    PexelsProvider.name: PexelsProvider,
    PixabayProvider.name: PixabayProvider,
    OpenverseProvider.name: OpenverseProvider,
}


def provider_status() -> list[dict]:
    """What each configured provider needs, and whether it has it."""
    status = []
    for name in config.VISUAL_PROVIDERS:
        cls = _PROVIDERS.get(name)
        if cls is None:
            status.append({"name": name, "known": False, "ready": False,
                           "detail": "unknown provider name"})
            continue
        ready = cls.is_configured()
        status.append({
            "name": name,
            "known": True,
            "ready": ready,
            "kinds": list(cls.kinds),
            "detail": "ready" if ready else cls.missing_key_message(),
        })
    return status


# ---------------------------------------------------------------------------
# Filtering and ranking
# ---------------------------------------------------------------------------

def crop_to_aspect(width: int, height: int, aspect: float = TARGET_ASPECT) -> tuple[int, int]:
    """Pixels left after a centre crop of `width` x `height` to `aspect`."""
    if width <= 0 or height <= 0:
        return (0, 0)
    if width / height > aspect:            # too wide - crop the sides
        return (int(round(height * aspect)), height)
    return (width, int(round(width / aspect)))   # too tall - crop top and bottom


def meets_resolution_floor(
    width: int, height: int, *, min_width: int | None = None, min_height: int | None = None
) -> bool:
    """True if the 9:16 crop of this source still clears the floor.

    This is the whole point of the check: a 1920x1080 clip is "1080p" but only
    608px wide once cropped to vertical, so it would be upscaled by 78% in the
    final frame. A 3840x2160 clip leaves 1215x2160 and passes comfortably.
    """
    crop_w, crop_h = crop_to_aspect(width, height)
    floor_w = config.VISUAL_MIN_WIDTH if min_width is None else min_width
    floor_h = config.VISUAL_MIN_HEIGHT if min_height is None else min_height
    return crop_w >= floor_w and crop_h >= floor_h


def _orientation(candidate: Candidate) -> str:
    if candidate.height > candidate.width:
        return "portrait"
    if candidate.height == candidate.width:
        return "square"
    return "landscape"


def reject_reason(candidate: Candidate, settings: dict) -> str:
    """Why this candidate is unusable, or "" if it is fine."""
    if not candidate.download_url:
        return "no download url"
    if candidate.kind not in settings["kinds"]:
        return f"{candidate.kind} not enabled"
    if not candidate.width or not candidate.height:
        return "provider reported no dimensions"
    if _orientation(candidate) == "landscape" and not settings["allow_landscape"]:
        return "landscape source (VISUAL_ALLOW_LANDSCAPE is off)"
    if not meets_resolution_floor(
        candidate.width, candidate.height,
        min_width=settings["min_width"], min_height=settings["min_height"],
    ):
        crop_w, crop_h = crop_to_aspect(candidate.width, candidate.height)
        return (
            f"{candidate.width}x{candidate.height} crops to {crop_w}x{crop_h}, "
            f"under the {settings['min_width']}x{settings['min_height']} floor"
        )
    return ""


def score_candidate(
    candidate: Candidate, beat: Beat, *, provider_rank: int, recently_used: bool
) -> float:
    """Rank a usable candidate. Higher is better.

    Weighted for a 9:16 reel: real vertical video that covers the whole beat,
    at comfortable resolution, from a preferred provider, that has not been in
    a recent video.
    """
    crop_w, crop_h = crop_to_aspect(candidate.width, candidate.height)
    score = 0.0

    orientation = _orientation(candidate)
    score += {"portrait": 3.0, "square": 1.0, "landscape": 0.0}[orientation]

    # Resolution headroom, capped so an 8K plate does not outrank a good match.
    score += min(crop_h / max(1, config.VISUAL_MIN_HEIGHT), 2.0)

    if candidate.kind == "video":
        score += 2.0
        if candidate.duration:
            if candidate.duration >= beat.duration:
                score += 1.5
            else:
                score += 1.5 * (candidate.duration / max(beat.duration, 0.1))
            if candidate.duration > 30:
                score -= 0.5      # long clips are usually montages, not shots

    score += max(0.0, 1.0 - 0.35 * provider_rank)
    if recently_used:
        score -= 4.0
    return round(score, 4)


# ---------------------------------------------------------------------------
# Download cache and cross-run usage ledger
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: dict) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically, so a crash cannot corrupt the index or ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


class AssetCache:
    """Content-addressed store of everything ever downloaded.

    Keyed by `provider:id` so a repeat search skips the network, and indexed by
    sha256 as well so the same file offered by two providers is stored once and
    recognised as a duplicate.
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or config.VISUALS_CACHE_DIR)
        self.index_path = self.root / "index.json"
        self._data = _read_json(self.index_path, {"assets": {}, "hashes": {}})
        self._data.setdefault("assets", {})
        self._data.setdefault("hashes", {})

    def get(self, key: str) -> dict | None:
        """Cached entry for `key`, or None if absent or the file has vanished."""
        entry = self._data["assets"].get(key)
        if not entry:
            return None
        path = self.root / entry["file"]
        if not path.is_file():
            self._data["assets"].pop(key, None)
            return None
        return {**entry, "path": path}

    def key_for_hash(self, digest: str) -> str | None:
        return self._data["hashes"].get(digest)

    def path_for(self, candidate: Candidate, url: str) -> Path:
        """Where a download for this candidate belongs inside the cache."""
        stem = _safe_stem(candidate.provider_id) or sha1(url.encode()).hexdigest()[:10]
        suffix = _extension_for(url) or ".bin"
        variant = sha1(url.encode()).hexdigest()[:8]
        return self.root / candidate.provider / f"{stem}_{variant}{suffix}"

    def put(self, key: str, path: Path, meta: dict) -> dict:
        entry = {**meta, "file": str(path.relative_to(self.root)).replace(os.sep, "/")}
        self._data["assets"][key] = entry
        if meta.get("sha256"):
            self._data["hashes"].setdefault(meta["sha256"], key)
        return {**entry, "path": path}

    def save(self) -> None:
        _write_json(self.index_path, self._data)


class UsageLedger:
    """Remembers which stock assets recent runs used.

    Without this, the highest-ranked clip for "ocean waves" opens every ocean
    video ever generated. Assets used within the last N runs are pushed down the
    ranking, and only chosen if nothing fresh is available.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.VISUALS_USAGE_FILE)
        self._data = _read_json(self.path, {"run_counter": 0, "runs": {}, "assets": {}})
        for field_name, default in (("run_counter", 0), ("runs", {}), ("assets", {})):
            self._data.setdefault(field_name, default)
        #: Numbers handed out for runs not yet written to disk. Without this,
        #: every recently_used() call would burn a fresh number and the cooldown
        #: would expire itself part-way through a single run.
        self._pending: dict[str, int] = {}

    def run_index(self, run_id: str) -> int:
        """Sequence number for `run_id`, allocated once and reused on re-runs."""
        existing = self._data["runs"].get(run_id)
        if existing:
            return int(existing["index"])
        if run_id not in self._pending:
            self._pending[run_id] = int(self._data["run_counter"]) + 1
        return self._pending[run_id]

    def recently_used(self, key: str, *, cooldown: int, run_id: str) -> bool:
        if cooldown <= 0:
            return False
        entry = self._data["assets"].get(key)
        if not entry:
            return False
        if run_id in (entry.get("runs") or []):
            return False        # this run's own earlier pick - not a repeat
        age = self.run_index(run_id) - int(entry.get("last_run_index", 0))
        return age <= cooldown

    def record(self, run_id: str, keys: list[str]) -> None:
        index = self.run_index(run_id)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._data["run_counter"] = max(int(self._data["run_counter"]), index)
        self._data["runs"][run_id] = {"index": index, "at": stamp, "assets": keys}
        for key in keys:
            entry = self._data["assets"].setdefault(key, {"uses": 0, "runs": []})
            if run_id not in entry["runs"]:
                entry["runs"].append(run_id)
                entry["uses"] = int(entry.get("uses", 0)) + 1
            entry["last_run_index"] = index
            entry["last_used"] = stamp
        _write_json(self.path, self._data)


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _extension_for(url: str, content_type: str = "") -> str:
    """Pick a file extension from the URL, falling back to the content type."""
    path = url.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(path).suffix.lower()
    if suffix and 2 <= len(suffix) <= 5:
        return suffix
    return _CONTENT_TYPE_EXTENSIONS.get(content_type.split(";", 1)[0].strip().lower(), "")


def _safe_stem(text: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned[:limit]


def download(session, url: str, destination: Path, *, provider: str) -> dict:
    """Stream `url` to `destination` atomically.

    Returns {"bytes", "sha256", "content_type"}. Raises ProviderError - never
    fatal - for dead links, HTML error pages served as media, and oversized
    files, so the caller can simply move on to the next candidate.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    try:
        response = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    except requests.exceptions.RequestException as exc:
        raise ProviderError(
            f"{provider} download failed: {type(exc).__name__}: {exc}", provider=provider
        ) from exc

    with response:
        if response.status_code == 429:
            # Providers hand back URLs on hosts with their own limits (Openverse
            # points at Wikimedia, for one). Marching through the next twenty
            # candidates only deepens the hole.
            raise ProviderRateLimitError(
                f"{provider} rate limited the download: {url}", provider=provider
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"{provider} download URL returned {response.status_code}: {url}",
                provider=provider,
            )

        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith(_ALLOWED_CONTENT_PREFIXES):
            raise ProviderError(
                f"{provider} served {content_type!r} instead of media: {url}",
                provider=provider,
            )

        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_DOWNLOAD_BYTES:
            raise ProviderError(
                f"{provider} file is {declared / 1e6:.0f}MB, over the "
                f"{MAX_DOWNLOAD_BYTES / 1e6:.0f}MB cap",
                provider=provider,
            )

        digest = sha256()
        written = 0
        try:
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise ProviderError(
                            f"{provider} file exceeded the "
                            f"{MAX_DOWNLOAD_BYTES / 1e6:.0f}MB cap mid-download",
                            provider=provider,
                        )
                    handle.write(chunk)
                    digest.update(chunk)
            if written == 0:
                raise ProviderError(
                    f"{provider} download URL returned an empty body: {url}",
                    provider=provider,
                )
            os.replace(partial, destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    return {
        "bytes": written,
        "sha256": digest.hexdigest(),
        "content_type": content_type.split(";", 1)[0].strip(),
    }


def probe_image_size(path: Path) -> tuple[int, int] | None:
    """Real pixel size of a downloaded image, if Pillow is installed.

    Providers occasionally report the original's dimensions while serving a
    smaller rendition; measuring the file catches that instead of handing step 5
    something that has to be upscaled.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - a corrupt file is just unusable
        return None


def _link_into_run(source: Path, destination: Path) -> str:
    """Put a cached file into the run directory without copying bytes if possible."""
    if destination.exists():
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)   # different volume, or no link support
        return "copy"


# ---------------------------------------------------------------------------
# Run identity and inputs
# ---------------------------------------------------------------------------

def _slugify(text: str, limit: int = 40) -> str:
    normalised = unicodedata.normalize("NFKD", text or "")
    ascii_only = normalised.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug[:limit].strip("-") or "reel"


def script_fingerprint(script: str) -> str:
    """Stable hash of the narration, used to key the per-run directory."""
    return sha1(" ".join(script.split()).encode("utf-8")).hexdigest()[:12]


def make_run_id(topic: str, script: str) -> str:
    """Directory name for this script: readable prefix, content hash suffix.

    The hash means the same script always lands in the same directory, which is
    what makes a re-run free.
    """
    label = _slugify(topic or " ".join(script.split()[:6]))
    return f"{label}-{script_fingerprint(script)[:8]}"


def _resolve_script(script) -> tuple[str, str]:
    """Accept step 2's dict or a bare string. Returns (text, topic)."""
    if isinstance(script, dict):
        text = str(script.get("script") or "")
        topic = str(script.get("topic") or "")
    else:
        text, topic = str(script or ""), ""
    text = text.strip()
    if not text:
        raise ValueError("script is empty - nothing to source visuals for")
    return text, topic.strip()


def _resolve_duration(audio, duration_seconds, script_text: str) -> tuple[float, str]:
    """Work out how long the narration runs, and say where the number came from.

    Accepts step 3's dict, a path to the .mp3, a number, or nothing at all. A
    measured duration is always preferred: the beat timings handed to step 5 are
    only as good as this number.
    """
    if isinstance(audio, dict):
        if audio.get("duration_seconds"):
            return float(audio["duration_seconds"]), "measured (step 3 result)"
        audio = audio.get("audio_path")

    if isinstance(audio, (int, float)) and not isinstance(audio, bool):
        return float(audio), "caller-supplied"

    if audio:
        path = Path(audio)
        if not path.is_file():
            raise ValueError(f"audio file not found: {path}")
        from pipeline import tts   # step 3 already knows how to measure an mp3

        return tts.read_duration_seconds(path), f"measured from {path.name}"

    if duration_seconds:
        return float(duration_seconds), "caller-supplied"

    words = len(script_text.split())
    try:
        from pipeline import script_gen

        return float(script_gen.estimate_duration(script_text)), "estimated from word count"
    except Exception:  # noqa: BLE001 - fall back to the same 150wpm assumption
        return max(1.0, words * 60 / 150), "estimated from word count"


def _settings_fingerprint(settings: dict) -> str:
    """Hash of the knobs that would change what gets sourced."""
    relevant = {
        k: settings[k] for k in (
            "providers", "assets_per_beat", "headroom", "min_width", "min_height",
            "kinds", "allow_landscape", "query_mode", "image_duration",
        )
    }
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return sha1(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sourcing
# ---------------------------------------------------------------------------

def _init_providers(names, session, warnings: list[str]) -> list[Provider]:
    """Instantiate the configured providers that actually have their keys."""
    providers: list[Provider] = []
    for name in names:
        cls = _PROVIDERS.get(name)
        if cls is None:
            warnings.append(
                f"unknown provider {name!r} in VISUAL_PROVIDERS - known: "
                + ", ".join(sorted(_PROVIDERS))
            )
            continue
        if not cls.is_configured():
            warnings.append(f"skipping {name} - {cls.missing_key_message()}")
            continue
        providers.append(cls(session))
    return providers


def _search_provider(
    provider: Provider, query: str, kind: str, warnings: list[str]
) -> list[Candidate]:
    """One search, with a single backoff on a rate limit. Never raises."""
    for attempt in (0, 1):
        try:
            return provider.search(query, kind, SEARCH_PAGE_SIZE)
        except ProviderRateLimitError as exc:
            if attempt == 0:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            provider.disabled_reason = str(exc)
            warnings.append(f"{provider.name} benched for this run: {exc}")
        except ProviderError as exc:
            if exc.fatal:
                provider.disabled_reason = str(exc)
                warnings.append(f"{provider.name} benched for this run: {exc}")
            else:
                warnings.append(f"{provider.name} search {query!r} failed: {exc}")
        return []
    return []


def _gather_candidates(
    providers: list[Provider], beat: Beat, settings: dict, warnings: list[str]
) -> list[tuple[Candidate, int, str]]:
    """Search every provider for a beat. Returns (candidate, provider_rank, query)."""
    pool: list[tuple[Candidate, int, str]] = []
    seen: set[str] = set()
    usable = 0

    # Judge sufficiency on candidates that would actually survive the filter.
    # Counting raw hits instead would never trip: one provider returns a page of
    # 15, and half of those can be landscape 1080p that the 9:16 floor rejects.
    needed = min(settings["assets_per_beat"] * CANDIDATE_MARGIN, CANDIDATE_POOL_SIZE)

    for query in (beat.query, *beat.alt_queries):
        if not query:
            continue
        for rank, provider in enumerate(providers):
            if provider.disabled_reason:
                continue
            for kind in settings["kinds"]:
                if kind not in provider.kinds:
                    continue
                for candidate in _search_provider(provider, query, kind, warnings):
                    if candidate.key in seen:
                        continue
                    seen.add(candidate.key)
                    pool.append((candidate, rank, query))
                    if not reject_reason(candidate, settings):
                        usable += 1
                if usable >= needed:
                    break      # no need to also search stills for this beat
            if usable >= needed:
                # Providers are in preference order, so a good haul from an
                # earlier one means the fallbacks are not needed - and their
                # quota (Openverse allows 200 requests a day) is not spent.
                break
        if usable >= needed or len(pool) >= CANDIDATE_POOL_SIZE:
            break     # the first query that satisfies the beat is the best match
    return pool


def _rank_pool(
    pool: list[tuple[Candidate, int, str]],
    beat: Beat,
    settings: dict,
    ledger: UsageLedger,
    run_id: str,
    state: "_RunState",
    rejections: dict[str, int],
) -> list[tuple[float, Candidate, str]]:
    """Filter a beat's pool and sort what survives, best first."""
    ranked: list[tuple[float, Candidate, str]] = []
    for candidate, rank, query in pool:
        if candidate.key in state.taken_keys:
            rejections["already used in this run"] = (
                rejections.get("already used in this run", 0) + 1
            )
            continue
        if candidate.key in state.failed_keys:
            continue
        reason = reject_reason(candidate, settings)
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        stale = ledger.recently_used(
            candidate.key, cooldown=settings["reuse_cooldown"], run_id=run_id
        )
        score = score_candidate(
            candidate, beat, provider_rank=rank, recently_used=stale
        )
        ranked.append((score, candidate, query))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


@dataclass
class _RunState:
    """What this run has already tried, so no later pass repeats the work.

    The headroom top-up walks each beat's leftover candidates again. Without
    `failed_keys` it re-downloads - and re-fails on - everything that did not
    work the first time, which on a live run meant tens of megabytes of wasted
    traffic and the same warning six times over.
    """

    taken_keys: set[str] = dataclass_field(default_factory=set)
    taken_hashes: set[str] = dataclass_field(default_factory=set)
    failed_keys: set[str] = dataclass_field(default_factory=set)
    #: Hosts whose downloads started returning 429, keyed by hostname rather
    #: than by provider: Openverse alone hands back files from Wikimedia, Flickr
    #: and a dozen museums, and one throttling host must not cost us the others.
    rate_limited: set[str] = dataclass_field(default_factory=set)
    ordinals: dict[int, int] = dataclass_field(default_factory=dict)

    def next_ordinal(self, beat_index: int) -> int:
        """Per-beat counter for filenames, continuous across top-up passes."""
        value = self.ordinals.get(beat_index, 0)
        self.ordinals[beat_index] = value + 1
        return value


def _download_host(url: str) -> str:
    """Hostname a download URL points at, for per-host rate-limit tracking."""
    return urlsplit(url).netloc.lower()


def _download_with_backoff(session, candidate: Candidate, target: Path) -> dict:
    """One download, retried once after a rate limit before giving up."""
    for attempt in (0, 1):
        try:
            return download(
                session, candidate.download_url, target, provider=candidate.provider
            )
        except ProviderRateLimitError:
            if attempt == 0:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise
    raise AssertionError("unreachable")


def _acquire(
    ranked: list[tuple[float, Candidate, str]],
    beat: Beat,
    *,
    want: int,
    session,
    cache: AssetCache,
    run_dir: Path,
    settings: dict,
    state: _RunState,
    warnings: list[str],
) -> list[dict]:
    """Download up to `want` assets for a beat, skipping anything that fails.

    Candidates that 404, serve HTML, or turn out to be duplicates are stepped
    over and remembered as failures, so a later pass does not walk into them
    again. The beat only fails if every candidate does.
    """
    acquired: list[dict] = []
    attempts = 0

    for score, candidate, query in ranked:
        if len(acquired) >= want:
            break
        if candidate.key in state.taken_keys or candidate.key in state.failed_keys:
            continue
        if _download_host(candidate.download_url) in state.rate_limited:
            continue
        if attempts >= MAX_DOWNLOAD_ATTEMPTS_PER_BEAT:
            warnings.append(
                f"beat {beat.index}: gave up after {attempts} failed downloads"
            )
            break

        cached = cache.get(candidate.key)
        if cached:
            digest = cached.get("sha256", "")
            if digest and digest in state.taken_hashes:
                continue
            source_path, meta, was_cached = cached["path"], cached, True
        else:
            attempts += 1
            target = cache.path_for(candidate, candidate.download_url)
            try:
                result = _download_with_backoff(session, candidate, target)
            except ProviderRateLimitError as exc:
                host = _download_host(candidate.download_url)
                state.rate_limited.add(host)
                warnings.append(
                    f"{host} is rate limiting downloads - stopped fetching from "
                    f"it for the rest of this run ({exc})"
                )
                continue
            except ProviderError as exc:
                state.failed_keys.add(candidate.key)
                warnings.append(f"beat {beat.index}: {exc}")
                continue

            if result["sha256"] in state.taken_hashes:
                state.failed_keys.add(candidate.key)
                target.unlink(missing_ok=True)
                continue

            existing_key = cache.key_for_hash(result["sha256"])
            if existing_key and existing_key != candidate.key:
                warnings.append(
                    f"beat {beat.index}: {candidate.key} is byte-identical to "
                    f"{existing_key}, skipped"
                )
                state.failed_keys.add(candidate.key)
                target.unlink(missing_ok=True)
                continue

            width, height = candidate.width, candidate.height
            if candidate.kind == "image":
                measured = probe_image_size(target)
                if measured is None:
                    pass                      # Pillow absent: trust the provider
                elif not meets_resolution_floor(
                    *measured,
                    min_width=settings["min_width"], min_height=settings["min_height"],
                ):
                    warnings.append(
                        f"beat {beat.index}: {candidate.key} downloaded at "
                        f"{measured[0]}x{measured[1]}, under the floor - discarded"
                    )
                    state.failed_keys.add(candidate.key)
                    target.unlink(missing_ok=True)
                    continue
                else:
                    width, height = measured

            meta = cache.put(candidate.key, target, {
                "provider": candidate.provider,
                "kind": candidate.kind,
                "width": width,
                "height": height,
                "duration": candidate.duration,
                "sha256": result["sha256"],
                "bytes": result["bytes"],
                "content_type": result["content_type"],
                "source_url": candidate.source_url,
                "download_url": candidate.download_url,
                "creator": candidate.creator,
                "license": candidate.license,
                "attribution": candidate.attribution,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            source_path, was_cached = target, False

        run_name = (
            f"beat{beat.index:02d}_{state.next_ordinal(beat.index)}_"
            f"{candidate.provider}_{_safe_stem(candidate.provider_id, 24)}"
            f"{source_path.suffix}"
        )
        run_path = run_dir / run_name
        try:
            link_mode = _link_into_run(source_path, run_path)
        except OSError as exc:
            warnings.append(f"beat {beat.index}: could not place {run_name}: {exc}")
            state.failed_keys.add(candidate.key)
            continue

        state.taken_keys.add(candidate.key)
        if meta.get("sha256"):
            state.taken_hashes.add(meta["sha256"])

        acquired.append({
            "beat_index": beat.index,
            "kind": candidate.kind,
            "path": str(run_path.resolve()),
            "relative_path": run_name,
            "width": int(meta.get("width") or candidate.width),
            "height": int(meta.get("height") or candidate.height),
            "duration": meta.get("duration") if candidate.kind == "video" else None,
            "provider": candidate.provider,
            "provider_id": candidate.provider_id,
            "provider_key": candidate.key,
            "query": query,
            "score": score,
            "source_url": candidate.source_url,
            "download_url": candidate.download_url,
            "creator": candidate.creator,
            "creator_url": candidate.creator_url,
            "license": candidate.license,
            "attribution": candidate.attribution,
            "attribution_required": _PROVIDERS[candidate.provider].attribution_required,
            "sha256": meta.get("sha256", ""),
            "bytes": meta.get("bytes", 0),
            "from_cache": was_cached,
            "placement": link_mode,
        })

    return acquired


def _usable_seconds(asset: dict, image_duration: float) -> float:
    """How much screen time this asset can honestly contribute."""
    if asset["kind"] == "video" and asset.get("duration"):
        return min(float(asset["duration"]), COVERAGE_CLIP_CAP_SECONDS)
    if asset["kind"] == "video":
        return COVERAGE_CLIP_CAP_SECONDS / 2   # provider gave no duration
    return image_duration


def _apply_placement(assets: list[dict], beats: list[Beat], image_duration: float) -> None:
    """Suggest where each asset sits inside its beat, and label its role.

    Only a suggestion - step 5 owns pacing - but it means a naive assembler can
    lay the run out end to end without doing any arithmetic of its own.
    """
    by_beat: dict[int, list[dict]] = {}
    for asset in assets:
        by_beat.setdefault(asset["beat_index"], []).append(asset)

    for beat in beats:
        group = by_beat.get(beat.index, [])
        planned = [a for a in group if a["role"] != "spare"] or group
        share = beat.duration / max(1, len(planned))
        for asset in group:
            if asset not in planned:
                # Spares exist to give step 5 slack, not to claim a slot.
                asset["suggested_start"] = None
                asset["suggested_duration"] = None
        for position, asset in enumerate(planned):
            asset["suggested_start"] = round(beat.start + position * share, 3)
            asset["suggested_duration"] = round(
                min(share, _usable_seconds(asset, image_duration)), 3
            )


def source_visuals(
    script,
    audio=None,
    *,
    topic: str | None = None,
    duration_seconds: float | None = None,
    run_id: str | None = None,
    providers=None,
    assets_per_beat: int | None = None,
    headroom: float | None = None,
    query_mode: str | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    reuse_cooldown: int | None = None,
    refresh: bool = False,
    dry_run: bool = False,
    runs_dir: Path | str | None = None,
) -> dict:
    """Source the visual assets for one script.

    Args:
        script: step 2's result dict, or the narration text.
        audio: step 3's result dict, a path to the .mp3, or a duration in
            seconds. Anything that yields a measured length is preferred - the
            beat timings are scaled to it.
        topic: overrides the topic carried in step 2's dict.
        duration_seconds: used only when `audio` gives nothing.
        run_id: directory name for this run. Defaults to a slug of the topic
            plus a hash of the script, so re-runs land in the same place.
        providers, assets_per_beat, headroom, query_mode, min_width,
            min_height, reuse_cooldown: per-call overrides of the config
            defaults. Left alone, everything comes from config.py.
        refresh: ignore an existing manifest and source again.
        dry_run: plan the beats and queries, download nothing.
        runs_dir: parent of the per-run directory. Defaults to config.

    Returns:
        A manifest dict - see MANIFEST_VERSION notes in the module docstring.
        The keys step 5 needs are `asset_dir`, `beats` and `assets`; every asset
        carries an absolute `path`, a `relative_path`, its real dimensions, its
        beat, and its attribution.

    Raises:
        ValueError: the script is empty, or the duration is unusable.
        NoAssetsError: every beat failed - nothing was downloaded at all.
    """
    script_text, script_topic = _resolve_script(script)
    topic = (topic or script_topic or "").strip()
    narration_duration, duration_source = _resolve_duration(
        audio, duration_seconds, script_text
    )
    if narration_duration <= 0:
        raise ValueError(f"narration duration must be positive, got {narration_duration}")

    kinds = ("video", "image") if config.VISUAL_ALLOW_IMAGES else ("video",)
    settings = {
        "providers": list(providers or config.VISUAL_PROVIDERS),
        "assets_per_beat": int(
            assets_per_beat if assets_per_beat is not None
            else config.VISUAL_ASSETS_PER_BEAT
        ),
        "headroom": float(
            headroom if headroom is not None else config.VISUAL_DURATION_HEADROOM
        ),
        "min_width": int(min_width if min_width is not None else config.VISUAL_MIN_WIDTH),
        "min_height": int(
            min_height if min_height is not None else config.VISUAL_MIN_HEIGHT
        ),
        "reuse_cooldown": int(
            reuse_cooldown if reuse_cooldown is not None
            else config.VISUAL_REUSE_COOLDOWN_RUNS
        ),
        "query_mode": (query_mode or config.VISUAL_QUERY_MODE).strip().lower(),
        "kinds": list(kinds),
        "allow_landscape": bool(config.VISUAL_ALLOW_LANDSCAPE),
        "image_duration": float(config.VISUAL_IMAGE_DURATION),
    }
    if settings["assets_per_beat"] < 1:
        raise ValueError("assets_per_beat must be at least 1")
    if settings["headroom"] <= 0:
        raise ValueError("headroom must be positive")

    run_id = run_id or make_run_id(topic, script_text)
    run_dir = Path(runs_dir or config.VISUALS_RUNS_DIR) / run_id
    fingerprint = _settings_fingerprint(settings)
    script_hash = script_fingerprint(script_text)

    if not refresh and not dry_run:
        reused = _load_existing_run(run_dir, script_hash, fingerprint)
        if reused:
            return reused

    warnings: list[str] = []
    beats = segment_script(script_text, narration_duration)
    warnings.extend(build_queries(beats, topic, settings["query_mode"]))

    target_duration = round(narration_duration * settings["headroom"], 3)
    base = {
        "manifest_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": topic,
        "script": script_text,
        "script_hash": script_hash,
        "settings_hash": fingerprint,
        "settings": settings,
        "narration_duration": round(narration_duration, 3),
        "duration_source": duration_source,
        "target_duration": target_duration,
        "frame": {
            "width": config.VIDEO_WIDTH,
            "height": config.VIDEO_HEIGHT,
            "fps": config.VIDEO_FPS,
        },
    }

    if dry_run:
        return {
            **base,
            "asset_dir": str(run_dir),
            "manifest_path": None,
            "beats": [{**b.to_dict(), "asset_indexes": [], "status": "planned"}
                      for b in beats],
            "assets": [],
            "attributions": [],
            "sourced_duration": 0.0,
            "coverage_ratio": 0.0,
            "warnings": warnings,
            "stats": {"dry_run": True, "beats": len(beats)},
            "reused_run": False,
        }

    session = _new_session()
    active = _init_providers(settings["providers"], session, warnings)
    if not active:
        raise NoAssetsError(
            "no usable providers. " + " ".join(warnings)
            + " openverse needs no key - check VISUAL_PROVIDERS in .env."
        )

    cache = AssetCache()
    ledger = UsageLedger()
    run_dir.mkdir(parents=True, exist_ok=True)

    state = _RunState()
    rejections: dict[str, int] = {}
    assets: list[dict] = []
    beat_records: list[dict] = []
    leftovers: dict[int, list[tuple[float, Candidate, str]]] = {}

    for beat in beats:
        record = {**beat.to_dict(), "asset_indexes": [], "status": "failed", "notes": []}
        try:
            pool = _gather_candidates(active, beat, settings, warnings)
            ranked = _rank_pool(
                pool, beat, settings, ledger, run_id, state, rejections
            )
            if not ranked:
                record["notes"].append(
                    f"no usable candidates from {len(pool)} search hits"
                )
                warnings.append(
                    f"beat {beat.index} ({beat.query!r}): no usable candidates "
                    f"from {len(pool)} hits"
                )
                beat_records.append(record)
                continue

            acquired = _acquire(
                ranked, beat,
                want=settings["assets_per_beat"], session=session, cache=cache,
                run_dir=run_dir, settings=settings, state=state, warnings=warnings,
            )
            leftovers[beat.index] = [
                row for row in ranked
                if row[1].key not in state.taken_keys
                and row[1].key not in state.failed_keys
            ]
        except Exception as exc:  # noqa: BLE001 - one beat must not kill the run
            warnings.append(
                f"beat {beat.index} failed: {type(exc).__name__}: {exc}"
            )
            beat_records.append(record)
            continue

        if not acquired:
            record["notes"].append("every candidate failed to download")
            warnings.append(f"beat {beat.index}: every candidate failed to download")
            beat_records.append(record)
            continue

        for position, asset in enumerate(acquired):
            asset["role"] = "primary" if position == 0 else "alternate"
            asset["index"] = len(assets)
            record["asset_indexes"].append(asset["index"])
            assets.append(asset)

        record["status"] = (
            "ok" if len(acquired) >= settings["assets_per_beat"] else "partial"
        )
        beat_records.append(record)

    cache.save()

    if not assets:
        # Carry the diagnosis into the exception: this is the one exit where the
        # manifest is never written, so the warnings would otherwise be lost and
        # the caller left with "it did not work".
        detail = list(dict.fromkeys(warnings))[:8]
        raise NoAssetsError(
            "no visuals could be sourced for any beat. "
            + (" | ".join(detail) if detail else "no warnings were recorded.")
        )

    # --- headroom top-up ---------------------------------------------------
    # Step 5 needs slack to pace cuts, so keep pulling the next-best leftover
    # candidate - spreading across beats - until the target is met.
    sourced = sum(_usable_seconds(a, settings["image_duration"]) for a in assets)
    if sourced < target_duration:
        for _ in range(MAX_TOTAL_ASSETS):
            if sourced >= target_duration or len(assets) >= MAX_TOTAL_ASSETS:
                break
            beat_order = sorted(
                beats,
                key=lambda b: (
                    len([a for a in assets if a["beat_index"] == b.index]), b.index
                ),
            )
            progressed = False
            for beat in beat_order:
                remaining = [
                    row for row in leftovers.get(beat.index, [])
                    if row[1].key not in state.taken_keys
                    and row[1].key not in state.failed_keys
                ]
                if not remaining:
                    leftovers[beat.index] = []
                    continue
                extra = _acquire(
                    remaining, beat,
                    want=1, session=session, cache=cache, run_dir=run_dir,
                    settings=settings, state=state, warnings=warnings,
                )
                if not extra:
                    leftovers[beat.index] = []
                    continue
                for asset in extra:
                    asset["role"] = "spare"
                    asset["index"] = len(assets)
                    assets.append(asset)
                    sourced += _usable_seconds(asset, settings["image_duration"])
                    for record in beat_records:
                        if record["index"] == beat.index:
                            record["asset_indexes"].append(asset["index"])
                progressed = True
                break
            if not progressed:
                break
        cache.save()

    sourced = round(
        sum(_usable_seconds(a, settings["image_duration"]) for a in assets), 3
    )
    coverage = round(sourced / narration_duration, 3) if narration_duration else 0.0
    if sourced < target_duration:
        warnings.append(
            f"sourced {sourced:.1f}s against a {target_duration:.1f}s target "
            f"({coverage:.2f}x narration) - step 5 will have less slack than asked for"
        )

    _apply_placement(assets, beats, settings["image_duration"])
    ledger.record(run_id, [a["provider_key"] for a in assets])

    # The same problem can surface on several passes; report each one once.
    warnings = list(dict.fromkeys(warnings))

    attributions = [a["attribution"] for a in assets]
    manifest = {
        **base,
        "asset_dir": str(run_dir.resolve()),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "beats": beat_records,
        "assets": assets,
        "attributions": attributions,
        "sourced_duration": sourced,
        "coverage_ratio": coverage,
        "warnings": warnings,
        "stats": {
            "beats": len(beats),
            "beats_ok": sum(1 for b in beat_records if b["status"] == "ok"),
            "beats_partial": sum(1 for b in beat_records if b["status"] == "partial"),
            "beats_failed": sum(1 for b in beat_records if b["status"] == "failed"),
            "assets": len(assets),
            "from_cache": sum(1 for a in assets if a["from_cache"]),
            "downloaded_bytes": sum(
                a["bytes"] for a in assets if not a["from_cache"]
            ),
            "by_provider": _count_by(assets, "provider"),
            "by_kind": _count_by(assets, "kind"),
            "searches": {p.name: p.searches for p in active},
            "benched_providers": {
                p.name: p.disabled_reason for p in active if p.disabled_reason
            },
            "rejected_candidates": rejections,
        },
        "reused_run": False,
    }

    _write_json(run_dir / "manifest.json", manifest)
    _write_attribution_file(run_dir, manifest)
    return manifest


def _count_by(assets: list[dict], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for asset in assets:
        counts[asset[field_name]] = counts.get(asset[field_name], 0) + 1
    return counts


def _write_attribution_file(run_dir: Path, manifest: dict) -> None:
    """Human-readable credits, so they survive outside the JSON."""
    lines = [
        f"Visual credits - {manifest['topic'] or manifest['run_id']}",
        f"Generated {manifest['created_at']}",
        "",
    ]
    for asset in manifest["assets"]:
        flag = " [attribution required]" if asset["attribution_required"] else ""
        lines.append(f"{asset['relative_path']}{flag}")
        lines.append(f"    {asset['attribution']}")
    (run_dir / "attribution.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_existing_run(run_dir: Path, script_hash: str, fingerprint: str) -> dict | None:
    """Reuse a previous run of the same script under the same settings.

    Every referenced file has to still be on disk; one missing asset and the run
    is re-sourced rather than handing step 5 a broken path.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path, {})
    if (
        manifest.get("script_hash") != script_hash
        or manifest.get("settings_hash") != fingerprint
        or not manifest.get("assets")
    ):
        return None
    for asset in manifest["assets"]:
        if not Path(asset["path"]).is_file():
            return None
    manifest["reused_run"] = True
    return manifest


def assets_for_beat(manifest: dict, beat_index: int) -> list[dict]:
    """Assets belonging to one beat, in the order they should be used."""
    return [a for a in manifest["assets"] if a["beat_index"] == beat_index]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(manifest: dict) -> None:
    rule = "-" * 78
    stats = manifest["stats"]
    print(rule)
    print(f"RUN     : {manifest['run_id']}"
          + ("  (reused, nothing re-downloaded)" if manifest["reused_run"] else ""))
    print(f"TOPIC   : {manifest['topic'] or '(none given)'}")
    print(f"AUDIO   : {manifest['narration_duration']}s "
          f"[{manifest['duration_source']}]")
    print(f"TARGET  : {manifest['target_duration']}s of footage "
          f"({manifest['settings']['headroom']}x narration)")
    if not stats.get("dry_run"):
        print(f"SOURCED : {manifest['sourced_duration']}s "
              f"({manifest['coverage_ratio']}x narration) across "
              f"{stats['assets']} assets")
        print(f"DIR     : {manifest['asset_dir']}")
    print(rule)

    for beat in manifest["beats"]:
        text = beat["text"]
        if len(text) > 62:
            text = text[:59] + "..."
        marker = {"ok": "+", "partial": "~", "failed": "!", "planned": "."}
        print(f"{marker.get(beat['status'], '?')} beat {beat['index']:>2} "
              f"[{beat['start']:>6.2f} - {beat['end']:>6.2f}]  {text}")
        print(f"    query ({beat['query_source']}): {beat['query']!r}"
              + (f"  alts: {beat['alt_queries']}" if beat["alt_queries"] else ""))
        for index in beat["asset_indexes"]:
            asset = manifest["assets"][index]
            length = (f"{asset['duration']:.0f}s" if asset.get("duration")
                      else asset["kind"])
            print(f"      - [{asset['role']:<9}] {asset['relative_path']}  "
                  f"{asset['width']}x{asset['height']} {length} "
                  f"via {asset['provider']}"
                  + ("  (cached)" if asset["from_cache"] else ""))
        for note in beat.get("notes", []):
            print(f"      ! {note}")

    print(rule)
    if not stats.get("dry_run"):
        print(f"beats   : {stats['beats_ok']} ok / {stats['beats_partial']} partial "
              f"/ {stats['beats_failed']} failed")
        print(f"assets  : {stats['by_provider']}  kinds={stats['by_kind']}  "
              f"cached={stats['from_cache']}")
        print(f"searches: {stats['searches']}  "
              f"downloaded={stats['downloaded_bytes'] / 1e6:.1f}MB")
        if stats["rejected_candidates"]:
            print("rejected:")
            for reason, count in sorted(
                stats["rejected_candidates"].items(), key=lambda kv: -kv[1]
            ):
                print(f"    {count:>3} x {reason}")
    if manifest["warnings"]:
        print("warnings:")
        for warning in manifest["warnings"]:
            print(f"    - {warning}")
    print(rule)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Source stock visuals for a narration script (step 4)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("script", nargs="?", help="the narration text")
    source.add_argument("-f", "--script-file", type=Path,
                        help="read the narration from a text file instead")
    source.add_argument("--providers-status", action="store_true",
                        help="show which providers are configured, then exit")

    parser.add_argument("-a", "--audio", type=Path, default=None,
                        help="narration .mp3 from step 3; its measured length "
                             "scales the beat timings")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="narration length in seconds, if there is no audio yet")
    parser.add_argument("-t", "--topic", default=None, help="topic, for query context")
    parser.add_argument("--assets-per-beat", type=int, default=None)
    parser.add_argument("--headroom", type=float, default=None,
                        help="multiple of the narration length to source")
    parser.add_argument("--query-mode", choices=("llm", "heuristic"), default=None)
    parser.add_argument("--provider", action="append", dest="providers", default=None,
                        help="override VISUAL_PROVIDERS (repeatable)")
    parser.add_argument("--run-id", default=None, help="per-run directory name")
    parser.add_argument("--refresh", action="store_true",
                        help="re-source even if this script already has a manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan beats and queries without downloading anything")
    parser.add_argument("--json", action="store_true",
                        help="print the manifest as JSON instead of a summary")
    args = parser.parse_args(argv)

    if args.providers_status:
        for entry in provider_status():
            mark = "ready" if entry["ready"] else "not ready"
            print(f"{entry['name']:<12} {mark:<10} {entry['detail']}")
        return 0

    if args.script_file:
        try:
            script = args.script_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: could not read {args.script_file}: {exc}", file=sys.stderr)
            return 1
        topic = args.topic or args.script_file.stem.replace("_", " ").replace("-", " ")
    else:
        script, topic = args.script, args.topic

    try:
        manifest = source_visuals(
            script,
            args.audio,
            topic=topic,
            duration_seconds=args.duration,
            run_id=args.run_id,
            providers=args.providers,
            assets_per_beat=args.assets_per_beat,
            headroom=args.headroom,
            query_mode=args.query_mode,
            refresh=args.refresh,
            dry_run=args.dry_run,
        )
    except (ValueError, config.MissingConfigError, VisualsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        _print_result(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
