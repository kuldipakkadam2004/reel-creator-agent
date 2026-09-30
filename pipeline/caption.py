"""Step 6 - the Instagram post text: caption copy plus hashtags.

Takes the narration script (step 2's, or the copy step 5 carried into the run
directory) and writes the words that go in Instagram's caption field:

    post.txt   the exact caption, ready to hand to the Graph API
    post.json  the same thing in parts - hook, body, CTA, tags, credits,
               character counts, and every warning raised on the way

Both land in output/<date>/<run id>/ next to reel.mp4, so step 7 picks up the
video and its caption together.

Nothing here touches the burned-in captions from step 5 - those are pixels
inside the video. This is the post copy underneath it.

What happens, in order:

  1. the script, topic and niche are read from whatever this was handed:
     step 2's dict, a run directory, assembly.json, or a bare .txt;
  2. Gemini writes the copy and a tiered set of hashtags in one JSON call,
     through step 2's client rather than a second one. POST_MODE=heuristic, a
     missing key or a failed call all fall back to keyword extraction instead;
  3. the draft is then held to Instagram's real constraints in code, because a
     model cannot be trusted with a character count: the hook is cut to
     POST_HOOK_MAX_CHARS at a sentence boundary so it never reads as a
     truncated line, tags are normalised, deduped, banned-checked and mixed
     across tiers, banned words are re-checked and must-include phrases added;
  4. footage whose licence requires a credit gets one, in the form that licence
     expects (POST_ATTRIBUTION);
  5. anything still over the 2200-character limit is trimmed from the bottom -
     courtesy credits first; the hook and required credits, never.

Every knob - tone, tag count, attribution, banned words - lives in config.py.

Standalone use:
    python pipeline/caption.py -r output/2026-08-19/ocean-1a2b3c
    python pipeline/caption.py -f output/scripts/ocean.json --hashtags 12
    python pipeline/caption.py -f output/scripts/ocean.txt --dry-run --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

# Allow both `python pipeline/caption.py` and `python -m pipeline.caption`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import script_gen, tagvolume  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables. The voice, the tag count and the attribution policy are config.py's
# job - what follows is mechanical: what the files are called, how much slack to
# leave the model, and the word lists the no-API fallback needs.
# ---------------------------------------------------------------------------

#: Bumped when the shape of post.json changes, so step 7 can tell.
#: 2 added "hashtag_volumes" - the post count behind every tag.
#: 3 added "alt_text" and "alt" - the description of step 5's cover frame.
#: 4 added "hashtag_rejected" - the tags that were priced and thrown out. They
#: used to exist only as a sentence in the log, which made the estimator's
#: rejections the one part of the tag pipeline nobody could check or seed
#: against. `python pipeline/tagvolume.py --worklist` reads this field.
#: 5 added "claims" - what the fact-check cut from the copy. See
#: `check_post_claims`.
POST_VERSION = 5

#: Written into the run directory. The `captions.*` files beside them are step
#: 5's burned-in subtitles - deliberately a different word.
POST_TEXT_NAME = "post.txt"
POST_JSON_NAME = "post.json"

#: Only written when POST_HASHTAG_PLACEMENT=comment. Kept out of post.txt so
#: that file is always exactly the caption and nothing else - step 7 posting
#: it verbatim can never put the tag block in the caption by accident.
POST_COMMENT_NAME = "post_comment.txt"

#: The cover's alt text, on its own for the same reason as the comment file:
#: it goes in a different Instagram field from the caption, and posting by hand
#: means copying from somewhere. Only written when there is a cover to describe.
POST_ALT_NAME = "post_alt.txt"

#: Written by step 5. Where the script, topic and per-asset credits come from
#: when this is pointed at a run directory.
ASSEMBLY_NAME = "assembly.json"

#: Step 4's, read the same way if this is pointed at a visuals run instead.
MANIFEST_NAME = "manifest.json"

#: Gemini sampling. Warmer than step 4's query pass - this is copy, not search
#: terms - and a little cooler than step 2's script.
TEMPERATURE = 0.9
MAX_OUTPUT_TOKENS = 2048
THINKING_BUDGET = 0

#: How the tag slots are split. Broad tags place the post in a subject area,
#: niche ones reach the people who actually care, and mid does the work of
#: saying what this particular video is. Ratios, so the mix survives a changed
#: POST_HASHTAG_COUNT.
HASHTAG_MIX = (("broad", 0.25), ("mid", 0.45), ("niche", 0.30))

#: Ask each tier for a few more than its quota, so dedupe and the banned list
#: have something to eat without leaving the mix short.
HASHTAG_OVERASK = 3

#: When this share of a pool priced purely by estimate comes back dead, the
#: estimator is out of its depth on the subject rather than the generator
#: having returned a pool of junk. Nothing is dropped in that case - see
#: _weigh_the_verdict.
ESTIMATE_DOUBT_SHARE = 0.5

#: With volume banding on, the band filter eats the pool as well - and it is
#: the hungriest of the three, because a whole tier of the model's "niche"
#: guesses can turn out to be dead tags. Ask for more up front; an unused
#: suggestion costs nothing, a short tag block costs reach.
HASHTAG_OVERASK_BANDED = 6

#: Longest tag worth having. Instagram accepts more; nobody searches it.
HASHTAG_MAX_CHARS = 30

#: Paragraph count asked of the model.
BODY_MIN_PARAGRAPHS = 1
BODY_MAX_PARAGRAPHS = 3

#: Runs kept in the hashtag ledger.
LEDGER_KEEP_RUNS = 50

#: Instagram's cap on @mentions in one caption. Only warned about - the copy
#: never adds mentions by itself, they arrive via POST_MUST_INCLUDE.
MENTION_LIMIT = 20


class CaptionError(RuntimeError):
    """Raised when there is no usable script, or nowhere to write the post."""


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

#: The ranges step 2 strips from a script. Here they are counted rather than
#: removed - a caption may keep a couple (POST_ALLOW_EMOJI / POST_MAX_EMOJI).
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # pictographs, emoticons, symbols
    "\U00002600-\U000027bf"  # misc symbols and dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"
    "]"
)

#: Zero-width joiners, skin tones and variation selectors, so a multi-codepoint
#: emoji is not left half-deleted behind a stripped one.
_EMOJI_GLUE_RE = re.compile(
    "[\U0000fe00-\U0000fe0f\U0000200d\U0001f3fb-\U0001f3ff]"
)

#: Labels a model sometimes prefixes to a field despite being told not to.
_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?(?:caption|hook|body|cta|post|hashtags?)"
    r"(?:\s*\(.*?\))?(?:\*\*)?\s*[:\-]\s*",
    flags=re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"'”’]?\s+")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]+")

#: A hook must not end on one of these. It leaves the reader mid-thought, which
#: is exactly what Instagram's own "... more" already does to a caption.
_DANGLING_WORDS = frozenset("""
a an and as at because before but by for from how if in into like of on or so
than that the then to until upon when where which while who with without your
""".split())

#: Words that make a poor hashtag on their own.
_STOPWORDS = frozenset("""
about after all also and any are around because been before being both but can
could did does doing down each even ever every few for from get gets got had
has have having her here hers him his how into its itself just like made make
makes many may more most much must not now off once one only other our out
over own same she should some such than that the their them then there these
they thing things this those through too under until very was way were what
when where which while who why will with would you your
""".split())

#: Instagram tags allow letters, digits and underscores. Everything else -
#: spaces, punctuation, emoji - is stripped rather than the tag thrown away.
_TAG_STRIP_RE = re.compile(r"[^\w]", flags=re.UNICODE)

_MENTION_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9._]+")


def strip_markup(text: str) -> str:
    """Remove the markdown, fences and labels a model adds on its own."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"[*_`]{1,3}(?=\S)", "", cleaned)      # **bold**, __italic__
    cleaned = re.sub(r"(?<=\S)[*_`]{1,3}", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = "\n".join(_LABEL_RE.sub("", line) for line in cleaned.splitlines())
    cleaned = cleaned.strip()
    if (len(cleaned) >= 2 and cleaned[0] in "\"'“‘"
            and cleaned[-1] in "\"'”’"):
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, keeping the punctuation."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(str(text or "").strip())]
    return [p for p in parts if p]


def count_emoji(text: str) -> int:
    return len(_EMOJI_RE.findall(text or ""))


def apply_emoji_budget(fields: list[str], *, allowed: bool,
                       maximum: int) -> tuple[list[str], list[str]]:
    """Hold a run of copy to its emoji budget, spending it top-down.

    The hook is worth an emoji far more than the third paragraph is, so the
    budget is spent in reading order and the surplus stripped from the end.
    """
    budget_total = max(0, maximum) if allowed else 0
    seen = sum(count_emoji(field) for field in fields)
    if seen <= budget_total:
        return list(fields), []

    budget = budget_total
    trimmed: list[str] = []
    for field in fields:
        matches = list(_EMOJI_RE.finditer(field))
        if len(matches) <= budget:
            budget -= len(matches)
            trimmed.append(field)
            continue
        surplus = matches[budget:]
        budget = 0
        for match in reversed(surplus):
            field = field[:match.start()] + field[match.end():]
        field = _EMOJI_GLUE_RE.sub("", field)
        trimmed.append(re.sub(r"[ \t]{2,}", " ", field).strip())

    note = (f"stripped {seen - budget_total} emoji - the copy came back with "
            f"{seen} against a budget of {budget_total}"
            if budget_total else
            f"stripped {seen} emoji (POST_ALLOW_EMOJI is off)")
    return trimmed, [note]


def _undangle(text: str) -> str:
    """Trim a line back off a dangling connector, leaving a finished thought."""
    cleaned = text.strip()
    if cleaned.endswith((".", "!", "?", "…", ":")):
        return cleaned                      # already a finished sentence
    cleaned = cleaned.rstrip(",;:-–— ")
    words = cleaned.split()
    while words and words[-1].lower().strip("’'") in _DANGLING_WORDS:
        words.pop()
    return " ".join(words).rstrip(",;:-–— ")


def fit_hook(text: str, limit: int) -> tuple[str, list[str]]:
    """Cut the first line down to `limit` without leaving it mid-sentence.

    Instagram shows only the opening of a caption before "... more", so for
    most of the people who see the post this line *is* the caption. It is
    shortened at a sentence break first, a clause break second, and at a word
    boundary only as a last resort - never with an ellipsis, which is what the
    platform's own truncation already looks like.
    """
    hook = strip_markup(" ".join(str(text or "").split()))
    if not hook:
        return "", ["the generated hook was empty"]
    if len(hook) <= limit:
        return _undangle(hook), []

    sentences = split_sentences(hook)
    kept: list[str] = []
    for sentence in sentences:
        if len(" ".join(kept + [sentence])) > limit:
            break
        kept.append(sentence)
    if kept:
        return _undangle(" ".join(kept)), [
            f"hook was {len(hook)} chars; kept the first {len(kept)} "
            f"sentence(s) to fit {limit}"
        ]

    first = sentences[0] if sentences else hook
    clause = ""
    for match in re.finditer(r"\s*[,;:–—]\s+", first):
        piece = first[:match.start()].strip()
        if piece and len(piece) <= limit:
            clause = piece
    if clause:
        return _undangle(clause), [
            f"hook was {len(hook)} chars with no sentence break inside "
            f"{limit}; cut at a clause boundary"
        ]

    cut = first[:limit + 1].rsplit(" ", 1)[0]
    return _undangle(cut), [
        f"hook was {len(hook)} chars with no sentence or clause break inside "
        f"{limit}; cut at a word boundary - check it still reads as a hook"
    ]


def normalise_tag(raw: str) -> str | None:
    """Turn whatever the model wrote into a tag Instagram will accept.

    Casing is kept - #DeepSeaCreatures reads better than the flattened version
    and matches the same tag either way - but anything Instagram would refuse
    is stripped rather than the tag thrown away.
    """
    text = str(raw or "").strip().lstrip("#").strip()
    text = _TAG_STRIP_RE.sub("", text)
    if not text or text.isdigit() or len(text) < 2:
        return None
    if len(text) > HASHTAG_MAX_CHARS:
        return None
    return text


def _stem(raw: str) -> str:
    """Lowercase a word and drop any contraction or possessive tail.

    Without this "It’s" survives as "its" and normalises into the tag
    #its, which says nothing about anything.
    """
    return re.split(r"['’]", str(raw).lower(), maxsplit=1)[0].strip("-")


def _is_content_word(word: str) -> bool:
    return len(word) >= 4 and word not in _STOPWORDS and not word.isdigit()


def keywords(text: str, limit: int = 6) -> list[str]:
    """The most tag-worthy words in `text`, best first. No API, no deps."""
    scores: dict[str, float] = {}
    order: list[str] = []
    for raw in _WORD_RE.findall(text or ""):
        word = _stem(raw)
        if not _is_content_word(word):
            continue
        if word not in scores:
            order.append(word)
            scores[word] = min(len(word), 12) / 12.0
            if raw[0].isupper() and raw != "I":
                scores[word] += 0.4
        scores[word] += 0.3                   # repetition is the real signal
    return sorted(order, key=lambda w: (-scores[w], order.index(w)))[:limit]


def phrases(text: str, limit: int = 4) -> list[str]:
    """Two-word phrases that stand on their own, for the long-tail tier.

    Only words that sit next to each other in the text count - "deep ocean"
    and "total darkness" are things people search for, whereas pairing the two
    highest-scoring words in a script gives a tag nobody has ever typed.
    """
    scores: dict[str, float] = {}
    order: list[str] = []
    # Punctuation ends a run: a pair either side of a comma is not a phrase.
    for run in re.split(r"[^\w\s'’-]+", text or ""):
        words = [_stem(w) for w in _WORD_RE.findall(run)]
        for first, second in zip(words, words[1:]):
            if not (_is_content_word(first) and _is_content_word(second)):
                continue
            phrase = f"{first}{second}"
            if len(phrase) > HASHTAG_MAX_CHARS:
                continue
            if phrase not in scores:
                order.append(phrase)
                scores[phrase] = 0.0
            scores[phrase] += 1.0
    return sorted(order, key=lambda p: (-scores[p], order.index(p)))[:limit]


# ---------------------------------------------------------------------------
# What this was handed
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """Everything the copy is written from, wherever it arrived from."""

    script: str = ""
    topic: str = ""
    niche: str = ""
    run_id: str = ""
    run_dir: Path | None = None
    credits: list[dict] = dataclass_field(default_factory=list)
    #: Step 5's cover record, when this came from an assembly.json that has one.
    #: The alt text is written from this and nothing else - it is a description
    #: of that image, so with no cover there is nothing to describe.
    cover: dict = dataclass_field(default_factory=dict)
    script_path: Path | None = None
    assembly_path: Path | None = None
    warnings: list[str] = dataclass_field(default_factory=list)


def _read_json(path: Path, default: dict) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically, so a crash cannot corrupt the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _slug(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug[:limit].strip("-")


def _from_mapping(data: dict) -> Source:
    """Read step 2's script dict, step 4's manifest or step 5's assembly.

    All three carry `topic` and `script` under the same names; only the last
    two carry `attributions`, and only step 5's names an output directory or a
    `cover` (and only from assembly_version 2 on).
    """
    output_dir = data.get("output_dir")
    cover = data.get("cover")
    return Source(
        script=str(data.get("script") or "").strip(),
        topic=str(data.get("topic") or "").strip(),
        niche=str(data.get("niche") or "").strip(),
        run_id=str(data.get("run_id") or "").strip(),
        run_dir=Path(output_dir) if output_dir else None,
        credits=[c for c in (data.get("attributions") or [])
                 if isinstance(c, dict)],
        cover=cover if isinstance(cover, dict) else {},
    )


def _from_directory(directory: Path) -> Source:
    """Read a run directory: step 5's assembly.json, or step 4's manifest."""
    for name in (ASSEMBLY_NAME, MANIFEST_NAME):
        path = directory / name
        if not path.is_file():
            continue
        data = _read_json(path, {})
        if not data.get("script"):
            continue
        resolved = _from_mapping(data)
        resolved.assembly_path = path
        resolved.run_dir = directory
        return resolved

    raise CaptionError(
        f"{directory} holds no {ASSEMBLY_NAME} or {MANIFEST_NAME} with a "
        f"script in it. Point -r at a run directory written by step 5, or "
        f"pass the script itself with -f."
    )


def _from_file(path: Path) -> Source:
    """Read a .json script/assembly, or a .txt script with its sidecar."""
    if path.suffix.lower() == ".json":
        data = _read_json(path, {})
        if not data.get("script"):
            raise CaptionError(f"{path} has no 'script' field to write about.")
        resolved = _from_mapping(data)
        if data.get("assembly_version") or data.get("manifest_version"):
            resolved.assembly_path = path
            resolved.run_dir = resolved.run_dir or path.parent
        else:
            resolved.script_path = path
        return resolved

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaptionError(f"could not read {path}: {exc}") from exc

    resolved = Source(script=text.strip(), script_path=path)
    # Step 2 writes ocean.txt and ocean.json side by side; the sidecar is the
    # only place the topic survives, so prefer it over guessing from the name.
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        resolved.topic = str(_read_json(sidecar, {}).get("topic") or "").strip()
    if not resolved.topic:
        resolved.topic = path.stem.replace("_", " ").replace("-", " ").strip()
    return resolved


def resolve_source(source) -> Source:
    """Accept a dict, a run directory, a script file, or the script itself."""
    if isinstance(source, Source):
        resolved = source
    elif isinstance(source, dict):
        resolved = _from_mapping(source)
    elif isinstance(source, Path) or (
            isinstance(source, str) and _looks_like_path(source)):
        path = Path(source)
        resolved = _from_directory(path) if path.is_dir() else _from_file(path)
    else:
        resolved = Source(script=str(source or "").strip())

    resolved.script = strip_markup(resolved.script)
    if not resolved.script:
        raise CaptionError("the script is empty - there is nothing to caption.")
    return resolved


def _looks_like_path(text: str) -> bool:
    """True when a string is a path that exists, rather than a script."""
    if not text or len(text) > 260 or "\n" in text:
        return False
    try:
        return Path(text).exists()
    except OSError:
        return False


def _default_run_id(source: Source) -> str:
    """Same shape step 4 uses: a slug of the topic plus a hash of the script."""
    stem = _slug(source.topic) or _slug(" ".join(source.script.split()[:6]))
    digest = sha1(source.script.encode("utf-8")).hexdigest()[:8]
    return f"{stem or 'post'}-{digest}"


# ---------------------------------------------------------------------------
# Cross-run hashtag ledger
# ---------------------------------------------------------------------------

class HashtagLedger:
    """Which tags the last few posts used.

    Not a ban list. A tag that genuinely belongs on this post still goes on it;
    this only breaks ties, so consecutive posts do not carry an identical block
    of eight - which is what makes a feed read as automated.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.POST_HASHTAG_USAGE_FILE)
        self._data = _read_json(self.path, {"version": 1, "runs": []})
        if not isinstance(self._data.get("runs"), list):
            self._data["runs"] = []

    def recent(self, runs: int, *, exclude: str = "") -> list[str]:
        """Tags from the last `runs` posts, this run's own history aside.

        Re-running the same run id must not see its own previous tags as
        "recently used", or a re-run would drift away from its best answer.
        """
        if runs <= 0:
            return []
        entries = [e for e in self._data["runs"]
                   if isinstance(e, dict) and e.get("run_id") != exclude]
        tags: list[str] = []
        for entry in entries[-runs:]:
            tags.extend(str(t) for t in (entry.get("hashtags") or []))
        return list(dict.fromkeys(tags))

    def record(self, run_id: str, tags: list[str]) -> None:
        entries = [e for e in self._data["runs"]
                   if isinstance(e, dict) and e.get("run_id") != run_id]
        entries.append({
            "run_id": run_id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hashtags": list(tags),
        })
        self._data["runs"] = entries[-LEDGER_KEEP_RUNS:]

    def save(self) -> list[str]:
        """Persist. A ledger that cannot be written is a warning, not a failure."""
        try:
            _write_json(self.path, self._data)
        except OSError as exc:
            return [f"could not update the hashtag ledger at {self.path}: {exc}"]
        return []


# ---------------------------------------------------------------------------
# The draft: what the copy says, before it is held to any limit
# ---------------------------------------------------------------------------

@dataclass
class Draft:
    hook: str = ""
    body: list[str] = dataclass_field(default_factory=list)
    cta: str = ""
    hashtags: dict[str, list[str]] = dataclass_field(default_factory=dict)
    #: Only asked for, and only ever filled in, when POST_ALT_MODE=llm. It is a
    #: rephrasing of words step 5 recorded, not a description of an image the
    #: model has seen - which it has not.
    alt: str = ""
    source: str = "llm"


SYSTEM_INSTRUCTION = """\
You write the caption that sits under a short-form vertical video on Instagram.

You are not writing the video - it already exists. The viewer reads you either \
before it plays or after it has already landed, so the caption's job is to \
earn the tap on "... more" and then be worth having read.

You write like a person who knows the subject, not like a brand account. Short \
lines. Concrete detail over adjectives. No engagement bait, no "Did you know", \
no rows of emoji, no motivational filler, no restating what the video just said.

You always return valid JSON and nothing else."""


PROMPT_TEMPLATE = """\
Write the Instagram caption for this reel.

Topic: {topic}
Account subject area: {niche}

The narration the viewer hears, in full:
\"\"\"
{script}
\"\"\"

Fields:

"hook" - the first line of the caption, and the only part Instagram shows \
before it cuts to "... more". At most {hook_max} characters. It has to work \
alone: a finished thought, never the opening half of a sentence, never ending \
on a comma or a dangling connector. Do not restate the video's first line word \
for word - the viewer is already getting that.

"body" - {body_min} to {body_max} short paragraphs, one string per paragraph. \
Give the reader something the narration does not: the detail behind it, the \
number, the caveat, why it is worth knowing. Do not summarise the video.

"cta" - one closing line: {cta}. Use "" if it would only be filler.

"hashtags" - {tag_total} tags in three tiers, every one of them earned by what \
THIS script is about rather than by the subject in general:
  "broad": {broad} big tags for the subject area
  "mid":   {mid} tags for the specific subject of this video
  "niche": {niche_n} long-tail tags someone deep in this topic would follow
Each tag: no "#", no spaces, no punctuation, no emoji, {tag_max} characters at \
most. Do not repeat a word across tiers.
{alt_field}
Tone: {tone}
{rules}
Return ONLY this JSON object:
{{"hook": "...", "body": ["...", "..."], "cta": "...",{alt_json}
  "hashtags": {{"broad": ["..."], "mid": ["..."], "niche": ["..."]}}}}"""


#: Asked for only when POST_ALT_MODE=llm. The instruction is deliberately
#: narrow: the model has not seen the cover, and the one failure that matters
#: here is a confident description of something that is not in the picture.
ALT_FIELD_TEMPLATE = """
"alt" - alt text for the reel's cover image, at most {alt_max} characters. \
All that is known about that image is this: {depicts}. Rewrite ONLY those \
words into one plain, literal sentence describing the picture. Do not name a \
species, place, colour or action that is not in them, do not guess at what is \
happening, do not mention the video, and do not write a second caption. If \
they are already a sentence, leave them as they are.
"""


def tag_quota(count: int) -> dict[str, int]:
    """Split `count` slots across the tiers, remainder to the widest share."""
    count = max(0, count)
    quota = {tier: int(count * ratio) for tier, ratio in HASHTAG_MIX}
    widest = [tier for tier, _ in sorted(HASHTAG_MIX, key=lambda pair: -pair[1])]
    index = 0
    while sum(quota.values()) < count:          # the rounding remainder
        quota[widest[index % len(widest)]] += 1
        index += 1
    return quota


def build_prompt(script: str, *, topic: str, niche: str, hook_max: int,
                 count: int, tone: str, cta: str, rules: list[str],
                 alt_depicts: str = "") -> str:
    """Render the prompt. Every constraint in it also exists in code below."""
    quota = tag_quota(count)
    overask = (HASHTAG_OVERASK_BANDED if config.TAG_VOLUME_ENABLED
               else HASHTAG_OVERASK)
    asked = {tier: quota[tier] + overask if quota[tier] else 0
             for tier in quota}
    rule_block = "".join(f"- {rule}\n" for rule in rules if rule)
    alt_field = (ALT_FIELD_TEMPLATE.format(alt_max=config.POST_ALT_MAX_CHARS,
                                           depicts=alt_depicts)
                 if alt_depicts else "")
    return PROMPT_TEMPLATE.format(
        alt_field=alt_field,
        alt_json=' "alt": "...",' if alt_depicts else "",
        topic=topic or "(not given - take it from the script)",
        niche=niche or "(not given - take it from the script)",
        script=script,
        hook_max=hook_max,
        body_min=BODY_MIN_PARAGRAPHS,
        body_max=BODY_MAX_PARAGRAPHS,
        cta=cta or "leave it empty",
        tag_total=sum(asked.values()),
        broad=asked["broad"], mid=asked["mid"], niche_n=asked["niche"],
        tag_max=HASHTAG_MAX_CHARS,
        tone=tone,
        rules=f"\nAlso:\n{rule_block}" if rule_block else "",
    )


def _prompt_rules(*, banned_words, banned_tags, must_include, recent_tags,
                  emoji_allowed: bool, emoji_max: int) -> list[str]:
    """The run-specific rules bolted onto the prompt, from config.py."""
    rules = []
    if emoji_allowed and emoji_max:
        rules.append(f"At most {emoji_max} emoji in the whole caption, and only "
                     f"where one earns its place. None in the hashtags.")
    else:
        rules.append("No emoji anywhere in the caption.")
    if banned_words:
        rules.append("These words must not appear anywhere in the copy: "
                     + ", ".join(sorted(banned_words)) + ".")
    if banned_tags:
        rules.append("Never use these hashtags: "
                     + ", ".join("#" + t for t in sorted(banned_tags)) + ".")
    if must_include:
        rules.append("Work these in verbatim, without changing the wording: "
                     + "; ".join(must_include) + ".")
    if recent_tags:
        rules.append("Recent posts already used these tags - prefer fresh ones "
                     "unless a tag genuinely belongs on this post: "
                     + ", ".join("#" + t for t in recent_tags[:24]) + ".")
    return rules


def _parse_draft_json(raw: str) -> Draft:
    """Read the model's JSON object into a Draft, tolerantly."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output was not a JSON object")

    body_raw = data.get("body")
    if isinstance(body_raw, str):
        body_raw = [p for p in body_raw.split("\n\n")]
    body = [strip_markup(p) for p in (body_raw or []) if str(p or "").strip()]

    tiers: dict[str, list[str]] = {}
    hashtags = data.get("hashtags")
    if isinstance(hashtags, dict):
        for tier, _ in HASHTAG_MIX:
            tiers[tier] = [str(t) for t in (hashtags.get(tier) or [])
                           if str(t or "").strip()]
    elif isinstance(hashtags, list):     # a flat list: treat it as the mid tier
        tiers = {"broad": [], "mid": [str(t) for t in hashtags], "niche": []}

    draft = Draft(
        hook=strip_markup(data.get("hook") or ""),
        body=body[:BODY_MAX_PARAGRAPHS],
        cta=strip_markup(data.get("cta") or ""),
        hashtags=tiers,
        alt=strip_markup(data.get("alt") or ""),
    )
    if not draft.hook and not draft.body:
        raise ValueError("model returned neither a hook nor a body")
    return draft


def llm_draft(script: str, *, topic: str, niche: str, count: int,
              rules: list[str],
              alt_depicts: str = "") -> tuple[Draft | None, list[str]]:
    """One Gemini call for the whole caption. Never raises; returns warnings.

    Goes through step 2's client (`script_gen.call_gemini`) rather than
    building a second one, so the key lookup and the model-family quirks stay
    in one place.
    """
    prompt = build_prompt(
        script, topic=topic, niche=niche,
        hook_max=config.POST_HOOK_MAX_CHARS, count=count,
        tone=config.POST_TONE, cta=config.POST_CTA, rules=rules,
        alt_depicts=alt_depicts,
    )
    try:
        raw = script_gen.call_gemini(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_budget=THINKING_BUDGET,
            json_mode=True,
        )
    except config.MissingConfigError as exc:
        return None, [f"POST_MODE=llm but {exc} Wrote the caption from "
                      f"keywords instead."]
    except script_gen.ScriptGenerationError as exc:
        return None, [f"Gemini caption generation failed ({exc}). Wrote the "
                      f"caption from keywords instead."]

    try:
        return _parse_draft_json(raw), []
    except (ValueError, TypeError) as exc:
        return None, [f"Gemini returned an unusable caption ({exc}). Wrote the "
                      f"caption from keywords instead."]


#: Findings that take a sentence out of the post. One kind wider than step 2's
#: blocking pair, on purpose - see `check_post_claims`.
POST_CLAIM_DROP = ("contradiction", "false", "unsupported")


def check_post_claims(draft: Draft, *, topic: str,
                      mode: str | None = None) -> tuple[dict, list[str]]:
    """Fact-check what the caption says, and cut what fails. Never raises.

    The body is asked for "the detail behind it, the number, the caveat" - so
    it adds facts the narration never made, and until this pass none of them
    were checked. The narration survived a Pro fact-check that refuses whole
    topics, and the text posted beside it went out on one unchecked call.
    That stayed theoretical until the hunger stones post of 2026-09-30, which
    said "The oldest authenticated carving dates to 1417". The sources say the
    reverse: the 1417 mark was worn off, and 1616 is the oldest that can
    still be read.

    Same checker as step 2 (`script_gen.check_claims`), same switch
    (VERIFY_MODE), and a different remedy. A flagged sentence is dropped
    rather than repaired or refused. A caption is optional in a way the
    narration is not: losing one sentence of it costs nothing a viewer
    misses, and a rewrite would be a fresh unchecked call.

    That is also why `unsupported` drops here, though it only reports in
    step 2. There it cannot block, because blocking on it would refuse most
    interesting topics outright - see `script_gen._VERIFY_BLOCKING`. Here all
    it costs is one sentence, and a specific claim the checker does not
    believe is what the body is most likely to have invented.

    Only the hook and the body are checked. The CTA is a question to the
    reader, and the tags are not sentences.

    Returns (report, warnings). The report goes into post.json as `claims`.
    """
    setting = str(mode or config.VERIFY_MODE or "off").strip().lower()
    report = {"mode": setting, "checked": False, "dropped": [], "kept": []}
    if setting != "llm":
        return report, []

    # Every sentence of the copy, tagged with where it came from, so a finding
    # can be traced back to the one paragraph it has to come out of.
    located: list[tuple[int, str]] = []          # (-1 for the hook, paragraph)
    for sentence in split_sentences(draft.hook):
        located.append((-1, sentence))
    for number, paragraph in enumerate(draft.body):
        for sentence in split_sentences(paragraph):
            located.append((number, sentence))
    if len(located) < 2:
        report["checked"] = True
        return report, []

    checked = script_gen.check_claims(
        topic, " ".join(sentence for _, sentence in located)
    )
    report["checked"] = checked["checked"]
    warnings = [w.replace("The script was", "The caption was")
                for w in checked["warnings"]]

    doomed: set[int] = set()
    for finding in checked["findings"]:
        # The best match, not the first - "If you see me, then weep." sits
        # inside a sentence about the 1616 inscription, and a finding about
        # that sentence must not cost the post its hook.
        at = script_gen.quoted_sentence(
            finding["sentence"], [sentence for _, sentence in located])
        if at is None:
            continue
        entry = {"sentence": located[at][1], "kind": finding["kind"],
                 "problem": finding["problem"],
                 "from": "hook" if located[at][0] < 0 else "body"}
        if finding["kind"] not in POST_CLAIM_DROP:
            report["kept"].append(entry)
            continue
        if at in doomed:
            continue
        doomed.add(at)
        report["dropped"].append(entry)
        warnings.append(
            f"dropped a {finding['kind']} claim from the {entry['from']}: "
            f"{entry['sentence']!r} - {finding['problem']}"
        )
    if not doomed:
        return report, warnings

    hook = " ".join(sentence for at, (where, sentence) in enumerate(located)
                    if where < 0 and at not in doomed)
    body = []
    for number in range(len(draft.body)):
        kept = [sentence for at, (where, sentence) in enumerate(located)
                if where == number and at not in doomed]
        if kept:
            body.append(" ".join(kept))
    if not hook and body:
        # The hook is the only line shown before "... more", so an empty one
        # would open the post on the credits. The next finished paragraph
        # takes its place, and `fit_hook` holds it to the hook's limit.
        hook = body.pop(0)
        warnings.append("the whole hook was dropped; the first body paragraph "
                        "now opens the caption")
    draft.hook, draft.body = hook, body
    return report, warnings


def heuristic_draft(script: str, *, topic: str, niche: str,
                    count: int) -> Draft:
    """A caption with no API call behind it: the script's own best lines.

    Worse than the model on every axis - it can only reuse what the narration
    already said - but it always works, and a post with a flat caption beats a
    run that died at step 6.
    """
    sentences = split_sentences(script)
    hook = next((s for s in sentences if len(s) <= config.POST_HOOK_MAX_CHARS),
                sentences[0] if sentences else topic)

    rest = [s for s in sentences if s != hook]
    body: list[str] = []
    for start in range(0, min(len(rest), 4), 2):
        paragraph = " ".join(rest[start:start + 2]).strip()
        if paragraph:
            body.append(paragraph)
        if len(body) >= BODY_MAX_PARAGRAPHS - 1:
            break

    terms = keywords(f"{topic} {script}", limit=max(6, count * 2))
    niche_terms = keywords(niche, limit=2) if niche else []
    quota = tag_quota(count)
    broad = list(dict.fromkeys(niche_terms + keywords(topic, limit=3)))
    mid = [t for t in terms if t not in broad]
    pairs = [p for p in phrases(f"{topic}. {script}", limit=count)
             if p not in broad and p not in mid]

    return Draft(
        hook=hook,
        body=body,
        cta="",                       # POST_CTA describes a style, not a line
        hashtags={
            "broad": broad[:quota["broad"] + HASHTAG_OVERASK],
            "mid": mid[:quota["mid"] + HASHTAG_OVERASK],
            "niche": pairs[:quota["niche"] + HASHTAG_OVERASK],
        },
        source="heuristic",
    )


# ---------------------------------------------------------------------------
# Banned words
# ---------------------------------------------------------------------------

def banned_hits(draft: Draft, words) -> list[str]:
    """Which banned words the copy actually used. Whole words, case-blind."""
    if not words:
        return []
    text = " ".join([draft.hook, *draft.body, draft.cta]).lower()
    return [w for w in words
            if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE)]


def scrub_banned(draft: Draft, words) -> tuple[Draft, list[str]]:
    """Last resort when the model keeps using a banned word.

    Offending paragraphs and CTAs are dropped outright. A hook is not - a
    caption with no first line is worse than one that needs a human's eye - so
    that case is reported loudly and left for the operator to fix.
    """
    warnings: list[str] = []

    def offends(text: str) -> list[str]:
        return [w for w in words
                if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE)]

    kept = [p for p in draft.body if not offends(p)]
    if len(kept) != len(draft.body):
        warnings.append(
            f"dropped {len(draft.body) - len(kept)} paragraph(s) using banned "
            f"word(s) from POST_BANNED_WORDS"
        )
    draft.body = kept or draft.body[:1]

    if draft.cta and offends(draft.cta):
        warnings.append("dropped the CTA: it used a banned word")
        draft.cta = ""

    hook_hits = offends(draft.hook)
    if hook_hits:
        warnings.append(
            f"the hook still uses banned word(s) {', '.join(hook_hits)} - "
            f"left in place rather than mangling the first line. Fix it by "
            f"hand before posting."
        )
    return draft, warnings


# ---------------------------------------------------------------------------
# Hashtags
# ---------------------------------------------------------------------------

def _reading(volumes: dict, tag: str):
    """The reading for `tag`, however it was keyed."""
    return volumes.get(tag) or volumes.get(tag.lower())


def _weigh_the_verdict(pool: dict, dead: list[str], volumes: dict, *,
                       offered: list[str]) -> tuple[dict, list[str], list[str]]:
    """Decide whether the dead really are dead, or the estimator is lost.

    The estimator prices a tag by the words in it, so a subject it has no
    vocabulary for looks exactly like a subject full of coinages. #MariusHills
    and #RomanEmpire are the same shape to it - a name it does not hold and an
    ordinary word - and only one of them is a bad tag.

    A measured count is evidence and is acted on. A guess that condemns most of
    the pool is not evidence about the tags; it is evidence about the guesser.
    In that case nothing is dropped - the suspect tags go to the back of the
    queue instead, and the run says plainly that the reading is unreliable.
    """
    if not dead:
        return pool, dead, []

    readings = [_reading(volumes, tag) for tag in offered]
    guessed = [r for r in readings if r is not None and r.source == "estimate"]
    all_guesswork = readings and len(guessed) == len(readings)
    most_of_it = len(dead) >= max(2, ESTIMATE_DOUBT_SHARE * len(offered))

    if all_guesswork and most_of_it:
        # Biggest first: even a guess ranks them against each other.
        recovered = sorted(
            dead, key=lambda t: -((_reading(volumes, t) or
                                   tagvolume.Volume(t, 0, "", "")).count)
        )
        pool = dict(pool)
        pool["niche"] = list(pool.get("niche") or []) + recovered
        return pool, [], [
            f"{len(dead)} of {len(offered)} tags priced as too small to find, "
            f"which is more likely the estimator having no vocabulary for this "
            f"subject than {len(dead)} bad tags - kept, but ranked last. Seed "
            f"real counts in {config.TAG_VOLUME_FILE.name} to settle it."
        ]

    measured = [t for t in dead if (_reading(volumes, t) or
                                    tagvolume.Volume(t, 0, "estimate", "")
                                    ).source != "estimate"]
    return pool, dead, [
        f"dropped {len(dead)} tag(s) with too few posts to be found: "
        + ", ".join("#" + t for t in dead)
        + ("" if len(measured) == len(dead) else
           f" (estimated - seed real counts in "
           f"{config.TAG_VOLUME_FILE.name} to overrule)")
    ]


def _relax_floor(dead: list[str], volumes: dict, *, count: int,
                 chosen: list[str], add) -> list[str]:
    """Take back the largest of the dead tags rather than post a short block.

    Only down to TAG_VOLUME_FLOOR * TAG_VOLUME_RELAX, only in descending order
    of size, and never quietly: these are tags the filter already judged too
    small to be found, and the operator should see that the mix needed them.
    """
    floor = config.TAG_VOLUME_FLOOR * config.TAG_VOLUME_RELAX
    ranked = sorted(
        ((tag, (volumes.get(tag) or volumes.get(tag.lower())).count)
         for tag in dead
         if volumes.get(tag) or volumes.get(tag.lower())),
        key=lambda pair: -pair[1],
    )
    admitted: list[str] = []
    for tag, size in ranked:
        if len(chosen) >= count or size < floor:
            break
        if add(tag, "below-floor"):
            admitted.append(tag)
    if not admitted:
        return []
    return [
        f"put back {len(admitted)} tag(s) below the volume floor to reach "
        f"{count}: " + ", ".join("#" + t for t in admitted)
        + " - small, but the alternative was a shorter block"
    ]


def band_pool(pool: dict, volumes: dict) -> tuple[dict, list[str]]:
    """Re-bucket a tiered pool by the band each tag's post count puts it in.

    The generator's tiers are a guess about words; this replaces them with the
    only thing that decides whether a tag can be found. A tag the model filed
    under "niche" that turns out to carry two million posts is a broad tag, and
    one it filed under "mid" carrying four hundred is not a tag at all.

    Returns a pool keyed by band, and the tags dropped as dead.
    """
    banded: dict[str, list[str]] = {band: [] for band in tagvolume.BANDS}
    for tier, _ in HASHTAG_MIX:
        for tag in pool.get(tier) or []:
            volume = volumes.get(tag) or volumes.get(tag.lower())
            # No reading at all is not evidence of a bad tag - keep it in the
            # band the generator meant, rather than inventing a verdict.
            band = volume.band if volume else tier
            if tag not in banded[band]:
                banded[band].append(tag)
    return banded, list(banded["dead"])


def select_hashtags(tiers: dict, *, count: int, always=(), banned_tags=(),
                    banned_words=(), recent=(),
                    volumes: dict | None = None,
                    rejected: list | None = None
                    ) -> tuple[list[str], dict, list[str]]:
    """Pick the final tag set: mixed across bands, deduped, freshest first.

    Returns the tags in posting order, a {tag: tier} map for post.json, and any
    warnings. Tags from POST_HASHTAGS_ALWAYS come first and are never dropped
    for a generated one.

    With TAG_VOLUME_ENABLED the tiers the generator returned are only a
    starting pool: every tag is priced (see pipeline/tagvolume.py), dead ones
    are dropped, and the rest are filled into the mix by the band they measure
    at rather than the tier they were labelled with. `volumes` lets the caller
    pass readings it has already taken; otherwise they are looked up here.

    `rejected` is an out-list, filled with what was dropped and why. Nothing
    used to survive the drop but a sentence in the log: 9 of 19 tags went in
    one run and the only record of which ones was a warning nobody can query.
    That is the wrong way round - the estimator's rejections are the exact
    thing worth auditing, because it prices a tag by how ordinary its words
    are and a subject it has no vocabulary for reads as a subject full of
    coinages. Written into post.json, they become a seeding worklist
    (`python pipeline/tagvolume.py --worklist`).
    """
    warnings: list[str] = []
    banned_set = {str(t).lower().lstrip("#") for t in banned_tags}
    words = tuple(str(w).lower() for w in banned_words)
    recent_set = {str(t).lower() for t in recent}

    chosen: list[str] = []
    tier_of: dict[str, str] = {}
    seen: set[str] = set()
    dropped: list[str] = []

    def refused(tag: str) -> bool:
        low = tag.lower()
        return low in banned_set or any(w and w in low for w in words)

    def add(tag: str, tier: str) -> bool:
        low = tag.lower()
        if low in seen:
            return False
        seen.add(low)
        chosen.append(tag)
        tier_of[tag] = tier
        return True

    for raw in always:
        tag = normalise_tag(raw)
        if not tag:
            continue
        if refused(tag):
            dropped.append(tag)
            continue
        add(tag, "always")

    pool: dict[str, list[str]] = {}
    for tier, _ in HASHTAG_MIX:
        items: list[str] = []
        for raw in tiers.get(tier) or []:
            tag = normalise_tag(raw)
            if not tag:
                continue
            if refused(tag):
                dropped.append(tag)
                continue
            items.append(tag)
        pool[tier] = items

    # --- price every tag, and re-bucket by what it measured at -------------
    banding = config.TAG_VOLUME_ENABLED
    dead: list[str] = []
    if banding:
        every = list(dict.fromkeys(t for tier, _ in HASHTAG_MIX
                                   for t in pool.get(tier, [])))
        if volumes is None:
            volumes, volume_warnings = tagvolume.lookup(every)
            warnings += volume_warnings
        pool, dead = band_pool(pool, volumes or {})
        pool, dead, doubt_warnings = _weigh_the_verdict(pool, dead, volumes,
                                                        offered=every)
        warnings += doubt_warnings
    order = [tier for tier, _ in HASHTAG_MIX]
    bands_for = (lambda tier: tagvolume.TIER_BANDS[tier]) if banding \
        else (lambda tier: (tier,))

    # A saturated tag places the post in a subject area; a block of them just
    # enters the post in competitions it cannot win.
    saturated_used = 0

    def spendable(tag: str, band: str) -> bool:
        nonlocal saturated_used
        if band != "saturated":
            return True
        if saturated_used >= config.TAG_VOLUME_MAX_SATURATED:
            return False
        saturated_used += 1
        return True

    def drain(band: str, tier: str, want: int, taken: int) -> int:
        """Take up to `want` tags from one band, freshest first."""
        candidates = pool.get(band) or []
        fresh = [t for t in candidates if t.lower() not in recent_set]
        stale = [t for t in candidates if t.lower() in recent_set]
        for tag in fresh + stale:
            if taken >= want or len(chosen) >= count:
                break
            if tag.lower() in seen or not spendable(tag, band):
                continue
            # Labelled with what it measured at, not with the slot that wanted
            # it - post.json should say what a tag is, not what it was for.
            if add(tag, band if banding else tier):
                taken += 1
        return taken

    quota = tag_quota(max(0, count - len(chosen)))
    for tier in order:
        taken = 0
        for band in bands_for(tier):
            if taken >= quota[tier] or len(chosen) >= count:
                break
            taken = drain(band, tier, quota[tier], taken)

    if len(chosen) < count:                 # a band came up short - top up
        for tier in ("mid", "niche", "broad"):
            for band in bands_for(tier):
                if len(chosen) >= count:
                    break
                drain(band, tier, count, len(chosen))

    if banding and len(chosen) < count and dead and config.TAG_VOLUME_RELAX < 1:
        warnings += _relax_floor(dead, volumes or {}, count=count,
                                 chosen=chosen, add=add)

    if rejected is not None:
        # After _relax_floor, so a tag put back to fill the block is not
        # recorded as dropped. `seen` holds everything that made it in.
        for tag in dict.fromkeys(dropped):
            rejected.append({"tag": tag, "reason": "banned", "count": None,
                             "source": ""})
        for tag in dict.fromkeys(dead):
            if tag.lower() in seen:
                continue
            reading = _reading(volumes or {}, tag)
            rejected.append({
                "tag": tag,
                "reason": "below the volume floor",
                "count": reading.count if reading else None,
                "source": reading.source if reading else "",
            })

    if dropped:
        warnings.append("dropped banned tag(s): "
                        + ", ".join("#" + t for t in dict.fromkeys(dropped)))
    if len(chosen) < count:
        warnings.append(
            f"only {len(chosen)} of {count} hashtags survived normalising and "
            f"deduping - the generator did not return enough usable tags"
        )
    if len(chosen) > config.POST_HASHTAG_LIMIT:
        warnings.append(
            f"trimmed to Instagram's limit of {config.POST_HASHTAG_LIMIT} tags"
        )
        chosen = chosen[:config.POST_HASHTAG_LIMIT]

    reused = [t for t in chosen if t.lower() in recent_set]
    if recent_set and len(reused) == len(chosen) and chosen:
        warnings.append(
            "every tag here was used by a recent post - the set is not varying"
        )
    return chosen, {t: tier_of[t] for t in chosen}, warnings


# ---------------------------------------------------------------------------
# Alt text for step 5's cover frame
#
# The description that goes in Instagram's "Write alt text" field. It does two
# jobs at once and they do not pull against each other: a screen reader reads
# it aloud, and Instagram reads it for what the picture is of.
#
# It is a description of an IMAGE, not a second caption. No hook, no selling,
# no hashtags - and short, because the field is short and because a screen
# reader user is listening to it rather than skimming it.
#
# Everything it says comes from what step 5 recorded about the cover it chose.
# ---------------------------------------------------------------------------

#: Cut here rather than mid-word when the description runs long.
_ALT_TRIM_RE = re.compile(r"[\s,;:]+\S*$")

#: Below this there is no room left to say anything about the picture, so the
#: description is dropped rather than trimmed to a stub.
_ALT_MIN_PICTURE_CHARS = 16


def _sentence_case(text: str) -> str:
    """Capitalise the first letter and leave the rest of the casing alone.

    `.capitalize()` would lower-case the rest, which turns a provider's "Blue
    Ocean Medusa" into something that reads as though it were retyped badly.
    """
    text = str(text or "").strip()
    return text[:1].upper() + text[1:] if text else ""


def _fit_alt(text: str, limit: int) -> str:
    """Trim alt text to the limit at a word boundary, ending in a full stop."""
    text = " ".join(str(text or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    trimmed = _ALT_TRIM_RE.sub("", cut) or cut
    return trimmed.rstrip(" ,;:") + "."


def build_alt_text(cover: dict, *, limit: int, mode: str = "literal",
                   rewritten: str = "") -> tuple[str, dict, list[str]]:
    """Describe the cover image. Returns (text, report, warnings).

    Two facts go in, in this order of importance:

      1. any text burned into the cover - the hook card. This is the half that
         matters most: a viewer using a screen reader gets nothing at all from
         a cover that is mostly a headline unless the alt text says what the
         headline said;
      2. what the picture underneath shows, in the words step 5 recorded.

    `rewritten` is the model's phrasing of (2) when POST_ALT_MODE=llm. It only
    ever replaces the wording of the depiction - never the card text, which is
    quoted exactly because it is a quotation.
    """
    warnings: list[str] = []
    report = {
        "mode": mode, "source": "none", "chars": 0, "limit": limit,
        "from_cover": "", "depicts": "", "has_card_text": False,
    }
    if not isinstance(cover, dict) or not cover:
        return "", report, warnings

    depicts = " ".join(str(cover.get("depicts") or "").split())
    source = str(cover.get("depicts_source") or "none")
    if rewritten:
        rephrased = " ".join(str(rewritten).split())
        if rephrased:
            depicts, source = rephrased, "llm"

    card = " ".join(str(cover.get("hook_card_text") or "").split())
    report.update({
        "source": source,
        "from_cover": str(cover.get("filename") or ""),
        "depicts": depicts,
        "has_card_text": bool(card),
    })

    if not depicts and not card:
        warnings.append(
            "step 5's cover record describes nothing, so the post has no alt "
            "text - the cover is the one thing a screen reader cannot reach"
        )
        return "", report, warnings

    # Two sentences rather than one clause. `depicts` arrives either as a
    # provider's lower-case slug ("medusa blue ocean") or, in llm mode, as a
    # capitalised sentence - and there is no join that reads correctly for both
    # ("... over A jellyfish drifts in blue water"). Kept apart, each is
    # grammatical whichever it is, and quoting the card in its own sentence is
    # how a screen reader is normally told an image has words on it.
    picture = f"{_sentence_case(depicts)}." if depicts else ""
    on_image = ""
    if card:
        # A card too long for the field on its own is trimmed *inside* the
        # quotes. Trimming the sentence instead leaves "Text on the image.",
        # which tells a listener there are words and not what they say.
        room = limit - len('Text on the image: "".')
        quoted = _sentence_case(card)
        if room > 0 and len(quoted) > room:
            quoted = _fit_alt(quoted, room).rstrip(".")
        on_image = f'Text on the image: "{quoted}".'

    if picture and on_image:
        # The card text is the half a screen reader cannot get any other way,
        # so the picture gives up its characters first rather than last - and
        # is dropped outright rather than left as a stub too short to mean
        # anything ("Bl. Text on the image: ...").
        room = limit - len(on_image) - 1
        trimmed = _fit_alt(picture, room) if len(picture) > room else picture
        if room < _ALT_MIN_PICTURE_CHARS:
            trimmed = ""
        text = f"{trimmed} {on_image}".strip() if trimmed else on_image
    else:
        text = picture or on_image

    fitted = _fit_alt(text, limit)
    if len(f"{picture} {on_image}".strip()) > limit:
        warnings.append(
            f"alt text did not fit {limit} characters and was trimmed; the "
            f"text on the cover is kept ahead of the description of it"
        )
    report["chars"] = len(fitted)
    return fitted, report, warnings


# ---------------------------------------------------------------------------
# Attribution carried through from step 4 (via step 5's assembly.json)
# ---------------------------------------------------------------------------

@dataclass
class Credits:
    """The credit block, split by whether a licence actually demands it."""

    header: str = ""
    required: list[str] = dataclass_field(default_factory=list)
    courtesy: str = ""
    mode: str = "required"
    assets: int = 0

    def block(self, *, courtesy: bool = True) -> str:
        lines = list(self.required)
        if courtesy and self.courtesy:
            lines.append(self.courtesy)
        if not lines:
            return ""
        if not self.required and courtesy and self.courtesy:
            return f"{self.header} {self.courtesy}".strip()
        return "\n".join([self.header, *lines]) if self.header else "\n".join(lines)


def _credit_line(credit: dict) -> str:
    """One asset's credit, in the form its licence asks for.

    Step 4 already built the string a CC licence wants - title, creator,
    licence and source - so it is used as given rather than rebuilt from parts.
    """
    line = str(credit.get("attribution") or "").strip()
    if line:
        return line
    creator = str(credit.get("creator") or "unknown creator").strip()
    provider = str(credit.get("provider") or "").strip()
    licence = str(credit.get("license") or "").strip()
    url = str(credit.get("source_url") or "").strip()
    parts = [f'"{credit.get("kind") or "asset"}" by {creator}']
    if provider:
        parts.append(f"via {provider}")
    if licence:
        parts.append(f"({licence})")
    return " ".join(parts) + (f" - {url}" if url else "")


def build_credits(credits: list[dict], mode: str, *, header: str,
                  maximum: int) -> tuple[Credits, list[str]]:
    """Turn step 4's per-asset credits into the block that goes in the caption.

    `required` credits are what a licence demands (CC-BY and friends) and are
    never dropped - not by the cap, not by the character trimmer. `courtesy` is
    the one-line "via Pexels" that most stock licences do not ask for.
    """
    warnings: list[str] = []
    if mode == "off" or not credits:
        return Credits(mode=mode, assets=len(credits)), warnings

    required = [c for c in credits if c.get("attribution_required")]
    lines = [_credit_line(c) for c in required]
    lines = list(dict.fromkeys(line for line in lines if line))

    if len(lines) > maximum:
        warnings.append(
            f"{len(lines)} credits are required by their licence, over the "
            f"POST_ATTRIBUTION_MAX of {maximum} - all of them are in the "
            f"caption anyway, because dropping one would breach the licence"
        )

    courtesy = ""
    if mode == "all":
        providers = list(dict.fromkeys(
            str(c.get("provider") or "").strip()
            for c in credits if not c.get("attribution_required")
        ))
        providers = [p for p in providers if p]
        if providers:
            courtesy = "via " + ", ".join(p.capitalize() for p in providers)

    return Credits(header=header, required=lines, courtesy=courtesy,
                   mode=mode, assets=len(credits)), warnings


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose(draft: Draft, *, tags: list[str], credits: Credits,
            must_include: list[str], placement: str,
            limit: int) -> tuple[str, dict, list[str]]:
    """Assemble the caption and hold it under Instagram's character limit.

    Over the limit, blocks are given up from the bottom in the order they can
    best be spared: the courtesy credit, then body paragraphs, then the CTA,
    then tags one at a time. The hook and any licence-required credit are never
    given up - if only those are left and it is still too long, the caption is
    cut and the run says so.
    """
    warnings: list[str] = []
    hook = draft.hook.strip()
    body = [p.strip() for p in draft.body if p.strip()]
    cta = draft.cta.strip()
    extras = [p for p in must_include if p]
    kept_tags = list(tags)
    courtesy = True

    def render() -> str:
        blocks = [hook, *body]
        if cta:
            blocks.append(cta)
        blocks.extend(extras)
        credit_block = credits.block(courtesy=courtesy)
        if credit_block:
            blocks.append(credit_block)
        if placement == "caption" and kept_tags:
            blocks.append(" ".join("#" + tag for tag in kept_tags))
        return "\n\n".join(block for block in blocks if block).strip()

    text = render()
    while len(text) > limit:
        if courtesy and credits.courtesy:
            courtesy = False
            warnings.append("dropped the courtesy credit line to fit the "
                            f"{limit}-character limit")
        elif len(body) > 1:
            body.pop()
            warnings.append(f"dropped a body paragraph to fit the "
                            f"{limit}-character limit")
        elif cta:
            cta = ""
            warnings.append(f"dropped the CTA to fit the {limit}-character limit")
        elif placement == "caption" and kept_tags:
            kept_tags.pop()
            warnings.append(f"dropped a hashtag to fit the {limit}-character limit")
        elif body:
            body.pop()
            warnings.append(f"dropped the last body paragraph to fit the "
                            f"{limit}-character limit")
        else:
            text = text[:limit].rsplit(" ", 1)[0].strip()
            warnings.append(
                f"the hook and the required credits alone run past {limit} "
                f"characters; the caption was cut. Check it before posting."
            )
            break
        text = render()

    parts = {
        "hook": hook,
        "body": body,
        "cta": cta,
        "must_include": extras,
        "credits": credits.block(courtesy=courtesy),
        "hashtags": kept_tags,
    }
    return text, parts, warnings


def _echoes_script(hook: str, script: str) -> bool:
    """True when the hook is just the narration's opening line again."""
    def flatten(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    first = (split_sentences(script) or [""])[0]
    a, b = flatten(hook), flatten(first)
    if not a or not b:
        return False
    return a == b or (len(a) > 24 and (a in b or b in a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_caption(source, *, run_dir=None, run_id: str | None = None,
                     topic: str | None = None, niche: str | None = None,
                     hashtag_count: int | None = None, placement: str | None = None,
                     attribution=None, mode: str | None = None, alt=None,
                     write: bool = True, overwrite: bool = True) -> dict:
    """Write the Instagram post text for a script.

    Args:
        source: step 2's script dict, step 5's assembly dict, a run directory,
            a .json/.txt script file, or the script text itself.
        run_dir: where to write. Defaults to the run directory the source came
            from, else output/<date>/<run id>/.
        run_id: overrides the id taken from the source, which names that
            directory.
        topic / niche: override what the source says. `niche` falls back to
            config.POST_NICHE.
        hashtag_count: defaults to config.POST_HASHTAG_COUNT.
        placement: "caption" (tags close the caption) or "comment" (tags are
            kept out and handed to step 7 for the first comment).
        attribution: "required" / "all" / "off", or a bool for the config
            default vs. off. Defaults to config.POST_ATTRIBUTION.
        mode: "llm" or "heuristic". Defaults to config.POST_MODE.
        alt: "literal" or "llm" for the cover's alt text, False for none.
            Defaults to config.POST_ALT_MODE.
        write: write post.txt and post.json. False plans everything and touches
            no disk.
        overwrite: replace an existing post.txt.

    Returns:
        The post manifest, also written as post.json: the caption text, its
        parts, the tags and their tiers, the credits, character counts and
        every warning raised.

    Raises:
        CaptionError: no usable script, or the destination cannot be written.
    """
    resolved = resolve_source(source)
    warnings: list[str] = list(resolved.warnings)

    if topic:
        resolved.topic = topic.strip()
    if niche is not None:
        resolved.niche = niche.strip()
    niche_value = resolved.niche or config.POST_NICHE

    mode = str(mode or config.POST_MODE).strip().lower()
    if mode not in ("llm", "heuristic"):
        warnings.append(f"unknown POST_MODE {mode!r}; used 'llm'")
        mode = "llm"

    placement = str(placement or config.POST_HASHTAG_PLACEMENT).strip().lower()
    if placement not in ("caption", "comment"):
        warnings.append(
            f"unknown POST_HASHTAG_PLACEMENT {placement!r}; used 'caption'"
        )
        placement = "caption"

    if attribution is None:
        attribution_mode = config.POST_ATTRIBUTION
    elif isinstance(attribution, bool):
        attribution_mode = (config.POST_ATTRIBUTION if attribution else "off")
        if attribution and attribution_mode == "off":
            attribution_mode = "required"
    else:
        attribution_mode = str(attribution)
    attribution_mode = attribution_mode.strip().lower()
    if attribution_mode not in ("required", "all", "off"):
        warnings.append(
            f"unknown POST_ATTRIBUTION {attribution_mode!r}; used 'required'"
        )
        attribution_mode = "required"

    # --- alt text mode ------------------------------------------------------
    # Settled before the draft, because `llm` is one more field in the caption
    # call rather than a request of its own - so it has to be known before that
    # call is built.
    if alt is False or not config.POST_ALT_ENABLED:
        alt_mode = "off"
    else:
        alt_mode = str(alt or config.POST_ALT_MODE).strip().lower()
        if alt_mode not in ("literal", "llm"):
            warnings.append(f"unknown POST_ALT_MODE {alt_mode!r}; used 'literal'")
            alt_mode = "literal"

    cover = resolved.cover if isinstance(resolved.cover, dict) else {}
    if alt_mode != "off" and not cover:
        # Not a failure: this is what a bare script, or a run directory written
        # before step 5 chose covers, looks like.
        warnings.append(
            "no cover frame in this run's assembly.json, so there is no image "
            "to write alt text for - re-run step 5 to choose one"
        )
        alt_mode = "off"

    alt_depicts = (str(cover.get("depicts") or "")
                   if alt_mode == "llm" and mode == "llm" else "")

    count = int(config.POST_HASHTAG_COUNT if hashtag_count is None
                else hashtag_count)
    if count < 0:
        count = 0
    if count > config.POST_HASHTAG_LIMIT:
        warnings.append(
            f"asked for {count} hashtags; Instagram takes at most "
            f"{config.POST_HASHTAG_LIMIT}"
        )
        count = config.POST_HASHTAG_LIMIT

    identifier = run_id or resolved.run_id or _default_run_id(resolved)
    destination = Path(run_dir) if run_dir else (
        resolved.run_dir or config.video_run_dir(identifier)
    )

    ledger = HashtagLedger()
    recent = ledger.recent(config.POST_HASHTAG_HISTORY_RUNS, exclude=identifier)

    # --- draft the copy ----------------------------------------------------
    rules = _prompt_rules(
        banned_words=config.POST_BANNED_WORDS,
        banned_tags=config.POST_HASHTAGS_BANNED,
        must_include=config.POST_MUST_INCLUDE,
        recent_tags=recent,
        emoji_allowed=config.POST_ALLOW_EMOJI,
        emoji_max=config.POST_MAX_EMOJI,
    )

    draft: Draft | None = None
    if mode == "llm":
        draft, llm_warnings = llm_draft(
            resolved.script, topic=resolved.topic, niche=niche_value,
            count=count, rules=rules, alt_depicts=alt_depicts,
        )
        warnings += llm_warnings
    if draft is None:
        draft = heuristic_draft(resolved.script, topic=resolved.topic,
                                niche=niche_value, count=count)

    # A banned word is worth one more call - the model can usually write around
    # it, and the deterministic scrub below can only delete.
    hits = banned_hits(draft, config.POST_BANNED_WORDS)
    if hits and draft.source == "llm":
        warnings.append(
            f"the copy used banned word(s) {', '.join(hits)}; asked again"
        )
        retry, retry_warnings = llm_draft(
            resolved.script, topic=resolved.topic, niche=niche_value,
            count=count,
            rules=rules + [f"The previous attempt used {', '.join(hits)}. "
                           f"Do not use those words in any form."],
            alt_depicts=alt_depicts,
        )
        warnings += retry_warnings
        if retry is not None and not banned_hits(retry, config.POST_BANNED_WORDS):
            draft = retry
    if banned_hits(draft, config.POST_BANNED_WORDS):
        draft, scrub_warnings = scrub_banned(draft, config.POST_BANNED_WORDS)
        warnings += scrub_warnings

    if draft.source == "heuristic" and mode == "llm":
        warnings.append(
            "this caption came from keyword extraction, not the model - it can "
            "only reuse what the narration already said"
        )

    # Before the hook is fitted, so a hook that loses a sentence here is still
    # held to its limit below. A keyword draft is the narration's own lines,
    # which step 2 already checked, so there is nothing new in it to check.
    claims = {"mode": "off", "checked": False, "dropped": [], "kept": []}
    if draft.source == "llm":
        claims, claim_warnings = check_post_claims(draft, topic=resolved.topic)
        warnings += claim_warnings

    # --- hold it to the platform's constraints -----------------------------
    hook, hook_warnings = fit_hook(draft.hook, config.POST_HOOK_MAX_CHARS)
    warnings += hook_warnings
    if hook and _echoes_script(hook, resolved.script):
        warnings.append(
            "the hook repeats the narration's opening line - the viewer is "
            "hearing that already"
        )
    draft.hook = hook

    fields, emoji_warnings = apply_emoji_budget(
        [draft.hook, *draft.body, draft.cta],
        allowed=config.POST_ALLOW_EMOJI, maximum=config.POST_MAX_EMOJI,
    )
    warnings += emoji_warnings
    draft.hook, draft.cta = fields[0], fields[-1]
    draft.body = fields[1:-1]

    volumes: dict = {}
    if config.TAG_VOLUME_ENABLED:
        offered = [normalise_tag(raw) for tier, _ in HASHTAG_MIX
                   for raw in (draft.hashtags.get(tier) or [])]
        volumes, volume_warnings = tagvolume.lookup(
            [tag for tag in offered if tag]
        )
        warnings += volume_warnings

    rejected_tags: list[dict] = []
    tags, tag_tiers, tag_warnings = select_hashtags(
        draft.hashtags, count=count,
        always=config.POST_HASHTAGS_ALWAYS,
        banned_tags=config.POST_HASHTAGS_BANNED,
        banned_words=config.POST_BANNED_WORDS,
        recent=recent,
        volumes=volumes,
        rejected=rejected_tags,
    )
    warnings += tag_warnings

    credits, credit_warnings = build_credits(
        resolved.credits, attribution_mode,
        header=config.POST_ATTRIBUTION_HEADER,
        maximum=config.POST_ATTRIBUTION_MAX,
    )
    warnings += credit_warnings

    written = " ".join([draft.hook, *draft.body, draft.cta]).lower()
    missing = [phrase for phrase in config.POST_MUST_INCLUDE
               if phrase.lower() not in written]
    if missing:
        warnings.append(
            "appended must-include phrase(s) the copy left out: "
            + "; ".join(missing)
        )

    caption, parts, compose_warnings = compose(
        draft, tags=tags, credits=credits, must_include=missing,
        placement=placement, limit=config.POST_MAX_CHARS,
    )
    warnings += compose_warnings

    tags = parts["hashtags"]
    first_comment = (" ".join("#" + tag for tag in tags)
                     if placement == "comment" and tags else None)

    # --- alt text for the cover --------------------------------------------
    alt_text, alt_report, alt_warnings = "", {
        "mode": "off", "source": "none", "chars": 0,
        "limit": config.POST_ALT_MAX_CHARS, "from_cover": "", "depicts": "",
        "has_card_text": False,
    }, []
    if alt_mode != "off":
        rewritten = draft.alt if alt_mode == "llm" else ""
        if alt_mode == "llm" and not rewritten:
            warnings.append(
                "POST_ALT_MODE=llm but the model returned no alt text; wrote "
                "it from step 5's own words instead"
            )
        alt_text, alt_report, alt_warnings = build_alt_text(
            cover, limit=config.POST_ALT_MAX_CHARS, mode=alt_mode,
            rewritten=rewritten,
        )
        warnings += alt_warnings

    mentions = _MENTION_RE.findall(caption + " " + (first_comment or ""))
    if len(mentions) > MENTION_LIMIT:
        warnings.append(
            f"{len(mentions)} @mentions, over Instagram's limit of "
            f"{MENTION_LIMIT}"
        )

    post = {
        "post_version": POST_VERSION,
        "run_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": resolved.topic,
        "niche": niche_value,
        "mode": draft.source,
        "caption": caption,
        "hook": parts["hook"],
        "body": parts["body"],
        "cta": parts["cta"],
        "must_include": parts["must_include"],
        "hashtags": ["#" + tag for tag in tags],
        "hashtag_tiers": {"#" + tag: tag_tiers.get(tag, "?") for tag in tags},
        "hashtag_volumes": {
            "#" + tag: (volumes.get(tag) or volumes.get(tag.lower())).as_dict()
            for tag in tags
            if volumes.get(tag) or volumes.get(tag.lower())
        },
        "hashtag_rejected": rejected_tags,
        "hashtag_placement": placement,
        "first_comment": first_comment,
        # Goes in Instagram's "Write alt text" field, not the caption. Empty
        # when there is no cover to describe.
        "alt_text": alt_text,
        "alt": alt_report,
        # What the fact-check cut from the copy, and what it flagged but left.
        "claims": claims,
        "attribution": {
            "mode": attribution_mode,
            "assets": credits.assets,
            "required": len(credits.required),
            "block": parts["credits"],
        },
        "chars": {
            "caption": len(caption),
            "limit": config.POST_MAX_CHARS,
            "remaining": config.POST_MAX_CHARS - len(caption),
            "hook": len(parts["hook"]),
            "hook_limit": config.POST_HOOK_MAX_CHARS,
        },
        "counts": {
            "hashtags": len(tags),
            "words": script_gen.count_words(
                " ".join([parts["hook"], *parts["body"], parts["cta"]])
            ),
            "paragraphs": len(parts["body"]),
            "emoji": count_emoji(caption),
            "mentions": len(mentions),
        },
        "source": {
            "script_path": str(resolved.script_path) if resolved.script_path else None,
            "assembly_path": (str(resolved.assembly_path)
                              if resolved.assembly_path else None),
            "script_chars": len(resolved.script),
        },
        "output_dir": str(destination.resolve()) if write else str(destination),
        "post_text_path": None,
        "post_json_path": None,
        "post_comment_path": None,
        "post_alt_path": None,
        "warnings": list(dict.fromkeys(warnings)),
    }

    if not write:
        return post

    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CaptionError(f"could not create {destination}: {exc}") from exc

    text_path = destination / POST_TEXT_NAME
    json_path = destination / POST_JSON_NAME
    if text_path.exists() and not overwrite:
        raise CaptionError(
            f"{text_path} already exists. Pass overwrite=True to replace it."
        )

    post["post_text_path"] = str(text_path.resolve())
    post["post_json_path"] = str(json_path.resolve())

    # post.txt is the caption and nothing else - the trailing newline is the
    # file's, not the caption's. Step 7 should post `caption` from post.json.
    text_path.write_text(caption + "\n", encoding="utf-8")

    comment_path = destination / POST_COMMENT_NAME
    if first_comment:
        comment_path.write_text(first_comment + "\n", encoding="utf-8")
        post["post_comment_path"] = str(comment_path.resolve())
    else:
        # A leftover from a run with a different placement would be posted
        # as a duplicate tag block by step 7.
        comment_path.unlink(missing_ok=True)

    alt_path = destination / POST_ALT_NAME
    if alt_text:
        alt_path.write_text(alt_text + "\n", encoding="utf-8")
        post["post_alt_path"] = str(alt_path.resolve())
    else:
        # Same reasoning as the comment file: a leftover from an earlier run
        # describes an earlier cover, and would be pasted against this one.
        alt_path.unlink(missing_ok=True)

    json_path.write_text(json.dumps(post, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    ledger.record(identifier, tags)
    ledger_warnings = ledger.save()
    if ledger_warnings:
        post["warnings"] = list(dict.fromkeys(post["warnings"] + ledger_warnings))
        json_path.write_text(json.dumps(post, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return post


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _console_safe() -> None:
    """Keep a cp1252 Windows console from dying on one emoji in the copy."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _print_result(post: dict) -> None:
    rule = "-" * 68
    chars, counts = post["chars"], post["counts"]
    print(rule)
    print(f"RUN     : {post['run_id']}")
    print(f"TOPIC   : {post.get('topic') or '(none)'}"
          + (f"   NICHE: {post['niche']}" if post.get("niche") else ""))
    print(f"MODE    : {post['mode']}")
    print(f"LENGTH  : {chars['caption']}/{chars['limit']} chars "
          f"({chars['remaining']} left), hook {chars['hook']}/{chars['hook_limit']}")
    print(f"COPY    : {counts['paragraphs']} paragraph(s), {counts['words']} words, "
          f"{counts['emoji']} emoji")
    print(f"TAGS    : {counts['hashtags']} in the {post['hashtag_placement']}")
    attribution = post["attribution"]
    print(f"CREDITS : {attribution['mode']} - {attribution['required']} required "
          f"of {attribution['assets']} asset(s)")
    alt = post.get("alt") or {}
    if post.get("alt_text"):
        print(f"ALT     : {alt['chars']}/{alt['limit']} chars, "
              f"{alt['mode']} from {alt['source']}")
        print(f"          {post['alt_text']}")
    elif alt.get("mode") not in (None, "off"):
        print("ALT     : none")
    if post["post_text_path"]:
        print(f"WROTE   : {post['post_text_path']}")
        print(f"          {post['post_json_path']}")
        if post.get("post_comment_path"):
            print(f"          {post['post_comment_path']}")
        if post.get("post_alt_path"):
            print(f"          {post['post_alt_path']}")
    else:
        print("WROTE   : nothing (dry run)")

    print(rule)
    print("HOOK  (all Instagram shows before \"... more\")")
    print(f"  {post['hook']}")
    print(rule)
    print(post["caption"])
    if post["first_comment"]:
        print(rule)
        print("FIRST COMMENT")
        print(post["first_comment"])
    print(rule)
    # With volume banding on the tier a tag is filed under is the band it
    # measured at, so printing both would say the same thing twice.
    sizes = post.get("hashtag_volumes") or {}
    print("TAGS BY BAND" if sizes else "TAGS BY TIER")
    for tag, tier in post["hashtag_tiers"].items():
        reading = sizes.get(tag)
        measure = (f"~{reading['count']:>12,}  {reading['source']}"
                   if reading else "")
        print(f"  {tier:<11} {tag:<26}{measure}")

    if post["warnings"]:
        print(rule)
        print("WARNINGS")
        for warning in post["warnings"]:
            print(f"  - {warning}")
    print(rule)


def main(argv: list[str] | None = None) -> int:
    _console_safe()
    parser = argparse.ArgumentParser(
        description="Write the Instagram caption and hashtags for a reel."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-r", "--run-dir", type=Path,
                        help="step 5 run directory (reads its assembly.json "
                             "and writes the post beside reel.mp4)")
    source.add_argument("-f", "--script-file", type=Path,
                        help="script .json from step 2, or a .txt")
    source.add_argument("-s", "--script", help="the narration text itself")

    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="where to write (default: the run directory, else "
                             "output/<date>/<run id>/)")
    parser.add_argument("--run-id", default=None,
                        help="override the run id taken from the source")
    parser.add_argument("--topic", default=None, help="override the topic")
    parser.add_argument("--niche", default=None,
                        help=f"account subject area "
                             f"(default: {config.POST_NICHE or 'from the script'})")
    parser.add_argument("-n", "--hashtags", type=int, default=None,
                        help=f"how many tags (default: {config.POST_HASHTAG_COUNT})")
    parser.add_argument("--placement", default=None, choices=("caption", "comment"),
                        help=f"where the tags go "
                             f"(default: {config.POST_HASHTAG_PLACEMENT})")
    parser.add_argument("--attribution", default=None,
                        choices=("required", "all", "off"),
                        help=f"credit policy (default: {config.POST_ATTRIBUTION})")
    parser.add_argument("--no-attribution", action="store_true",
                        help="shorthand for --attribution off")
    parser.add_argument("--mode", default=None, choices=("llm", "heuristic"),
                        help=f"how the copy is written (default: {config.POST_MODE})")
    parser.add_argument("--alt", default=None, choices=("literal", "llm"),
                        help=f"how the cover's alt text is written "
                             f"(default: {config.POST_ALT_MODE})")
    parser.add_argument("--no-alt", action="store_true",
                        help="do not write alt text for the cover")
    parser.add_argument("--dry-run", action="store_true",
                        help="write nothing to disk")
    parser.add_argument("--json", action="store_true",
                        help="print the post manifest as JSON")
    args = parser.parse_args(argv)

    given = args.run_dir or args.script_file or args.script
    try:
        post = generate_caption(
            given,
            run_dir=args.output_dir,
            run_id=args.run_id,
            topic=args.topic,
            niche=args.niche,
            hashtag_count=args.hashtags,
            placement=args.placement,
            attribution="off" if args.no_attribution else args.attribution,
            mode=args.mode,
            alt=False if args.no_alt else args.alt,
            write=not args.dry_run,
        )
    except (CaptionError, config.MissingConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(post, indent=2, ensure_ascii=False))
    else:
        _print_result(post)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
