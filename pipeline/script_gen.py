"""Step 2 - narration script generation via the Gemini API.

Takes a topic string and returns a narration script written for text-to-speech:
no stage directions, no emoji, no markdown - just the words to be spoken.

The opening sentence gets a second pass of its own. A reel is decided inside
its first second and a half, so the first line is not left to whatever the
scriptwriter happened to produce: HOOK_MODE=llm writes several openings, scores
them here in code against the one already there, and promotes a winner only if
it actually beats it. The chosen hook also yields a short on-screen form -
`hook_card` - which step 5 burns over the opening of the reel.

The sentences after it get a pass of their own too. A reel is decided in its
first second and a half and *lost* over the thirty after it, so STRUCTURE_MODE
measures the shape of the whole script - whether the close pays the opening off
instead of repeating it, whether anything turns in the middle, whether the last
line is short enough to land - and buys one rewrite for a script that has none.
Both passes score in code, and both keep the incumbent on a tie.

The script is written to output/scripts/ as a .txt (the words alone, which is
what step 3 speaks) and a .json sidecar (the topic and timings, which is what
step 6 reads to know what the reel is about).

Standalone use:
    python pipeline/script_gen.py "why the ocean is salty"
    python pipeline/script_gen.py "the history of coffee" --duration 30
    python pipeline/script_gen.py "the history of coffee" -o output/scripts/coffee.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

# Allow both `python pipeline/script_gen.py` and `python -m pipeline.script_gen`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables - tweak these to change output style without touching the logic.
# ---------------------------------------------------------------------------

#: Average speaking rate used to convert word count <-> seconds.
#: Spoken words per minute, as actually delivered - the base rate multiplied by
#: the speed step 3 will read at. Derived rather than written down twice: they
#: are the same quantity, and a pipeline that speeds the narration up without
#: lengthening the script makes every reel short of the length it was asked
#: for.
WORDS_PER_MINUTE = round(config.SCRIPT_WORDS_PER_MINUTE * config.TTS_SPEED)

#: How far either side of that a script may land and still be the length it was
#: asked for. Both the scriptwriter prompt and the structure pass's rewrite
#: budget are stated in these terms, so a rewrite is held to the brief the
#: original was written to rather than to one of its own.
WORD_BAND_LOW = 0.85
WORD_BAND_HIGH = 1.15

#: Gemini sampling temperature. Higher = more varied phrasing.
TEMPERATURE = 0.95

#: Ceiling on response length. Generous - scripts are short.
MAX_OUTPUT_TOKENS = 2048

#: Thinking budget for 2.5 models. 0 disables it (faster, fine for short copy).
#: Skipped for `pro` models, which cannot disable thinking.
THINKING_BUDGET = 0

#: Accepted duration range, in seconds.
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 180

# --- retry ------------------------------------------------------------------
# `call_gemini` below is the one Gemini call the whole project shares, so a
# blip here does more than fail step 2: step 4 quietly drops to keyword search
# queries and step 6 to heuristic copy, and the reel comes out worse with only
# a warning line to say why. Cheap retries buy all three steps their good path.

#: Attempts per call, including the first.
MAX_ATTEMPTS = 3

#: Exponential backoff between attempts: 1s, then 2s.
RETRY_BASE_SECONDS = 1.0

#: Worth trying again: rate limits, and the 500/503s flash models return when
#: the service is under load. Everything else - 400 malformed request, 401 bad
#: key, 404 unknown model - fails identically on a second attempt.
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


SYSTEM_INSTRUCTION = """\
You are a scriptwriter for short-form vertical video (Instagram Reels, \
TikTok, YouTube Shorts). You write narration that is spoken aloud by a \
voiceover artist, never read on screen.

Your writing is punchy, concrete and conversational. You use short sentences. \
You favour vivid specifics over generalities, and you never pad."""


PROMPT_TEMPLATE = """\
Write a voiceover narration script about: {topic}

The video is {duration_seconds} seconds long, so the script must be roughly \
{target_words} words (between {min_words} and {max_words} words).

Rules:
- Open with a hook that lands in the first 1-2 seconds. Make the viewer need \
to hear the next sentence. No throat-clearing, no "in this video", no \
"have you ever wondered".
- The opening must leave a question open - stated or implied - that the script \
does not answer until its final line. Do not answer it early.
- Turn somewhere in the middle: a point where what the viewer assumed stops \
being true. "But", "except", "the part nobody mentions".
- Write for the ear, not the eye. Short sentences. Natural spoken rhythm.
- Be specific. Use real details, numbers or images rather than vague claims.
- Never stop to ask a question you are about to answer yourself.
- Every claim must survive the evidence you give for it. If the facts you cite do not support the line you opened with, change the opening rather than asserting past them - a viewer who hears the numbers contradict the claim stops trusting the rest.
- Say a thing once. Never follow a sentence with the same sentence negated ("they don't sleep less than us. We sleep less than them"): heard once, at speed, that reads as a contradiction and the viewer stops to work out which way round it goes.
- End on a short line that answers the opening in DIFFERENT words - not a \
restatement of it, not a summary of the script. It should run back into the \
opening line closely enough that a viewer watching twice hears the two as \
continuous. Do not ask the viewer to like, follow or comment.

Output format - this matters, the text goes straight into a text-to-speech engine:
- Return ONLY the words to be spoken.
- No title, no headings, no labels such as "Hook:" or "Narrator:".
- No stage directions, no shot descriptions, no bracketed cues, no timestamps.
- No emoji, no markdown, no asterisks, no quotation marks around the script.
- Write numbers and symbols the way they should be said aloud."""


def build_prompt(topic: str, duration_seconds: int) -> str:
    """Render the user-facing prompt for a topic and target duration."""
    minimum, maximum = word_band(duration_seconds)
    return PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        duration_seconds=duration_seconds,
        target_words=words_for_duration(duration_seconds),
        min_words=minimum,
        max_words=maximum,
    )


class ScriptGenerationError(RuntimeError):
    """Raised when Gemini fails, refuses, or returns an unusable script."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def target_duration(duration_seconds: int | None = None) -> int:
    """The length to write to: what was asked for, or config's default.

    None means nobody had an opinion. That is the case worth naming, because it
    used to resolve to a hardcoded 45 - a number no topic had argued for, long
    enough that a one-fact reel spent its last fifteen seconds restating
    itself. Step 0 now suggests a length per topic and passes it in `-d`; this
    is what happens when step 2 is called without one.
    """
    return (config.SCRIPT_DEFAULT_SECONDS if duration_seconds is None
            else duration_seconds)


def words_for_duration(duration_seconds: int) -> int:
    """How many spoken words fit into `duration_seconds` at WORDS_PER_MINUTE."""
    return max(1, round(duration_seconds * WORDS_PER_MINUTE / 60))


def word_band(duration_seconds: int) -> tuple[int, int]:
    """The word count a script of this length may come back at, low to high."""
    target = words_for_duration(duration_seconds)
    return max(1, int(target * WORD_BAND_LOW)), int(target * WORD_BAND_HIGH)


def count_words(text: str) -> int:
    return len(text.split())


def estimate_duration(text: str) -> int:
    """Rough spoken length of `text`, in seconds, at WORDS_PER_MINUTE."""
    return round(count_words(text) * 60 / WORDS_PER_MINUTE)


_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # pictographs, emoticons, symbols
    "\U00002600-\U000027bf"  # misc symbols & dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "\U00002190-\U000021ff"  # arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U00002b00-\U00002bff"
    "]+",
    flags=re.UNICODE,
)

#: Labels a model sometimes prefixes to a line despite instructions.
_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?(?:script|narration|voiceover|vo|narrator|hook|opening)"
    r"(?:\s*\(.*?\))?(?:\*\*)?\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


def clean_script(raw: str) -> str:
    """Strip anything that would be spoken aloud by mistake.

    Removes code fences, bracketed stage directions, markdown emphasis, emoji
    and leading speaker labels, then normalises whitespace. Parentheses are
    left alone - they are more often part of a real sentence than a cue.
    """
    text = raw.strip()

    # ```...``` fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # [stage direction], [0:05], [upbeat music]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # markdown emphasis and headings
    text = re.sub(r"[*_`#]+", "", text)

    text = _EMOJI_RE.sub("", text)

    # per-line labels ("Narrator:", "Hook -")
    text = "\n".join(_LABEL_RE.sub("", line) for line in text.splitlines())

    # a wrapping pair of quotes around the whole script
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()

    # collapse runs of spaces, and blank-line runs down to a single break
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def _extract_text(response) -> str:
    """Pull the script text out of a Gemini response, or explain why it's absent."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        raise ScriptGenerationError(
            f"Gemini blocked the prompt (block_reason={block_reason}). "
            "Try rewording the topic."
        )

    text = getattr(response, "text", None)
    if text and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ScriptGenerationError("Gemini returned no candidates.")

    finish_reason = getattr(candidates[0], "finish_reason", None)
    finish_message = getattr(candidates[0], "finish_message", None)
    detail = f" ({finish_message})" if finish_message else ""
    raise ScriptGenerationError(
        f"Gemini returned an empty script (finish_reason={finish_reason}){detail}."
    )


def _build_generation_config(model: str, *, system_instruction: str,
                             temperature: float, max_output_tokens: int,
                             thinking_budget: int | None, json_mode: bool):
    """Assemble GenerateContentConfig, adapting to the model family."""
    from google.genai import types

    kwargs = {
        "system_instruction": system_instruction,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    # Only 2.5 flash / flash-lite accept a zero thinking budget; pro rejects it.
    if thinking_budget is not None and "pro" not in model.lower():
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kwargs)


def _is_retryable(exc: Exception) -> bool:
    """True when the identical call could plausibly succeed on a second try."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return getattr(exc, "code", None) in RETRY_STATUS_CODES

    # Transport-level failures - connect timeouts, dropped reads, protocol
    # errors. google-genai speaks HTTP over httpx, so they arrive as its
    # exceptions rather than the SDK's own.
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TransportError)


def _as_generation_error(exc: Exception, model: str) -> ScriptGenerationError:
    """Wrap whatever the SDK raised, keeping the provider's own message."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None) or getattr(exc, "status", "?")
        return ScriptGenerationError(
            f"Gemini API call failed [{code}] for model {model!r}: {exc}"
        )
    return ScriptGenerationError(
        f"Gemini API call failed for model {model!r}: "
        f"{type(exc).__name__}: {exc}"
    )


def call_gemini(prompt: str, *, system_instruction: str,
                temperature: float = TEMPERATURE,
                max_output_tokens: int = MAX_OUTPUT_TOKENS,
                thinking_budget: int | None = THINKING_BUDGET,
                json_mode: bool = False,
                model: str | None = None,
                api_key: str | None = None) -> str:
    """One Gemini call, and the only place the SDK is spoken to.

    Later steps that need the model (step 6's caption copy) come through here
    rather than building a second client: the key lookup, the model-family
    quirks, the failure-to-exception mapping and the retries stay in one place.

    Rate limits and transient server errors are retried up to MAX_ATTEMPTS
    times with exponential backoff; a refusal or a bad key is raised at once.

    Args:
        prompt: the user turn.
        system_instruction: who the model is being asked to be.
        json_mode: ask for `application/json` back. The response is still text -
            the caller parses it, and should not assume it is well formed.
        model / api_key: default to config.GEMINI_MODEL and the .env key.

    Raises:
        MissingConfigError: no GEMINI_API_KEY and none passed.
        ScriptGenerationError: the call failed or came back empty.
    """
    model = model or config.GEMINI_MODEL
    api_key = api_key or config.require(*config.REQUIRED_SCRIPT_GEN)["GEMINI_API_KEY"]

    from google import genai

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=_build_generation_config(
                    model,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=thinking_budget,
                    json_mode=json_mode,
                ),
            )
        except Exception as exc:  # API errors, network errors, SDK issues
            if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                raise _as_generation_error(exc, model) from exc
            delay = RETRY_BASE_SECONDS * 2 ** (attempt - 1)
            # stderr, so a caller piping the script to a file still gets it
            print(
                f"gemini: {type(exc).__name__} - retrying in {delay:g}s "
                f"(attempt {attempt + 1} of {MAX_ATTEMPTS})",
                file=sys.stderr,
            )
            time.sleep(delay)
        else:
            # A blocked prompt or an empty candidate is settled, not transient:
            # _extract_text raises from out here so it is never retried.
            return _extract_text(response)

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# The hook
#
# One extra Gemini call writes several openings; everything that decides
# between them happens here, in code. That split is deliberate, and it is the
# same one step 6 makes with its character limits: a model asked to rank its
# own output will happily explain why all five are excellent. A scoring
# function will not, it is the same function every run, and it can say what it
# did and why.
# ---------------------------------------------------------------------------

_HOOK_SYSTEM_INSTRUCTION = """\
You write opening lines for short-form vertical video.

An opening line has one job: make the next sentence unskippable. You write \
them concrete, specific and short. You never warm up, never announce what the \
video is about, and never ask a question the viewer can answer with "no".

You always return valid JSON and nothing else."""


_HOOK_PROMPT_TEMPLATE = """\
Topic of the video: {topic}

This is how the narration currently opens:
"{original}"

This is the rest of the narration, which must follow on from whatever you write:
{rest}

Write {count} DIFFERENT replacement opening lines. Each one has to lead \
naturally into the rest of the narration above.

Rules for every line:
- One sentence. At most {max_words} words.
- It is spoken aloud, so write it for the ear.
- Be concrete: a number, an image, a named thing, a claim that sounds wrong \
until it is explained. Something the viewer can picture or check.
- No throat-clearing. Never open with "in this video", "have you ever", \
"did you know", "let me tell you", "imagine if", "welcome to" or "today".
- No hype adjectives: amazing, incredible, mind-blowing, insane, crazy.
- Make the {count} lines genuinely different from each other - different \
angles on the same material, not one sentence reworded {count} times.

For each line also write "card": the same hook cut down to at most \
{card_words} words, to be shown on screen as large text. It must still read \
as a finished thought, not as a truncated sentence.

And write "bridge": the narration's next sentence, rewritten so it follows \
from YOUR opening instead of the one it was written for. This is the sentence \
immediately after the opening above:
"{second}"
Rewrite it if your line makes it repeat itself, answer itself, or read as a \
non-sequitur. Return "" for bridge if it already follows on cleanly. Keep any \
fact it carries - you are re-joining it, not cutting it.

Return ONLY a JSON array, {count} objects long:
[{{"hook": "...", "card": "...", "bridge": "..."}}]"""


#: Openings that spend the only second that mattered saying nothing. Matched
#: against the start of the line, lowercased.
_HOOK_THROAT_CLEARING = (
    "in this video", "have you ever", "did you know", "let me tell you",
    "today we", "today i", "welcome to", "i want to talk", "this is a video",
    "let's talk about", "lets talk about", "have you wondered",
    "ever wondered", "in today's", "in todays", "here's why", "heres why",
    "imagine if", "picture this",
)

#: Words that sound like meaning without carrying any. A hook built out of
#: these is a hook about nothing.
_HOOK_VAGUE = frozenset({
    "amazing", "incredible", "unbelievable", "mindblowing", "insane", "crazy",
    "journey", "unlock", "unleash", "innovation", "revolutionary",
    "gamechanger", "ultimate", "essential", "powerful", "transform",
    "fascinating", "interesting", "important", "epic", "wild", "stunning",
})

#: Openers that put something at stake inside the first two words.
_HOOK_STRONG_OPENERS = (
    "nobody", "no one", "everyone", "everything", "nothing", "most people",
    "you", "your", "there is", "there are", "it took", "every", "scientists",
    "we", "they", "this",
)

_HOOK_NUMBER_RE = re.compile(r"\b\d")
_HOOK_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")

#: A sentence carrying a negation. Contractions are matched whole because the
#: word pattern keeps them whole - "don't" arrives as one token, not two - and
#: both the straight and the curly apostrophe have to be allowed for, because
#: the scriptwriter returns whichever the model felt like typing.
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|neither|nor|cannot|"
    r"\w+n['’]t)\b",
    re.IGNORECASE,
)

#: A closing sentence this short, after a longer one, is a turn rather than a
#: second line: "It doesn't.", "Here's why.", "Nobody knows."
_HOOK_TURN_MAX_WORDS = 4
_HOOK_SENTENCE_RE = re.compile(r"[.!?…]+")

#: How much of a body sentence the hook can already have said before the hook
#: is spending the opening twice.
#:
#: Measured over the twelve scripts on disk. The deep-ocean reel opens "95% of
#: the biggest living space on Earth is unexplored." and then, one sentence
#: later, "Ninety-five percent of it is unexplored." - the same proposition,
#: respelled, as its own standalone sentence. It scores 0.67 here. The next
#: highest script scores 0.33 and has no restatement in it, so the line is
#: drawn halfway between the one real fault and the worst innocent case rather
#: than tuned onto either.
_HOOK_ECHO_ABOVE = 0.50

#: How far into the body a restatement still counts as one. Past the third
#: sentence the hook is being picked up again deliberately, which is a
#: callback, not a stutter.
_HOOK_ECHO_SENTENCES = 3

#: A body sentence shorter than this is emphasis, not restatement. The Inca
#: reel opens "The Inca built 100,000 miles of road without the wheel." and
#: follows it with "Without the wheel." - a three-word fragment repeating the
#: phrase on purpose, which is the strongest thing in that script. It is the
#: same line `_HOOK_TURN_MAX_WORDS` draws one rule above, for the same reason.
_HOOK_ECHO_MIN_WORDS = 4

#: Both sides need this many content words before their ratio means anything.
#: `_overlap` divides by the smaller set, so a two-word sentence sharing one
#: word scores 0.50 on its own - which is how "That's the pressure down there."
#: reached the threshold without restating anything.
_HOOK_ECHO_MIN_CONTENT = 3

#: Punctuation that ends a clause - somewhere a hook card can stop and still
#: sound like a finished thought.
_HOOK_CLAUSE_END = ",;:—–"


def _echoes_body(hook: str, body: str) -> str:
    """The sentence near the top of `body` that restates `hook`, or "".

    The defect this exists for, from the shipped deep-ocean reel:

        95% of the biggest living space on Earth is unexplored.
        It's our deep ocean.
        Ninety-five percent of it is unexplored.

    The hook pass replaced sentence one and left sentence three sitting there
    saying it again, because nothing in `score_hook` had ever been shown the
    script - it scored openings as free-standing sentences, and an opening is
    not a free-standing sentence, it is the first half of a join.

    Only the first `_HOOK_ECHO_SENTENCES` are read, only sentences long enough
    to be restatements rather than emphasis are considered, and both sides need
    `_HOOK_ECHO_MIN_CONTENT` content words before their ratio is believed. See
    those constants for the measurements behind each.
    """
    words = _content_words(hook)
    if len(words) < _HOOK_ECHO_MIN_CONTENT:
        return ""

    for sentence in split_sentences(body)[:_HOOK_ECHO_SENTENCES]:
        if len(_HOOK_WORD_RE.findall(sentence)) < _HOOK_ECHO_MIN_WORDS:
            continue
        theirs = _content_words(sentence)
        if len(theirs) < _HOOK_ECHO_MIN_CONTENT:
            continue
        if _overlap(words, theirs) >= _HOOK_ECHO_ABOVE:
            return sentence
    return ""


def score_hook(text: str, *, max_words: int | None = None,
               body: str = "") -> dict:
    """Score one candidate opening. Higher is better.

    Returns the score and the parts it was built from, so a run can say why the
    line it chose beat the four it did not. Nothing here is a hard filter: a
    candidate that breaks every rule still gets a number, because a run where
    every candidate is poor still has to open with one of them.

    `body` is the narration this hook would be fixed to - the script minus its
    opening sentence. Passing it turns on the one rule that cannot be read off
    the hook alone: whether the line is about to be said twice. Everything else
    scores a hook as a sentence; this scores it as a *join*, which is where the
    pass was doing its damage, because `apply_hook` swaps sentence one and
    leaves sentence two exactly as it was.
    """
    limit = config.HOOK_MAX_WORDS if max_words is None else max_words
    clean = (text or "").strip()
    lowered = clean.lower()
    words = _HOOK_WORD_RE.findall(clean)
    parts: dict[str, float] = {}

    if not words:
        return {"score": -99.0, "components": {"empty": -99.0}}

    # Length. 6-12 words is an opening line; below that it is usually a
    # fragment, and above it stops being an opening and becomes a paragraph.
    #
    # Worth 0.4 rather than the 1.5 it used to be, because it is hygiene and
    # not a virtue. Measured across the 55 candidates this pass has generated,
    # 80% of them land in the band and 92% of the winners did - the prompt asks
    # for one short sentence, so almost everything arrives short. Paying 1.5
    # for it lifted the whole field by the same amount and left the genuinely
    # discriminating terms fighting over the remainder.
    count = len(words)
    if count < 4:
        parts["too short"] = -1.5
    elif 6 <= count <= 12:
        parts["length"] = 0.4
    if count > limit:
        parts["over the word ceiling"] = round(-0.6 * (count - limit), 3)

    # Sentence count, charged only where it is evidence of something.
    #
    # This used to be a flat -2.0 per extra sentence, on the reasoning that a
    # second sentence means the hook has eaten the line after it. That is a
    # real failure, but it is a failure of *length*, and length is already
    # charged for above - so the rule was billing twice for one fault and
    # taking the setup-then-turn down with it. "The deep sea should crush
    # every creature flat. It doesn't. Here's why." scored -4 here and came
    # last of five; "Your six hours of sleep? That's barely half what a
    # gorilla gets." lost to a flat statement of fact. Both are openings a
    # reel would be lucky to have.
    #
    # So extra sentences are charged when the hook is over the ceiling - which
    # is what having eaten the next line actually looks like - and a third
    # sentence is charged a little regardless, because that is a paragraph.
    sentences = [s.strip() for s in _HOOK_SENTENCE_RE.split(clean) if s.strip()]
    extra = len(sentences) - 1
    if extra > 0 and count > limit:
        parts["more than one sentence"] = -2.0 * extra
    elif extra > 1:
        parts["more than two sentences"] = round(-0.7 * (extra - 1), 3)

    # A short sentence knocking down the one before it is the turn itself:
    # the setup stated flat, then taken away in three words. Scored for the
    # same reason `a turn mid-reel` is scored in the structure pass - it is
    # the shape that stops a scroll.
    #
    # The negation is what makes it a turn rather than a trailing clause. "It
    # doesn't" reverses the sentence before it; "It has for ages" is the same
    # length and the same position and only continues it, which is why length
    # and position alone are not enough to tell them apart.
    for sentence in sentences[1:]:
        length = len(_HOOK_WORD_RE.findall(sentence))
        if 0 < length <= _HOOK_TURN_MAX_WORDS and (count - length) >= length:
            if _NEGATION_RE.search(sentence):
                parts["turns on itself"] = 0.8
                break

    if any(lowered.startswith(opener) for opener in _HOOK_THROAT_CLEARING):
        parts["throat-clearing"] = -4.0

    vague = sum(1 for word in words if word.lower().strip("-'’") in _HOOK_VAGUE)
    if vague:
        parts["vague words"] = round(-1.2 * vague, 3)

    # A number, worth 0.8 rather than 1.5: a real virtue, and now priced like
    # the other real virtues instead of above them. At 1.5 it was the single
    # largest discriminating term in the scorer, and it showed - a digit
    # appears in 24% of candidates but in 42% of the winners. On the deep-ocean
    # reel it bought "95% of the biggest living space on Earth is unexplored."
    # a win over "We know more about the surface of Mars than we do this
    # place.", which is the better line by any reading and lost only for having
    # no digit in it.
    if _HOOK_NUMBER_RE.search(clean):
        parts["a number"] = 0.8

    # Direct address. "You" makes the line about the viewer rather than about
    # the subject, which is most of what stops a scroll.
    if any(word.lower() in ("you", "your", "you're", "youre") for word in words):
        parts["second person"] = 1.0

    if "throat-clearing" not in parts:
        if any(lowered.startswith(opener + " ") for opener in _HOOK_STRONG_OPENERS):
            parts["strong opener"] = 0.8

    # A capitalised word that does not open a sentence is a named thing - a
    # place, a person, a product. Specific by construction.
    #
    # Sentence-initial words are excluded rather than just the very first word
    # in the line, because a two-sentence hook capitalises the word after the
    # full stop too: "Rain is the reason your ocean tastes of salt. It has for
    # ages." was scoring for naming something, and the something was "It".
    named = False
    for sentence in sentences:
        tokens = _HOOK_WORD_RE.findall(sentence)
        if any(token[:1].isupper() for token in tokens[1:]):
            named = True
            break
    if named:
        parts["names something"] = 0.7

    # A question is fine, but only the kind that cannot be shrugged off. The
    # throat-clearing list has already caught "have you ever ..."; what is left
    # is usually a real one.
    if clean.endswith("?") and count <= limit:
        parts["asks something"] = 0.3

    # Whether the body is about to say this again. Charged -1.5: the fault
    # spends two of a forty-second reel's seconds saying nothing new, and the
    # charge has to be big enough to actually move a ranking, which at these
    # weights means bigger than any single virtue and smaller than
    # throat-clearing.
    echoed = _echoes_body(clean, body)
    if echoed:
        parts["the body says it again"] = -1.5

    return {"score": round(sum(parts.values()), 3), "components": parts}


def hook_card_text(hook: str, max_words: int | None = None) -> str:
    """The on-screen short form of a spoken hook.

    Cut at the last clause break that fits, because a clause is a finished
    thought and reads better than the same words guillotined; cut at the word
    limit otherwise. Used both when the model supplied no card of its own and
    to hold the one it did supply to the limit.
    """
    limit = max(1, config.HOOK_CARD_MAX_WORDS if max_words is None else max_words)
    clean = re.sub(r"\s+", " ", (hook or "").strip())
    if not clean:
        return ""

    words = clean.split()
    if len(words) <= limit:
        return clean.rstrip(_HOOK_CLAUSE_END + ". ")

    # Prefer to stop where the sentence itself pauses, taking the longest such
    # stop that still fits.
    for position in range(limit, 1, -1):
        if words[position - 1][-1:] in _HOOK_CLAUSE_END:
            return " ".join(words[:position]).rstrip(_HOOK_CLAUSE_END + " ")

    return " ".join(words[:limit]).rstrip(_HOOK_CLAUSE_END + ". ")


def split_first_sentence(script: str) -> tuple[str, str]:
    """Split a script into (first sentence, everything after it).

    A script with no sentence ending at all is treated as all opening and no
    body, which is what a one-line script is.
    """
    text = (script or "").strip()
    if not text:
        return "", ""

    for match in re.finditer(r"[.!?…]+[\"'’)\]]*(?:\s+|$)", text):
        head = text[:match.end()].strip()
        # A full stop three characters in is an abbreviation, not an ending.
        if len(_HOOK_WORD_RE.findall(head)) >= 3:
            return head, text[match.end():].strip()
    return text, ""


def _parse_hook_json(raw: str) -> list[dict]:
    """Read the model's JSON array of openings, tolerantly."""
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

    parsed: list[dict] = []
    for entry in data:
        if isinstance(entry, str):
            entry = {"hook": entry}
        if not isinstance(entry, dict):
            continue
        # clean_script is the same scrubber the narration goes through - these
        # lines end up spoken by the same engine, so they get the same pass.
        hook = " ".join(clean_script(str(entry.get("hook") or "")).split())
        if hook:
            parsed.append({
                "hook": hook,
                "card": " ".join(clean_script(str(entry.get("card") or "")).split()),
                "bridge": " ".join(
                    clean_script(str(entry.get("bridge") or "")).split()
                ),
            })
    return parsed


#: A bridge may not run away with the script. It replaces one sentence of
#: narration, and a much longer one changes the timing step 5 plans against.
_BRIDGE_MAX_GROWTH = 2.0

_DIGITS_RE = re.compile(r"\d+")


def _usable_bridge(bridge: str, second: str) -> str:
    """The model's rewrite of sentence two, or "" to keep the one already there.

    Every rejection falls back to the original sentence, which is why nothing
    here needs to warn: the worst case is the seam this pass was trying to
    repair, which is exactly where the pipeline stood before it existed.

    Three checks, in the order they are cheap:

    - Nothing offered, or the same sentence returned. The prompt asks for ""
      when the line already follows on, and most of the time it should.
    - It grew. A rewrite is a re-join, not a second draft, and a sentence twice
      the length is the model continuing the script instead of mending it.
    - It dropped a number. "Keep any fact it carries" is asked for in the
      prompt and therefore checked here: a digit in the original that is not in
      the rewrite is a fact deleted on the way through, which is the one
      failure that would be invisible in the finished reel.
    """
    candidate = " ".join((bridge or "").split())
    original = " ".join((second or "").split())
    if not candidate or not original:
        return ""
    if candidate.casefold() == original.casefold():
        return ""

    grown = len(_HOOK_WORD_RE.findall(candidate))
    was = len(_HOOK_WORD_RE.findall(original))
    if was and grown > was * _BRIDGE_MAX_GROWTH:
        return ""

    if set(_DIGITS_RE.findall(original)) - set(_DIGITS_RE.findall(candidate)):
        return ""
    return candidate


def generate_hooks(topic: str, script: str, *, count: int | None = None) -> dict:
    """Write and score replacement openings for `script`. Never raises.

    Returns {"candidates": [...], "warnings": [...]} with candidates best
    first. Every failure path - no key, a refused call, unparseable JSON -
    comes back as an empty candidate list and a warning, because a reel that
    opens with the scriptwriter's own line is worth immeasurably more than no
    reel at all.
    """
    wanted = max(1, config.HOOK_VARIANTS if count is None else count)
    original, rest = split_first_sentence(script)
    if not original:
        return {"candidates": [],
                "warnings": ["the script is empty - no hook was written"]}

    second, _ = split_first_sentence(rest)
    prompt = _HOOK_PROMPT_TEMPLATE.format(
        topic=topic or "(not given)",
        original=original,
        rest=rest or "(the opening is the whole script)",
        second=second or "(there is no sentence after the opening)",
        count=wanted,
        max_words=config.HOOK_MAX_WORDS,
        card_words=config.HOOK_CARD_MAX_WORDS,
    )

    try:
        raw = call_gemini(
            prompt,
            system_instruction=_HOOK_SYSTEM_INSTRUCTION,
            json_mode=True,
        )
        entries = _parse_hook_json(raw)
    except (ScriptGenerationError, config.MissingConfigError, ValueError) as exc:
        return {"candidates": [],
                "warnings": [f"the hook pass failed ({exc}). "
                             f"Kept the scriptwriter's opening."]}

    if not entries:
        return {"candidates": [],
                "warnings": ["the hook pass returned no usable openings. "
                             "Kept the scriptwriter's opening."]}

    candidates = []
    for entry in entries:
        # Scored against the body it would be joined to, not in the abstract.
        scored = score_hook(entry["hook"], body=rest)
        candidates.append({
            "text": entry["hook"],
            "card": hook_card_text(entry["card"] or entry["hook"]),
            "bridge": _usable_bridge(entry.get("bridge", ""), second),
            "score": scored["score"],
            "components": scored["components"],
        })
    # Ties break towards the shorter line: same score, less to sit through.
    candidates.sort(key=lambda c: (-c["score"], len(c["text"])))
    return {"candidates": candidates, "warnings": []}


def _separator_after(text: str, head: str) -> str:
    """The whitespace `text` puts between its first sentence and the rest.

    A script written in paragraphs stays in paragraphs, which is a shape step
    4 reads when it cuts the script into beats.
    """
    tail = text.strip()[len(head):]
    return "\n\n" if tail.startswith("\n\n") else (
        "\n" if tail.startswith("\n") else " "
    )


def apply_hook(script: str, hook: str, bridge: str = "") -> str:
    """Put `hook` at the top of `script` in place of its first sentence.

    `bridge` additionally replaces sentence *two*. Swapping only sentence one
    is what this pass used to do, and it is how a well-scored opening still
    produced a bad script: the line after it had been written to follow the
    sentence that was just deleted. The shipped salt reel opened "The ocean
    contains 50 quadrillion tons of salt." and went on "Especially when all
    the water that flows into it is fresh." - a subordinate clause with
    nothing left to attach to.
    """
    original, rest = split_first_sentence(script)
    opening = " ".join((hook or "").split())
    if not opening:
        return script
    if not rest:
        return opening

    mended = " ".join((bridge or "").split())
    if mended:
        second, remainder = split_first_sentence(rest)
        if second:
            rest = (f"{mended}{_separator_after(rest, second)}{remainder}"
                    if remainder else mended)

    return f"{opening}{_separator_after(script, original)}{rest}"


def choose_hook(topic: str, script: str, *, mode: str | None = None) -> dict:
    """Run the hook pass and report what it decided. Never raises.

    The scriptwriter's own opening is scored on the same scale as the
    candidates and stands as the incumbent, so a generated line has to actually
    beat it. A weak batch therefore leaves the script alone rather than making
    it worse, which is the whole reason the scoring lives in code.
    """
    setting = (mode or config.HOOK_MODE or "off").strip().lower()
    original, rest = split_first_sentence(script)
    # The incumbent is scored against the same body on the same scale. It is
    # not exempt from the restatement rule: a scriptwriter that opened on a
    # line its own next paragraph repeats has made the same mistake, and the
    # whole point of the incumbent is that it competes honestly.
    incumbent = score_hook(original, body=rest)
    report = {
        "mode": setting,
        "original": original,
        "original_score": incumbent["score"],
        "chosen": None,
        "card": hook_card_text(original),
        "bridge": "",
        "replaced": False,
        "candidates": [],
        "warnings": [],
    }

    if setting != "llm":
        return report

    result = generate_hooks(topic, script)
    report["candidates"] = result["candidates"]
    report["warnings"] = list(result["warnings"])
    if not result["candidates"]:
        report["mode"] = "failed"
        return report

    best = result["candidates"][0]
    if best["score"] <= incumbent["score"]:
        report["warnings"].append(
            f"none of the {len(result['candidates'])} generated openings beat "
            f"the scriptwriter's own (scored {incumbent['score']:g}) - kept it"
        )
        return report

    report["chosen"] = best["text"]
    report["card"] = best["card"] or hook_card_text(best["text"])
    report["bridge"] = best.get("bridge", "")
    report["replaced"] = True
    return report


# ---------------------------------------------------------------------------
# The structure
#
# The hook pass fixes sentence one. Nothing looked at sentences two onward, and
# it showed. A shipped reel about deep-sea pressure opened on "Snailfish live
# 11 kilometres down" and closed on "Some snailfish live eleven kilometres
# down" - the ending restating the opening as a fact rather than paying it off,
# so thirty seconds of watching arrived back where it started. The same script
# handed the viewer "So how do fish survive?" at sentence four, which is the
# throat-clearing `score_hook` charges -4 for at position 0 and never sees at
# position 12.
#
# So three things are asked for in the prompt and - because asking is not
# checking - measured here afterwards: the close returns to what the opening
# opened without simply repeating it, there is a turn somewhere in the middle,
# and the last line is short enough to land. A script scoring below
# STRUCTURE_MIN_SCORE buys one rewrite of its body, and that rewrite is kept
# only if it scores better. That is the hook pass's incumbent rule, for the
# hook pass's reason: a pass that can only improve a script can never be the
# reason there is no reel.
# ---------------------------------------------------------------------------

#: Sampling temperature for the rewrite. Below the scriptwriter's own, because
#: this call is not looking for new material - the facts are settled, and what
#: is wanted is the same ones in a better order.
STRUCTURE_TEMPERATURE = 0.8

#: One sentence: anything up to a terminator, plus whatever closing quotes or
#: brackets ride on the end of it, plus a trailing fragment that never got a
#: full stop. Sentences are *matched* rather than split apart, because a split
#: pattern wide enough to cover the closing quote consumes it - `he called it
#: "the salt trap."` would come back with the quote deleted.
_SENTENCE_RE = re.compile(r"[^.!?…]*[.!?…]+[\"'’)\]]*|[^.!?…]+$")

#: Fewer words than this and the "sentence" is an abbreviation that happened to
#: end in a full stop, so it is folded back into its neighbour. The same call
#: `split_first_sentence` makes, for the same reason - and with the same trade:
#: a genuine one-word sentence is folded in with it.
_SENTENCE_MIN_WORDS = 3

#: A script shorter than this has no middle to put a turn in, and rewriting it
#: for shape is rewriting it for nothing.
_STRUCTURE_MIN_SENTENCES = 4

#: How much of the end counts as the close. A reel is paid off in its last
#: breath or two, not its last clause - measuring loop closure against the
#: final sentence alone misses the script that restates its opening and then
#: adds a line of summary after it, which is the common shape of the failure.
_STRUCTURE_CLOSE_SENTENCES = 3

#: ...but never more than this share of the whole script, so the rule scales
#: down to a short one instead of calling half of it "the close".
_STRUCTURE_CLOSE_MAX_FRACTION = 1 / 3

#: Below this overlap between the opening and the close, the ending is about
#: something else: the script stops rather than lands, and whatever the opening
#: promised is never collected.
#:
#: Read it as "one word in common is enough". The overlap is measured against
#: the smaller set, which is nearly always the opening's own handful of content
#: words, so 0.10 clears an opening of up to ten of them on a single shared
#: word. It is set that low because synonyms are invisible here: a script
#: opening on "a tiny bit of salt to the sea" and closing on "why the ocean is
#: so salty" has closed its loop and shares exactly one word while doing it.
#: Demanding two called that script a drifter.
_STRUCTURE_DRIFT_BELOW = 0.10

#: Above it, the close is the opening again in slightly different words. That
#: is the worse failure of the two - it spends the payoff on something the
#: viewer already heard, and it is what a model does when asked for an ending
#: with no instruction about what an ending is for.
_STRUCTURE_RESTATES_ABOVE = 0.70

#: Where a turn has to land to count, as a fraction of the way through. Before
#: the first quarter it is part of the opening; past 70% it is the payoff
#: arriving, which the close is already scored for.
_STRUCTURE_TURN_START = 0.25
_STRUCTURE_TURN_END = 0.70

#: A last line longer than this is a summary. A payoff lands in one breath.
_STRUCTURE_LANDING_MAX_WORDS = 12

#: The point where what the viewer assumed stops being true.
_STRUCTURE_TURN_RE = re.compile(
    r"\b(?:but|except|until|turns out|here'?s the thing|the catch|"
    r"the real (?:reason|secret|problem|answer)|what nobody|nobody tells|"
    r"no one tells|and then|the problem is|the twist|the strange part|"
    r"the odd part|which is where|that changed|then it)\b",
    re.IGNORECASE,
)

#: A question the script asks itself in order to answer it - "So how do fish
#: survive?". The discourse marker is what marks it as filler: without one,
#: "Why does the salt stay behind?" is a real question and scores nothing
#: either way. Matched only away from the opening, where score_hook rules.
_STRUCTURE_FILLER_QUESTION_RE = re.compile(
    r"^(?:so|but|now|and|then|ok|okay)\b[^?]*\?$", re.IGNORECASE
)

#: How alike two neighbouring sentences have to be before saying one of them
#: negated reads as the script arguing with itself rather than contrasting two
#: things. High on purpose: a real contrast brings in a new subject to hold
#: against the old one and cannot score this close, while the failure restates
#: the same handful of words with the polarity flipped.
_STRUCTURE_SEESAW_OVERLAP = 0.75

#: ...and both sides must actually say something. Without this, "It doesn't."
#: pairs with whatever precedes it - which is the *good* version of this shape,
#: a flat claim knocked down in three words, and must never be charged for.
_STRUCTURE_SEESAW_MIN_WORDS = 2

#: Words carrying no subject matter. Dropped before the opening and the close
#: are compared, because two English sentences always share "the" and "is", and
#: that overlap says nothing about whether one answers the other.
_STRUCTURE_STOPWORDS = frozenset("""
a an the this that these those there here it its their them they
i you your yours we us our he she his her him who whom whose which what
is are was were be been being am do does did doing have has had having
can could will would shall should may might must
of in on at to from by for with without into onto over under about as
and or but nor so yet if then than because while when where how why
not no too very just only also even more most much many some any
one up down out off again make makes made get gets got
""".split())

#: Enough of the number words to make "eleven" and "11" the same token, which
#: is exactly the case the restatement rule has to catch: a model rewriting its
#: own opening line spells the digits out the second time.
_STRUCTURE_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}


def split_sentences(text: str) -> list[str]:
    """Split a script into sentences, folding abbreviations back together.

    Paragraph breaks are flattened - this is for measuring and never for
    writing back, and `apply_hook` is the one place that has to preserve the
    shape step 4 reads.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return []

    sentences: list[str] = []
    for piece in (p.strip() for p in _SENTENCE_RE.findall(clean)):
        if not piece:
            continue
        short = len(_HOOK_WORD_RE.findall(piece)) < _SENTENCE_MIN_WORDS
        previous_short = bool(sentences) and (
            len(_HOOK_WORD_RE.findall(sentences[-1])) < _SENTENCE_MIN_WORDS
        )
        if sentences and (short or previous_short):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _fold_number_word(word: str) -> str:
    """`word` as digits if it is a number word, else `word` unchanged.

    The plain lookup covers "eleven"; this also covers "ninety-five", which it
    did not, and which is the case the restatement rule exists for. English
    hyphenates its compound numbers from twenty-one to ninety-nine, so a model
    respelling "95%" in words writes the one form the fold could not read -
    measured on the deep-ocean reel, whose opening was restated verbatim one
    sentence later and scored 0.17, indistinguishable from a script with no
    restatement in it at all.

    Only the tens-plus-units shape is folded. The parts of a hyphenated number
    are added in English only there; "one-hundred" is multiplicative, and
    summing it would produce 101.
    """
    direct = _STRUCTURE_NUMBER_WORDS.get(word)
    if direct is not None:
        return direct
    if "-" not in word:
        return word
    parts = word.split("-")
    if len(parts) != 2:
        return word
    tens, units = (_STRUCTURE_NUMBER_WORDS.get(p) for p in parts)
    if tens is None or units is None:
        return word
    tens, units = int(tens), int(units)
    if 20 <= tens <= 90 and tens % 10 == 0 and 1 <= units <= 9:
        return str(tens + units)
    return word


def _content_words(text: str) -> set[str]:
    """The words in `text` that carry its subject matter.

    Lowercased, de-pluralised and with number words folded onto their digits,
    so the comparison is between what two passages are *about* rather than
    between how they were typed. The stemming is crude - a trailing "s" comes
    off - but it is applied to both sides of every comparison, so it can only
    make a real match easier to see.
    """
    words: set[str] = set()
    for raw in _HOOK_WORD_RE.findall(text or ""):
        word = raw.lower().strip("'’-")
        if not word or word in _STRUCTURE_STOPWORDS:
            continue
        word = _fold_number_word(word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) > 1 or word.isdigit():
            words.add(word)
    return words


def _overlap(left: set[str], right: set[str]) -> float:
    """How much of the smaller set the two share, 0.0-1.0.

    The overlap coefficient rather than Jaccard: a close three sentences long,
    compared against an opening of one, should not be marked as unrelated for
    the crime of being longer.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def closing_sentences(sentences: list[str]) -> list[str]:
    """The passage at the end that counts as the close."""
    if not sentences:
        return []
    room = max(1, int(len(sentences) * _STRUCTURE_CLOSE_MAX_FRACTION))
    return sentences[-min(_STRUCTURE_CLOSE_SENTENCES, room):]


def seesaw_pairs(sentences: list[str]) -> list[tuple[int, int]]:
    """Neighbouring sentences that say the same thing, one of them negated.

    The shape this catches, from a real run about sleep:

        But they don't get less sleep than the rest of us.
        We get less sleep than them.

    Two sentences, the same three content words, the polarity flipped between
    them. Whether or not it resolves to a contradiction on a second reading is
    beside the point - it is heard once, at speed, and what it does to a viewer
    is stop them working out which way round the claim goes.

    Distinguished from the *good* version of the same move by how much the two
    sentences have in common. A real contrast brings in a new subject to hold
    against the old one - "The deep sea should crush every creature flat. It
    doesn't." shares no content words at all and is one of the strongest turns
    available. Only a restatement can score above the threshold.
    """
    found: list[tuple[int, int]] = []
    for index in range(len(sentences) - 1):
        first, second = sentences[index], sentences[index + 1]
        negated = (
            bool(_NEGATION_RE.search(first)),
            bool(_NEGATION_RE.search(second)),
        )
        if negated[0] == negated[1]:
            continue                     # both or neither: not a flip
        # Negations are dropped before the comparison so the flip itself
        # cannot be what makes the two sentences look different.
        left = {w for w in _content_words(first)
                if not _NEGATION_RE.fullmatch(w)}
        right = {w for w in _content_words(second)
                 if not _NEGATION_RE.fullmatch(w)}
        if min(len(left), len(right)) < _STRUCTURE_SEESAW_MIN_WORDS:
            continue
        if _overlap(left, right) >= _STRUCTURE_SEESAW_OVERLAP:
            found.append((index, index + 1))
    return found


def score_structure(script: str) -> dict:
    """Score the shape of a whole script. Higher is better.

    Returns the score and the parts it was built from - the contract
    `score_hook` keeps, and for the same reason: step 9 stores the names of the
    components that fired and never the arithmetic, so a run can still be
    argued with a year after the weights moved.
    """
    sentences = split_sentences(script)
    if len(sentences) < 2:
        return {"score": 0.0, "components": {"too short to have a shape": 0.0}}

    parts: dict[str, float] = {}
    opening = _content_words(sentences[0])
    close = _content_words(" ".join(closing_sentences(sentences)))
    overlap = _overlap(opening, close)

    # Exactly one of these three fires. They are three readings of a single
    # measurement, not three separate rules.
    if overlap > _STRUCTURE_RESTATES_ABOVE:
        parts["the ending restates the opening"] = -2.5
    elif overlap < _STRUCTURE_DRIFT_BELOW:
        parts["the ending drifts"] = -1.5
    else:
        parts["closes the loop"] = 2.0

    # Where each sentence starts, as a fraction of the way through, so a turn
    # is placed by how far into the reel it is heard rather than by which
    # sentence number it happens to be.
    lengths = [len(_HOOK_WORD_RE.findall(s)) for s in sentences]
    total = sum(lengths) or 1
    running = 0
    for sentence, length in zip(sentences, lengths):
        position = running / total
        running += length
        if position < _STRUCTURE_TURN_START:
            continue
        if position > _STRUCTURE_TURN_END:
            break
        if _STRUCTURE_TURN_RE.search(sentence):
            parts["a turn mid-reel"] = 1.2
            break

    if len(_HOOK_WORD_RE.findall(sentences[-1])) <= _STRUCTURE_LANDING_MAX_WORDS:
        parts["lands on a short line"] = 0.5

    filler = sum(1 for sentence in sentences[1:]
                 if _STRUCTURE_FILLER_QUESTION_RE.match(sentence))
    if filler:
        parts["asks itself a question"] = round(-1.0 * filler, 3)

    # Charged heavily enough to pull an otherwise well-shaped script under
    # STRUCTURE_MIN_SCORE on its own. A script that argues with itself has a
    # worse problem than a long last line, and the shape rules cannot see it:
    # the sleep reel closed its loop, landed short, scored 2.5 and shipped.
    seesaws = seesaw_pairs(sentences)
    if seesaws:
        parts["argues with itself"] = round(-2.0 * len(seesaws), 3)

    return {"score": round(sum(parts.values()), 3), "components": parts}


def structure_faults(components: dict) -> str:
    """What is wrong with a script, in words the rewrite prompt can use.

    The measurement is handed to the call that has to fix it. A rewrite told
    only "make this better" moves whatever it feels like moving, and there is
    no reason to spend a call finding that out again every time.
    """
    said = {
        "the ending restates the opening":
            "the ending repeats the opening instead of paying it off",
        "the ending drifts":
            "the ending has nothing to do with what the opening promised",
        "asks itself a question":
            "it stops in the middle to ask itself a question",
        "argues with itself":
            "two sentences next to each other say the same thing with the "
            "polarity flipped, so the claim reads as a contradiction",
    }
    faults = [said[name] for name in components if name in said]
    if "a turn mid-reel" not in components:
        faults.append("nothing turns in the middle - it is one flat run of facts")
    if "lands on a short line" not in components:
        faults.append("the last line is too long to land")
    return "; ".join(faults) if faults else "nothing measurable - tighten it"


_STRUCTURE_SYSTEM_INSTRUCTION = """\
You rewrite narration for short-form vertical video so that it holds attention \
all the way to its last word.

You are given an opening line that has already been chosen and is not yours to \
change. You rewrite what comes after it: the same facts, the same length, in \
an order that has a shape - something turning in the middle, and an ending \
that pays off the opening rather than repeating it.

You return only the words to be spoken."""


_STRUCTURE_PROMPT_TEMPLATE = """\
Rewrite the narration below so that it holds attention all the way to the end.

Topic: {topic}

This opening line has already been chosen. Reproduce it EXACTLY, word for \
word, as the first sentence of your rewrite:
"{opening}"

The narration as it stands:
{script}

What is wrong with it: {faults}

Rewrite everything after the opening line so that:
- the opening leaves a question open - stated or implied - that the narration \
does not answer until its final line. Do not answer it early.
- somewhere in the middle there is a turn: the point where what the viewer \
assumed stops being true. "But", "except", "the part nobody mentions".
- the final line answers the opening in DIFFERENT words. It must not restate \
the opening and it must not summarise the script.
- the final line runs back into the opening line. Read the last line and the \
first line one after the other: they should sound continuous, because a viewer \
who watches it twice hears them that way.
- it never stops to ask itself a question it is about to answer.
- no sentence is followed by the same sentence negated. Say each thing once, in the direction that makes the claim clearest.
- the claims survive the facts quoted for them. Where a number does not support the line it sits under, cut the claim rather than asserting past it.

Keep the same facts and invent no new numbers. Keep the whole thing between \
{min_words} and {max_words} words.

Output format - this matters, the text goes straight into a text-to-speech engine:
- Return ONLY the words to be spoken, the opening line included.
- No title, no headings, no labels such as "Hook:" or "Narrator:".
- No stage directions, no shot descriptions, no bracketed cues, no timestamps.
- No emoji, no markdown, no asterisks, no quotation marks around the script.
- Write numbers and symbols the way they should be said aloud."""


def _same_opening(left: str, right: str) -> bool:
    """Whether two opening lines are the same sentence typed twice."""
    def key(text: str) -> str:
        return " ".join(w.lower() for w in _HOOK_WORD_RE.findall(text or ""))
    return key(left) == key(right)


def revise_structure(topic: str, script: str,
                     duration_seconds: int | None = None) -> dict:
    """Rewrite a script's body around its existing opening. Never raises.

    Returns {"script": str, "warnings": [...]}. An empty script means the
    rewrite could not be had, which every caller reads as "keep what you have".
    """
    opening, _ = split_first_sentence(script)
    if not opening:
        return {"script": "",
                "warnings": ["the script is empty - nothing to shape"]}

    low, high = word_band(target_duration(duration_seconds))
    prompt = _STRUCTURE_PROMPT_TEMPLATE.format(
        topic=topic or "(not given)",
        opening=opening,
        script=script,
        faults=structure_faults(score_structure(script)["components"]),
        min_words=low,
        max_words=high,
    )

    try:
        raw = call_gemini(
            prompt,
            system_instruction=_STRUCTURE_SYSTEM_INSTRUCTION,
            temperature=STRUCTURE_TEMPERATURE,
        )
    except (ScriptGenerationError, config.MissingConfigError) as exc:
        return {"script": "",
                "warnings": [f"the structure pass failed ({exc}). "
                             f"Kept the script as written."]}

    revised = clean_script(raw)
    if not revised:
        return {"script": "",
                "warnings": ["the structure pass returned nothing usable. "
                             "Kept the script as written."]}

    # The opening was handed over as fixed, so whether it came back fixed is
    # checkable rather than assumed. A model that rewrote it anyway has thrown
    # away the line the hook pass scored five candidates to choose.
    warnings: list[str] = []
    returned, _ = split_first_sentence(revised)
    if not _same_opening(returned, opening):
        revised = apply_hook(revised, opening)
        warnings.append("the rewrite changed the opening it was told to keep - "
                        "put the chosen one back")

    return {"script": revised, "warnings": warnings}


def choose_structure(topic: str, script: str,
                     duration_seconds: int | None = None, *,
                     mode: str | None = None) -> dict:
    """Score a script's shape, rewrite it when it is poor, report what happened.

    Never raises, and never hands back a script that scores worse than the one
    it was given: a rewrite has to beat the incumbent on the same scale and
    come back the length it was asked for, or it is dropped with a warning and
    the original stands.
    """
    setting = (mode or config.STRUCTURE_MODE or "off").strip().lower()
    incumbent = score_structure(script)
    report = {
        "mode": setting,
        "score": incumbent["score"],
        "components": incumbent["components"],
        "revised": False,
        "revised_script": "",
        "revised_score": None,
        "revised_components": {},
        "warnings": [],
    }

    if setting != "llm":
        return report

    sentences = split_sentences(script)
    if len(sentences) < _STRUCTURE_MIN_SENTENCES:
        report["warnings"].append(
            f"the script is {len(sentences)} sentence(s) - too short to have a "
            f"middle, so its shape was left alone"
        )
        return report

    # A script that already has the shape does not buy a second call. The
    # threshold sits at the score of a script that closes its loop and does
    # nothing else, so everything failing to close one is rewritten and nothing
    # that closes one is rewritten for polish.
    if incumbent["score"] >= config.STRUCTURE_MIN_SCORE:
        return report

    result = revise_structure(topic, script, duration_seconds)
    report["warnings"].extend(result["warnings"])
    if not result["script"]:
        report["mode"] = "failed"
        return report

    revised = result["script"]
    scored = score_structure(revised)
    report["revised_score"] = scored["score"]
    report["revised_components"] = scored["components"]

    # Length is step 0's decision, and step 5 has already been told it. A
    # rewrite that comes back half as long again is a different reel.
    if duration_seconds is not None:
        low, high = word_band(duration_seconds)
        words = count_words(revised)
        if not low <= words <= high:
            report["warnings"].append(
                f"the rewrite came back {words} words against a {low}-{high} "
                f"word budget - kept the original"
            )
            return report

    if scored["score"] <= incumbent["score"]:
        report["warnings"].append(
            f"the rewrite scored {scored['score']:g} against the original's "
            f"{incumbent['score']:g} - kept the original"
        )
        return report

    report["revised_script"] = revised
    report["revised"] = True
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The claims
#
# The hook pass fixes sentence one and the structure pass fixes the order. Both
# ask whether the script is *shaped* like a reel; neither asks whether it is
# true, and nothing else in the pipeline does either. Two shipped reels show
# what that costs.
#
# The glacier reel opens "This mountain of ice is moving at your walking
# speed." and says, eleven sentences later, "This happens slowly. A few
# centimetres a day." Those differ by a factor of about four million. The body
# is right; the opening was written by the hook pass, and nothing compared the
# two.
#
# The Oakland Zoo reel is worse. It narrates a crimson mountain lion, gives it
# a real cougar's name, puts it in the wrong mountains, and closes on "She's
# now safe at the Oakland Zoo. Come see her." There is no such animal. That
# script is not badly shaped - it is well shaped, and every shape rule in this
# module scores it fine.
#
# So this pass asks a model two questions in one call: does the script disagree
# with itself, and does it assert anything that is not true. What comes back is
# checked the way every other pass here is checked - against the script, in
# code - because a finding that does not quote a sentence the script actually
# contains is a finding about some other script.
#
# A script with faults buys one rewrite, on the structure pass's terms: it is
# kept only if the rewrite clears what the first draft did not. What survives
# that is refused outright, as a ScriptGenerationError - which step 8 already
# treats as this topic being a dead end, and falls through to the next
# candidate on the board. A day with no reel costs a day. A day with a reel
# that invents an animal at a named zoo costs more than that.
# ---------------------------------------------------------------------------

#: Sampling temperature for both calls in this pass. Low: neither one is
#: looking for phrasing, and a checker that answers differently every time is
#: not a check.
VERIFY_TEMPERATURE = 0.2

#: Findings that stop a script. `unsupported` deliberately does not: it is the
#: model saying it cannot confirm something, which is true of most interesting
#: claims and would leave the pipeline unable to ship anything at all. Only
#: what it will state is wrong, or what the script says twice in two different
#: ways, is worth losing a topic over.
_VERIFY_BLOCKING = ("contradiction", "false")

_VERIFY_KINDS = _VERIFY_BLOCKING + ("unsupported",)

#: How much of a quoted sentence has to match one in the script before
#: the finding is believed to be about the script. Not 1.0: the quote
#: comes back through JSON and a model straightens apostrophes and drops
#: commas on the way. Not lower: a finding that half-matches two
#: different sentences is a finding about neither.
_VERIFY_QUOTE_MATCH = 0.8


_VERIFY_SYSTEM_INSTRUCTION = """\
You fact-check narration written for short-form video.

You are shown a script that is about to be recorded and published. You find \
the claims in it that are wrong, and the places where it disagrees with \
itself. You are terse, you quote the sentence at fault, and you do not \
comment on style, structure, pacing or word choice - other passes do that and \
your findings are acted on automatically.

You always return valid JSON and nothing else."""


_VERIFY_PROMPT_TEMPLATE = """\
Topic of the video: {topic}

The narration, one sentence per line:
{numbered}

Find every place where this script:
- contradicts itself. Two sentences that cannot both be true - especially two \
different magnitudes given for the same quantity, where one of them has been \
rounded to something vivid.
- states something that is not true. Invented people, animals, places, \
events or institutions; a real name attached to the wrong thing; a figure \
that is wrong by an order of magnitude.

Do NOT report: anything you merely cannot verify but which is plausible; \
simplifications that are fair for a general audience; approximations that are \
roughly right; matters of style or emphasis.

For each finding return:
- "sentence": the sentence at fault, copied EXACTLY from the script above.
- "kind": "contradiction" if the script disagrees with itself, "false" if the \
claim is wrong, "unsupported" if it is the sort of specific claim that would \
need a source and you do not believe it.
- "problem": one sentence saying what is wrong. Name the other sentence if \
this is a contradiction.

Return ONLY a JSON array, empty if the script is sound:
[{{"sentence": "...", "kind": "...", "problem": "..."}}]"""


_VERIFY_FIX_SYSTEM_INSTRUCTION = """\
You repair narration for short-form video that has been fact-checked.

You are given a script and the specific things wrong with it. You fix exactly \
those, change nothing else, and never paper over a fault by making the \
sentence vaguer - you correct it or you cut it.

You return only the words to be spoken."""


_VERIFY_FIX_TEMPLATE = """\
The narration below is about to be recorded. A fact-check found these faults:

{faults}

The narration:
{script}

Rewrite it so that every fault above is gone. Rules:
- Fix the faults and change nothing else. Sentences that were not flagged \
come through word for word.
- Where two sentences disagree, keep the one that is true and correct the \
other. Do not split the difference and do not delete both.
- Where a claim is false, cut it or replace it with what is actually the \
case. Do not make it vague to make it survivable.
- Do not add new numbers, names or claims that were not already there.
- Keep the first sentence exactly as it is UNLESS it is one of the faults.
- Keep the length within a few words of what it is now.

Output format - this matters, the text goes straight into a text-to-speech engine:
- Return ONLY the words to be spoken.
- No title, no headings, no labels such as "Hook:" or "Narrator:".
- No stage directions, no bracketed cues, no timestamps.
- No emoji, no markdown, no asterisks, no quotation marks around the script.
- Write numbers and symbols the way they should be said aloud."""


def _numbered_sentences(script: str) -> str:
    """The script one sentence per line, numbered.

    Numbered so a finding about a contradiction can point at both halves of
    it, and split the same way `score_structure` splits, so a sentence quoted
    back can be matched against the same list it was taken from.
    """
    return "\n".join(f"{i}. {s}"
                      for i, s in enumerate(split_sentences(script), start=1))


def _parse_verify_json(raw: str) -> list[dict]:
    """Read the model's JSON array of findings, tolerantly."""
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

    findings = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        sentence = " ".join(str(entry.get("sentence") or "").split())
        problem = " ".join(str(entry.get("problem") or "").split())
        kind = str(entry.get("kind") or "").strip().lower()
        if not sentence or kind not in _VERIFY_KINDS:
            continue
        findings.append({"sentence": sentence, "kind": kind,
                         "problem": problem or "no reason given"})
    return findings


def _finding_is_about(script: str, sentence: str) -> bool:
    """Whether `sentence` is really in `script`.

    A finding that quotes a sentence the script does not contain is a finding
    about something the model wrote itself, and acting on it means rewriting a
    script to fix a fault that was never in it. Compared on content words so
    that a quote which lost a comma or straightened an apostrophe on the way
    through still matches - the same tokens every other comparison in this
    module is made on.
    """
    quoted = _content_words(sentence)
    if not quoted:
        return False
    for candidate in split_sentences(script):
        theirs = _content_words(candidate)
        if theirs and _overlap(quoted, theirs) >= _VERIFY_QUOTE_MATCH:
            return True
    return False


def check_claims(topic: str, script: str) -> dict:
    """Fact-check `script`. Never raises.

    Returns {"findings": [...], "warnings": [...]}. Every failure path comes
    back as no findings and a warning: a checker that cannot be reached must
    not be the reason a reel does not exist, and a script nobody could check
    is exactly where the pipeline stood before this pass was written.
    """
    sentences = split_sentences(script)
    if len(sentences) < 2:
        return {"findings": [], "warnings": []}

    prompt = _VERIFY_PROMPT_TEMPLATE.format(
        topic=topic or "(not given)",
        numbered=_numbered_sentences(script),
    )
    try:
        raw = call_gemini(
            prompt,
            system_instruction=_VERIFY_SYSTEM_INSTRUCTION,
            temperature=VERIFY_TEMPERATURE,
            json_mode=True,
        )
        found = _parse_verify_json(raw)
    except (ScriptGenerationError, config.MissingConfigError, ValueError) as exc:
        return {"findings": [],
                "warnings": [f"the claim check failed ({exc}). "
                             f"The script was not fact-checked."]}

    findings, warnings = [], []
    for finding in found:
        if _finding_is_about(script, finding["sentence"]):
            findings.append(finding)
        else:
            warnings.append(
                "the claim check quoted a sentence that is not in the script "
                f"({finding['sentence'][:60]!r}) - ignored"
            )
    return {"findings": findings, "warnings": warnings}


def blocking_findings(findings: list[dict]) -> list[dict]:
    """The findings that are worth losing a topic over."""
    return [f for f in findings if f.get("kind") in _VERIFY_BLOCKING]


def claim_faults(findings: list[dict]) -> str:
    """The findings as the repair prompt wants them."""
    return "\n".join(
        f'- "{f["sentence"]}" - {f["problem"]}' for f in findings
    )


def revise_claims(script: str, findings: list[dict]) -> dict:
    """Buy one rewrite that fixes `findings`. Never raises."""
    prompt = _VERIFY_FIX_TEMPLATE.format(
        faults=claim_faults(findings),
        script=script,
    )
    try:
        raw = call_gemini(
            prompt,
            system_instruction=_VERIFY_FIX_SYSTEM_INSTRUCTION,
            temperature=VERIFY_TEMPERATURE,
        )
    except (ScriptGenerationError, config.MissingConfigError) as exc:
        return {"script": "", "warnings": [f"the claim repair failed ({exc})"]}

    revised = clean_script(raw)
    if not revised:
        return {"script": "",
                "warnings": ["the claim repair returned nothing usable"]}
    return {"script": revised, "warnings": []}


def verify_claims(topic: str, script: str, *, mode: str | None = None) -> dict:
    """Fact-check `script` and repair it once if it needs it. Never raises.

    Returns the report step 8 and the saved metadata read:

        mode      what ran - "llm", "off", or "failed"
        findings  what the first check found
        remaining what is still wrong after the repair, if one was bought
        revised   whether the script was rewritten
        script    the script as it now stands
        blocked   whether what remains is bad enough to refuse

    `blocked` is reported rather than raised so that the decision belongs to
    the caller: `generate_script` turns it into a ScriptGenerationError, and
    anything reading a saved script can see the same thing without one.
    """
    setting = (mode or config.VERIFY_MODE or "off").strip().lower()
    report = {"mode": setting, "findings": [], "remaining": [],
              "revised": False, "script": script, "blocked": False,
              "warnings": []}
    if setting != "llm":
        return report

    checked = check_claims(topic, script)
    report["findings"] = checked["findings"]
    report["warnings"] = list(checked["warnings"])

    blocking = blocking_findings(checked["findings"])
    if not blocking:
        return report

    repair = revise_claims(script, blocking)
    report["warnings"].extend(repair["warnings"])
    if not repair["script"]:
        report["remaining"] = blocking
        report["blocked"] = True
        return report

    # Asking is not checking: the repair is re-checked, and kept only if it
    # actually cleared the faults. A rewrite that fixes one contradiction and
    # introduces another is not an improvement, and this is the pass that must
    # not wave anything through.
    rechecked = check_claims(topic, repair["script"])
    report["warnings"].extend(rechecked["warnings"])
    still = blocking_findings(rechecked["findings"])
    if len(still) < len(blocking):
        report["script"] = repair["script"]
        report["revised"] = True
        report["remaining"] = still
        report["blocked"] = bool(still)
        return report

    report["remaining"] = blocking
    report["blocked"] = True
    report["warnings"].append(
        "the rewrite did not clear the fact-check - kept the original"
    )
    return report


def generate_script(topic: str, duration_seconds: int | None = None, *,
                    hooks: str | bool | None = None,
                    structure: str | bool | None = None,
                    verify: str | bool | None = None) -> dict:
    """Generate a TTS-ready narration script for `topic`.

    Args:
        topic: what the reel is about.
        duration_seconds: target spoken length. None takes
            config.SCRIPT_DEFAULT_SECONDS - see `target_duration`.
        hooks: the hook pass. None follows config.HOOK_MODE; False skips it;
            a string forces a mode ("llm" or "off").
        structure: the structure pass, on the same three-way contract against
            config.STRUCTURE_MODE.
        verify: the claim check, on the same three-way contract against
            config.VERIFY_MODE.

    Returns:
        {"topic", "script", "estimated_duration", "hook_card", "hook",
        "structure", "claims"} - where `hook`, `structure` and `claims` are the
        reports of what those passes decided (each score and the parts it was
        made of) and `hook_card` is the short on-screen form step 5 burns over
        the opening.

    Raises:
        ValueError: the topic is empty or the duration is out of range.
        ScriptGenerationError: the API call failed or returned nothing usable,
            or the script asserted something the claim check would not pass and
            could not be repaired.

    No pass ever raises from inside itself: all three are improvements on a
    script that already exists, so any failure within one leaves that script
    exactly as it was. The claim check is the one that can end the run, and it
    does so here rather than in the pass, on a report the pass returns.

    They run in this order because the structure pass builds an ending that
    pays off whatever the script opens with, and the hook pass changes what it
    opens with. Reversed, the hook pass would swap the opening out from under
    an ending written to close it.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    duration_seconds = target_duration(duration_seconds)
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"duration_seconds must be between {MIN_DURATION_SECONDS} and "
            f"{MAX_DURATION_SECONDS}, got {duration_seconds}"
        )

    prompt = build_prompt(topic, duration_seconds)
    raw = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION)
    script = clean_script(raw)

    if not script:
        raise ScriptGenerationError(
            "Gemini's response contained no usable narration after cleaning. "
            f"Raw response was: {raw[:200]!r}"
        )

    def _mode(flag) -> str | None:
        return "off" if flag is False else (flag if isinstance(flag, str) else None)

    hook = choose_hook(topic.strip(), script, mode=_mode(hooks))
    if hook["replaced"]:
        script = apply_hook(script, hook["chosen"], hook.get("bridge", ""))

    shape = choose_structure(topic.strip(), script, duration_seconds,
                             mode=_mode(structure))
    if shape["revised"]:
        script = shape["revised_script"]

    # Last, because it is the only pass that reads the script for what it says
    # rather than for how it is built, and the two above it rewrite the words.
    # Checking before them would be checking a draft that no longer exists.
    claims = verify_claims(topic.strip(), script, mode=_mode(verify))
    script = claims["script"]
    if claims["blocked"]:
        raise ScriptGenerationError(
            "the script did not survive its fact-check: "
            + "; ".join(f'"{f["sentence"]}" - {f["problem"]}'
                        for f in claims["remaining"])
        )

    return {
        "topic": topic.strip(),
        "script": script,
        "estimated_duration": estimate_duration(script),
        "hook_card": hook["card"],
        "hook": hook,
        "structure": shape,
        "claims": claims,
    }


# ---------------------------------------------------------------------------
# Saving - where step 3 and step 6 come looking
# ---------------------------------------------------------------------------

def slugify(text: str, limit: int = 48) -> str:
    """Filename-safe stem for a topic. Same shape step 4 gives its run ids."""
    normalised = unicodedata.normalize("NFKD", text or "")
    ascii_only = normalised.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug[:limit].strip("-") or "script"


#: Written side by side, so the pair is claimed or free together.
_SAVED_SUFFIXES = (".txt", ".json")


def _free_stem(directory: Path, stem: str) -> str:
    """`stem`, or the first `stem-2`, `stem-3` ... with both names unclaimed.

    TEMPERATURE is high, so running one topic twice gives two different scripts
    - and step 4 keys its run directory off the script's own hash, making the
    second one a second reel rather than a replacement. Overwriting would throw
    away a script whose audio and visuals are still on disk.
    """
    n = 1
    while True:
        candidate = stem if n == 1 else f"{stem}-{n}"
        if not any((directory / (candidate + s)).exists()
                   for s in _SAVED_SUFFIXES):
            return candidate
        n += 1


def save_script(result: dict, output_path: str | Path | None = None) -> dict:
    """Write a generated script where the rest of the pipeline expects it.

    Two files side by side:
      <stem>.txt   the words alone - `tts.py -f` reads this and speaks it, so
                   nothing but narration may go in it
      <stem>.json  the whole result dict. The topic only survives here, and
                   step 6 reads it to know what the reel is about

    Args:
        result: what `generate_script` returned.
        output_path: the .txt to write. Defaults to a slug of the topic under
            output/scripts/, numbered rather than overwritten.

    Returns:
        `result` with `script_path` and `metadata_path` added.

    Raises:
        ValueError: output_path does not end in .txt.
        OSError: the files could not be written.
    """
    if output_path is None:
        config.SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = _free_stem(config.SCRIPTS_DIR, slugify(result.get("topic", "")))
        destination = config.SCRIPTS_DIR / f"{stem}.txt"
    else:
        destination = Path(output_path)
        if destination.suffix.lower() != ".txt":
            raise ValueError(
                f"output_path must end in .txt, got {output_path!r}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

    sidecar = destination.with_suffix(".json")
    saved = {
        **result,
        "script_path": str(destination.resolve()),
        "metadata_path": str(sidecar.resolve()),
    }

    destination.write_text(result["script"] + "\n", encoding="utf-8")
    sidecar.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: dict, target: int) -> None:
    rule = "-" * 68
    words = count_words(result["script"])
    print(rule)
    print(f"TOPIC   : {result['topic']}")
    print(f"TARGET  : {target}s  (~{words_for_duration(target)} words)")
    print(f"ACTUAL  : {result['estimated_duration']}s  ({words} words @ "
          f"{WORDS_PER_MINUTE} wpm)")
    hook = result.get("hook") or {}
    if hook.get("mode") == "llm" and hook.get("candidates"):
        verdict = ("chosen" if hook["replaced"]
                   else f"kept the original ({hook['original_score']:g})")
        print(f"HOOK    : best of {len(hook['candidates'])} - {verdict}")
    shape = result.get("structure") or {}
    if shape.get("mode") in ("llm", "failed"):
        verdict = (f"rewritten, {shape['score']:g} -> {shape['revised_score']:g}"
                   if shape.get("revised")
                   else f"{shape['score']:g} - left as written")
        print(f"SHAPE   : {verdict}")
    if result.get("hook_card"):
        print(f"CARD    : {result['hook_card']}")
    if result.get("script_path"):
        print(f"SCRIPT  : {result['script_path']}")
        print(f"META    : {result['metadata_path']}")
    print(rule)
    print(result["script"])
    print(rule)
    if hook.get("candidates"):
        print("HOOKS  (scored here, not by the model - best first)")
        for candidate in hook["candidates"]:
            mark = "*" if candidate["text"] == hook.get("chosen") else " "
            detail = ", ".join(f"{name} {value:+g}"
                               for name, value in candidate["components"].items())
            print(f" {mark}{candidate['score']:>6.2f}  {candidate['text']}")
            print(f"         {detail}")
        print(f"  {hook['original_score']:>6.2f}  {hook['original']}"
              f"   <- the scriptwriter's own")
        print(rule)
    if shape.get("components"):
        print("SHAPE  (scored here, not by the model)")
        columns = (("as written", shape["score"], shape["components"]),)
        if shape.get("revised_components"):
            columns += (("rewritten", shape["revised_score"],
                         shape["revised_components"]),)
        for label, score, components in columns:
            mark = "*" if (label == "rewritten") == bool(shape.get("revised")) else " "
            detail = ", ".join(f"{name} {value:+g}"
                               for name, value in components.items()) or "nothing fired"
            print(f" {mark}{score:>6.2f}  {label}")
            print(f"         {detail}")
        print(rule)
    for warning in list(hook.get("warnings", [])) + list(shape.get("warnings", [])):
        print(f"note: {warning}", file=sys.stderr)
    if result.get("script_path"):
        # The exact next command, so the path never has to be retyped.
        print("next:")
        print(f"  python pipeline/tts.py -f {_relative(result['script_path'])}")
        print(rule)
    print("dict:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _relative(path: str) -> str:
    """Project-relative path if it is inside the project, else as given."""
    try:
        return Path(path).relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a short-form video narration script with Gemini."
    )
    parser.add_argument("topic", help="what the reel is about")
    parser.add_argument(
        "-d", "--duration", type=int, default=None,
        help=f"target spoken length in seconds (default: "
             f"{config.SCRIPT_DEFAULT_SECONDS}). `python pipeline/topics.py` "
             f"prints a length with each topic and puts it in the command it "
             f"suggests - that one is argued for, this one is only a fallback",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output .txt path (default: output/scripts/<topic-slug>.txt). "
             "A .json sidecar is written beside it either way.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the script without writing it to disk",
    )
    parser.add_argument(
        "--no-hook", action="store_true",
        help="skip the hook pass and keep the scriptwriter's opening",
    )
    parser.add_argument(
        "--no-structure", action="store_true",
        help="skip the structure pass and keep the script's own shape",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip the claim check and publish whatever the script asserts",
    )
    args = parser.parse_args(argv)
    duration = target_duration(args.duration)

    try:
        result = generate_script(
            args.topic, duration,
            hooks=False if args.no_hook else None,
            structure=False if args.no_structure else None,
            verify=False if args.no_verify else None,
        )
        if not args.no_save:
            result = save_script(result, args.output)
    except (ValueError, config.MissingConfigError, ScriptGenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write the script: {exc}", file=sys.stderr)
        return 1

    _print_result(result, duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
