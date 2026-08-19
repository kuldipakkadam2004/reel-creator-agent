"""Step 2 - narration script generation via the Gemini API.

Takes a topic string and returns a narration script written for text-to-speech:
no stage directions, no emoji, no markdown - just the words to be spoken.

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
WORDS_PER_MINUTE = 150

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
- Write for the ear, not the eye. Short sentences. Natural spoken rhythm.
- Be specific. Use real details, numbers or images rather than vague claims.
- End on a line that resolves the hook or leaves a thought worth sitting with. \
Do not ask the viewer to like, follow or comment.

Output format - this matters, the text goes straight into a text-to-speech engine:
- Return ONLY the words to be spoken.
- No title, no headings, no labels such as "Hook:" or "Narrator:".
- No stage directions, no shot descriptions, no bracketed cues, no timestamps.
- No emoji, no markdown, no asterisks, no quotation marks around the script.
- Write numbers and symbols the way they should be said aloud."""


def build_prompt(topic: str, duration_seconds: int) -> str:
    """Render the user-facing prompt for a topic and target duration."""
    target_words = words_for_duration(duration_seconds)
    return PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        duration_seconds=duration_seconds,
        target_words=target_words,
        min_words=max(1, int(target_words * 0.85)),
        max_words=int(target_words * 1.15),
    )


class ScriptGenerationError(RuntimeError):
    """Raised when Gemini fails, refuses, or returns an unusable script."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def words_for_duration(duration_seconds: int) -> int:
    """How many spoken words fit into `duration_seconds` at WORDS_PER_MINUTE."""
    return max(1, round(duration_seconds * WORDS_PER_MINUTE / 60))


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
# Public API
# ---------------------------------------------------------------------------

def generate_script(topic: str, duration_seconds: int = 45) -> dict:
    """Generate a TTS-ready narration script for `topic`.

    Returns:
        {"topic": str, "script": str, "estimated_duration": int}

    Raises:
        ValueError: the topic is empty or the duration is out of range.
        ScriptGenerationError: the API call failed or returned nothing usable.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")
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

    return {
        "topic": topic.strip(),
        "script": script,
        "estimated_duration": estimate_duration(script),
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
    if result.get("script_path"):
        print(f"SCRIPT  : {result['script_path']}")
        print(f"META    : {result['metadata_path']}")
    print(rule)
    print(result["script"])
    print(rule)
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
        "-d", "--duration", type=int, default=45,
        help="target spoken length in seconds (default: 45)",
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
    args = parser.parse_args(argv)

    try:
        result = generate_script(args.topic, args.duration)
        if not args.no_save:
            result = save_script(result, args.output)
    except (ValueError, config.MissingConfigError, ScriptGenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write the script: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
