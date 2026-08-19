"""Step 2 - narration script generation via the Gemini API.

Takes a topic string and returns a narration script written for text-to-speech:
no stage directions, no emoji, no markdown - just the words to be spoken.

Standalone use:
    python pipeline/script_gen.py "why the ocean is salty"
    python pipeline/script_gen.py "the history of coffee" --duration 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
    quirks and the failure-to-exception mapping stay in one place.

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
    from google.genai import errors

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
    except errors.APIError as exc:
        code = getattr(exc, "code", None) or getattr(exc, "status", "?")
        raise ScriptGenerationError(
            f"Gemini API call failed [{code}] for model {model!r}: {exc}"
        ) from exc
    except Exception as exc:  # network errors, malformed key, SDK issues
        raise ScriptGenerationError(
            f"Gemini API call failed for model {model!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return _extract_text(response)


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
    print(rule)
    print(result["script"])
    print(rule)
    print("dict:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a short-form video narration script with Gemini."
    )
    parser.add_argument("topic", help="what the reel is about")
    parser.add_argument(
        "-d", "--duration", type=int, default=45,
        help="target spoken length in seconds (default: 45)",
    )
    args = parser.parse_args(argv)

    try:
        result = generate_script(args.topic, args.duration)
    except (ValueError, config.MissingConfigError, ScriptGenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
