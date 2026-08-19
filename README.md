# Reel Creator Agent

Local automation agent that generates short-form (9:16) videos for manual
posting to Instagram Reels.

Built incrementally — see the step table below for what exists today.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
python config.py          # self-check: paths + which keys are set
```

Run modules from the project root:

```bash
python pipeline/script_gen.py "why the ocean is salty" -d 35
python pipeline/tts.py -f output/scripts/ocean.txt -o output/audio/ocean.mp3
python pipeline/visuals.py -f output/scripts/ocean.txt -a output/audio/ocean.mp3
python pipeline/assemble.py -a output/audio/ocean.mp3 -v assets/visuals/runs/ocean-1a2b3c
python pipeline/caption.py -r output/2026-08-19/ocean-1a2b3c
```

`python pipeline/visuals.py --providers-status` shows which stock providers have
their keys; `--dry-run` plans the beats and search queries without downloading.
`python pipeline/assemble.py --dry-run` prints the shot list and writes the
captions without spending minutes on an encode. `python pipeline/caption.py
--dry-run` prints the post text without writing it into the run directory.

## Layout

```
config.py           paths, .env loading, per-step key validation
pipeline/           pipeline modules (one per step)
assets/visuals/cache/   downloaded media, content-addressed
assets/visuals/runs/    one directory per run: media + manifest.json
assets/visuals/usage.json  which stock asset each run used
assets/fonts/       caption fonts
output/scripts/     generated narration scripts
output/audio/       ElevenLabs .mp3 renders
output/<date>/<run>/  one finished reel: .mp4, captions, attribution, post text
output/captions/hashtag_usage.json  which tags recent posts used
tests/
```

Every path is derived from `config.py`'s own location, so the project can be
moved or deployed without editing code.

## Build steps

| # | Module | Status |
|---|--------|--------|
| 1 | Scaffolding, config | done |
| 2 | Script generation (Gemini) | done |
| 3 | TTS (ElevenLabs) | done |
| 4 | Visual sourcing (Pexels / Pixabay / Openverse) | done |
| 5 | Video assembly + burned-in captions | done |
| 6 | Instagram caption + hashtags (Gemini) | done |
| 7 | Telegram delivery | not started |
| 8 | Manual end-to-end pipeline | not started |

Deliberately out of scope for now: scheduling, the FastAPI server, and
auto-posting to Instagram.

## Step 4 — visual sourcing

`pipeline/visuals.py` turns a script plus its narration .mp3 into an ordered set
of downloaded assets and a `manifest.json` for step 5.

* the script is cut into **visual beats**, timed against the *measured* length of
  the .mp3 rather than an estimate, so beat boundaries line up with the audio;
* each beat gets a search query — Gemini writes them by default
  (`VISUAL_QUERY_MODE=llm`), falling back to keyword extraction if the call fails;
* candidates are filtered by a **crop-aware resolution floor**: the test is what
  survives a centre crop to 9:16, so a 4K landscape clip passes and a 1080p one
  does not;
* downloads are content-addressed in `assets/visuals/cache/` and hardlinked into
  the run directory, so re-running a script costs no bytes and no API calls;
* a usage ledger keeps the same stock clip from opening every video.

A failed beat is recorded and skipped, never fatal. Every asset carries its
provider, licence and attribution; `attribution.txt` is written alongside.

What step 5 consumes from the returned dict (and `manifest.json`):

| key | meaning |
|-----|---------|
| `asset_dir` | the run directory holding every file |
| `assets` | flat list in playback order — `path`, `kind`, `width`, `height`, `duration`, `beat_index`, `role`, `suggested_start`, `suggested_duration` |
| `beats` | `start`/`end`/`text`/`query` per beat, plus `asset_indexes` into `assets` |
| `sourced_duration` / `target_duration` | how much footage there is vs. how much was asked for |

## Step 5 — video assembly

`pipeline/assemble.py` turns the narration .mp3 and step 4's run directory into
a finished 9:16 .mp4 with captions burned in, written to
`output/<date>/<run id>/`.

* step 4's **beat order and beat timings are the edit** — nothing here re-reads
  the script to work out segmentation. Each beat's window is divided into shots
  of about `SHOT_TARGET_SECONDS`, capped by how many assets that beat got;
* clips are centre-cropped to 9:16, trimmed to their slot, and **looped** when
  they are too short; stills get a slow push-in. A clip with room to spare cuts
  in a little way past its start, where stock footage stops fading up;
* captions are timed by **ElevenLabs forced alignment** — the script is already
  known, so the words are exact rather than transcribed. `whisper` and a
  no-network `estimate` are the fallbacks, tried in that order;
* the whole edit is **one ffmpeg pass** — a single `filter_complex` scales,
  concatenates, burns the captions in with libass and muxes the audio, so the
  picture is encoded once.

Mismatches degrade instead of failing. A beat with no usable footage gives its
time to a neighbour; an unreadable file is dropped from the edit; too little
footage is looped and reused (and two identical shots in a row are merged into
one, since that is not a cut); too much is left on the shelf. Every one of those
is reported in `warnings`.

ffmpeg has to be **built with libass** or the captions cannot be burned in. With
none on `PATH`, the binary from the `imageio-ffmpeg` wheel is used.

What lands in `output/<date>/<run id>/`:

| file | contents |
|------|----------|
| `reel.mp4` | the finished 1080x1920 reel |
| `captions.ass` / `.srt` | the burned-in captions, and a plain copy |
| `alignment.json` | every word with its timestamp, and which backend timed it |
| `attribution.txt` | credits for the footage that made the cut |
| `assembly.json` | the shot-by-shot timeline plus **step 4's per-asset attribution**, carried through for steps 6 and 7 |

## Step 6 — Instagram caption + hashtags

`pipeline/caption.py` turns the script into the **post text** — the words that
go in Instagram's caption field. Nothing to do with step 5's burned-in
captions, which are pixels inside the video.

* one Gemini call writes the hook, body, CTA and a tiered set of hashtags, and
  it goes through **step 2's client** (`script_gen.call_gemini`) rather than a
  second one. `POST_MODE=heuristic`, a missing key or a failed call all fall
  back to keyword extraction, which needs no API at all;
* the **first line is treated as the whole caption**, because for most viewers
  it is: Instagram cuts to "... more" at ~125 characters, so the hook is fitted
  to `POST_HOOK_MAX_CHARS` at a sentence break, then a clause break, and only
  ever at a word boundary as a last resort — never left as a truncated line or
  ending on a dangling connector;
* **the model is not trusted with the limits**. Character counts, the 2200-char
  cap, the 30-tag cap, tag normalising, the banned lists and the must-include
  phrases are all enforced in code after the call, whatever the prompt asked for;
* hashtags are **mixed across tiers** — broad to place the post, mid to say what
  it is, niche for the people already deep in the subject — and a cross-run
  ledger keeps consecutive posts from carrying an identical block;
* footage whose licence requires a credit gets one in the form that licence
  expects, carried through from step 4 via step 5's `assembly.json`. Pexels and
  Pixabay require none, so most runs add nothing (`POST_ATTRIBUTION`).

Over the character limit, blocks are given up from the bottom: the courtesy
credit first, then body paragraphs, then the CTA, then tags. The hook and any
licence-required credit are never given up.

What it adds to `output/<date>/<run id>/`:

| file | contents |
|------|----------|
| `post.txt` | the caption, exactly as it should be posted, and nothing else |
| `post.json` | the same in parts: hook, body, CTA, tags and their tiers, credits, character counts, warnings |
| `post_comment.txt` | the tag block — only when `POST_HASHTAG_PLACEMENT=comment` |

## Secrets

All keys live in `.env` (git-ignored) and are read only through `config.py`.
Nothing is ever hardcoded.
