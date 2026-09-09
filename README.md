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
python pipeline/topics.py                 # step 0: what should this reel be about?
python pipeline/script_gen.py "why the ocean is salty" -d 40 -o output/scripts/ocean.txt
python pipeline/tts.py -f output/scripts/ocean.txt -o output/audio/ocean.mp3
python pipeline/visuals.py -f output/scripts/ocean.json -a output/audio/ocean.mp3
python pipeline/assemble.py -a output/audio/ocean.mp3 -v assets/visuals/runs/ocean-1a2b3c
python pipeline/caption.py -r output/2026-08-19/ocean-1a2b3c
```

Or all of it in one command, choosing its own topic — and once a day by itself:

```bash
python pipeline/run.py --dry-run          # step 8: what would it make?
python pipeline/run.py                    # step 8: make it
python pipeline/schedule.py install       # every day at 17:00, from now on
```

And once you have posted one, tell it how that went:

```bash
python pipeline/perform.py                          # step 9: what needs numbers
python pipeline/perform.py cd1bccae --views 4200 --watched 61 --saves 30
python pipeline/perform.py cd1bccae --skipped footage
python pipeline/perform.py --report                 # what that has added up to
```

`python pipeline/topics.py` prints a ranked board of topic candidates and the
`script_gen.py` command for each — including how long that reel should be, in
`-d`; `--take N` marks one as made so no later board offers it again,
`--sources-status` shows which sources have their keys, `--no-probe` skips the
stock-footage check, `--niche-mode` sets how hard the niche is enforced, and
`--no-phrases` makes the whole step spend nothing on Gemini.
`python pipeline/script_gen.py` writes the script to
`output/scripts/<topic-slug>.txt` and prints the exact `tts.py` command to run
next; `-d` is the target length (step 0 suggests one per topic; without it the
fallback is `SCRIPT_DEFAULT_SECONDS`), `-o` picks the path and `--no-save`
prints without writing.
`python pipeline/visuals.py --providers-status` shows which stock providers have
their keys; `--dry-run` plans the beats and search queries without downloading.
`python pipeline/assemble.py --dry-run` prints the shot list, the cover frame it
would take and why, and writes the captions — without spending minutes on an
encode; `--cover hook_card|footage` overrules where the cover comes from and
`--no-cover` writes none. `python pipeline/caption.py --dry-run` prints the post
text without writing it into the run directory; `--alt llm` has the model
rephrase the cover's alt text and `--no-alt` leaves it out.

## Layout

```
config.py           paths, .env loading, per-step key validation
pipeline/           pipeline modules (one per step)
pipeline/run.py     step 8: all of it, start to finish, unattended
pipeline/schedule.py  registers step 8 with Windows Task Scheduler
pipeline/perform.py step 9: how the reel did, and what that changes
pipeline/instagram.py  step 9's optional provider: insights, read not typed
assets/visuals/cache/   downloaded media, content-addressed
assets/visuals/runs/    one directory per run: media + manifest.json
assets/visuals/usage.json  which stock asset each run used
assets/fonts/       caption fonts
assets/music/       background tracks you drop in, + optional <track>.txt credits
assets/sfx/         one-shots you drop in: whoosh*, impact*
output/topics/usage.json  which topics recent boards showed, and which were made
output/scripts/     generated narration scripts: <name>.txt (the words,
                    which step 3 speaks) + <name>.json (topic, timings and the
                    hook report, which steps 4 and 6 read)
output/scripts/refused/  drafts the fact-check would not pass, with the report
                    that refused them: every finding, and what each repair
                    round came back with. Nothing downstream reads it
output/audio/       ElevenLabs .mp3 renders
output/<date>/<run>/  one finished reel: .mp4, captions, attribution, post text
output/captions/hashtag_usage.json  which tags recent posts used
output/captions/hashtag_volume.json known post counts per tag, and the
                    cache behind TAG_VOLUME_PROVIDER
output/logs/        one transcript per step-8 run, pruned after 30 days
output/runs.json    one line per step-8 run: what it made, cost and how it ended
output/<date>/<run>/features.json  what that reel was, in the terms step 9
                    compares: which components fired, not what they scored
output/performance.json  one entry per reel: what it was, and what it did
tests/
```

Every path is derived from `config.py`'s own location, so the project can be
moved or deployed without editing code.

## Build steps

| # | Module | Status |
|---|--------|--------|
| 0 | Topic scouting (Reddit / Reddit RSS / YouTube / Trends / Wikipedia) | done |
| 1 | Scaffolding, config | done |
| 2 | Script generation (Gemini) | done |
| 3 | TTS (ElevenLabs) | done |
| 4 | Visual sourcing (Pexels / Pixabay / Openverse) | done |
| 5 | Video assembly + burned-in captions + cover frame | done |
| 6 | Instagram caption + hashtags + cover alt text (Gemini) | done |
| 7 | Telegram delivery | not started |
| 8 | End-to-end pipeline + daily scheduling | done |
| 9 | Performance ledger — how the reel did, and what that changes | done |

Deliberately out of scope for now: the FastAPI server, and auto-posting to
Instagram. Step 8 generates a reel a day and stops — what gets published, and
when, is still a decision a person makes.

### Tests

```bash
python -m pytest -q          # 1340 tests, about 10 seconds
```

Offline by construction: no key, no network, no ffmpeg. `.github/workflows/`
runs exactly that line on every push, with no secrets configured — and the lack
of secrets is the point rather than an omission. With a `GEMINI_API_KEY` in
`.env` the visual relevance screener defaults to `llm`, and three of the step 4
tests were quietly making real model calls at 40-49 seconds each; they pin it
offline in their own fixture now, so the suite exercises the same path on every
machine and costs nothing to run.

## Step 0 — topic scouting

`pipeline/topics.py` is where the topic comes from when you have not brought
one. It reads a few public sources, scores the candidates **in code**, and
prints a ranked board with the exact `script_gen.py` command for the one you
pick.

It does not rank for you. The decision of what to make is the highest-leverage
one in the pipeline, and a model asked to rank its own candidates will explain
why all eight are excellent — the same reason step 2 scores its own hooks and
step 6 enforces its own character limits after the call. Every number on the
board is printed with the candidate it belongs to:

```
 1.   4.42  why the ocean is salty
           redditrss r/todayilearned top #4  4h old
           velocity +1.71, still climbing +0.8, niche match +2, explainer shape +1.2,
           curiosity (contradiction, scale) +1, footage +1.5
           footage: 6 of 8 results name 'ocean salty' on pexels
           length: 45s   base 30, explainer +8, discussion +2.7, footage +2
```

There is exactly one judgement it delegates, and it is a closed one: **whether
a candidate is about your subject at all**. See the niche gate below for why
that could not stay in code. The model is not shown the scores, does not rank,
does not choose, and does not know how many places are on the board.

* **being on the niche is a gate, not a score.** Every source here is ranked by
  motion before this code sees it, so a national feed always has something
  moving faster than anything in one subject. Priced as a penalty, "off the
  account's subject entirely" simply loses to "moving quickly" — measured on
  one board of an ocean-and-earth-science account, eight of ten candidates were
  off-niche and the top three were a hockey fixture, a Bollywood sequel and a
  wrestler. Under the default `TOPIC_NICHE_MODE=gate` an off-niche candidate is
  held off the board while there is anything on-niche to show, and the board
  backfills and warns rather than coming back empty;
* **whether it is on the niche is decided by one Gemini call**, because shared
  words cannot decide it. `TOPIC_NICHE_MATCH=terms` is free and it is the
  stricter thing by a long way: on one 59-candidate board it placed *nothing*
  on-niche, "NASA has found evidence of caves beneath the Moon" among the
  misses — squarely earth science, not one word in common with it. The same
  mechanism scored a hockey **World** Cup as a full match on an account whose
  keywords included "natural **world**". Both are one error: a question about
  meaning answered by comparing spellings. Any failure falls back to words, and
  a candidate kept on the model's judgement alone says so on the board;
* **curiosity is scored, not just availability.** Velocity, footage, shape and
  length all measure whether a topic *can be made*; none of them says the reel
  is worth watching. A contradiction ("despite", "nobody knows", "cannot"), an
  extreme ("deepest", "only", "first") or a number each pay, and hype in the
  source's own title costs — the same words step 2 scores down in a hook. These
  weights are the one prior in the file that is not measured, and they are the
  first thing a performance ledger should overwrite;
* **velocity is a rate, not a total.** A post with 8,000 upvotes gathered over
  a week is a good post, not a moving one, and a reel about it arrives after
  the wave. Rates are not comparable across sources — 300 upvotes an hour and
  300 YouTube views a day are not the same event — so each is normalised
  against its own reference before the scores meet. It is deliberately no
  longer the largest positive on the board: the feed had already sorted itself
  by motion, so velocity mostly re-states the source's own ordering;
* **the sources are asked different questions.** Your own subreddits are the
  strongest, and not because the numbers are better: a post title is a hook a
  human wrote and several thousand people already voted on, so what comes back
  is a *framing* rather than a noun. They are read as `top` of the week rather
  than `hot`, because a reel that costs a Gemini call, an ElevenLabs render and
  a full encode is not going to beat anyone to a breaking story — and because
  a mechanism that was true last month is still true (`TOPIC_REDDIT_LISTING`).
  Reddit is also the one source that wants credentials in practice — it used to
  serve its public `.json` endpoints to anything with a User-Agent and now
  answers an unauthenticated read with a 403 block page, so it wants a free
  "script" app from <https://www.reddit.com/prefs/apps>; the `.rss` feeds
  `redditrss` reads still answer without one. YouTube is deliberately **not**
  the trending endpoint — that is music and celebrity whatever niche you point
  it at — but a keyword search bounded to the last month and ranked by **views
  per day**, which finds what has already worked in short form. Google Trends
  is the official RSS rather than `pytrends`, which scrapes an internal
  endpoint and breaks every few months;
* **search intent is not scroll-stop intent**, and the trends source is scored
  accordingly. Google Trends measures what people type when they already want
  an answer; a reel wins on what stops someone who was not looking for
  anything. It earns its place by occasionally surfacing the one story in your
  subject that everyone is asking about today;
* **proper nouns cost, which is the opposite sign to step 2's hook scoring** —
  and both are right. A hook names something because a specific image stops a
  scroll. A topic that is mostly names is a fixture, an election or a product
  launch: it decays in hours and no stock library holds a frame of it;
* topics shown on recent boards are scored down, ones already **made** are
  dropped, and ones whose script the **fact-check refused** are charged for it,
  through a ledger in `output/topics/usage.json` — the same pattern as step 4's
  asset usage and step 6's hashtags.

  The refusal half is newer than the rest, and it is worth saying why. A topic
  is marked used at the first moment money has gone on it, which a refused
  draft never reaches, so the entire cost of a refusal was the −0.6 for having
  been on a board — and a refused script costs a ~150s Pro call and leaves
  nothing behind. Measured across the logs: *"what the sound of the deep ocean
  actually is"* was chosen and refused **five times** over 26 boards before a
  draft survived. (The phrase pass rewords the line every run, so the log shows
  three spellings of it; the ledger key was stable throughout, and the problem
  was never the deduping.) A first refusal now costs `REFUSED_PENALTY` (−1.2),
  which puts a backlog line under the floor unless it is carrying enough else
  to be worth a second call, and `TOPIC_MAX_REFUSALS` (2) drops it. Charged
  once however many times, and the count is a filter rather than more
  arithmetic — [board #41](#step-0--topic-scouting) deadlocked on two penalties
  priced independently, and a number the scorer never sees cannot do that
  again.

A source that is unconfigured, rate-limited or broken is a warning; the board
is still produced from whatever answered. Only every source failing is fatal.

**The board is only as good as what you point it at.** Two things dominate, and
neither is in the scoring: whether `TOPIC_SUBREDDITS` is actually about your
subject, and whether `TOPIC_NICHE` describes one. `r/todayilearned` and
`r/science` will fill a board every day, but they are about everything, so most
of what they return is something the niche gate then throws away — a warning
saying "only 2 candidates matched the niche" is usually a subreddit problem
rather than a scoring one. Two or three subreddits genuinely about your subject
beat ten large ones.

**Trends alone is a weak board**, and worth saying plainly: it is national
general-news, so with no Reddit or YouTube credentials a run will mostly return
fixtures and share prices, score them all negative and reject most of them for
being off the niche or having no footage. That is the scoring working, not
failing — but it is not a board worth picking from. Set up Reddit first.

### Does the footage exist?

This is the check that makes step 0 part of *this* pipeline rather than a
generic trend reader, and it is the one that decides whether any of this is
worth having.

Step 4 sources from stock libraries. Half of what is trending anywhere is
proper nouns no stock library holds a single frame of, and a reel about one of
those does not fail — it comes out as generic filler laid over words that
needed specific pictures. That is the expensive kind of wrong: it costs a
Gemini call, an ElevenLabs render and a full encode before anyone can see that
it went wrong.

**The obvious implementation does not work, and it is worth saying why.** Ask
Pexels how many results it holds and it answers about 2,500 — for `zzqxwv`.
Stock APIs do not return nothing; they fall back to a generic pool, so the
total is the same large number whatever you search for. Measured: "ocean"
8,000, "shawver" 2,526, a random string 2,560. A count-based filter would have
passed every unshootable topic on the board.

**Reading the results instead does not work either, and that took measuring.**
The obvious repair is to read the clips that came back rather than the count:
`provider.search` returns titles and source URLs, and Pexels names each clip in
its own link even when the alt text is empty —
`serene-ocean-surface-with-gentle-waves`. Counting how many returned clips name
what was asked for looks like a real measurement. It is not.

Measured on 2026-09-02 over 439 results, from six known-good topics and five
known-bad ones (nonsense phrases plus real fixtures out of the ledger — a
Seahawks trade, a State Farm dividend, an LA mayoral result):

| | Pexels | Pixabay |
|---|---|---|
| `how a spider builds a web across a gap` (real) | 20/20 | 20/20 |
| `why a purple hexagon dreams of accounting` | 7/20 | **20/20** |
| `what makes a spreadsheet taste like violin` | 17/20 | **20/20** |

Across the whole evergreen backlog the effective count averaged **19.82 of 20,
median 20, with 56 of 62 lines scoring exactly 20**. A component that is the
same number for every candidate does no ranking at all — and worse, the hard
filter below needs a genuine zero to fire, so it had quietly stopped firing.

No threshold recovers it. Requiring two query words, or scaling the bar by how
many words the result describes itself with, was tried at 8, 10 and 12: good
and bad overlap under every one. Per-result term hits average 0.86 for good
topics against 0.60 for bad on Pexels, 1.40 against 1.22 on Pixabay. The reason
is that `probe_query` strips a topic to three keywords, so `State Farm
dividend` honestly retrieves farm footage — **the results are right and the
query was the wrong question**, and no amount of reading them fixes a question
that was wrong before it was asked.

So the question is now asked of **something that can read the topic**. Under
the default `TOPIC_SOURCEABILITY=llm` the model rates each shortlisted topic
0–3 on whether an ordinary stock library could illustrate it, batched into one
call the way the niche pass is. `TOPIC_SOURCEABILITY=probe` still selects the
keyword probe for anyone who wants it, and `off` skips the question entirely.

It stays a **strong positive and a weak negative**, and the scoring still says
so. A topic the model rates highly is definitely shootable; a low rating may
still be shootable through a synonym — `bioluminescence` returns jellyfish
footage, which is exactly right and matches not one letter of the query. So a
low rating scores *down* rather than dropping, and the one hard filter on the
board takes **two** signals: a zero rating **and** proper nouns in the topic.
Neither is sound alone — the Atlantic is a proper noun every library has — but
together they are conclusive: a named subject nothing can show is a person, a
fixture or a product, and there is no footage of it to find.

Only the top `TOPIC_PROBE_LIMIT` candidates are checked, because there is no
point asking whether a topic that lost on every other count is shootable. Under
`probe` that also saves requests: providers are asked in step 4's own
preference order and the walk stops at the first one holding enough, so a topic
Pexels can shoot never spends an Openverse request — which matters, as
Openverse allows 200 anonymous requests a day.

### How long should it be?

Length used to be a flag with a number in it. `-d 45` was the default, and it
was applied to a mechanism worth forty seconds and a single fact worth twenty
alike — which is where the reach went. **Completion is what the ranking runs
on**, so a fact stretched to 45 seconds is fifteen seconds of restating it that
the viewer leaves during, and a 30-second reel watched to the end beats a
45-second one abandoned at 15 twice over: once on the watch time, once on the
signal.

So step 0 decides the length too, from the signals it already has — printed
with the parts it was built from, like every other number on the board, and
carried into the `script_gen.py` command as `-d`:

```
           length: 45s   base 30, explainer +8, discussion +2.7, footage +2
```

* **a mechanism is given longer than a fixture.** "Why the ocean is salty" is
  setup, turn and payoff — three beats where "Braves at Twins on Sunday" is
  one, and the extra seconds on the fixture go to restating it;
* **an argument earns a middle to put it in.** The comments-to-reach ratio that
  scores a topic up for being discussed also says there is a second side to
  put, which is what the middle of a reel is for;
* **thin footage shortens the reel rather than filling it.** This is the
  constraint that belongs to *this* pipeline: fewer distinct clips spread over
  more seconds is the same shots coming round again, which reads as padding
  whatever the narration is doing. It is deliberately asymmetric — thin footage
  costs three times what plentiful footage pays, because plenty removes an
  objection rather than making a case;
* **and it shortens rather than vetoes.** A probe that found nothing may only
  mean the library knew a synonym this does not — `bioluminescence` returns
  jellyfish — so a miss it cannot be trusted to veto is not allowed to drive a
  reel to the floor. The conclusive case already has its own hard filter above;
* the result is rounded to `SCRIPT_DURATION_STEP_SECONDS` and clamped into
  `SCRIPT_MIN_SECONDS`..`SCRIPT_MAX_SECONDS`. **31 seconds is not a more
  considered answer than 30**, and printing it as one invites the number to be
  read as a measurement rather than a judgement.

The decision runs **after** the phrase pass below, not before. The rewrite is
what turns "TIL that the ocean is salty because rivers carry minerals into it"
into "why the ocean is salty" — a fact into a mechanism, which is most of what
the length turns on. Scoring runs before the rewrite because it has to rank
candidates that are about to be discarded; this runs after, because it only
concerns the ones that survived and it reads their final wording.

`-d` still overrides it, and `SCRIPT_MIN`/`MAX_SECONDS` bound the suggestion
rather than the flag. A topic that reaches step 2 without coming through a
board — you brought your own — gets `SCRIPT_DEFAULT_SECONDS`.

### From headline to topic

A Reddit title is not a topic. `clean_title` strips the furniture (`TIL`,
`ELI5`, `[OC]`, trailing outlet credits) locally, and then one Gemini call
rewrites the whole shortlist into topic phrases — one call, not one per
candidate, and only after the ranking is settled.

**It rewrites and nothing else.** The model is never shown the scores, never
asked to rank or reorder, and a rewrite that shares too little with its
original is discarded here rather than trusted, because a model that has
drifted will not report that it drifted. `--no-phrases` skips it — and the
niche pass with it, so the whole step spends nothing; the cleaned headlines are
usable topics on their own, and the niche falls back to shared words.

## Step 2 — the hook

`pipeline/script_gen.py` writes the narration, and then writes its opening line
again.

It is handed a target length rather than choosing one — step 0 decides that,
and passes it in `-d`. Called without one, it writes to
`SCRIPT_DEFAULT_SECONDS` (30) rather than the 45 it used to assume.

A reel is decided inside its first second and a half, and all of that rests on
one sentence. The scriptwriter prompt has always asked for a hook, but asking
is not checking. `HOOK_MODE=llm` spends a second Gemini call on **five
alternative openings**, each of which has to lead into the script's existing
second sentence.

**Nothing about the choice is left to the model.** `score_hook` ranks the
candidates here, in code — the same split step 6 makes with its character
limits, and for the same reason: a model asked to rank its own output will
explain why all five are excellent. The score is built from parts that can be
read back, and every run prints them:

```
   2.50  Rain is the reason your ocean tastes like it does.
         length +1.5, second person +1
  -1.20  Have you ever wondered why the ocean is salty?
         length +1.5, throat-clearing -4, second person +1, asks something +0.3
```

Throat-clearing (`have you ever`, `in this video`, `did you know`) and hype
adjectives cost; a number, a named thing and direct address pay.

**What the parts are worth was measured, and two of them were wrong.** Across
the 55 candidates the pass had generated at the time, `length` fired on 80% of
all candidates and 92% of the winners — it is hygiene, not a virtue, because
the prompt asks for one short sentence and almost everything arrives short.
Paid at 1.5 it lifted the whole field by the same amount and left the terms
that actually discriminate fighting over the remainder. A digit fired on 24% of
candidates and 42% of winners: it was the largest single discriminating term in
the scorer, and it decided hooks on its own. `length` is now 0.4 and `a number`
0.8, level with the other real virtues. What that was costing:

```
   3.70  95% of the biggest living space on Earth is unexplored.
         length +1.5, a number +1.5, names something +0.7
   1.50  We know more about the surface of Mars than we do this place.
         strong opener +0.8, names something +0.7
```

The first opened a shipped reel. It won for having a digit in it.

**Two sentences are not one of the costs.** They used to be, at a flat −2.0
each, on the reasoning that a second sentence means the hook has eaten the line
after it. That is a real failure — but it is a failure of *length*, which the
word rules already charge for, so the rule was billing twice for one fault and
taking the strongest opening shape in the format down with it:

```
  -1.80  The deep sea should crush every creature flat. It doesn't. Here's why.
         length +1.5, more than one sentence -4, names something +0.7
   3.00  Snailfish live 11 kilometres down, where pressure is extreme.
         length +1.5, a number +1.5
```

The first came last of five and the flat statement of fact won. Extra sentences
are now charged only when the hook is *also* over the word ceiling — which is
what having eaten the next line actually looks like — with a smaller charge for
a third sentence, because that is a paragraph. A short sentence that **reverses
the one before it** (`turns on itself`, +0.8) pays instead: the negation is what
separates "It doesn't" from "It has for ages", which is the same length in the
same position and merely continues the line.

**A weight can only choose from the field it is shown.** `second person` is
the largest single virtue in the scorer at 1.0 — above `a number`, `strong
opener` and `names something` — and for a long time nothing in the prompt
asked for one. Of the 135 candidates on disk, 10 (7%) put the viewer in the
sentence, and 4 of those 10 won their script: a 40% win rate against a field
average of 16%. The scorer was not turning them down. It was shown one about
every third run, and the runs it was not shown one look like this:

```
   2.00  They entombed the reactor in 400,000 cubic meters of concrete.
         length +0.4, a number +0.8, strong opener +0.8
```

That won a field of five with no viewer anywhere in it. The prompt now asks
for **one or two of the five** to put the viewer inside the fact — one or two
rather than all five, because much of what the pass chooses between is
material the viewer is not standing in, and a `you` reached for where the
fact has no honest use for one buys 1.0 and spends more than that on sounding
like an advert.

**An opening is not a free-standing sentence — it is the first half of a
join**, and for a long time nothing here had ever been shown the script. Every
rule above scores a hook on its own, while `apply_hook` swapped sentence one
and left sentence two exactly as it was. Three shipped reels show what that
produced:

```
95% of the biggest living space on Earth is unexplored.
It's our deep ocean. Ninety-five percent of it is unexplored.   <- said twice

Ocean salt comes from rocks. And there's no drain.
... So where does all the salt come from?                       <- answered, then asked

The ocean contains 50 quadrillion tons of salt.
Especially when all the water that flows into it is fresh.      <- nothing to attach to
```

Two changes. `score_hook` now takes the body it would be joined to and charges
−1.5 for **the body says it again**: the hook's content words against the first
three sentences after it, at an overlap of 0.50 or more. Measured over the
twelve scripts on disk the deep-ocean opening scores 0.67 there and the next
highest script scores 0.33 with no restatement in it, so the line is drawn
halfway between the one real fault and the worst innocent case. Sentences under
four words are skipped, because the Inca reel's `"...without the wheel." /
"Without the wheel."` is emphasis and the best thing in that script; both sides
need three content words, because `_overlap` divides by the smaller side and a
two-word sentence sharing one word scores 0.50 on its own.

The same Gemini call now also returns a **bridge**: sentence two, rewritten to
follow from the new opening, or `""` if it already does. It is checked before
it is used — a bridge that grew past twice the length is the model continuing
the script rather than mending it, and one that dropped a digit the original
carried has deleted a fact on the way through. Either way the original sentence
stands, which is exactly where the pipeline was before the bridge existed.

**The scriptwriter's own opening is scored on the same scale and stands as
the incumbent**, so a weak batch of candidates leaves the script alone instead
of making it worse — which is the whole reason the scoring is not a prompt. It
is not exempt from the restatement rule either: a scriptwriter that opened on a
line its own next paragraph repeats has made the same mistake.

The winner also yields `hook_card`: the same hook cut to `HOOK_CARD_MAX_WORDS`
for step 5 to burn over the opening. It rides through step 4's manifest, so
step 5 never has to go back to step 2's sidecar for it.

Every failure path — no key, a refused call, unparseable JSON — costs a warning
and nothing else. The hook pass improves a script that already exists, so it
can never be the reason there is no reel. `--no-hook` skips it.

### Step 2 — the shape

The hook pass buys the first second and a half. **A reel is decided at the top
and lost in the middle**, and everything after sentence one used to be whatever
the scriptwriter returned in one pass, unchecked. It showed. Scored across the
seven scripts this repo had already made, four of them had no ending:

```
kept       3.70  why-the-ocean-is-salty          closes the loop, a turn mid-reel, lands short
kept       2.50  nasa-evidence-of-caves          closes the loop, lands short
rewrite   -1.00  oakland-zoo-crimson-lion        the ending drifts, lands short
rewrite   -1.80  how-deep-sea-creatures-survive  the ending restates the opening, ...
```

That last one is the case worth reading. It opened on "Snailfish live 11
kilometres down" and closed on "Some snailfish live eleven kilometres down" —
thirty seconds of watching that arrive back where they started. Along the way
it stopped to ask itself "So how do fish survive?", which is exactly the
throat-clearing `score_hook` charges −4 for at position 0 and never saw at
position 12.

`STRUCTURE_MODE=llm` asks the scriptwriter for a shape in the first prompt and
then **checks whether it got one**, the same split the hook pass makes. Four
components, each measurable without a model:

| component | | what is measured |
|-----------|--|------------------|
| `closes the loop` | **+2.0** | the close returns to the opening's subject in different words |
| `the ending restates the opening` | **−2.5** | it returns in *the same* words — the payoff spent on something already heard |
| `the ending drifts` | **−1.5** | it never returns at all; the script stops rather than lands |
| `a turn mid-reel` | +1.2 | a "but"/"except"/"the part nobody mentions" between 25% and 70% of the way through |
| `lands on a short line` | +0.5 | the last line fits in one breath |
| `asks itself a question` | −1.0 each | "So how do they manage it?" — a question handed over in order to be answered |
| `argues with itself` | −2.0 each | two neighbouring sentences saying the same thing with the polarity flipped |

`argues with itself` is the one component that is not about shape. A reel
about sleep closed its loop, landed on a short line, scored 2.5 and shipped
with this in the middle of it:

> But they don't get less sleep than the rest of us. We get less sleep than them.

Heard once, at speed, that reads as a contradiction, and the shape rules cannot
see it — the script has a perfectly good shape. It is caught by measuring how
much two neighbouring sentences have in common when exactly one of them carries
a negation: above 75% overlap the second is the first restated backwards. The
threshold is what separates it from the *good* version of the move, which is
one of the strongest turns available — "The deep sea should crush every
creature flat. It doesn't." shares no content words at all, so it never fires.

The first three are one measurement read three ways — the word overlap between
the opening and the closing passage — so **exactly one of them ever fires**.
Number words are folded onto their digits before the comparison, because that
restating deep-sea script spelled "11" out as "eleven" the second time and
would otherwise have scored as a clean loop.

A script scoring at or above `STRUCTURE_MIN_SCORE` (2.0) **buys no second
call**. The threshold is deliberately the score of a script that closes its
loop and does nothing else, which makes the line it draws one sentence long:
close your loop and you are never rewritten for polish; fail to and you always
are, whatever else the script has going for it.

Below it, one Gemini call rewrites the body — and is **told what was wrong**,
in the words of the components that fired, rather than being asked to make it
better. What comes back is kept only if it clears three checks in code:

* the opening it was told to keep came back unchanged. It is handed over as
  fixed, so whether it *stayed* fixed is checkable rather than assumed — a
  model that rewrote it anyway has thrown away the line the hook pass scored
  five candidates to choose, and **the whole rewrite is dropped**. It used to
  be repaired instead, by pasting the chosen opening back over the rewrite's
  own. That repair is not available here: pasting swaps sentence one and
  leaves sentence two written to follow the sentence just deleted, which is
  the seam the hook pass carries a `bridge` to mend, reopened where there is
  no bridge left to mend it with. The Chernobyl reel shipped that way — the
  bridge `And the cost of that concrete was just the beginning.` was written
  to set up a rebuttal three lines later, the rewrite dropped it, the paste
  could not put it back, and `Not even close.` went out answering nothing.
  Falling back leaves a worse-shaped script that still says something;
* the length is still the one step 0 asked for. A rewrite half as long again is
  a different reel, and step 5 has already been told the old number;
* **it scores better than the script it replaces.** Same incumbent rule as the
  hook pass, same reason: a pass that can only improve a script can never be
  the reason there is no reel.

The rejected rewrite is scored and kept in the sidecar anyway, so step 9 can
ask whether the pass was worth running rather than only seeing its successes.
`--no-structure` skips it; `PERFORMANCE_EXPLORE=structure` runs it on every
other reel, which is the only way to find out whether any of this is true.

The two passes run **hook first, then structure**. The structure pass writes an
ending that pays off whatever the script opens with, so it has to see the
opening the hook pass settled on; reversed, the hook pass would swap the
opening out from under an ending written to close it.

### Step 2 — the take-away

The two passes above buy attention. The hook wins the first second, the shape
holds the thirty after it. Neither asks what the viewer is *holding* when it
stops, and the account says that is where it fails: across the seven reels
posted to 2026-09-08, watch-through runs 0.38 to 0.68 — the retention those
passes were built for, working — and **saves and shares are 0 on every one of
them**. A reel nobody keeps and nobody sends is shown to nobody else, whatever
its watch-through.

What gets kept is a reel that changed what the viewer will notice tomorrow;
what gets sent is one they can retell in a sentence. Both need the script to
hand something over — where they will meet the thing, what it looks like when
they do, or what it is called — so `TAKEAWAY_MODE` scores exactly that:

| component | worth | what it is |
| --- | --- | --- |
| puts the viewer in front of it | +2.0 | `next time you…`, `you'll see it in…` |
| names the thing | +1.2 | `it's called creep`, `they call it the Wood Wide Web` |
| the viewer is in the close | +0.8 | second person in the last few sentences |

Measured over the 29 scripts in `output/scripts/` before any of this existed:
**1 of 29** puts the viewer in front of the thing, 3 name it, and **23 score
nothing at all**. The endings are not bad — they are good writing that hands
over nothing. `It's the tilt.` `The ocean has no drain.` `The green just got
out of the way.` Every one of those closes its loop and lands short, which is
every virtue the shape pass knows how to pay for.

A 1-in-29 base rate is the hook pass's second-person finding again: a virtue
the scorer would pay for that the prompt never asked for. So it ships as
**both** — `PROMPT_TEMPLATE` asks for a take-away and this pass checks, because
asking is not checking.

Two things are deliberately **not** scored. There is no penalty for an ending
that hands nothing over — absence simply scores nothing and falls under
`TAKEAWAY_MIN_SCORE` — because the last lexicon written to judge endings here
flagged five shipped closes to catch one bad one. And a figure with a unit is
not paid for, though it looks like it belongs: it fires on 15 of the 29
scripts, which describes the house style rather than separating one reel from
another.

The pass runs **after the structure pass**, and that is the mirror of the
hook/structure argument: everything it is asked for lands in the closing
sentences, which is exactly where the shape pass measures the payoff. So it
writes into the ending the shape pass just built — and then re-scores that
shape, refusing any rewrite that hands the viewer something by flattening the
ending. A hand-over bought that way trades the line they would have remembered
for the thing they were supposed to keep.

**The weights are a prior, not a fit.** Nothing in the ledger has ever been
saved, so there was nothing to fit them to; read them as an ordering. That is
why `PERFORMANCE_EXPLORE=takeaway` is the trial currently running — the `off`
arm is the only thing that can ever say whether the pass earned its place.
`--no-takeaway` skips it.

### Step 2 — the claims

The three passes above ask whether the script is *shaped* like a reel and
whether it hands anything over. None of them asks whether it is true, and until
`VERIFY_MODE` nothing else in the pipeline did either. Two shipped reels are
what that cost.

The glacier reel opens `This mountain of ice is moving at your walking speed.`
and says, eleven sentences later, `This happens slowly. A few centimetres a
day.` Those differ by a factor of about four million. The body is right — the
opening was written by the hook pass, and nothing compared the two.

The Oakland Zoo reel is worse:

```
This mountain lion looks like it walked out of a painting. Now imagine one
that's crimson red. That's what scientists found in California's Santa Cruz
Mountains. Her name is P-Twenty-Two. ... She's now safe at the Oakland Zoo.
Come see her.
```

There is no such animal. P-22 was a real cougar, in Griffith Park in Los
Angeles, and was not red. The reel invents an animal, gives it a real animal's
name, puts it in the wrong mountains and closes by telling viewers to visit a
**named real institution** to see it. That script is not badly shaped — it is
well shaped, and every rule in the passes above scores it fine.

`VERIFY_MODE=llm` spends one Gemini call asking two questions: does the script
disagree with itself, and does it assert anything untrue. What comes back is
checked the way everything else here is checked — in code, against the script.
**A finding has to quote a sentence the script actually contains** (content-word
overlap ≥ 0.8, so a quote that straightened an apostrophe on the way through
still matches). A finding about a sentence that is not there is a finding about
some other script, and acting on it means repairing a fault that was never in
this one.

Findings are `contradiction`, `false`, `unsupported` or `figurative`. **Only
the first two block.** `unsupported` is the model saying it cannot confirm
something, which is true of most interesting claims and would leave the
pipeline unable to ship anything at all; it is reported and nothing more.

`figurative` is there for the same reason, on evidence from
`output/scripts/refused/`. A hook is compressed on purpose — *"a silent water
pump with no moving parts"*, *"the fruit is the mother, but the seed had a
father"* — and read literally a compression like that is always false of
something. Given only the first three kinds, a checker told to find claims
that are wrong had nowhere to file a metaphor, so it filed it under `false`.
On 2026-09-06 that lost a run both of its topics: the tree hook was refused
over stomata, the repair softened it to *"almost no moving parts"* and it was
refused again on the same ground — no wording of that metaphor survives being
read as mechanics. Three earlier fixes to this pass all bought the repair more
room, and none of them could have helped, because the sentence was never the
fault. What still blocks is the reel that invents an animal: *"the
deep-freezing kills them instantly"* stays `false`, a literal claim about a
process, wrong by days, that one qualifier repairs.

A blocked script buys **one repair**, on the same incumbent rule the other two
passes use — the rewrite is re-checked, and kept only if it actually cleared
fewer faults than it started with. What survives that is refused outright as a
`ScriptGenerationError`, which step 8 already treats as this topic being a dead
end: it falls through to the next candidate on the board and makes a different
reel. A day with no reel costs a day. A day with a reel that invents an animal
at a named zoo costs more than that.

Every failure path — no key, a refused call, unparseable JSON — costs a warning
and never a block. A checker that cannot be reached must not be the reason a
reel does not exist, which is exactly where the pipeline stood before this pass
was written. `--no-verify` skips it.

The pass runs **last**, after all three rewrites: it is the only one that reads
the script for what it says rather than for how it is built, and checking
before the others would be checking a draft that no longer exists.

**And the card is re-cut after it.** The on-screen `hook_card` is cut from the
opening the hook pass chose, which is settled *before* the check runs — so a
claim the checker threw out of the narration was still being burned over the
reel, the cover and the alt text, where it is the most-read text in the whole
thing. On run `how-a-forest-talks-through-its-roots-531029b5` the checker
called `a network more complex than your own brain` false and the repair
rewrote the narration to `hides a vast, hidden network` — and the card, the
cover headline and the alt text all still read **A network smarter than you**.
The overclaim ended up *more* prominent after the repair than before it.

The card is now re-cut from the repaired opening, and only when that opening
actually moved: if sentence one came through the check untouched then the card
is exactly as true as the sentence it was cut from, and re-cutting would trade
a card the model wrote for a mechanical one for nothing. The re-cut is recorded
in the claim report's warnings.

**What it does not do.** It is given the script and the topic, not the source
material step 0 found. It catches a script arguing with itself and a claim the
model will state is wrong; it is not a citation check.

## Step 3 — the read

`pipeline/tts.py` speaks the script. One ElevenLabs request, written to
`output/audio/`, and the only thing downstream cares about is the file and its
measured length — step 4 times its beats against the real narration rather than
step 2's estimate.

**Delivery is configurable, and was not.** `stability`, `similarity_boost`,
`style`, `use_speaker_boost` and `speed` were ElevenLabs' own defaults written
into the module as constants — a neutral read for a format that is not neutral.
They are `TTS_*` knobs in `config.py` now, because a narrator is worth trying
variants of and step 9 can only run a trial on a setting it can set. Stability
moves to 0.4, the expressive end of the 0.4–0.6 narration band, and style off
zero to 0.25.

`TTS_SPEED` defaults to 1.08. It is the one number in the block that is a
judgement about the format rather than a measurement of this pipeline —
short-form narration sits a little above a natural read — and nothing in
`output/` can confirm it, because every reel so far was made at 1.0. Worth
running as a `PERFORMANCE_EXPLORE` arm rather than taken on trust.

**Speed and script length are the same quantity, so they are written down
once.** Step 2's `WORDS_PER_MINUTE` is `SCRIPT_WORDS_PER_MINUTE × TTS_SPEED`,
derived rather than stored, because a pipeline that speeds the narration up
without lengthening the script makes every reel short of the duration step 0
asked for. At 1.08 a 40-second reel asks for 108 words instead of 100 and still
lands on 40 seconds — denser and faster, not shorter. The base rate stays at
150: measured over the thirteen reels on disk, script word count against the
rendered narration's real duration comes out at 153.2 wpm mean and 152.7
median, so the assumption was already sound.

## Step 4 — visual sourcing

`pipeline/visuals.py` turns a script plus its narration .mp3 into an ordered set
of downloaded assets and a `manifest.json` for step 5.

* the script is cut into **visual beats**, timed against the *measured* length of
  the .mp3 rather than an estimate, so beat boundaries line up with the audio;
* each beat gets a search query — Gemini writes them by default
  (`VISUAL_QUERY_MODE=llm`), falling back to keyword extraction if the call fails;
* **what a clip shows is checked against what the beat says**
  (`VISUAL_RELEVANCE_MODE=llm`). Everything else here scores how well a clip is
  *shot* — vertical, sharp, long enough — and a stock library always returns
  something, so a query it cannot answer comes back full of well-shot clips of
  the wrong subject that score identically to the right ones. One Gemini call
  screens every beat's shortlist against its narration, reading the provider's
  own words for each file; rejects are demoted below the keepers rather than
  discarded, and a beat left with nothing is searched again on its fallback
  query. Lenient by design: a "grey-headed flying fox" is a match for narration
  about bats, and only a different *subject* — a bear where the words are about
  people — is a reject. The one exception is culture: another civilisation's
  artefact under a named people ("intricate stone carvings of ancient indian
  temple" over narration about Inca stonework) is rejected, because it states
  something false about a real people. A **modern stand-in is not** that error
  and is kept — a present-day rope bridge over narration about rope bridges is
  read as illustration, and for most of history there is no other footage to be
  had. An earlier draft rejected both and left three beats of a ten-beat reel
  with nothing at all;
* candidates are filtered by a **crop-aware resolution floor**: the test is what
  survives a centre crop to 9:16, so a 4K landscape clip passes and a 1080p one
  does not;
* **the opening beat is not scored like the others.** Everywhere else a still is
  honest filler; on the first shot it is the reel announcing that nothing is
  going to happen, over the second and a half in which the viewer decides. Beat
  0 pays a premium for motion and for a clip long enough to hold without looping
  back to its own first frame — enough of one that a landscape clip which has to
  be cropped beats a perfectly framed vertical still;
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
  the script to work out segmentation. Each beat's window is divided into shots,
  capped by how many assets that beat got;
* the shot length **follows a curve rather than a constant**. One rate for the
  whole reel is the obvious thing to do and the wrong one: a cut buys attention
  and costs a moment's re-orientation, and the exchange rate is not fixed. At
  the top there is no attention yet to lose, so cuts are nearly free; in the
  middle the viewer is following an argument and a cut a second is noise over
  the top of it. So `SHOT_HOOK_SECONDS` (1.4s) at t=0, easing out to
  `SHOT_TARGET_SECONDS` (3.2s) over `SHOT_PACING_RAMP_SECONDS`, then tightening
  to `SHOT_OUTRO_SECONDS` across the close so the reel lands rather than fades.
  The count comes from **integrating** the curve across each beat, not sampling
  it once — a long opening beat's midpoint already sits past the ramp, and
  sampling would quietly cancel the feature on exactly the beat that matters.
  `SHOT_MIN_SECONDS` gives way to the curve rather than fighting it;
* **no shot may overrun the curve by more than `SHOT_MAX_OVERRUN` (1.25x).**
  This is the ceiling that actually binds. `SHOT_MAX_SECONDS` sits at 6.0s
  while the curve asks for 3.2s, so it never fires, and shot length was really
  being decided by the rounding: a window between 1.0 and 1.49 targets long
  rounds down to one shot and then holds for all of it. Measured across the
  thirteen reels on disk, the longest shot in the whole set is 4.81s against a
  3.2s target — 1.50x, exactly the rounding boundary — and **28 of the 32 shots
  over 3.5s were beats holding one clip while a second already-downloaded clip
  for that same beat went unused.** The footage was there; the arithmetic threw
  it away. Charging the overrun against the curve rather than against a
  constant is the only ceiling that means the same thing at the top of a reel
  and in the middle of one. Over those thirteen reels it takes the 34 shots
  past 3.5s down to 5, the longest from 4.77s to 3.99s, and the median from
  2.66s to 2.40s, while nothing lands under the 1.6s floor. It can never invent
  footage: the count is still capped at twice what step 4 sourced for the beat;
* **cuts land on words.** Step 5 already knows when every word is spoken,
  because it aligns the narration to write the captions — so each cut is nudged
  up to `SHOT_SNAP_MAX_SECONDS` onto the nearest silence between two words, and
  onto the attack of a word when there is no silence in reach. Without it a cut
  falls wherever dividing a beat by a shot count happened to put it, which is
  reliably halfway through a word: the picture changes mid-syllable and the edit
  reads as something that went wrong. This is why the alignment now runs
  **before** the timeline is planned — it decides where the cuts go, so it
  cannot be worked out afterwards;
* **the opening cuts land on the beat instead.** A cut on a beat is not heard
  as a cut at all — it is heard as the track doing it, which is the whole trick
  behind an edit that feels scored rather than assembled. It is confined to the
  first `SHOT_BEAT_WINDOW_SECONDS` (8s), where the music's fade-in has just
  finished, the narration has not built up and the cutting is at its fastest;
  past that the bed has settled under the voice and cutting to a tempo nobody
  is listening for buys nothing. The grid claims its cuts **before** the word
  pass runs, because the other order does not work — the words would find a
  nearer pause for every cut the music had just placed. **A beat falling inside
  a spoken word is refused**, and that boundary drops through to the word pass:
  cutting mid-word is the defect the pass above exists to prevent, and the
  music does not get to override it. Needs a tempo written beside the track —
  see [the music bed](#the-music-bed);
* clips are centre-cropped to 9:16, trimmed to their slot, and **looped** when
  they are too short. A clip with room to spare cuts in a little way past its
  start, where stock footage stops fading up;
* **every shot moves** — see [motion](#motion) below;
* captions are timed by **ElevenLabs forced alignment** — the script is already
  known, so the words are exact rather than transcribed. `whisper` and a
  no-network `estimate` are the fallbacks, tried in that order;
* **the cue pops as it arrives, and the payoff word is made.** The highlight
  already keeps up with the voice; what it did not do is *arrive*, so three
  words replacing three other words in the same place at the same size read as
  a slideshow. Each cue now lands at `CAPTION_POP_SCALE` (112%) and settles
  over `CAPTION_POP_MS` (90ms). A **payoff word** — the number the sentence was
  built to deliver — gets its own colour and a harder 124% pop as it is spoken.
  The detector reads number *words*, not digits: step 2 is told to write
  numbers the way they should be said aloud, and across every script on disk
  there are 39 digit characters in twelve reels against "quadrillion",
  "ninety-five" and "thousand" spelled out. Two constraints keep it from
  reading as an animation: **a pop never interrupts a pop** (two accents inside
  one settle is a wobble, and "ninety-five percent" is two payoffs a breath
  apart), and **a cue with no room left does not pop**. Scaling happens after
  libass has chosen the line breaks, so popping a cue that already fills the
  frame wraps it onto three rows and snaps it back inside 90ms — measured on
  the salt reel, where "50 QUADRILLION TONS" did exactly that. Cues with room
  pop the full amount; the rest pop less, or not at all, which costs nothing —
  fitting every cue to its popped size instead would shrink every caption in
  the reel permanently for the sake of 90 milliseconds;
* the narration is **loudness-normalised to -14 LUFS**, Instagram's own
  working point, so two reels cut a week apart do not land at different
  volumes. ElevenLabs renders at whatever level the voice sits at — one real
  narration measured -24.4 LUFS in and -15.6 out. `AUDIO_NORMALIZE=false`
  leaves it alone;
* a **music bed** is mixed under the narration, **ducking** out of its way
  while the voice speaks and lifting in the gaps between lines. The bed is
  loudness-matched too, so a quiet track and a mastered one sit in the same
  place;
* the reel **cuts straight in** — no fade up from black. A reel is judged in
  its first half-second, which is exactly the stretch a fade spends showing
  nothing. The tail still fades;
* a **hook card** is burned over the opening: step 2's hook in big type, high in
  the frame, while the narration says it. This is the muted-viewer path — feeds
  autoplay without sound, and the running captions below are a transcript
  arriving three words at a time, which is fine once someone is watching and
  useless for deciding whether to. It rides in the same ASS file as the captions
  rather than a separate `drawtext` pass, because libass is already being run
  over the picture;
* **whooshes ride the cuts** and an impact lands on the open — see below;
* a **cover frame** is written alongside the reel — see [the cover
  frame](#the-cover-frame) below;
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
| `cover.jpg` | the still Instagram shows in the grid, at the same 1080x1920 |
| `cover.ass` | the card-only subtitle file the cover was drawn through — kept, because when a cover looks wrong this is the thing to read |
| `captions.ass` / `.srt` | the burned-in captions, and a plain copy |
| `alignment.json` | every word with its timestamp, and which backend timed it |
| `attribution.txt` | credits for the footage that made the cut |
| `assembly.json` | the shot-by-shot timeline, the music bed used, every effect and where it landed, **which frame became the cover and why**, plus **step 4's per-asset attribution**, carried through for steps 6 and 7 |

### Motion

Stills have always moved. Clips never did — they were scaled, centre-cropped
and left alone, so a reel cut entirely from stock video was a sequence of
static frames that happened to have things moving inside them. **Every reel
this has made was cut entirely from stock video**: 16 shots in the last two,
not one of them with any camera move. The difference between an edit and a
slideshow is that the camera is doing something, and a slow push costs one
filter.

* the **opening shot pushes in harder** than the rest (`SHOT_MOTION_HOOK_ZOOM`,
  1.18) and never pulls back. It is on screen for the second and a half in
  which the viewer decides, and it is the frame the cover is drawn from — a
  pull-back there moves away from the subject at exactly the moment the reel is
  asking to be watched;
* every other clip moves by `SHOT_MOTION_ZOOM` (1.08), **pushing and pulling in
  turn**. One direction repeated sixteen times is a mechanical tic rather than
  an edit, which is the thing being fixed and not a milder version of it.
  `SHOT_MOTION=push` moves them all the same way, `off` restores the static
  picture;
* the move is small on purpose. 1.08 over three seconds is under 3% a second,
  which reads as a camera; past ~1.15 on a long shot it reads as a zoom, and
  the crop eats real resolution;
* **`IMAGE_ZOOM` still governs stills and `SHOT_MOTION` governs clips**, because
  they are different problems: a still has no motion of its own at all, while a
  clip only needs enough to stop reading as a locked-off frame. Turning clip
  motion off has never been a reason to freeze the stills. A still does take its
  turn in the alternation count, so the clips either side of one carry on in the
  order they would have had — it simply does not honour the "pull" half;
* motion is **planned, not decided at render time**. Two neighbours showing the
  same picture are folded into one shot first, and that fold renumbers
  everything after it; reading the direction off the pre-merge numbering would
  leave two shots running the same way where the seam was.

**Why `zoompan` and not something cheaper.** A zoom needs the crop rectangle to
change size every frame, and ffmpeg's `crop` evaluates its width and height once
at configuration time — only `x`/`y` move per frame, which buys a pan and not a
push. `zoompan` is the filter that does it, and on a clip it runs with `d=1`,
emitting one output frame per input frame.

That last detail is the whole risk in this feature, because **`d=1` counts input
frames**: a 24fps source left unnormalised hands back 24 frames labelled as 30,
and the shot plays a quarter too fast. So the `fps` filter goes *before*
`zoompan` rather than after it. Measured against the static path on real 24fps
and 30fps sources, seeked and looped:

```
case                      want  moving  static
4K 30fps clip, 1.4s         42      41      41
HD 24fps clip, 3.0s         90      89      89
still, 2.6s                 78      78      77   <- zoompan is the exact one
clip w/ -ss, 3.0s           90      89      89
looped clip, 3.0s           90      89      89
```

Every clip case is **frame-identical moving or static**, so nothing about the
timeline, the captions or the sound moves when this is switched on. (The
one-frame shortfall against `want` is the `trim` boundary and predates this; it
does not accumulate — every finished reel still lands exactly on its planned
duration, because the graph takes its length from the audio.)

Each shot is scaled to `MOTION_OVERSAMPLE` (1.5×) before `zoompan` sees it,
because zoompan quantises its zoom to whole input pixels and a slow push out of
a 1:1 source advances in visible steps. It is why a moving shot carries no
`crop` to the output size — zoompan's own `s=` lands it there at the end of the
chain. `assembly.json` records the direction and distance of every shot under
`motion`, and `stats.moving_shots` counts them.

### The cover frame

The reel earns the watch. The cover earns the tap that gets it played at all —
it is the only part of a reel that is judged with no motion and no sound, in a
grid of other people's covers.

**It is not a frame grabbed out of `reel.mp4`.** That was the obvious
implementation and it does not work: the running captions are burned into every
frame of the finished file. Measured on a real run, the first cue is up at
0.079s and there is one on screen essentially continuously — so a grab from the
video puts three words out of the middle of a sentence across the bottom of the
tile. `KILOMETRES DOWN, WHERE` is not a poster.

So the cover is rebuilt from the source asset with the edit's own scale and
crop, and the hook card is redrawn over it through the same `build_ass` path the
video uses — which is the point of doing it that way rather than with a
`drawtext` pass. The headline on the cover cannot drift from the headline in the
reel, because the same code sets both. The one difference is the fade:
`HOOK_CARD_FADE_MS` is suppressed, since a cover is a single frame at t=0 and in
the reel that instant is the middle of the fade-in.

**Which frame.** Every shot is scored — a different question from step 4's
`score_candidate`, which asks whether a clip can carry a beat:

* how long the assembler gave it, capped at `SHOT_TARGET_SECONDS`. A long slot
  means the beat had footage worth holding on;
* whether it needs cropping at all. Native vertical shows the whole picture as
  it was shot instead of a slice out of the middle;
* resolution headroom over `VISUAL_MIN_HEIGHT`;
* **minus** for a looped shot (too short for its slot — a scrap, not a hero
  shot) and for a reused one (already on screen elsewhere, so nothing
  distinctive about it);
* **minus** a little for how far down the reel it sits. Only shots opening
  inside `COVER_SEARCH_SECONDS` are candidates at all: the cover should show
  what the reel promises, and a frame from the payoff at 0:48 is a picture of
  the answer to a question the browser has not been asked.

The frame is then taken ~45% into that shot and never within 0.25s of either
cut, so the picture has settled and the subject has not yet left frame. A clip
is seeked to the same in-point the edit uses, so the cover is a frame the reel
genuinely contains.

**Card or footage.** `COVER_SOURCE=auto` prefers the hook card, for the same
reason the card exists at all: the grid is a browse surface, and a line of type
says what a reel is about where a beautiful shot of open water only says that it
is blue. In card mode the frame comes from the best of the shots the card
actually sits over — a shot catching only the last fraction of the card is
skipped, since there is barely any of it in the window.

`auto` falls back to footage when the card would not survive being cropped.
Instagram centre-crops covers differently in the grid, the profile and search,
and has changed which more than once, so `COVER_SAFE_ASPECT` is a knob rather
than a constant. At the 1080x1920 default the card sits 320px down:

| crop | visible band | the card |
|------|--------------|----------|
| 4:5 | 285–1635px | fits |
| 3:4 | 240–1680px | fits |
| 1:1 | 420–1500px | **first line cut off** |

`hook_card` keeps the card regardless and says so; `footage` never uses it.

Everything about the choice — which shot, which frame, which mode, the reason in
words, and whether the card cleared the crop — goes into `assembly.json` under
`cover`, so step 9 can put it against how the reel actually did. `--dry-run`
makes the whole decision and writes no file, exactly as it plans a timeline
without encoding one.

### The music bed

Tracks come from `assets/music/` and nowhere else — nothing is downloaded and
nothing is generated, so whatever you put there is yours to have licensed.

* an **empty `assets/music/` means no music**, which is how a fresh clone
  arrives. Drop a track in and the next reel has a bed;
* with several tracks, step 5 **rotates** between them
  (`MUSIC_REUSE_COOLDOWN_RUNS`) so the same bed does not sit under every reel.
  `--music <file>` pins one, `--no-music` goes without;
* the bed is normalised to `MUSIC_LOUDNESS_TARGET` (-30 LUFS — about 16 LU
  under the narration). **That is the knob that decides how loud the music
  is**, and normalising means the level does not depend on how hot the track
  you happened to download was mastered;
* **ducking** is a compressor keyed off the narration, so the bed drops while
  the voice speaks. Measured on a real narration, the bed sits about 5 dB
  higher in the pauses than under speech. That is close to the ceiling: the
  reduction is bounded by how far speech rises over `MUSIC_DUCK_THRESHOLD`, so
  raising `MUSIC_DUCK_RATIO` past 8 changes nothing;
* a **`track.txt` beside `track.mp3`** holding a credit line is treated as a
  credit that has to be used — it goes into `attribution.txt` and step 6 puts
  it in the post text, because a credit that only lives next to the .mp4 is not
  a credit. No sidecar means no credit is claimed, and the licence is yours to
  check;
* the same sidecar can carry the track's **tempo**, which is what the opening
  cuts are snapped to:

  ```
  "Autumn prelude" by zero-project (CC BY 3.0)
  bpm: 92
  offset: 0.21
  ```

  `offset` is where the first downbeat sits inside the file, for a track that
  does not open on one; any downbeat will do, since it is folded back inside a
  single beat. **A settings line never reaches the credit** — that is the part
  worth being careful about, because the credit is copied verbatim into
  `attribution.txt` and from there into the public post text, so a stray `bpm:`
  in it would be published as part of somebody's name. A sidecar with no
  settings line is read exactly as it always was. `MUSIC_BPM` is the fallback
  for a track with none of its own, and a tempo outside 30-300 is treated as a
  typo and dropped with a warning rather than believed;
* **`bpm: none`** says the track was measured and has no constant grid in it.
  That is a different thing from having no `bpm:` line, which means nobody has
  looked, and the run says so differently: a bed with no pulse is not a job to
  go and do, so it is stated once instead of asking for a number every day
  about a file that will never have one.

  The five beds in `assets/music/` were measured by fitting a constant grid to
  their tracked beats and reading the residual — the same grid step 5 builds
  from one number and an offset, so the test is the thing the feature actually
  needs. Three have a tempo and two do not, and it is not a close call:

  | bed | tempo | residual |
  |---|---|---|
  | `documentary-background` | 100 | 7.5ms |
  | `millennium-sunset` | 120 | 7.9ms |
  | `silent-dreams` | 120 | 14.0ms |
  | `plathe-beatless-ambient-mix` | none | halves disagree, 135 vs 90 |
  | `autumn-prelude` | none | 1178ms |

  For scale, pure noise fits a grid to 110ms and a synthetic click track to
  6.7ms. `documentary-background` had previously been written down as `bpm:
  99.4` — the beat-tracker's estimate rather than a fit — and at that tempo the
  same beats fit to **165ms**, drifting a quarter of a second out of sync
  across a 40-second reel. It is 100.

A track that ffmpeg cannot read is dropped with a warning rather than failing
the reel, like everything else here.

### Sound effects

A whoosh across a cut is the cheapest production value in the pipeline: it costs
one small file, and it turns a change of picture into a deliberate edit.

One-shots come from `assets/sfx/` and nowhere else — nothing is downloaded and
nothing is generated, exactly like the music bed, so whatever you put there is
yours to have licensed. **An empty `assets/sfx/` means no effects at all**,
which is how a fresh clone arrives and what every reel cut before this sounded
like.

* the **role is read off the front of the filename**, so `whoosh.wav`,
  `whoosh-02.wav` and `whoosh_soft.mp3` are all whooshes and step 5 rotates
  between them. `impact*` lands on the first frame — the one effect with no cut
  to ride, which is the point of it. A file whose name matches no role is
  **reported**, not ignored: `wooosh.wav` is a typo, and silently doing nothing
  with it is how that typo survives for months;
* each file is **levelled to `SFX_PEAK_TARGET` (-14 dBFS) by measurement**, so a
  hot one-shot and a quiet one land in the same place instead of the mix
  depending on what you happened to download. Peak rather than LUFS on purpose:
  these are 200ms transients, and EBU R128 integrates over 400ms blocks, so it
  would hand back nothing usable for exactly the files this is for. Peak is also
  what a sound editor levels a one-shot by;
* a whoosh is placed **`SFX_LEAD_SECONDS` before its cut**, so the transient
  lands *on* the change of picture. One that arrives after it is heard as a
  separate noise instead of as the cut itself;
* **cuts closer together than `SFX_MIN_GAP_SECONDS` get one effect, not two.**
  The pacing curve opens on ~1.4s shots, and an effect on every one of them
  stops being punctuation and becomes noise. Raise it if the reel feels busy;
* the effects are mixed **over** the finished audio rather than through the
  ducking with it. They are transients whose whole job is to punch, and a
  compressor keyed off the narration would pull each one down at exactly the
  moment it fired.

A file that cannot be measured, or that turns out to be silent, is dropped with
a warning rather than failing the reel — and never amplified 70 dB into its own
noise floor to reach the target.

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
* hashtags are **mixed across bands** — broad to place the post, mid to say what
  it is, niche for the people already deep in the subject — and a cross-run
  ledger keeps consecutive posts from carrying an identical block. Which band a
  tag belongs to is decided by its post count, not by the generator's opinion:
  see [hashtag volume banding](#hashtag-volume-banding) below;
* footage whose licence requires a credit gets one in the form that licence
  expects, carried through from step 4 via step 5's `assembly.json`. Pexels and
  Pixabay require none, so most runs add nothing (`POST_ATTRIBUTION`);
* **alt text for step 5's cover** is written from what step 5 recorded about it
  — see [alt text](#alt-text) below.

Over the character limit, blocks are given up from the bottom: the courtesy
credit first, then body paragraphs, then the CTA, then tags. The hook and any
licence-required credit are never given up.

What it adds to `output/<date>/<run id>/`:

| file | contents |
|------|----------|
| `post.txt` | the caption, exactly as it should be posted, and nothing else |
| `post.json` | the same in parts: hook, body, CTA, tags with their band and post count, credits, the cover's alt text, character counts, warnings |
| `post_comment.txt` | the tag block — only when `POST_HASHTAG_PLACEMENT=comment` |
| `post_alt.txt` | the cover's alt text, for Instagram's own field — only when there is a cover to describe |

### Alt text

The description of step 5's cover that goes in Instagram's **Write alt text**
field — a different field from the caption, which is why it gets its own
`post_alt.txt` the way the tag block gets `post_comment.txt`. It does two jobs
and they do not pull against each other: a screen reader reads it aloud, and
Instagram reads it for what the picture is of.

It is a description of an **image**, not a second caption. No hook, no selling,
no hashtags, and short — `POST_ALT_MAX_CHARS` is Instagram's field limit, and it
is enforced in code after the call like every other limit here.

**Where the words come from matters more than how they read.** The obvious
source is step 4's `image_prompt`, which is already a one-line description of
what the beat shows — and it is the wrong one. It describes what the beat *asked
for*, and stock search does not always answer. From one real run:

| step 4 asked for | the clip it actually used |
|---|---|
| A snailfish swims in the deep ocean. | medusa blue ocean *(a jellyfish)* |
| A dark, high-pressure environment in the deep ocean. | serene scuba diving in clear waters *(bright)* |
| A microscopic view of an animal cell. | vertical stop motion of paper cutouts |

Alt text built from that would tell someone who cannot check it that they are
looking at a snailfish in a dark trench, when the cover is a jellyfish in bright
blue water. That is worse than no alt text. So the description comes from the
words the provider put on the file that was **actually used**, in this order:

1. the provider's own title — Pexels' alt text, Pixabay's tags, Openverse's
   title, kept by step 4 for exactly this;
2. the slug in its source URL, which names the clip even when the alt text is
   empty — `medusa-blue-ocean-13693981`;
3. the search query that found it;
4. the topic, and the run says so when it comes to that.

**Any text burned into the cover is quoted exactly**, and comes first when the
field runs short. A cover that is mostly a headline is unreadable to a screen
reader user unless the alt text says what the headline said, so the description
of the picture gives up its characters before the quotation does.

```
Medusa blue ocean. Text on the image: "Snailfish: extreme pressure".
```

The two stay separate sentences rather than one clause. The description arrives
either as a lower-case slug or, in `llm` mode, as a capitalised sentence, and
there is no join that reads correctly for both — `"..." over A jellyfish drifts
in blue water` is what gluing them together produces.

`POST_ALT_MODE=llm` has the caption call rephrase the description into a
sentence — one more field in the JSON it already returns, not a second request.
It only ever touches the description; the quoted headline is a quotation. The
default is `literal` because the model has not seen the picture: asked to
describe one, it writes something plausible, and the `image_prompt` table above
is what plausible looks like when it is wrong.

A run whose `assembly.json` has no cover — a bare script, or a reel cut before
step 5 chose covers — gets no alt text and a warning, rather than a generic
sentence about the topic.

### Hashtag volume banding

Asking a model for "niche" hashtags gets tags that are niche *as words*. That
is not the same thing as a tag anyone can find. Real output from this pipeline,
before this existed:

```
#MarineBiology #OceanLife #DeepSeaFish #Snailfish
#HighPressure #CellBiology #Piezophile #Osmolyte
```

`#Piezophile` and `#Osmolyte` are the long-tail slots, and both are empty — a
few hundred lifetime posts, no browsers, nothing to be found by. Two of eight
slots spent on characters. At the other end `#MarineBiology` has millions of
posts and a small account will never place in it.

`pipeline/tagvolume.py` puts a post count against every tag the generator
offers and re-files it by what it measures, not what it was labelled:

| band | posts | what it is for |
|------|-------|----------------|
| dead | below `TAG_VOLUME_FLOOR` | dropped — no browsers |
| niche | .. `TAG_VOLUME_NICHE_MAX` | small enough for a small account to rank in |
| mid | .. `TAG_VOLUME_MID_MAX` | the band that actually converts |
| broad | .. `TAG_VOLUME_BROAD_MAX` | places the post in a subject area |
| saturated | above that | capped at `TAG_VOLUME_MAX_SATURATED` — a token, not a strategy |

Counts come from the first of these that has one: a figure in
`output/captions/hashtag_volume.json` (hand-seeded entries never expire and
always win), then `TAG_VOLUME_PROVIDER` if you have a data source, then the
built-in estimator.

**There is no Instagram provider to point this at.** The Graph API's hashtag
node returns `id` and `name` and no post count, and caps an account at 30
unique hashtag lookups per rolling 7 days — less than one week of posts at
eight tags each. It cannot answer the question.

So the default is an **offline estimator**: it reads a tag as the words it is
built from and prices it on how ordinary they are and how many there are, with
prefixes and suffixes taken off so `#Bioluminescence` and `#Oceanic` are read
as the real words they are. It is a rarity model, not an oracle — coarse in the
middle, and every figure it produces is stamped `estimate` in `post.json`. Its
job is to throw out the empty tags, and it does that reliably.

Where it is weakest is a subject it has no vocabulary for, which looks exactly
like a subject full of coinages. So a verdict that condemns most of an
estimated pool is treated as evidence about the estimator rather than the tags:
nothing is dropped, the suspect tags are ranked last, and the run says so.

```
! 4 of 8 tags priced as too small to find, which is more likely the estimator
  having no vocabulary for this subject than 4 bad tags - kept, but ranked
  last. Seed real counts in hashtag_volume.json to settle it.
```

See what it makes of a set of tags without running anything else:

```
python pipeline/tagvolume.py Piezophile DeepSeaFish OceanLife Snailfish
```

And overrule it wherever it is wrong, with the figure Instagram shows on the
tag's own search screen. A seeded count is authoritative, beats the estimator
and never expires:

```
python pipeline/tagvolume.py --set marinebiology=4.2M oceanlife=430K trees=9,500
```

Worth an hour once, for the tags the account actually posts under — until then
every band in every caption is a guess, and two of the estimator's failures are
structural rather than merely coarse. Tags built from the same shape of words
collide on one figure: `#biology`, `#botany` and `#trees` all read as 3,162,277.
And a common term of art reads far smaller than it is — `#MarineBiology`, the
tag named above as having millions of posts, estimates at 112,201, which files
it in the one band a small account is told it can rank in.

Which hour, though — "seed the tags the account posts under" is advice you can
only follow if you know which those are, and the tags the estimator *threw out*
used to leave no trace but a sentence in the log. They are recorded now, in
`post.json` under `hashtag_rejected`, and `--worklist` reads every post back
and ranks them: thrown out first, then whatever the account leans on most,
skipping anything already seeded by hand.

```
python pipeline/tagvolume.py --worklist

12 of 55 tag(s) this account has used carry no measured count.
  tag                        thrown out  posted with   estimated
  #mycorrhiza                         3            0   ~900
  #ecology                            0            2   -
```

Those are the two ways a wrong number costs something: a tag dropped from every
caption that mentions the subject, and a tag that goes out on every post in the
wrong band. The `~900` is the estimate that condemned it, printed so a figure
that is obviously wrong is obvious.

## Step 8 — the whole thing, unattended

`pipeline/run.py` is steps 0 and 2-6 in one call: it scouts a board, picks a
topic, writes the script, narrates it, sources the footage, cuts the reel and
writes the post text. It adds no plumbing — every step already returns a dict
the next one accepts — so what it actually contributes is the two jobs a person
was doing between those six commands, plus a trace of what happened.

```bash
python pipeline/run.py                    # scout, choose, make it
python pipeline/run.py "why the ocean is salty" -d 40   # skip step 0
python pipeline/run.py --dry-run          # choose a topic, make nothing
python pipeline/run.py --status           # how the last runs went
```

**It does not post anything.** The reel and its caption land in
`output/<date>/<run id>/` and you publish them yourself. This replaces the
typing, not the judgement about what goes out.

### It reads yesterday's numbers first

Before anything else, `RUN_FETCH_INSIGHTS` has step 9 ask the provider about
every reel whose numbers are not final yet. This is not a convenience — it is
the only way the numbers are usable at all.

Step 9 can only compare a reading taken between `PERFORMANCE_MIN_AGE_HOURS`
and `PERFORMANCE_WINDOW_HOURS` after posting. That is a **24-hour slot per
reel**, and no hand-run schedule hits it. Read off the ledger on 2026-09-09: of
the seven posted reels, five had been read only past 48h and one only at 5.6h —
**six of seven sat outside the window they had to be inside**, and the 5.6h
reading had handed the worst reel in the ledger the highest watch-ratio in it.

A job that fires once every 24 hours lands inside any 24-hour span, so one
daily read guarantees each reel is asked about at least once while the answer
still counts. Step 8 is already the thing the scheduler runs every day.

Two details that are not obvious:

- **It runs before the once-per-day guard.** The reading is about reels posted
  on *earlier* days, so a day that makes no reel is still a day one of them has
  to be asked about. Behind the guard, a reel posted the evening before a
  skipped day would fall out of its window unread.
- **A reading that already counts is never taken again.** The report compares
  on the age of the *latest* reading, and consecutive daily readings are a day
  apart against a 24-hour-wide window — so a reel read at 29h would be read
  again at 53h, comparable for exactly one day and then silently not. This is
  only reachable once the read is automated, which is why it survived every
  earlier fix.

A provider that is down, rate-limited or misconfigured leaves a warning on the
run and nothing else: the numbers are bookkeeping and the reel is the job.
No provider configured at all is a choice rather than a fault, so it says
nothing. `RUN_FETCH_INSIGHTS=false` turns it off.

### Which topic it makes

The top of the board. That is not a shrug — step 0 has already gated the niche,
checked what survived for footage and ranked it, and a second ranking layered
over a good one would only be a worse ranking nobody can read back. The topic is marked made (step 0's `--take`) so no later board offers it
again.

`RUN_TOPIC_MIN_SCORE` is the one judgement left: the score a candidate has to
reach before an unattended run will make it. It is `0` — off — by default,
because a board that cleared the gate and the probe is worth making from. Raise
it if you would rather publish nothing on a thin day; the run then ends
`skipped` rather than `failed`, because nothing went wrong.

`--dry-run` scouts a real board with the real settings and stops before step 2,
which is how you see what it would pick. It deliberately does **not** turn step
0's Gemini calls off to save a few cents: `TOPIC_NICHE_MATCH=terms` decides the
niche on shared words, and a preview that chooses that way previews a topic the
real run will not make — on an ocean-science account it put a Formula One
championship top of the board. `--no-phrases` gives you the free version, and
says on the page that its choice is not the real one.

### When a step fails

| failure | what happens | why |
|---------|--------------|-----|
| step 4 finds no footage | try the next topic on the board | this topic cannot be shot; the alternatives are ranked and sitting right there |
| step 2 returns nothing usable | try the next topic | same — it is about these words |
| bad key, no credits, dead ffmpeg | stop the run | the next topic fails identically, after paying for a script to find out |
| step 6 cannot write the caption | keep the reel, end `partial` | the expensive artifacts are on disk and correct; the caption is one Gemini call to redo |

`RUN_TOPIC_ATTEMPTS` (default 2) caps the fall-through. A topic is marked made
at the moment real money goes on it — after the script, before the narration —
rather than at the end of a run that worked. That is deliberate: a topic step 4
cannot shoot would otherwise still be top of tomorrow's board, and a daily job
would buy the same script and the same narration every evening to fail in the
same place.

Both keys are checked **before** the board is scouted. A run that spends two
minutes on step 0 and then dies on a missing `ELEVENLABS_API_KEY` has paid to
discover something knowable at second zero.

### What it leaves behind

Nobody is watching stdout at 17:00, so:

| where | what |
|-------|------|
| `output/logs/<stamp>.log` | every line all six steps printed, pruned after `RUN_LOG_KEEP_DAYS` |
| `output/runs.json` | one line per run: status, topic, cost, how it ended |
| exit code | `0` for a reel or a deliberate skip, `1` for a failure — what the scheduler reads |

`RUN_ONCE_PER_DAY` (on by default) refuses to start when today already has a
reel. That is the money guard: the way the bill runs away is not one expensive
run, it is a catch-up run landing on top of a real one. `--force` overrides it.

### Running it at 17:00 every day

```bash
python pipeline/schedule.py install       # daily at SCHEDULE_TIME (17:00)
python pipeline/schedule.py install --time 18:30    # also how you change it
python pipeline/schedule.py status        # next run, last result
python pipeline/schedule.py run-now       # fire it once, now
python pipeline/schedule.py preview       # print the definition, install nothing
python pipeline/schedule.py uninstall
```

It hands a task definition to **Windows Task Scheduler** and gets out of the
way. A long-lived Python process sleeping until 17:00 would work until the
first reboot, and the thing that starts programs after a reboot is Task
Scheduler — so it is Task Scheduler, directly, with no daemon of ours in the
middle. Nothing needs to be running, and no admin rights are involved.

It goes in as XML rather than `schtasks /SC DAILY` because the settings that
make a *local* daily job work are not on the command line:

| setting | why |
|---------|-----|
| `StartWhenAvailable` | a desktop is asleep or off at 17:00 more often than not; without this, those evenings silently make nothing |
| `MultipleInstancesPolicy=IgnoreNew` | a catch-up run cannot land on top of a live one |
| `RestartOnFailure` | a provider having a bad ten minutes, or a laptop that has not found the wifi yet, is worth one more go |
| `ExecutionTimeLimit` | a hung run cannot hold the day open forever |
| `pythonw.exe` | no console window thrown over whatever is on screen at 17:00 — the transcript goes to `output/logs/` either way |

The task runs as you, only while you are logged in, with the interpreter that
installed it — `sys.executable`, not the word `python`, because the task runs
in no shell and has no `PATH` worth the name. Installing from inside a
virtualenv therefore pins that virtualenv.

Two things it does not solve. The machine has to be on at some point in the day
for the catch-up to fire (`SCHEDULE_WAKE=true` will wake it, off by default —
waking a sleeping computer every evening is more than a reel is worth). And
running whether or not you are logged in needs stored credentials, which is a
Task Scheduler setting to change by hand rather than something worth putting a
password prompt in this repo for.

On anything that is not Windows, `preview` prints the equivalent crontab line
instead. Installing it is one paste, and doing that behind someone's back on a
machine this code has never seen is not a favour.

## Step 9 — how it did

Every other step in this pipeline optimises a proxy. Step 0 scores velocity,
curiosity and footage; step 2 scores hooks; step 6 bands hashtags by post
count. Not one of those numbers had ever been checked against a view.
`pipeline/perform.py` is the column that was missing.

It is cheap to add because of something already true everywhere else — every
scoring decision here emits the parts it was made of, not just a total:

```json
"hook": {"score": 2.3, "components": {"length": 1.5, "strong opener": 0.8}}
```

So a finished reel is already a fully labelled example missing exactly one
column. Step 8 writes the label-less row to `features.json` as it makes the
reel; you supply the outcome afterwards.

```bash
python pipeline/perform.py                     # which reels are waiting
python pipeline/perform.py cd1bccae --views 4200 --watched 61 --saves 30
python pipeline/perform.py cd1bccae --skipped footage
python pipeline/perform.py cd1bccae --skipped hook --note "the card gave it away"
python pipeline/perform.py --reasons            # the skip vocabulary
python pipeline/perform.py --report
```

A reel is named by its run id, by the short hash on the end of it, by its
directory or by `latest`. Reels made before this step existed are adopted from
`output/` on first run — they carry the reel but not its components, because
step 0's scoring was never written into a run directory, and a backfilled row
says so about itself rather than reading as "nothing fired".

* **the components are stored, the scores are not.** Retune `EXPLAINER_BONUS`
  and every score in the history becomes uninterpretable, while "explainer
  shape fired on this reel" stays true forever. A row is a list of names; the
  arithmetic that turned them into 4.42 is deliberately not preserved;
* **a reel you decided not to post is data, and it arrives first.** It is a
  quality judgement you make the same day, for free, where an Instagram number
  takes 48 hours and a Meta app. `--skipped` records it, and for the first
  month it is very nearly the only signal that exists. It is also the one
  column no API will ever supply.

  It takes a **code**, from a fixed list, and that is a correction: it used to
  take free text, on the reasoning that a sentence written the same day carries
  more than a checkbox. It does, to a reader. It carried nothing to the report,
  which never opened the column — and what fourteen of the nineteen binned
  reels actually said was *"not good enough to post"*, fourteen times, which is
  a judgement with the evidence left out. A code is comparable across reels,
  which is the only property that matters here, and every code names a step you
  could go and change:

  | code | what it says |
  |---|---|
  | `footage` | the footage did not show what was being said |
  | `hook` | the opening did not land |
  | `claim` | something in it was wrong, or overstated |
  | `pacing` | it dragged, or the cuts fought the read |
  | `cover` | the cover frame or the text on it was weak |
  | `audio` | the read, the music or the mix |
  | `duplicate` | too close to a reel already posted |
  | `other` | a fault, and none of the above |
  | `untested` | never a candidate — a test run |

  `--note` keeps the sentence beside the code. Anything that is not a code is
  still accepted rather than refused — the sentence is the better record of
  what was wrong, it is just not comparable — and lands as `unlabelled`, which
  the report counts and prints the command to clear. Neither `unlabelled` nor
  `untested` votes in either table: a test run was never going to be posted
  whatever was in it, and a rejection whose reason was lost is not evidence
  about a reel;
* **views trend with follower count**, so nothing here compares raw numbers.
  Reel #40 beats reel #5 for reasons that have nothing to do with its hook.
  Every reel is scored as a ratio to the median of its own neighbours in time —
  equal numbers either side, so the newest reel is not scored at all until the
  reels after it exist. A baseline built only from older reels flatters
  whatever is newest, which is precisely the reel you are most curious about.
  `PERFORMANCE_METRIC` defaults to a watch figure for the same reason: it is
  the one that does not simply climb with the account;
* **the watch figure is `watch_ratio`, not `watch_through`.** Instagram's
  average watch time counts replays, so it can exceed the length of the reel —
  89.5 seconds of average watch time on a 37.59-second reel, measured here.
  A *share of the reel* cannot exceed 1.0, so `watch_through` is capped, and
  the cap handed the highest score in the ledger to the reel with three views.
  `watch_ratio` is the same division left uncapped: 2.38 stays 2.38. Below 1.0
  the two are identical and both are stored, so switching back costs nothing.
  A reel sitting on the cap is refused by the report rather than ranked as
  perfect;
* **a measurement has an age, at both ends.** A reel read at twelve hours and
  one read at a week are not the same measurement, and averaging them measures
  when you happened to look. `PERFORMANCE_WINDOW_HOURS` is 48 and anything
  older is kept but not compared — and `PERFORMANCE_MIN_AGE_HOURS` is 24,
  because a reel read the same evening it went out has not finished happening.
  The first reel read through the provider entered the ledger 5.6 hours old
  with three views, and that is not a verdict on its hook;
* **numbers are read more than once.** They used to be taken once, which meant
  whatever a reel happened to have the moment somebody ran `--fetch` was what
  it was judged on forever. A posted reel is asked about again until its
  reading is `PERFORMANCE_SETTLE_HOURS` (36) old, and each answer is *appended*
  — `measurements` holds what the reel had at every age it was read at, which
  is worth more than the last number alone.

### Without a trial there is nothing to compare

This is the one that bites, and it is not a statistics problem — it is a
pipeline problem. Left alone, step 2 replaces the hook every single time. After
a year the ledger holds 365 reels that all made the same choice, and it cannot
say whether it was the right one. Natural variance does not arrive at one reel
a day.

`PERFORMANCE_EXPLORE` fixes it by varying one thing deliberately:

| value | what alternates |
|-------|-----------------|
| `off` | nothing — the default |
| `hook` | the hook pass runs on every other reel |
| `structure` | the structure pass runs on every other reel |
| `duration` | one step shorter, then one step longer than step 0 suggested |

Arms **alternate**; they are not randomised. At thirty observations randomising
can hand you nineteen of one arm and eleven of the other by luck alone, which
is not a comparison, and there is no second month to spend discovering that.

### Reading the report

Four tables, because four different columns are being read:

```
WHY THEY WERE BINNED  (the same-day verdict; no API involved)
  footage      6    43%  the footage did not show what was being said
  hook         3    21%  the opening did not land
  claim        3    21%  something in it was wrong, or overstated
  pacing       2    14%  it dragged, or the cuts fought the read

KEPT OR BINNED  (1.00 = always posted; every reel counts)
  question                   answer         n    median   vs rest
  topic source               evergreen     11     0.91    +0.24

HOW THEY DID  (1.00 = an ordinary reel for this account)
  question                   answer         n    median   vs rest
  length                     40-50s        12     1.18x   +0.21
  trial: hook                on            11     1.09x   +0.14

SAVED OR SENT  (1.00 = every reel got a save or a share)
  question                   answer         n    median   vs rest
  trial: takeaway            on            11     0.27    +0.18
```

The first table is a tally and not a segment, deliberately. With fourteen
rejections spread over eight codes, crossing them with the component vocabulary
would print a table of segments of one that all look like findings — the exact
failure the other two tables refuse. *"Six of these were binned for the
footage"* is a small claim that is actually supported, and it points at the step
to go and change. It is also the table that fills up first: a rejection is a
labelled failure available the same day, where a view takes a week.

Nothing below `PERFORMANCE_MIN_SAMPLES` on **both** sides of a question is read
out of the other three — a segment of three that happened to do well is not a
finding, and the report says how far off it is rather than printing the number
anyway.

**Why SAVED OR SENT is a rate and not a metric.** Saves and shares are what the
take-away pass exists to move, so the obvious thing is
`PERFORMANCE_METRIC=saves` — and it is wrong in a way that looks like an
answer. `HOW THEY DID` divides each reel by the median of its nearest
neighbours and refuses a median of zero, so a metric that is zero on most reels
yields **no baseline, therefore no reading, for every reel, forever** — printed
identically to *"measured, no effect"*. It does not fix itself as saves arrive
either: more than half the surrounding reels would need one first. So saves and
shares are read as a yes/no and reported as a share of reels, which is defined
at zero and moves the first time one lands. While nothing has ever been saved
the report says so outright, so a column of zeros is never mistaken for a
finding.

Be realistic about what a daily pipeline can support. With ~20 possible
components and one reel a day, most segments never reach a readable n. The cuts
that fire on every reel and get there within a couple of months are **duration
band, topic source, hook replaced or kept, structure rewritten or kept,
take-away rewritten or kept, cover source, and whichever trial is running** —
those plus the kept/binned table are what this can actually tell you.

Hashtags are the exception to "~20 components": from `FEATURES_VERSION` 5 every
tag on a post is recorded by name and asked as its own question, which is a
great many questions and almost all of them unreadable for a long time. That is
the point of storing them — the count and the band tally that earlier rows kept
could not hold *"did #woodwideweb do anything"* whatever the tags were. The
spine tags in `POST_HASHTAGS_ALWAYS` are the ones that will reach a readable n
first, because they are the only ones on every post.

### What it changes

Nothing, by default. `PERFORMANCE_FEEDBACK` is off, and the honest use of this
for a long while is a report a person reads before moving a constant by hand.

Turned on — and only once the ledger holds `PERFORMANCE_MIN_REELS` reels — it
adds exactly one more named component to step 0's board, capped at
`PERFORMANCE_MAX_ADJUST` and printed with the evidence behind it, like every
other component:

```
 1.   4.42  why the ocean is salty
           velocity +1.71, niche match +2, explainer shape +1.2,
           measured (explainer shape 1.31x, n=14) +0.31
```

It never edits a weight in the source. An optimiser quietly rewriting
`EXPLAINER_BONUS` would buy slightly better tuning at the cost of the property
that makes step 0 worth reading at all — that every number on the board can be
argued with.

### Where the numbers come from

By default you type them in, and for the first few weeks that is the right
answer rather than a limitation — the column that matters most early on is
`--skipped`, and no API has it.

`PERFORMANCE_PROVIDER` is a dotted path to a callable handed the waiting run
ids and their features rows, returning `{run_id: {metric: value}}`. Anything it
cannot answer stays pending, exactly as if it had not been asked.

### Reading them off Instagram instead

`pipeline/instagram.py` is that callable. It reads insights and **cannot post**
— nothing in this pipeline can.

```bash
python pipeline/instagram.py --check      # is the token good, and whose?
python pipeline/instagram.py --media      # recent posts, as the API sees them
python pipeline/instagram.py --fetch      # record everything still waiting
python pipeline/instagram.py --pin cd1bccae 17912345678901234
python pipeline/instagram.py --refresh    # a fresh 60 days
```

Setting it up is about twenty minutes of Meta dashboard, once: switch the
account to **Professional** (a personal account has no insights API at all),
create an app at developers.facebook.com and add the Instagram product, add
yourself as its owner/tester, then exchange the short-lived token for a
long-lived one and put it in `.env` with the account's numeric id — the number
`--check` prints, not the @handle, which is not a node the API answers for.
Reading your own account this way needs **no App Review** — Meta lets an app's
own admins and testers exercise permissions in development mode, which is
exactly one person reading one account.

Four things about it are worth knowing before relying on it:

* **there are two Instagram APIs and the tokens do not cross.** Which host
  answers depends on how the token was minted: Instagram Login tokens (they all
  begin `IG`) are parseable only by `graph.instagram.com`, Facebook Login ones
  (`EAA...`) only by `graph.facebook.com`. The wrong host says *Cannot parse
  access token*, which reads exactly like a token that has lapsed and is
  nothing of the kind — so the host follows the token, and `INSTAGRAM_API_HOST`
  overrules it if that guess is ever wrong;
* **matching a reel to a post is the hard half.** Instagram has never heard of
  `cd1bccae`. What it has is a caption — and step 6 wrote that caption, so the
  first line of `post.txt` is the join key and the posting time is the
  tie-break. A match made on time alone is a guess, so how it was made is
  recorded next to the numbers and `--fetch` prints which ones to check.
  `--pin` overrules it for good. The waiting reels are matched **all at once**,
  and a post belongs to one reel: matched one at a time, the nearest later post
  is the nearest later post for every reel made that week, and the ones that
  were never posted each walk off with a copy of the numbers of the one that
  was. Measured the day this was first switched on: 7 posts, 7 exact caption
  matches, and 5 further reels handed somebody else's numbers by the fallback;
* **the metric names churn.** `plays` became `views`, the reels metrics have
  been consolidated more than once, and which ones a media node answers for
  depends on its type and on the API version. So the request is made from a
  candidate list, and whatever the API refuses is dropped and asked for again —
  four metrics out of six is useful, and an exception because Meta renamed a
  field is not. `INSTAGRAM_API_VERSION` is pinned for the same reason;
* **watch-through is derived, not reported.** Instagram gives an average watch
  time in milliseconds; step 9 ranks on how much of the reel that was. The
  reel's own length is in its features row, so the conversion is a division —
  which is why the provider is handed the features rows and not just ids. Two
  numbers come out of it, because the division does not stay under 1.0:
  Instagram counts replays, so a looped reel reports more watch time than it
  has reel. `watch_through` caps at 1.0 and `watch_ratio` does not, and the
  report ranks on the second;
* **a reel is read more than once.** `--fetch` asks about every reel with no
  numbers *and* every posted reel whose reading is younger than
  `PERFORMANCE_SETTLE_HOURS`, because a number taken the same evening is a
  number taken while the reel was still being shown. Each answer is appended to
  that reel's `measurements`, so the early reading survives the later one.

The token lasts sixty days. `--refresh` mints a new one and prints it to paste
back into `.env` — deliberately printed rather than written, because a program
that rewrites the one file your secrets live in is a program that can mangle
it. A scheduled run that finds the token expired says so and records nothing,
which is the right failure: the reels are still on disk and the numbers can
still be typed in.

## When a provider blips

Every API call the pipeline makes is retried on the failures worth retrying —
rate limits, and the 5xxs a service returns under load — with exponential
backoff, and raised at once on the ones that are settled: a bad key, an empty
balance, a refusal, an unknown model.

`script_gen.call_gemini` is the one Gemini call **three** steps share, so this
matters more than it looks: without it, a single 503 does not just fail step 2,
it drops step 4 to keyword search queries and step 6 to heuristic copy, and the
reel comes out quietly worse with only a warning line to say why.

Step 3 is the exception that proves the rule — a timeout there is **not**
retried. The narration may well have been generated and billed already, and
paying twice for one script is the worse outcome. Only refusals the API sent
back before synthesising anything are tried again.

## Secrets

All keys live in `.env` (git-ignored) and are read only through `config.py`.
Nothing is ever hardcoded.
