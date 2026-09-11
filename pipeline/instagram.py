"""Step 9's Instagram provider - the numbers, read rather than typed.

Plugs into `pipeline/perform.py` through PERFORMANCE_PROVIDER and does exactly
one thing: for reels that are waiting for numbers, find the post each one
became and read its insights. It cannot publish, and it is not asked to.

    PERFORMANCE_PROVIDER=pipeline.instagram.fetch

Five things about it are worth knowing before you rely on it:

* **there are two Instagram APIs and the tokens do not cross.** A token minted
  through Instagram Login (they all begin `IG`) is parseable only by
  graph.instagram.com, one from the Facebook Login flow (`EAA...`) only by
  graph.facebook.com, and the wrong host answers "Cannot parse access token" -
  which reads exactly like a token that has lapsed and is nothing of the kind.
  So the host follows the token, and `INSTAGRAM_API_HOST` overrules it;
* **it reads your own account and nothing else.** The Graph API's media node
  only answers for the account the token belongs to. There is no path here to
  anyone else's numbers, and none is wanted;
* **matching a reel to a post is the hard half.** Instagram has never heard of
  `cd1bccae`. What it has is a caption, and step 6 wrote that caption - so the
  first line of `post.txt` is the join key, and the timestamp is the tie-break.
  Neither is certain, so a match is always reported with how it was made and
  can be overruled by hand with `--pin`. All the waiting reels are matched at
  once and a post is one reel: the reels that were never posted would otherwise
  each take a copy of the numbers of the one that was;
* **the metric names churn.** `plays` became `views`, the reels-specific
  metrics have been consolidated more than once, and which ones a media node
  answers for depends on its type and on the API version. So the request is
  made from a candidate list and anything the API refuses is dropped and
  retried rather than failing the fetch - a provider that returns four of six
  metrics is useful, and one that raises because Meta renamed a field is not;
* **watch-through is derived, not reported.** Instagram gives an average watch
  time in milliseconds; step 9's default metric is the fraction of the reel
  that got watched. Dividing by the reel's own length - which `features.json`
  already knows - is the whole conversion, and it is why this provider is
  handed the features rows and not just a list of ids.

The token is a long-lived one and lasts sixty days. `--refresh` mints a new one
before it lapses; a scheduled run that finds it expired says so and records
nothing, which is the right failure - the reels are still on disk and the
numbers can still be typed in.

Standalone use:
    python pipeline/instagram.py --check       # is the token good, and whose?
    python pipeline/instagram.py --media       # recent posts, as the API sees them
    python pipeline/instagram.py --fetch       # record what is waiting, via step 9
    python pipeline/instagram.py --pin cd1bccae 17912345678901234
    python pipeline/instagram.py --refresh     # a fresh sixty days
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow both `python pipeline/instagram.py` and `python -m pipeline.instagram`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import perform, visuals  # noqa: E402

USER_AGENT = "reel-creator-agent/1.0"

TIMEOUT = (5, 30)

#: Metrics asked for, in Instagram's own names, mapped onto the ledger's.
#: Asked for as a set and pruned on refusal - see `_insights`. `views` is the
#: current name for what used to be `plays`; both are listed because which one
#: a given API version answers to is not knowable from here.
#:
#: `follows` and `profile_visits` are the outcome the rest of this file was
#: only ever measuring the run-up to. Every other metric here is attention -
#: how many were shown it, how long they stayed, whether they tapped. For an
#: account at zero followers the only number that compounds is whether a
#: viewer went and followed, and until 2026-09-10 nothing asked for it:
#: `perform.METRICS` carried a `follows` column and `--follows` parsed, but no
#: fetch ever filled either, so the column had been empty on every reel ever
#: made. Asking is free by construction - `_insights` reads a refusal for
#: which name offended and drops it - so whether this API version serves the
#: two at media level is answered by running it, and costs a warning if not.
METRIC_MAP = {
    "views": "views",
    "plays": "views",
    "reach": "reach",
    "likes": "likes",
    "comments": "comments",
    "saved": "saves",
    "shares": "shares",
    "ig_reels_avg_watch_time": "avg_watch_seconds",
    "follows": "follows",
    "profile_visits": "profile_visits",
}

#: The ones worth a second attempt on their own if the batch was refused.
CORE_METRICS = ("reach", "likes", "comments", "saved")


class InstagramError(RuntimeError):
    """Raised when the API cannot be used at all."""


class InstagramAuthError(InstagramError):
    """The token is missing, expired, or not entitled to what was asked."""


#: Meta runs two Instagram APIs and the tokens are not interchangeable. A token
#: minted through **Instagram Login** (they all begin "IG") is parseable only by
#: graph.instagram.com; one from the **Facebook Login** flow ("EAA...") only by
#: graph.facebook.com. The wrong host answers "Cannot parse access token",
#: which reads like a token that has lapsed and is nothing of the kind - so the
#: host follows the token unless INSTAGRAM_API_HOST overrules it.
INSTAGRAM_LOGIN_HOST = "https://graph.instagram.com"
FACEBOOK_LOGIN_HOST = "https://graph.facebook.com"


def _host() -> str:
    configured = (config.INSTAGRAM_API_HOST or "").strip().rstrip("/")
    if configured:
        return configured if "//" in configured else f"https://{configured}"
    token = (config.INSTAGRAM_ACCESS_TOKEN or "").strip()
    return INSTAGRAM_LOGIN_HOST if token.startswith("IG") else FACEBOOK_LOGIN_HOST


def _base() -> str:
    return f"{_host()}/{config.INSTAGRAM_API_VERSION}"


def _session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get(session, path: str, params: dict) -> dict:
    """GET a Graph API node, mapping Meta's error envelope onto exceptions.

    Meta answers a bad token with a 400 as readily as a 401, and puts the real
    story in the body, so the body is what is read rather than the status.
    """
    import requests

    url = path if path.startswith("http") else f"{_base()}/{path.lstrip('/')}"
    payload = {"access_token": config.INSTAGRAM_ACCESS_TOKEN, **params}
    try:
        response = session.get(url, params=payload, timeout=TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise InstagramError(f"instagram timed out after {TIMEOUT[1]}s") from exc
    except requests.exceptions.RequestException as exc:
        raise InstagramError(
            f"could not reach instagram: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        body = response.json()
    except ValueError as exc:
        text = (response.text or "").strip()[:200]
        raise InstagramError(
            f"instagram returned a non-JSON body [{response.status_code}]: {text!r}"
        ) from exc

    error = body.get("error") if isinstance(body, dict) else None
    if error:
        message = error.get("message") or "unknown error"
        code = error.get("code")
        subcode = error.get("error_subcode")
        if code in (190, 102, 463, 467) or "session" in message.lower():
            raise InstagramAuthError(
                f"instagram rejected the token: {message}. "
                f"python pipeline/instagram.py --refresh mints a new one, and "
                f"a token more than 60 days old has to be generated again in "
                f"the Meta app dashboard."
            )
        if code == 10 or code == 200 or subcode == 2108006:
            raise InstagramAuthError(
                f"instagram refused the request: {message}. This is usually a "
                f"permission the app was never granted "
                f"(instagram_business_manage_insights), or a personal rather "
                f"than a professional account."
            )
        if code == 4 or code == 17 or code == 32:
            raise InstagramError(
                f"instagram rate limited this app: {message}. It resets on a "
                f"rolling hour; nothing was recorded."
            )
        raise InstagramError(f"instagram: {message}"
                             + (f" [code {code}]" if code else ""))

    if response.status_code >= 400:
        raise InstagramError(
            f"instagram failed [{response.status_code}]: "
            f"{(response.text or '').strip()[:200]}"
        )
    return body if isinstance(body, dict) else {}


#: The fields `account` may not drop - without these it is not proving
#: anything, and a token that cannot answer them is a token to report.
REQUIRED_ACCOUNT_FIELDS = ("id", "username")


def require_credentials() -> None:
    config.require(*config.REQUIRED_INSTAGRAM)


# ---------------------------------------------------------------------------
# The account, and what it has posted
# ---------------------------------------------------------------------------

#: Asked for on the account node, most useful first. `followers_count` is
#: the denominator every reach figure in step 9 is implicitly divided by,
#: and the one number the report never had - see AUDIENCE in `_print_report`.
ACCOUNT_FIELDS = ("id", "username", "media_count", "followers_count",
                  "follows_count")


def account(session=None, *, fields=None) -> dict:
    """Who the token belongs to. The cheapest proof it works.

    Drops a field the API will not answer for rather than failing. The two
    Instagram APIs disagree about `followers_count` - the Graph API serves
    it on a business or creator account, and a token that cannot have it
    should still be able to prove itself, which is this call's first job.
    Same reasoning as `_refused`, at a smaller scale: what is asked for here
    is fixed and short, so one retry without the extras settles it.
    """
    require_credentials()
    session = session or _session()
    wanted = list(fields or ACCOUNT_FIELDS)
    try:
        return _get(session, str(config.INSTAGRAM_USER_ID),
                    {"fields": ",".join(wanted)})
    except InstagramAuthError:
        raise
    except InstagramError:
        core = [name for name in wanted if name in REQUIRED_ACCOUNT_FIELDS]
        if core == wanted:
            raise
        return _get(session, str(config.INSTAGRAM_USER_ID),
                    {"fields": ",".join(core)})


def audience(session=None) -> dict:
    """The account's own size, for the reach figures to be read against.

    Empty when the API would not say, which is not the same as zero - the
    report prints the difference, because "nobody follows this account" and
    "this token cannot see the follower count" are different findings.
    """
    try:
        body = account(session)
    except InstagramError:
        return {}
    out = {"read_at": perform._now()}
    for key, name in (("followers", "followers_count"),
                      ("following", "follows_count"),
                      ("posts", "media_count")):
        value = body.get(name)
        if isinstance(value, (int, float)):
            out[key] = int(value)
    if body.get("username"):
        out["username"] = str(body["username"])
    return out if len(out) > 1 else {}


def media(session=None, *, limit: int | None = None) -> list[dict]:
    """Recent posts, newest first, as the API describes them."""
    require_credentials()
    limit = config.INSTAGRAM_MEDIA_LIMIT if limit is None else limit
    body = _get(session or _session(), f"{config.INSTAGRAM_USER_ID}/media", {
        "fields": "id,caption,timestamp,media_type,media_product_type,permalink",
        "limit": max(1, limit),
    })
    return [item for item in (body.get("data") or []) if isinstance(item, dict)]


def _refused(message: str, wanted: list[str]) -> list[str]:
    """Which of the metrics asked for this API version would not answer.

    Meta says this two ways round. Sometimes it names the offender - "(#100)
    ig_reels_avg_watch_time is not valid for this media product type" - and the
    refused metric is the one the message mentions. Sometimes it names every
    metric it *would* have taken - "metric[1] must be one of the following
    values: impressions, shares, ..." - and then reading the message for names
    it mentions is exactly backwards: it would drop the seven good metrics and
    keep the one bad one. So the enumerated form is detected and inverted.

    Inverting also prunes faster than Meta complains. It reports one offending
    index per call, and the whole valid list is enough to drop all of them at
    once instead of a round trip each.
    """
    allowed = re.search(r"must be one of the following values:\s*(.+)",
                        message, re.IGNORECASE | re.DOTALL)
    if allowed:
        names = {name.strip(" .'\"") for name
                 in re.split(r"[,\s]+", allowed.group(1))}
        names.discard("")
        if names:
            return [name for name in wanted if name not in names]
    return [name for name in wanted if name in message]


def _insights(session, media_id: str) -> tuple[dict, list[str]]:
    """Insights for one post, with refused metrics dropped rather than fatal.

    Meta renames these between versions and answers for a different set
    depending on the media type. Asking for a metric a node does not have fails
    the *whole* call, so a refusal is read for which name offended, that name is
    dropped, and what is left is asked for again. Four metrics out of six is a
    useful answer; an exception because a field was renamed is not.
    """
    warnings: list[str] = []
    wanted = list(METRIC_MAP)

    while wanted:
        try:
            body = _get(session, f"{media_id}/insights",
                        {"metric": ",".join(wanted)})
        except InstagramAuthError:
            raise
        except InstagramError as exc:
            refused = _refused(str(exc), wanted)
            if not refused:
                # Nothing identifiable in the message: fall back to the metrics
                # that have survived every version of this API, once.
                if set(wanted) == set(CORE_METRICS):
                    warnings.append(f"{media_id}: {exc}")
                    return {}, warnings
                warnings.append(
                    f"{media_id}: asked for the full metric set and got "
                    f"\"{exc}\" - retrying with the core four"
                )
                wanted = list(CORE_METRICS)
                continue
            for name in refused:
                wanted.remove(name)
            warnings.append(
                f"{media_id}: this API version does not report "
                + ", ".join(refused)
            )
            continue

        out: dict[str, float] = {}
        for entry in body.get("data") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            values = entry.get("values") or []
            value = values[0].get("value") if values and isinstance(values[0], dict) else None
            if name in METRIC_MAP and value is not None:
                # Some metrics answer with a breakdown rather than a count -
                # `profile_activity` returns {action_type: n} - and a dict
                # written into the ledger is a number every reader downstream
                # would try to divide by. Only scalars are stored, and a
                # breakdown is skipped rather than flattened, because which
                # part of it was meant is not knowable from here. Checked
                # here rather than at the call site so it holds for whatever
                # is added to METRIC_MAP next.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    warnings.append(
                        f"{media_id}: {name} answered with a "
                        f"{type(value).__name__} rather than a count - not stored"
                    )
                    continue
                # `views` and `plays` both land on views; whichever the version
                # answered with is the one that counts, and the larger of two
                # answers is never the right tie-break, so first wins.
                out.setdefault(METRIC_MAP[name], value)
        return out, warnings

    return {}, warnings


# ---------------------------------------------------------------------------
# Matching a reel to the post it became
# ---------------------------------------------------------------------------

def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def caption_of(features: dict) -> str:
    """The first line of the post text step 6 wrote for this reel."""
    run_dir = features.get("run_dir") or ""
    if not run_dir:
        return ""
    try:
        return _first_line(
            (Path(run_dir) / "post.txt").read_text(encoding="utf-8")
        )
    except OSError:
        return ""


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


NO_MATCH = {"media_id": "", "how": "", "permalink": "", "posted_at": ""}


def _found(post: dict, how: str) -> dict:
    return {"media_id": post.get("id") or "", "how": how,
            "permalink": post.get("permalink") or "",
            "posted_at": post.get("timestamp") or ""}


def _caption_hit(features: dict, post: dict) -> bool:
    """Whether this post opens with the words step 6 wrote for this reel."""
    hook = _normalise(caption_of(features))
    if not hook:
        return False
    caption = _normalise(post.get("caption") or "")
    return bool(caption) and (caption.startswith(hook) or hook in caption)


def _gap(features: dict, post: dict):
    """How long after the reel was made this post went up, or None.

    None for a post outside INSTAGRAM_MATCH_HOURS and for one that predates the
    reel - a reel is posted after it is made, never before, so a post older
    than the reel is some other post that happens to be nearby.
    """
    made = perform._parse_stamp(features.get("created_at"))
    when = perform._parse_stamp(post.get("timestamp"))
    if not made or not when or when < made:
        return None
    gap = when - made
    window = timedelta(hours=max(1, config.INSTAGRAM_MATCH_HOURS))
    return gap if gap <= window else None


def _distance(features: dict, post: dict):
    """Absolute time between a reel and a post, for ranking caption matches."""
    made = perform._parse_stamp(features.get("created_at"))
    when = perform._parse_stamp(post.get("timestamp"))
    if not made or not when:
        return timedelta.max
    return abs(when - made)


def match(features: dict, posts: list[dict]) -> dict:
    """Which post this reel became, and how sure that is.

    Returns {"media_id", "how", "permalink", "posted_at"} - `how` being
    "caption" for a match on the words step 6 wrote, "timestamp" for the
    nearest post inside INSTAGRAM_MATCH_HOURS, or "" for no match at all. The
    how is carried through to the ledger because "we guessed by time" and "the
    caption is identical" are not the same claim and should not be recorded as
    one.

    This answers for one reel in isolation. `assign` is what `fetch` uses, and
    is the one to reach for whenever there is more than one reel in play.
    """
    for post in posts:
        if _caption_hit(features, post):
            return _found(post, "caption")

    best, best_gap = None, None
    for post in posts:
        gap = _gap(features, post)
        if gap is None:
            continue
        if best_gap is None or gap <= best_gap:
            best, best_gap = post, gap
    return _found(best, "timestamp") if best is not None else dict(NO_MATCH)


def assign(features_by_run: dict, posts: list[dict],
           pinned: dict | None = None) -> dict:
    """Match every waiting reel at once, letting no post be claimed twice.

    `match` cannot see the other reels, which is right when a caption settles
    it and badly wrong when nothing does: the nearest later post is the nearest
    later post for *every* reel made that week, so the reels that were never
    posted all end up wearing the numbers of the one that was. Measured on this
    account the day the provider was first switched on: 7 posts, 7 exact
    caption matches, and 5 further reels handed a copy of somebody else's
    numbers by the fallback.

    A post is one reel. So pins are honoured first, then the captions - which
    are evidence rather than a guess - and the timestamp fallback is left to
    divide up whatever is still unclaimed, closest gap first and one reel each.
    """
    claimed: dict[str, str] = {}
    out: dict[str, dict] = {}

    for run_id, media_id in (pinned or {}).items():
        if not media_id or run_id not in features_by_run:
            continue
        post = next((p for p in posts if (p.get("id") or "") == media_id), None)
        out[run_id] = (_found(post, "pinned") if post else
                       {**NO_MATCH, "media_id": media_id, "how": "pinned"})
        claimed[media_id] = run_id

    # Two runs of the same topic write near-identical first lines, so even a
    # caption can collide. The reel made nearest the post is the one posted.
    hits = [(run_id, post) for run_id, features in features_by_run.items()
            if run_id not in out
            for post in posts if _caption_hit(features, post)]
    hits.sort(key=lambda pair: _distance(features_by_run[pair[0]], pair[1]))
    for run_id, post in hits:
        media_id = post.get("id") or ""
        if run_id in out or media_id in claimed:
            continue
        out[run_id] = _found(post, "caption")
        claimed[media_id] = run_id

    guesses = []
    for run_id, features in features_by_run.items():
        if run_id in out:
            continue
        for post in posts:
            gap = _gap(features, post)
            if gap is not None:
                guesses.append((gap, run_id, post))
    guesses.sort(key=lambda triple: (triple[0], triple[1]))
    for _, run_id, post in guesses:
        media_id = post.get("id") or ""
        if run_id in out or media_id in claimed:
            continue
        out[run_id] = _found(post, "timestamp")
        claimed[media_id] = run_id

    return {run_id: out.get(run_id) or dict(NO_MATCH)
            for run_id in features_by_run}


def derive_watch_through(metrics: dict, features: dict) -> dict:
    """Turn an average watch time into how much of the reel that was.

    Instagram reports milliseconds watched; step 9 ranks on what share of the
    reel that was. The reel's own length is in its features row, so the
    conversion is division - and it is done here rather than in the ledger
    because it is this provider's unit problem, not step 9's.

    Two numbers come out of the one division, because the division does not
    stay under 1.0. Instagram's average watch time counts replays: the first
    reel read through this provider reported 89.5 seconds of average watch time
    on a 37.59-second reel, which is a real reading of a looped reel and not a
    bad one. `watch_through` is a *share of the reel* and so is capped, which
    made that the best score in the ledger; `watch_ratio` is the same division
    left alone, and is what the report ranks on. Below the cap they are equal.
    """
    out = dict(metrics)
    millis = out.pop("avg_watch_seconds", None)
    if millis is None:
        return out
    seconds = float(millis) / 1000.0
    out["avg_watch_seconds"] = round(seconds, 2)

    length = features.get("spoken_seconds")
    if length and float(length) > 0:
        ratio = seconds / float(length)
        out["watch_ratio"] = round(ratio, 4)
        out["watch_through"] = round(min(1.0, ratio), 4)
    return out


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------

def fetch(run_ids, features_by_run: dict) -> dict:
    """What PERFORMANCE_PROVIDER calls. {run_id: {metric: value}}.

    Anything that cannot be matched or measured is simply left out, and stays
    pending in step 9 exactly as if this had never been asked - a reel with no
    numbers is a reel to type numbers into, not an error.
    """
    require_credentials()
    session = _session()
    posts = media(session)
    if not posts:
        return {}

    ledger = perform.PerformanceLedger()
    # Once per fetch, not once per reel: it is one number about the account
    # and it does not change between two reels read in the same minute.
    ledger.set_audience(audience(session))
    out: dict[str, dict] = {}

    features_by_run = {run_id: features_by_run.get(run_id) or {}
                       for run_id in run_ids}
    pinned = {run_id: ledger.entry(run_id).get("media_id") or ""
              for run_id in run_ids}
    found_by_run = assign(features_by_run, posts, pinned)

    for run_id in run_ids:
        features = features_by_run[run_id]
        found = found_by_run.get(run_id) or dict(NO_MATCH)
        if not found["media_id"]:
            continue

        try:
            metrics, warnings = _insights(session, found["media_id"])
        except InstagramAuthError:
            raise
        except InstagramError:
            continue
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if not metrics:
            continue

        metrics = derive_watch_through(metrics, features)
        metrics = {name: value for name, value in metrics.items()
                   if name in perform.METRICS}
        if not metrics:
            continue

        # How the match was made travels with the numbers. A timestamp match is
        # a guess, and a guess that lands on the wrong reel is worse than no
        # number at all - so it is written down where it can be audited.
        record = ledger.reels.setdefault(run_id, {"run_id": run_id})
        record["media_id"] = found["media_id"]
        record["permalink"] = found["permalink"]
        # On a re-read the media id comes from the ledger, so `assign` reports
        # it as pinned whatever settled it the first time. Keep what did: a
        # timestamp match is a guess, and quietly promoting it to "pinned" on
        # the second fetch would hide the one row worth checking. A hand pin
        # is written by `pin` and survives because nothing overwrites it.
        if found["how"] != "pinned" or not record.get("matched_by"):
            record["matched_by"] = found["how"]
        # When the post went up, which is not when the reel was made - step 9
        # assumes the latter when nobody says otherwise, and the age of a
        # measurement is what decides whether two of them compare.
        if found["posted_at"]:
            record["posted_at"] = found["posted_at"]
        out[run_id] = metrics

    ledger.save()
    return out


def pin(run_id: str, media_id: str) -> list[str]:
    """Say by hand which post a reel became. Overrules the matcher for good."""
    ledger = perform.PerformanceLedger()
    resolved = perform.resolve_run(run_id, ledger=ledger)
    entry = ledger.reels.setdefault(resolved, {"run_id": resolved})
    entry["media_id"] = media_id.strip()
    entry["matched_by"] = "pinned"
    return ledger.save()


def refresh_token(session=None) -> str:
    """Mint a fresh sixty days from the current token, and return it.

    Deliberately printed rather than written into .env. This file is the one
    place secrets live and it is edited by a person; a program that rewrites it
    is a program that can mangle it, and the failure mode is an account this
    pipeline can no longer read at all.
    """
    require_credentials()
    body = _get(session or _session(), "refresh_access_token",
                {"grant_type": "ig_refresh_token"})
    token = body.get("access_token") or ""
    if not token:
        raise InstagramError(
            "instagram did not return a new token. A token can only be "
            "refreshed while it is still valid and at least 24 hours old; "
            "past that it has to be generated again in the Meta dashboard."
        )
    return token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_check() -> int:
    try:
        who = account()
    except config.MissingConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nBoth keys go in .env:")
        print("  INSTAGRAM_ACCESS_TOKEN=   long-lived token from your Meta app")
        print("  INSTAGRAM_USER_ID=        the professional account's numeric id")
        return 1
    print(f"token   : works")
    print(f"account : @{who.get('username', '?')}  (id {who.get('id', '?')})")
    print(f"posts   : {who.get('media_count', '?')}")
    print(f"version : {config.INSTAGRAM_API_VERSION}")
    return 0


def _print_media(limit: int) -> int:
    posts = media(limit=limit)
    if not posts:
        print("the account has no posts the API will show.")
        return 0
    print(f"{'posted':<12} {'type':<10} {'media id':<20} caption")
    for post in posts:
        when = (post.get("timestamp") or "")[:10]
        kind = (post.get("media_product_type") or post.get("media_type") or "")[:10]
        caption = _first_line(post.get("caption") or "")[:38]
        print(f"{when:<12} {kind.lower():<10} {post.get('id', ''):<20} {caption}")
    return 0


def _print_fetch() -> int:
    ledger = perform.PerformanceLedger()
    # Not just the reels with no numbers: a posted reel is re-read until its
    # reading has settled, because a number taken the same evening is a number
    # taken while the reel was still being shown.
    waiting = ledger.unsettled()
    if not waiting:
        print("nothing is waiting for numbers, and nothing read recently "
              "enough to be worth reading again.")
        return 0
    again = sum(1 for row in waiting if row.get("status") == "posted")

    recorded, warnings = perform.fetch(ledger=ledger, provider=fetch)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(f"recorded {recorded} of {len(waiting)} reel(s)"
          + (f" - {again} of them re-read, their numbers not yet "
             f"{config.PERFORMANCE_SETTLE_HOURS}h old." if again else "."))
    guessed = [run_id for run_id, entry in ledger.reels.items()
               if entry.get("matched_by") == "timestamp"]
    if guessed:
        print(f"\n{len(guessed)} matched on posting time rather than caption. "
              f"Check one is right, and pin it if not:")
        for run_id in guessed[:5]:
            print(f"  python pipeline/instagram.py --pin {run_id} <media id>")
    unmatched = len(waiting) - recorded
    if unmatched > 0:
        print(f"\n{unmatched} could not be matched to a post. If they were "
              f"never posted, say so - it is the more useful record:")
        print(f"  python pipeline/perform.py <run> --skipped "
              + "|".join(list(perform.SKIP_REASONS)[:4]) + "|...")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read reel insights off Instagram for step 9. "
                    "Reads only - it cannot post.",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the token and print whose account it is")
    parser.add_argument("--media", action="store_true",
                        help="list recent posts as the API sees them")
    parser.add_argument("--fetch", action="store_true",
                        help="record numbers for every reel still waiting")
    parser.add_argument("--pin", nargs=2, metavar=("RUN", "MEDIA_ID"),
                        help="say by hand which post a reel became")
    parser.add_argument("--refresh", action="store_true",
                        help="mint a fresh 60-day token and print it")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help=f"posts to read (default: "
                             f"{config.INSTAGRAM_MEDIA_LIMIT})")
    args = parser.parse_args(argv)

    if not any((args.check, args.media, args.fetch, args.pin, args.refresh)):
        return _print_check()

    try:
        if args.check:
            return _print_check()
        if args.refresh:
            token = refresh_token()
            print("a fresh 60 days. Paste this into .env as "
                  "INSTAGRAM_ACCESS_TOKEN:\n")
            print(token)
            return 0
        if args.pin:
            for warning in pin(args.pin[0], args.pin[1]):
                print(f"warning: {warning}", file=sys.stderr)
            print(f"pinned {args.pin[0]} to media {args.pin[1]}.")
            return 0
        if args.media:
            return _print_media(args.limit or config.INSTAGRAM_MEDIA_LIMIT)
        if args.fetch:
            return _print_fetch()
    except (InstagramError, perform.PerformanceError,
            config.MissingConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
