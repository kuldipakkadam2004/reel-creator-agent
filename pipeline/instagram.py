"""Step 9's Instagram provider - the numbers, read rather than typed.

Plugs into `pipeline/perform.py` through PERFORMANCE_PROVIDER and does exactly
one thing: for reels that are waiting for numbers, find the post each one
became and read its insights. It cannot publish, and it is not asked to.

    PERFORMANCE_PROVIDER=pipeline.instagram.fetch

Four things about it are worth knowing before you rely on it:

* **it reads your own account and nothing else.** The Graph API's media node
  only answers for the account the token belongs to. There is no path here to
  anyone else's numbers, and none is wanted;
* **matching a reel to a post is the hard half.** Instagram has never heard of
  `cd1bccae`. What it has is a caption, and step 6 wrote that caption - so the
  first line of `post.txt` is the join key, and the timestamp is the tie-break.
  Neither is certain, so a match is always reported with how it was made and
  can be overruled by hand with `--pin`;
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
METRIC_MAP = {
    "views": "views",
    "plays": "views",
    "reach": "reach",
    "likes": "likes",
    "comments": "comments",
    "saved": "saves",
    "shares": "shares",
    "ig_reels_avg_watch_time": "avg_watch_seconds",
}

#: The ones worth a second attempt on their own if the batch was refused.
CORE_METRICS = ("reach", "likes", "comments", "saved")


class InstagramError(RuntimeError):
    """Raised when the API cannot be used at all."""


class InstagramAuthError(InstagramError):
    """The token is missing, expired, or not entitled to what was asked."""


def _base() -> str:
    return f"https://graph.facebook.com/{config.INSTAGRAM_API_VERSION}"


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


def require_credentials() -> None:
    config.require(*config.REQUIRED_INSTAGRAM)


# ---------------------------------------------------------------------------
# The account, and what it has posted
# ---------------------------------------------------------------------------

def account(session=None) -> dict:
    """Who the token belongs to. The cheapest proof it works."""
    require_credentials()
    return _get(session or _session(), str(config.INSTAGRAM_USER_ID),
                {"fields": "id,username,media_count"})


def media(session=None, *, limit: int | None = None) -> list[dict]:
    """Recent posts, newest first, as the API describes them."""
    require_credentials()
    limit = config.INSTAGRAM_MEDIA_LIMIT if limit is None else limit
    body = _get(session or _session(), f"{config.INSTAGRAM_USER_ID}/media", {
        "fields": "id,caption,timestamp,media_type,media_product_type,permalink",
        "limit": max(1, limit),
    })
    return [item for item in (body.get("data") or []) if isinstance(item, dict)]


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
            refused = [name for name in wanted if name in str(exc)]
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


def match(features: dict, posts: list[dict]) -> dict:
    """Which post this reel became, and how sure that is.

    Returns {"media_id", "how", "permalink"} - `how` being "caption" for a
    match on the words step 6 wrote, "timestamp" for the nearest post inside
    INSTAGRAM_MATCH_HOURS, or "" for no match at all. The how is carried
    through to the ledger because "we guessed by time" and "the caption is
    identical" are not the same claim and should not be recorded as one.
    """
    hook = _normalise(caption_of(features))
    if hook:
        for post in posts:
            caption = _normalise(post.get("caption") or "")
            if caption and (caption.startswith(hook) or hook in caption):
                return {"media_id": post.get("id") or "", "how": "caption",
                        "permalink": post.get("permalink") or ""}

    made = perform._parse_stamp(features.get("created_at"))
    if not made:
        return {"media_id": "", "how": "", "permalink": ""}

    window = timedelta(hours=max(1, config.INSTAGRAM_MATCH_HOURS))
    best, best_gap = None, window
    for post in posts:
        # A reel is posted after it is made, never before, and a post that
        # predates the reel is some other post that happens to be nearby.
        when = perform._parse_stamp(post.get("timestamp"))
        if not when or when < made:
            continue
        gap = when - made
        if gap <= best_gap:
            best, best_gap = post, gap
    if best is None:
        return {"media_id": "", "how": "", "permalink": ""}
    return {"media_id": best.get("id") or "", "how": "timestamp",
            "permalink": best.get("permalink") or ""}


def derive_watch_through(metrics: dict, features: dict) -> dict:
    """Turn an average watch time into the fraction of the reel watched.

    Instagram reports milliseconds watched; step 9 ranks on what share of the
    reel that was. The reel's own length is in its features row, so the
    conversion is division - and it is done here rather than in the ledger
    because it is this provider's unit problem, not step 9's.
    """
    out = dict(metrics)
    millis = out.pop("avg_watch_seconds", None)
    if millis is None:
        return out
    seconds = float(millis) / 1000.0
    out["avg_watch_seconds"] = round(seconds, 2)

    length = features.get("spoken_seconds")
    if length and float(length) > 0:
        out["watch_through"] = round(min(1.0, seconds / float(length)), 4)
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
    out: dict[str, dict] = {}

    for run_id in run_ids:
        features = features_by_run.get(run_id) or {}
        entry = ledger.entry(run_id)

        pinned = entry.get("media_id") or ""
        found = ({"media_id": pinned, "how": "pinned",
                  "permalink": entry.get("permalink") or ""} if pinned
                 else match(features, posts))
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
        record["matched_by"] = found["how"]
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
    waiting = ledger.pending()
    if not waiting:
        print("nothing is waiting for numbers.")
        return 0

    recorded, warnings = perform.fetch(ledger=ledger, provider=fetch)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(f"recorded {recorded} of {len(waiting)} waiting reel(s).")
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
        print(f"  python pipeline/perform.py <run> --skipped \"why\"")
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
