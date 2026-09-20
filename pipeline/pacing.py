"""Shot pacing - how long a shot holds, and how many a window of narration gets.

Lifted out of step 5 because step 4 has to ask the same question before it can
answer its own. Step 5 decides how many shots a beat is cut into from the beat's
length and the pacing curve; step 4 decides how many clips to source for that
beat. While those two numbers came from different arithmetic - a curve on one
side, a flat VISUAL_ASSETS_PER_BEAT on the other - a long beat was routinely cut
into more shots than it had footage for, and the extra shots were filled by
cycling back over clips the viewer had just seen. Measured across the 21 reels
on disk, 15 had at least one beat in that state.

Nothing here touches the network, ffmpeg or the filesystem: it is arithmetic
over the SHOT_* settings, and both steps import it so the two answers cannot
drift apart again.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow both `python pipeline/pacing.py` and `python -m pipeline.pacing`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def shot_target_at(position: float, total: float) -> float:
    """How long a shot should hold, `position` seconds into a `total`-second reel.

    One rate for the whole reel is the obvious thing to do and the wrong one.
    A cut costs the viewer a moment's re-orientation and buys back a moment's
    attention, and the exchange rate is not constant: at the top there is no
    attention yet to lose, so cuts are nearly free and nearly all of what keeps
    someone watching; in the middle the viewer is following an argument, and a
    cut every second is just noise over the top of it. At the close the reel
    has to land rather than fade, so the rate comes back up.

    So: SHOT_HOOK_SECONDS at t=0, easing linearly out to SHOT_TARGET_SECONDS
    over SHOT_PACING_RAMP_SECONDS, then tightening to SHOT_OUTRO_SECONDS across
    the final SHOT_OUTRO_WINDOW_SECONDS.
    """
    steady = max(0.3, config.SHOT_TARGET_SECONDS)
    if (config.SHOT_PACING or "").strip().lower() != "curve":
        return steady

    target = steady
    ramp = max(0.0, config.SHOT_PACING_RAMP_SECONDS)
    if ramp > 0 and position < ramp:
        opening = max(0.3, config.SHOT_HOOK_SECONDS)
        target = opening + (steady - opening) * (position / ramp)

    window = max(0.0, config.SHOT_OUTRO_WINDOW_SECONDS)
    if window > 0 and total > 0 and position > total - window:
        into = min(1.0, (position - (total - window)) / window)
        closing = max(0.3, config.SHOT_OUTRO_SECONDS)
        # `min` rather than a straight assignment: on a reel short enough that
        # the ramp and the outro window overlap, whichever phase wants the
        # faster cutting gets it, instead of the outro undoing the opening.
        target = min(target, steady + (closing - steady) * into)

    return max(0.3, target)


def curve_shots_in(start: float, end: float, total: float,
                    steps: int = 64) -> float:
    """How many shot-lengths the pacing curve fits between two times.

    The integral of dt/target(t), which is the only honest way to ask the
    question when the target is not constant across the window. Sampling the
    curve once at the middle of the beat instead would quietly cancel the whole
    feature on exactly the beat that matters: a long opening beat's midpoint
    already sits past the ramp, so it would be handed the steady rate and the
    reel would open at the same pace as its middle.
    """
    span = end - start
    if span <= 0:
        return 0.0
    step = span / steps
    return sum(step / max(0.3, shot_target_at(start + (i + 0.5) * step, total))
               for i in range(steps))


def shot_count(window: float, available: int | None = None, *,
                position: float = 0.0, total: float = 0.0) -> int:
    """How many cuts a window of narration should get.

    Follows the pacing curve across the window, then respects the floor and
    both ceilings - the absolute SHOT_MAX_SECONDS and the SHOT_MAX_OVERRUN one
    that is measured against the curve. An asset may be used twice inside one
    beat if that is what a ceiling needs, but no further - past that the shot
    simply runs long.

    `available` is how many clips the beat actually has. Step 5 always knows
    it. Step 4 asks the question the other way round - it is deciding what to
    download - and passes None for "no footage constraint yet", which is the
    demand the curve would like met.

    The floor gives way to the curve rather than fighting it. SHOT_MIN_SECONDS
    exists to stop a short beat being chopped into flashes by the arithmetic;
    where the curve is deliberately asking for 1.4s shots through the opening,
    that is the edit and not an accident, so the floor steps aside.
    """
    count = max(1, int(round(curve_shots_in(position, position + window, total))))

    floor = min(config.SHOT_MIN_SECONDS,
                shot_target_at(position + window / 2, total))
    while count > 1 and window / count < floor:
        count -= 1
    if config.SHOT_MAX_SECONDS > 0:
        count = max(count, math.ceil(window / config.SHOT_MAX_SECONDS - 1e-6))

    # The ceiling that actually binds. SHOT_MAX_SECONDS sits at 6.0s while the
    # curve asks for 3.2s, so it never fires, and what decided shot length was
    # the rounding above: a window up to 1.49 targets long rounds to one shot
    # and then holds for all of it. This charges that overrun against the curve
    # instead of against a constant, which is the only ceiling that means the
    # same thing at the top of a reel and in the middle of one.
    if config.SHOT_MAX_OVERRUN > 0:
        ceiling = shot_target_at(position + window / 2, total) * config.SHOT_MAX_OVERRUN
        if ceiling > 0:
            count = max(count, math.ceil(window / ceiling - 1e-6))

    if available is None:
        return max(1, count)
    return max(1, min(count, max(1, available * 2)))
