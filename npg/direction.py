"""Uphill-first planning for the non-planar extrusion paths.

The flat side of the nozzle pushes a little material ahead of it when a line is
printed downhill, so a stroke should rather run uphill.  This module splits a
stroke into monotone sections and gives each one a print direction that goes up
the surface.

The split works on the real z of the stroke (no smoothing) with a hysteresis of
``min_rise``: every turn of the surface whose counter-movement exceeds
``min_rise`` becomes a cut.  Inside a section the height therefore never runs
against the print direction by more than ``min_rise`` – a curved line with
stair noise below the threshold stays in one piece, a wave across a slope gets
one cut per crest and trough.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 0.02          # mm – a section flatter than this counts as level (direction free)
MIN_H = 0.005       # mm – floor for the hysteresis (heightmap noise), also used for min_rise = 0


def _arc(P: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))] if len(P) > 1 else np.zeros(1)


def _turns(z: np.ndarray, h: float) -> list[tuple[int, float, bool]]:
    """Zigzag extrema with hysteresis ``h``: [(index, swing, is_peak)] of the confirmed turns.

    A turn is confirmed once the height has moved back by more than ``h`` from
    the running extreme; ``swing`` is that counter-movement's final size (the
    smaller of the two neighbouring rises), used to rank cuts if they must be
    capped.
    """
    n = len(z)
    ext: list[int] = []
    trend = 0                  # +1 rising, -1 falling, 0 unknown
    hi = lo = 0
    for i in range(1, n):
        v = z[i]
        if trend >= 0:
            if v > z[hi]:
                hi = i
            if trend == 0 and v < z[lo]:
                lo = i
            if trend == 0:
                if z[hi] - z[lo] > h:
                    trend = 1 if hi > lo else -1
                    if trend == 1:
                        lo = hi
                    else:
                        hi = lo
                continue
            if z[hi] - v > h:          # peak confirmed
                ext.append(hi)
                trend, lo = -1, i
        else:
            if v < z[lo]:
                lo = i
            if v - z[lo] > h:          # trough confirmed
                ext.append(lo)
                trend, hi = 1, i
    pts = [0] + ext + [n - 1]
    out = []
    for k in range(1, len(pts) - 1):
        a, b, c = pts[k - 1], pts[k], pts[k + 1]
        out.append((b, float(min(abs(z[b] - z[a]), abs(z[c] - z[b]))), bool(z[b] >= z[a])))
    return out


def plan(P: np.ndarray, z: np.ndarray, *, min_rise: float = 0.10, min_len: float = 0.5,
         max_parts: int = 0, smooth: float = 0.0) -> list[tuple[int, int, bool | None]]:
    """Split ``P``/``z`` into sections that can each be printed uphill.

    Returns ``[(i0, i1, rev)]`` covering the whole polyline in index order.
    ``rev`` is True (print from i1 back to i0), False, or None for a level
    section where either direction works.

    ``min_len`` only protects the two ends of the stroke: a turn closer than
    that to the start or end is not cut off (a tiny stub would just be a
    blob).  ``max_parts`` <= 0 means unlimited; a cap drops the weakest turns
    first and then leaves some downhill.  ``smooth`` is accepted for old
    designs and ignored – smoothing hid real dips.
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    h = max(float(min_rise), MIN_H)
    if n < 3:
        return [(0, n - 1, _dir(float(z[-1] - z[0]), h))]
    s = _arc(P)
    L = float(s[-1])

    # single-sample spikes are heightmap stair noise, not dips: find the turns on
    # a 3-point median (keeps every real dip of >= 2 samples), decide on raw z
    zm = z.copy()
    zm[1:-1] = np.median(np.stack([z[:-2], z[1:-1], z[2:]]), axis=0)
    turns = []
    for i, w, peak in _turns(zm, h):           # snap back to the raw extreme nearby
        a, b = max(i - 1, 0), min(i + 2, n)
        seg = z[a:b]
        turns.append((a + int(np.argmax(seg) if peak else np.argmin(seg)), w))
    cuts = [(i, w) for i, w in turns if s[i] >= min_len and L - s[i] >= min_len]
    if max_parts and max_parts > 0:
        while len(cuts) + 1 > max_parts:
            cuts.pop(min(range(len(cuts)), key=lambda k: cuts[k][1]))

    idx = [0] + [i for i, _ in cuts] + [n - 1]
    out: list[tuple[int, int, bool | None]] = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 1:
            continue
        out.append((a, b, _dir(float(z[b] - z[a]), h)))
    return out or [(0, n - 1, None)]


def _dir(rise: float, h: float = EPS) -> bool | None:
    t = min(EPS, h)
    if rise < -t:
        return True          # would run downhill – print it backwards
    if rise > t:
        return False
    return None


def order(sections: list[tuple[int, int, bool | None]], P: np.ndarray,
          start: tuple[float, float]) -> list[tuple[int, int, bool]]:
    """Nearest-section ordering; level sections may flip to shorten the travel."""
    rem = list(range(len(sections)))
    out: list[tuple[int, int, bool]] = []
    cx, cy = float(start[0]), float(start[1])
    while rem:
        best = None
        for k in rem:
            i0, i1, rev = sections[k]
            if rev is None:
                d0 = math.hypot(P[i0, 0] - cx, P[i0, 1] - cy)
                d1 = math.hypot(P[i1, 0] - cx, P[i1, 1] - cy)
                cand = (d0 if d0 <= d1 else d1, k, d1 < d0)
            else:
                j = i1 if rev else i0
                cand = (math.hypot(P[j, 0] - cx, P[j, 1] - cy), k, rev)
            if best is None or cand[0] < best[0]:
                best = cand
        _, k, rev = best
        i0, i1, _ = sections[k]
        out.append((i0, i1, bool(rev)))
        cx, cy = float(P[i0 if rev else i1, 0]), float(P[i0 if rev else i1, 1])
        rem.remove(k)
    return out
