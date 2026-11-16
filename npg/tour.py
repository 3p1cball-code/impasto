"""Travel-minimising order for the non-planar sections.

Each section is a short stroke with two ends.  Which end is printed first is not
only a print-quality question ("uphill first", see :mod:`npg.direction`) but also
a travel question: every gap between two sections is a retract, a hop and a
prime, and a long gap leaves a blob.  This module orders the sections and
chooses their direction so that the travels stay short, while a downhill
millimetre is charged like ``weight`` millimetres of travel.

``weight = 0`` orders purely by travel (downhill ignored), a large weight keeps
the uphill decision of the planner untouched.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _neighbours(P: np.ndarray, n: int, k: int) -> list[list[int]]:
    """nearest other sections per section, measured between any two ends"""
    if n < 2:
        return [[] for _ in range(n)]
    k = max(1, min(k, n - 1))
    tree = cKDTree(P)
    _, idx = tree.query(P, k=min(len(P), 2 * k + 2))
    idx = np.atleast_2d(idx)
    out = []
    for i in range(n):
        cand = np.unique(np.concatenate([idx[i], idx[i + n]])) % n
        out.append([int(c) for c in cand if c != i][:k])
    return out


_EXACT_N = 9   # up to this many sections the order is solved exactly (Held-Karp)


def _exact(ax, ay, bx, by, down_f, down_b, w, sx, sy):
    """optimal open tour with free direction per section, dynamic programming
    over (visited set, last section, last direction); 2^n * 2n states"""
    n = len(ax)
    hyp = math.hypot
    ent = [[(ax[i], ay[i]), (bx[i], by[i])] for i in range(n)]   # [f] -> entry
    ext = [[(bx[i], by[i]), (ax[i], ay[i])] for i in range(n)]   # [f] -> exit
    dw = [[down_f[i], down_b[i]] for i in range(n)]
    # gap[i][fi][j][fj]: travel from exit of (i, fi) to entry of (j, fj)
    gap = [[[[hyp(ent[j][fj][0] - ext[i][fi][0], ent[j][fj][1] - ext[i][fi][1])
              for fj in (0, 1)] for j in range(n)] for fi in (0, 1)] for i in range(n)]
    INF = float("inf")
    full = 1 << n
    D = [[INF] * (2 * n) for _ in range(full)]
    P = [[-1] * (2 * n) for _ in range(full)]
    for j in range(n):
        for fj in (0, 1):
            D[1 << j][2 * j + fj] = hyp(ent[j][fj][0] - sx, ent[j][fj][1] - sy) + w * dw[j][fj]
    for m in range(1, full):
        row = D[m]
        for s in range(2 * n):
            c = row[s]
            if c == INF:
                continue
            i, fi = divmod(s, 2)
            g = gap[i][fi]
            for j in range(n):
                if m >> j & 1:
                    continue
                mj = m | (1 << j)
                gj = g[j]
                for fj in (0, 1):
                    v = c + gj[fj] + w * dw[j][fj]
                    if v < D[mj][2 * j + fj] - 1e-12:
                        D[mj][2 * j + fj] = v
                        P[mj][2 * j + fj] = s
    last = min(range(2 * n), key=lambda s: D[full - 1][s])
    order = []
    m, s = full - 1, last
    while s >= 0:
        order.append(s)
        ps = P[m][s]
        m ^= 1 << (s // 2)
        s = ps
    order.reverse()
    ids = [s // 2 for s in order]
    fls = [bool(s % 2) for s in order]
    travel, down = 0.0, 0.0
    px, py = sx, sy
    for i, f in zip(ids, fls):
        e = ent[i][f]
        travel += hyp(e[0] - px, e[1] - py)
        px, py = ext[i][f]
        down += dw[i][f]
    return ids, fls, {"travel": travel, "downhill": down, "cost": travel + w * down}


def _solve(sections, start, weight: float, passes: int, cand: int, kicks: int = 0,
           starts=None):
    n = len(sections)
    if n == 0:
        return [], [], {"travel": 0.0, "downhill": 0.0, "cost": 0.0}
    # plain python floats: the inner loops run millions of times
    ax = [float(s[0][0]) for s in sections]
    ay = [float(s[0][1]) for s in sections]
    bx = [float(s[1][0]) for s in sections]
    by = [float(s[1][1]) for s in sections]
    down_f = [max(0.0, -float(s[2])) for s in sections]     # downhill for a -> b
    down_b = [max(0.0, float(s[2])) for s in sections]      # downhill for b -> a
    w = max(0.0, float(weight))
    sx, sy = float(start[0]), float(start[1])
    hyp = math.hypot
    if starts is None:
        starts = [w]

    if n <= _EXACT_N:
        return _exact(ax, ay, bx, by, down_f, down_b, w, sx, sy)

    # ---- nearest neighbour with a free direction -----------------------------
    A = np.array([ax, ay], dtype=np.float64).T
    B = np.array([bx, by], dtype=np.float64).T
    df = np.array(down_f)
    db = np.array(down_b)

    def nearest(wn):
        ids: list[int] = []
        fls: list[bool] = []
        free = np.ones(n, dtype=bool)
        px, py = sx, sy
        for _ in range(n):
            idx = np.nonzero(free)[0]
            cf = np.hypot(A[idx, 0] - px, A[idx, 1] - py) + wn * df[idx]
            cb = np.hypot(B[idx, 0] - px, B[idx, 1] - py) + wn * db[idx]
            if cf.min() <= cb.min():
                j = int(idx[int(np.argmin(cf))])
                ids.append(j)
                fls.append(False)
                px, py = bx[j], by[j]
            else:
                j = int(idx[int(np.argmin(cb))])
                ids.append(j)
                fls.append(True)
                px, py = ax[j], ay[j]
            free[j] = False
        return ids, fls

    nb = _neighbours(np.vstack([A, B]), n, cand)

    def entry(i, f):
        return (bx[i], by[i]) if f else (ax[i], ay[i])

    def exit_(i, f):
        return (ax[i], ay[i]) if f else (bx[i], by[i])

    def dwn(i, f):
        return down_b[i] if f else down_f[i]

    def improve(ids, fls, w=w):
        """first-improvement 2-opt + or-opt (blocks of 1-3, both directions)
        until no move helps or ``passes`` sweeps without progress are used up"""
        sweeps = 0
        while sweeps < passes:
            sweeps += 1
            moved = False
            # ---- 2-opt: reverse a block, every section inside flips ----------
            pos = {k: t for t, k in enumerate(ids)}
            ps = [0.0]
            for i, f in zip(ids, fls):
                ps.append(ps[-1] + dwn(i, not f) - dwn(i, f))   # flip delta prefix
            t = 0
            while t < n:
                if t > 0:
                    px, py = exit_(ids[t - 1], fls[t - 1])
                    cands = set(nb[ids[t - 1]]) | set(nb[ids[t]])
                else:
                    px, py = sx, sy
                    cands = set(nb[ids[t]])
                ex0, ey0 = entry(ids[t], fls[t])
                head = hyp(ex0 - px, ey0 - py)
                best = None
                for j in cands:
                    k = pos[j]
                    if k <= t:
                        continue
                    old = head
                    new = 0.0
                    if k + 1 < n:
                        qx, qy = entry(ids[k + 1], fls[k + 1])
                        kx, ky = exit_(ids[k], fls[k])
                        old += hyp(kx - qx, ky - qy)
                        new += hyp(ex0 - qx, ey0 - qy)      # flipped first exits at its old entry
                    nx0, ny0 = exit_(ids[k], fls[k])         # flipped last enters at its old exit
                    new += hyp(nx0 - px, ny0 - py)
                    delta = new - old + w * (ps[k + 1] - ps[t])
                    if delta < -1e-7 and (best is None or delta < best[0]):
                        best = (delta, k)
                if best is not None:
                    k = best[1]
                    ids[t:k + 1] = ids[t:k + 1][::-1]
                    fls[t:k + 1] = [not f for f in fls[t:k + 1][::-1]]
                    pos = {kk: tt for tt, kk in enumerate(ids)}
                    ps = [0.0]
                    for i, f in zip(ids, fls):
                        ps.append(ps[-1] + dwn(i, not f) - dwn(i, f))
                    moved = True
                t += 1
            # ---- or-opt: move a block of 1-3 sections, optionally reversed ---
            for L in (1, 2, 3):
                if L >= n:
                    break
                t = 0
                pos = {kk: tt for tt, kk in enumerate(ids)}
                while t + L <= n:
                    a, fa = ids[t], fls[t]
                    b, fb = ids[t + L - 1], fls[t + L - 1]
                    sa = entry(a, fa)                      # block entry
                    sb = exit_(b, fb)                      # block exit
                    flipd = sum(dwn(ids[u], not fls[u]) - dwn(ids[u], fls[u])
                                for u in range(t, t + L))
                    pre = exit_(ids[t - 1], fls[t - 1]) if t > 0 else (sx, sy)
                    post = entry(ids[t + L], fls[t + L]) if t + L < n else None
                    gain = -hyp(sa[0] - pre[0], sa[1] - pre[1])
                    if post is not None:
                        gain += hyp(post[0] - pre[0], post[1] - pre[1]) \
                            - hyp(post[0] - sb[0], post[1] - sb[1])
                    best = None
                    gaps = set()
                    for j in set(nb[a]) | set(nb[b]):
                        q = pos[j]
                        gaps.add(q)
                        gaps.add(q + 1)
                    gaps.add(0)
                    for g in gaps:
                        if t <= g <= t + L or g > n:
                            continue
                        gp = exit_(ids[g - 1], fls[g - 1]) if g > 0 else (sx, sy)
                        gq = entry(ids[g], fls[g]) if g < n else None
                        base = hyp(gq[0] - gp[0], gq[1] - gp[1]) if gq is not None else 0.0
                        for rev in (False, True):
                            en, ex = (sb, sa) if rev else (sa, sb)
                            add = hyp(en[0] - gp[0], en[1] - gp[1])
                            if gq is not None:
                                add += hyp(gq[0] - ex[0], gq[1] - ex[1])
                            delta = add - base + gain + (w * flipd if rev else 0.0)
                            if delta < -1e-7 and (best is None or delta < best[0]):
                                best = (delta, g, rev)
                    if best is not None:
                        _, g, rev = best
                        bi = ids[t:t + L]
                        bf = fls[t:t + L]
                        if rev:
                            bi = bi[::-1]
                            bf = [not f for f in bf[::-1]]
                        if g > t:
                            g -= L
                        del ids[t:t + L]
                        del fls[t:t + L]
                        ids[g:g] = bi
                        fls[g:g] = bf
                        pos = {kk: tt for tt, kk in enumerate(ids)}
                        moved = True
                    else:
                        t += 1
            if not moved:
                break
        return ids, fls

    def cost(ids, fls):
        tr = 0.0
        px, py = sx, sy
        dn = 0.0
        for i, f in zip(ids, fls):
            ex, ey = entry(i, f)
            tr += hyp(ex - px, ey - py)
            px, py = exit_(i, f)
            dn += dwn(i, f)
        return tr + w * dn

    # every start weight seeds a tour, which is then polished with the real
    # weight; the cheapest one goes on to the kicks
    best_c = None
    for ws in starts:
        ti, tf = nearest(ws)
        if ws != w:
            ti, tf = improve(ti, tf, ws)
        ti, tf = improve(ti, tf)
        c = cost(ti, tf)
        if best_c is None or c < best_c - 1e-9:
            ids, fls, best_c = ti, tf, c

    # ---- a few deterministic kicks (double bridge) + local search ------------
    if n >= 8 and kicks > 0:
        rng = np.random.default_rng(len(ids))
        for _ in range(kicks):
            p1, p2, p3 = sorted(rng.choice(np.arange(1, n), size=3, replace=False))
            ti = ids[:p1] + ids[p2:p3] + ids[p1:p2] + ids[p3:]
            tf = fls[:p1] + fls[p2:p3] + fls[p1:p2] + fls[p3:]
            ti, tf = improve(ti, tf)
            c = cost(ti, tf)
            if c < best_c - 1e-7:
                ids, fls, best_c = ti, tf, c

    travel = 0.0
    px, py = sx, sy
    for t, (i, f) in enumerate(zip(ids, fls)):
        ex, ey = entry(i, f)
        travel += hyp(ex - px, ey - py)
        px, py = exit_(i, f)
    downhill = float(sum(dwn(i, f) for i, f in zip(ids, fls)))
    info = {"travel": travel, "downhill": downhill, "cost": travel + w * downhill}
    return ids, fls, info


def _d2(p, q):
    return (p[0] - q[0], p[1] - q[1])


def plan(sections, start, *, weight: float = 10.0, passes: int | None = None, cand: int = 12,
         kicks: int | None = None):
    """Order ``sections`` from ``start``.

    ``sections`` is a sequence of ``(a, b, drop)``: the two end points in index
    order and the extra downhill (mm) that printing ``a -> b`` costs compared to
    printing ``b -> a``.

    Returns ``(order, flips, info)`` with ``order`` = section indices in print
    order, ``flips[k]`` = print ``order[k]`` backwards, and ``info`` holding
    ``travel`` (mm), ``downhill`` (mm) and ``cost`` (both combined).

    Two seed tours are built - nearest neighbour with the requested weight and
    purely by travel - both are polished with the requested weight and the
    better one goes on to ``kicks`` deterministic double-bridge perturbations
    (fixed seed, so the same input always gives the same order).
    """
    n = len(sections)
    w = max(0.0, float(weight))
    if passes is None:
        passes = 40 if n <= 400 else 15 if n <= 1500 else 6
    if kicks is None:
        kicks = 20 if n <= 150 else 10 if n <= 400 else 0
    return _solve(sections, start, w, passes, cand, kicks,
                  starts=[w, 0.0] if w > 0 else [0.0])
