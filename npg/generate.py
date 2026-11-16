"""Generates the single non-planar decoration layer as Bambu-compatible G-code."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

from . import direction, surface, toolchange
from .bambu import Job
from .parse import Toolpaths
from .surface import Grid


@dataclass
class Params:
    z_offset: float = 0.15        # nozzle height above the printed surface = bead height (mm);
                                  # negative = sink the nozzle a bit into the estimated surface
    line_width: float = 0.45      # mm
    flow: float = 1.0             # multiplier (on top of the filament flow ratio from Bambu)
    use_filament_flow_ratio: bool = True
    speed: float = 15.0           # print speed mm/s (capped by filament max volumetric speed)
    travel_speed: float = 150.0   # mm/s
    hop: float = 0.8              # travel lift above the highest surface point along the travel (mm)
    retract: float = -1.0         # mm, <0 = from Bambu config (retraction_length)
    retract_speed: float = 30.0   # mm/s
    step: float = 0.25            # resampling step so the nozzle follows the surface (mm)
    min_len: float = 0.5          # drop shorter strokes (mm)
    gap_flow: bool = True         # adapt extrusion to the real gap below the nozzle
    stack: bool = True            # lift over already printed non-planar lines (crossings)
    ramp: float = 0.5             # max dz/ds of crossing ramps
    accel: int = 2000             # M204 acceleration
    fan: int = -1                 # part fan %, <0 = Bambu filament fan_max_speed
    prime: float = 0.2            # extra mm filament after a filament change
    short_travel: float = 1.0     # travels shorter than this stay down, no retract (mm)
    purge_factor: float = 1.0     # multiplier on Bambu's purge volume for a filament change
    min_purge: float = 50.0       # mm3, at least this much is purged on every change
    color_order: list[int] = field(default_factory=list)  # explicit slot order (optional)
    dir_opt: bool = True          # print every section uphill where possible
    dir_min_rise: float = 0.10    # mm – a turn in the surface only cuts the stroke above this
    dir_end_len: float = 0.5      # mm – no cut closer than this to a stroke end
    dir_cap: int = 0              # max sections per stroke, 0 = unlimited
    # old keys dir_min_len / dir_max_parts / dir_smooth are ignored on purpose:
    # their old defaults (1.5 / 6 / 0.8) hid real dips and left downhill lines.

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                cur = getattr(p, k)
                if isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    v = int(round(float(v)))
                elif isinstance(cur, float):
                    v = float(v)
                setattr(p, k, v)
        return p


def _f(v: float, nd: int = 3) -> str:
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(d / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def _length(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


@dataclass
class Result:
    gcode: str
    moves: np.ndarray            # (N, 9) float32: type(0 extr/1 travel/2 change) slot x0 y0 z0 x1 y1 z1 line
    stats: dict
    max_z: float
    end_slot: int


class Writer:
    def __init__(self):
        self.lines: list[str] = []
        self.moves: list[tuple] = []
        self.x = self.y = self.z = None
        self.f = None

    def emit(self, s: str):
        self.lines.append(s)

    def move(self, x=None, y=None, z=None, e=None, f=None, kind=1, slot=-1, comment=""):
        parts = ["G1"]
        nx = self.x if x is None else x
        ny = self.y if y is None else y
        nz = self.z if z is None else z
        if x is not None and (self.x is None or abs(x - self.x) > 1e-4):
            parts.append("X" + _f(x))
        if y is not None and (self.y is None or abs(y - self.y) > 1e-4):
            parts.append("Y" + _f(y))
        if z is not None and (self.z is None or abs(z - self.z) > 1e-4):
            parts.append("Z" + _f(z))
        if e is not None:
            parts.append("E" + _f(e, 5))
        if f is not None and (self.f is None or abs(f - self.f) > 0.5):
            parts.append("F" + _f(f, 0))
            self.f = f
        if len(parts) == 1 or (len(parts) == 2 and parts[1][0] == "F"):
            if len(parts) == 2:
                self.f = None   # feedrate not emitted, force it on the next real move
            return
        self.emit(" ".join(parts) + (f" ; {comment}" if comment else ""))
        if None not in (self.x, self.y, self.z) and None not in (nx, ny, nz) and (x is not None or y is not None or z is not None):
            self.moves.append((kind, slot, self.x, self.y, self.z, nx, ny, nz, len(self.lines) - 1))
        self.x, self.y, self.z = nx, ny, nz


def generate(job: Job, tp: Toolpaths, g: Grid, paths: list[dict], params: Params, *,
             layer_num: int | None = None) -> Result:
    p = params
    if g.ZN is None:
        surface.analyze(g)
    n_fil = job.n_filaments
    w = p.line_width
    off = p.z_offset
    bead = max(abs(off), 0.02)     # nominal bead height for flow / speed calculation
    area_cache = {}

    # local mean surface for gap-adaptive flow
    valid = np.isfinite(g.H)
    sig = max(0.5, 0.25 * w / g.res)
    num = ndimage.gaussian_filter(np.where(valid, g.H, 0.0), sig, mode="constant")
    den = ndimage.gaussian_filter(valid.astype(np.float64), sig, mode="constant")
    HA = np.where(den > 0.2, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)
    HNP = np.full_like(g.H, -np.inf)          # top of already printed non-planar beads

    # ---- clean & group input --------------------------------------------------
    # The element order is the stacking order: element 1 is printed first (on the part),
    # every later element is laid onto whatever is below it at that point.
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    closed_flags: dict[tuple[int, int], list[bool]] = {}
    for item in paths:
        slot = int(item.get("slot", 0))
        if slot < 0 or slot >= n_fil:
            slot = 0
        idx = int(item.get("order", 0) or 0)
        pts = np.asarray(item.get("pts", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            continue
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-4]
        pts = pts[keep]
        if len(pts) < 2 or _length(pts) < p.min_len:
            continue
        closed = bool(item.get("closed", False)) or float(np.hypot(*(pts[0] - pts[-1]))) < 1e-3
        groups.setdefault((idx, slot), []).append(pts)
        closed_flags.setdefault((idx, slot), []).append(closed)

    cur_slot = tp.last_filament if tp.last_filament >= 0 else 0

    def colour_order(slots_: list[int]) -> list[int]:
        """Colour sequence with the least purge (Bambu flush matrix), starting from the loaded filament."""
        slots_ = sorted(set(slots_))
        if not slots_:
            return []
        if p.color_order:
            first = [s for s in p.color_order if s in slots_]
            return first + [s for s in slots_ if s not in first]
        if len(slots_) > 6:
            return slots_
        import itertools

        def cost(seq):
            c, prev = 0.0, cur_slot
            for s in seq:
                if s != prev:
                    c += max(p.min_purge, toolchange.purge_volume(job, prev, s) * p.purge_factor) + 1e-3
                prev = s
            return c
        return list(min(itertools.permutations(slots_), key=cost))

    work: list[tuple[int, int]] = []
    for idx in sorted({i for (i, _) in groups}):
        for slot in colour_order([s for (i, s) in groups if i == idx]):
            work.append((idx, slot))

    W = Writer()
    base_z = max(tp.max_z, max(tp.layer_z) if tp.layer_z else tp.max_z)
    zn_max = float(np.nanmax(g.ZN)) if np.isfinite(g.ZN).any() else base_z
    max_z_seen = base_z
    if layer_num is None:
        layer_num = len(tp.layer_z)            # 0-based index of the new layer
    total_layers = layer_num + 1

    retract_len = {s: (p.retract if p.retract >= 0 else job.float_at("retraction_length", s, 0.8)) for s in range(n_fil)}
    temps = job.floats("nozzle_temperature")
    fan_cfg = job.floats("fan_max_speed")
    mvs = job.floats("filament_max_volumetric_speed")
    ratios = job.floats("filament_flow_ratio")

    stats = {"length_mm": {}, "filament_mm": {}, "volume_mm3": {}, "strokes": {}, "toolchanges": 0,
             "purge_mm3": 0.0, "time_s": 0.0, "max_z": 0.0, "min_z": 1e9,
             "cuts": [], "dir_up_mm": 0.0, "dir_down_mm": 0.0, "dir_before_mm": 0.0,
             "dir_down_max": 0.0, "stacked_mm": 0.0, "elements": len({i for (i, _) in groups})}

    W.emit("; ===== NON-PLANAR LAYER (Non-Planar Gcoder) START =====")
    W.emit("; CHANGE_LAYER")
    W.emit(f"; Z_HEIGHT: {_f(zn_max + off)}")
    W.emit(f"; LAYER_HEIGHT: {_f(off)}")
    W.emit(f"; layer num/total_layer_count: {layer_num + 1}/{total_layers}")
    W.emit("; update layer progress")
    W.emit(f"M73 L{layer_num + 1}")
    W.emit(f"M991 S0 P{layer_num} ;notify layer change")
    W.emit("; FEATURE: Top surface")
    W.emit(f"; LINE_WIDTH: {_f(w)}")
    W.emit("G90")
    W.emit("M83")
    W.emit(f"M204 S{p.accel}")

    retracted = retract_len.get(cur_slot, 0.8)   # Bambu wipes+retracts at the end of the last layer
    safe_z = zn_max + max(p.hop, 0.4) + 0.4
    W.emit(f"G1 Z{_f(safe_z)} F1200 ; lift above the part")
    W.z = safe_z
    W.x, W.y = tp.end_pos[0], tp.end_pos[1]
    stats["time_s"] += abs(safe_z - tp.end_pos[2]) / 20.0

    def zn_line_max(x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / 0.4) + 1)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        z = surface.sample_max(g.ZN, g, xs, ys)
        z2 = surface.sample_max(HNP, g, xs, ys) if p.stack else z
        m = np.nanmax(np.r_[z, z2, -np.inf])
        return float(m) if math.isfinite(m) else base_z

    def fan_cmd(slot):
        pct = p.fan if p.fan >= 0 else (fan_cfg[slot] if slot < len(fan_cfg) else 100)
        W.emit(f"M106 P1 S{int(round(255 * max(0, min(100, pct)) / 100))}")

    first = True
    tc_count = 0
    for (idx, slot) in work:
        strokes = groups[(idx, slot)]
        closed = closed_flags[(idx, slot)]
        if slot != cur_slot:
            # ---- filament change, rendered by the printer's own template ----
            lift = max(W.z or 0, zn_max + 1.0)
            if retracted <= 0:
                W.emit(f"G1 E-{_f(retract_len[cur_slot], 5)} F{int(p.retract_speed * 60)}")
            W.move(z=lift, f=1200)
            tc_count += 1
            pv0 = toolchange.purge_volume(job, cur_slot, slot) * p.purge_factor
            txt, pv = toolchange.render_change(
                job, cur_slot, slot, max_layer_z=max(base_z, max_z_seen), layer_z=zn_max + off,
                layer_num=layer_num, pos=(W.x or 0, W.y or 0, W.z or 0), toolchange_count=tc_count,
                purge_vol=max(p.min_purge, pv0))
            tc_line = len(W.lines)
            W.emit("; FilamentChangeGCodeStart")
            for ln in txt.rstrip("\n").split("\n"):
                W.emit(ln)
            W.emit("; FilamentChangeGCodeEnd")
            W.moves.append((2, slot, W.x or 0, W.y or 0, W.z or 0, W.x or 0, W.y or 0, W.z or 0, tc_line))
            stats["toolchanges"] += 1
            stats["purge_mm3"] += pv
            fvs = job.float_at("filament_max_volumetric_speed", slot, 2.0) or 2.0
            stats["time_s"] += 50.0 + pv / fvs
            # position after the template: unknown XY near the purge chute, Z at max_layer_z+3
            W.x, W.y = -13.5, 90.0
            W.z = max(base_z, max_z_seen) + 3.0
            W.f = None
            retracted = job.float_at("retract_length_toolchange", slot, 2.0) + p.prime
            cur_slot = slot
            fan_cmd(slot)
        elif first:
            t = int(temps[slot]) if slot < len(temps) else 200
            W.emit(f"M104 S{t}")
            fan_cmd(slot)
        first = False

        fr = (ratios[slot] if (p.use_filament_flow_ratio and slot < len(ratios)) else 1.0) * p.flow
        fil_area = toolchange.filament_area(job, slot)
        vmax = mvs[slot] if slot < len(mvs) and mvs[slot] > 0 else 10.0
        nominal = toolchange.flow_mm3_per_mm(w, bead)
        v_print = min(p.speed, vmax / max(nominal, 1e-6))
        f_print = v_print * 60
        f_travel = p.travel_speed * 60
        rl = retract_len[slot]
        rf = int(p.retract_speed * 60)

        # ---- order strokes: greedy nearest neighbour with reversal / loop rotation ----
        remaining = list(range(len(strokes)))
        seq = []
        cx, cy = (W.x or 0.0), (W.y or 0.0)
        while remaining:
            best = None
            for k in remaining:
                s = strokes[k]
                if closed[k]:
                    d = np.hypot(s[:, 0] - cx, s[:, 1] - cy)
                    j = int(np.argmin(d))
                    cand = (float(d[j]), k, "rot", j)
                else:
                    d0 = math.hypot(s[0, 0] - cx, s[0, 1] - cy)
                    d1 = math.hypot(s[-1, 0] - cx, s[-1, 1] - cy)
                    cand = (d0, k, "fwd", 0) if d0 <= d1 else (d1, k, "rev", 0)
                if best is None or cand[0] < best[0]:
                    best = cand
            _, k, mode, j = best
            s = strokes[k]
            if mode == "rev":
                s = s[::-1]
            elif mode == "rot" and j:
                ring = s[:-1] if np.hypot(*(s[0] - s[-1])) < 1e-3 else s
                s = np.vstack([ring[j:], ring[:j], ring[j:j + 1]])
            seq.append(s)
            remaining.remove(k)
            cx, cy = s[-1]

        def emit_path(P, z, ds, gap, note="", dirz=None):
            """Travel to P[0] and extrude along it."""
            nonlocal retracted, max_z_seen
            if note:
                W.emit(f"; npg {note}")
            x0, y0, z0 = P[0, 0], P[0, 1], z[0]
            dist = math.hypot(x0 - (W.x or 0), y0 - (W.y or 0))
            if dist <= p.short_travel and retracted <= 0 and W.z is not None and abs(W.z - z0) < 1.0:
                W.move(x=x0, y=y0, z=max(z0, W.z) if dist > 0.05 else z0, f=f_travel, kind=1, slot=slot)
                if abs((W.z or z0) - z0) > 1e-4:
                    W.move(z=z0, f=f_print, kind=1, slot=slot)
                stats["time_s"] += dist / p.travel_speed
            else:
                if retracted <= 0:
                    W.emit(f"G1 E-{_f(rl, 5)} F{rf}")
                    retracted = rl
                    stats["time_s"] += rl / p.retract_speed
                tz = max(zn_line_max(W.x or x0, W.y or y0, x0, y0) + p.hop, z0 + p.hop)
                if W.z is None or tz > W.z:
                    W.move(z=tz, f=1200, kind=1, slot=slot)
                else:
                    tz = W.z
                W.move(x=x0, y=y0, f=f_travel, kind=1, slot=slot)
                W.move(z=z0, f=600, kind=1, slot=slot)
                stats["time_s"] += dist / p.travel_speed + 2 * abs(tz - z0) / 10.0
            if retracted > 0:
                W.emit(f"G1 E{_f(retracted, 5)} F{rf}")
                retracted = 0.0
                stats["time_s"] += 0.05
            W.f = None
            tot_e = 0.0
            for i in range(1, len(P)):
                hmid = 0.5 * (gap[i] + gap[i - 1])
                e = fr * toolchange.flow_mm3_per_mm(w, hmid) * ds[i] / fil_area
                if e <= 0:
                    continue
                W.move(x=P[i, 0], y=P[i, 1], z=z[i], e=e, f=f_print, kind=0, slot=slot)
                tot_e += e
            L = float(ds.sum())
            dz = np.diff(z if dirz is None else dirz)
            stats["dir_up_mm"] += float(ds[1:][dz > 1e-6].sum())
            stats["dir_down_mm"] += float(ds[1:][dz < -1e-6].sum())
            if len(dz) and float(dz.min()) < 0:
                stats["dir_down_max"] = max(stats["dir_down_max"], float(-dz.min()))
            stats["length_mm"][slot] = stats["length_mm"].get(slot, 0.0) + L
            stats["filament_mm"][slot] = stats["filament_mm"].get(slot, 0.0) + tot_e
            stats["volume_mm3"][slot] = stats["volume_mm3"].get(slot, 0.0) + tot_e * fil_area
            stats["strokes"][slot] = stats["strokes"].get(slot, 0) + 1
            stats["time_s"] += L / v_print
            stats["max_z"] = max(stats["max_z"], float(z.max()))
            stats["min_z"] = min(stats["min_z"], float(z.min()))
            max_z_seen = max(max_z_seen, float(z.max()))
            if p.stack:
                _stamp(HNP, g, P, z, w * 0.4)

        for s in seq:
            dense = _resample(s, p.step)
            zs = surface.sample_max(g.ZN, g, dense[:, 0], dense[:, 1])
            ok = np.isfinite(zs)
            # split at holes
            pieces = []
            start = None
            for i, o in enumerate(ok):
                if o and start is None:
                    start = i
                if (not o or i == len(ok) - 1) and start is not None:
                    end = i + 1 if o else i
                    if end - start >= 2:
                        pieces.append((start, end))
                    start = None
            for (a, b) in pieces:
                P = dense[a:b]
                if _length(P) < p.min_len:
                    continue
                z = zs[a:b] + off
                if p.stack:
                    hn = surface.sample(HNP, g, P[:, 0], P[:, 1], mode="nearest")
                    hn = np.where(np.isfinite(hn), hn, -np.inf)
                    z = np.maximum(z, hn + off)   # lay it onto the element below
                else:
                    hn = np.full(len(P), -np.inf)
                ds = np.r_[0.0, np.hypot(*np.diff(P, axis=0).T)]
                dz0 = np.diff(z)
                stats["dir_before_mm"] += float(ds[1:][dz0 < -1e-6].sum())

                # ---- how much of this element sits on material below ----
                P0, z0, hn0, ds0 = P, z, hn, ds
                on_top = z0 > (zs[a:b] + off + 0.02)     # sits on an element printed before
                if on_top.any():
                    stats["stacked_mm"] += float(ds0[1:][on_top[1:]].sum())
                for (ra, rb) in [(0, len(P0))]:
                    P, z, hn, ds = P0[ra:rb], z0[ra:rb], hn0[ra:rb], ds0[ra:rb]

                    # ---- uphill first: split the stroke where the surface turns ----
                    if p.dir_opt:
                        secs = direction.order(
                            direction.plan(P, z, min_rise=p.dir_min_rise, min_len=p.dir_end_len,
                                           max_parts=p.dir_cap),
                            P, (W.x or 0.0, W.y or 0.0))
                    else:
                        secs = [(0, len(P) - 1, False)]

                    prev_end = None
                    for k, (i0, i1, rev) in enumerate(secs):
                        sl = slice(i0, i1 + 1)
                        if rev:
                            Pp, zp, dsp, hp = P[sl][::-1], z[sl][::-1], ds[sl][::-1], hn[sl][::-1]
                        else:
                            Pp, zp, dsp, hp = P[sl], z[sl], ds[sl], hn[sl]
                        nk = min(3, max(1, len(zp) // 4))
                        rise, slen = float(zp[-nk:].mean() - zp[:nk].mean()), float(dsp.sum())
                        z_dir = zp.copy()                      # surface height for the uphill metric
                        if p.ramp > 0:
                            for i in range(1, len(zp)):
                                zp[i] = max(zp[i], zp[i - 1] - p.ramp * dsp[i])
                            for i in range(len(zp) - 2, -1, -1):
                                zp[i] = max(zp[i], zp[i + 1] - p.ramp * dsp[i + 1])
                        if p.gap_flow:
                            ha = surface.sample(HA, g, Pp[:, 0], Pp[:, 1])
                            ha = np.maximum(np.where(np.isfinite(ha), ha, -np.inf), hp)
                            gap = np.where(np.isfinite(ha), zp - ha, bead)
                            gap = np.clip(gap, 0.5 * bead, 2.0 * bead)
                        else:
                            gap = np.full(len(zp), bead)

                        note = ""
                        if p.dir_opt and len(secs) > 1:
                            way = "uphill" if rise > 1e-6 else ("downhill" if rise < -1e-6 else "flat")
                            note = f"section {k + 1}/{len(secs)}: {way} {rise:+.2f} mm over {slen:.1f} mm"
                            if prev_end is not None:
                                note += (f" - cut at X{prev_end[0]:.2f} Y{prev_end[1]:.2f} "
                                         f"Z{prev_end[2]:.2f}")
                        emit_path(Pp, zp, dsp, gap, note, dirz=z_dir)
                        if k < len(secs) - 1:
                            stats["cuts"].append({"x": round(float(Pp[-1, 0]), 2), "y": round(float(Pp[-1, 1]), 2),
                                                  "z": round(float(zp[-1]), 2), "rise": round(rise, 3),
                                                  "len": round(slen, 2)})
                        prev_end = (float(Pp[-1, 0]), float(Pp[-1, 1]), float(zp[-1]))
        # end of colour
    if retracted <= 0 and W.x is not None:
        W.emit(f"G1 E-{_f(retract_len[cur_slot], 5)} F{int(p.retract_speed * 60)}")
    end_z = max(W.z or 0, max_z_seen) + 1.0
    W.move(z=end_z, f=1200)
    W.emit(f"M204 S{int(job.float_at('default_acceleration', 0, 6000))}")
    W.emit("; ===== NON-PLANAR LAYER END =====")

    if stats["min_z"] > 1e8:
        stats["min_z"] = 0.0
    stats["dir"] = {"cuts": len(stats["cuts"]),
                    "up_mm": round(stats["dir_up_mm"], 2),
                    "down_mm": round(stats["dir_down_mm"], 2),
                    "before_mm": round(stats["dir_before_mm"], 2),
                    "down_max": round(stats["dir_down_max"], 3)}
    stats["end_z"] = end_z
    used_slots = sorted({s for (_, s) in groups})
    stats["colors"] = used_slots
    stats["speed_mm_s"] = {s: round(min(p.speed, (mvs[s] if s < len(mvs) and mvs[s] > 0 else 10) /
                                     max(toolchange.flow_mm3_per_mm(w, bead), 1e-6)), 2) for s in used_slots}
    mv = np.asarray(W.moves, dtype=np.float32).reshape(-1, 9)
    return Result(gcode="\n".join(W.lines) + "\n", moves=mv, stats=stats, max_z=end_z, end_slot=cur_slot)


def _stamp(HNP: np.ndarray, g: Grid, P: np.ndarray, z: np.ndarray, radius: float):
    dense = _resample(P, g.res * 0.7)
    # interpolate z onto dense points by arc length
    s0 = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s1 = np.r_[0.0, np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]
    zz = np.interp(s1 * (s0[-1] / max(s1[-1], 1e-9)), s0, z)
    cx = np.rint((dense[:, 0] - g.x0) / g.res).astype(np.int64)
    cy = np.rint((dense[:, 1] - g.y0) / g.res).astype(np.int64)
    r = max(0, int(round(radius / g.res)))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > (r + 0.25) ** 2:
                continue
            qx = cx + dx
            qy = cy + dy
            ok = (qx >= 0) & (qx < g.nx) & (qy >= 0) & (qy < g.ny)
            np.maximum.at(HNP, (qy[ok], qx[ok]), zz[ok].astype(np.float32))
