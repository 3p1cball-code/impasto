// Cell pattern (jittered Voronoi on a hex lattice) and the seeded randomness shared by the
// random colours. Everything is deterministic: the same seed gives the same cells and colours.
import { DEG } from "./geom.js";

// ------------------------------------------------------------------ seeded noise
// value in [0, 1) for (seed, a, b) – a pure hash, so a cell keeps its colour/shape when the
// rest of the pattern changes
export function hash01(seed, a, b = 0) {
  let h = (Math.imul(seed | 0, 0x9e3779b1) ^ Math.imul(a | 0, 0x85ebca6b) ^ Math.imul(b | 0, 0xc2b2ae35)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15; h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// ------------------------------------------------------------------ convex polygon tools
// keep the part of a CCW polygon with (x - m)·n <= -d   (n unit, pointing away from the site)
function clipHalf(poly, mx, my, nx, ny, d) {
  const out = [];
  const f = (p) => (p[0] - mx) * nx + (p[1] - my) * ny + d;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const fa = f(a), fb = f(b);
    if (fa <= 0) out.push(a);
    if ((fa <= 0) !== (fb <= 0)) {
      const t = fa / (fa - fb);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2;
}

// Voronoi cell of site s against its neighbours, every border pushed inwards by d
function core(s, nbrs, d, R0) {
  let poly = [[s[0] - R0, s[1] - R0], [s[0] + R0, s[1] - R0], [s[0] + R0, s[1] + R0], [s[0] - R0, s[1] + R0]];
  for (const q of nbrs) {
    const dx = q[0] - s[0], dy = q[1] - s[1], L = Math.hypot(dx, dy);
    if (L < 1e-9) continue;
    poly = clipHalf(poly, (s[0] + q[0]) / 2, (s[1] + q[1]) / 2, dx / L, dy / L, d);
    if (poly.length < 3) return null;
  }
  return area(poly) > 1e-6 ? poly : null;
}

// Minkowski sum of a convex CCW polygon with a disc of radius r: rounded corners
function rounded(poly, r, arcStep = 0.12) {
  if (!poly) return null;
  if (r <= 1e-6) return poly.map(p => p.slice());
  const n = poly.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = poly[(i - 1 + n) % n], p = poly[i], p1 = poly[(i + 1) % n];
    // outward normals of the incoming and outgoing edge (CCW: right of the edge)
    const a0 = Math.atan2(-(p[0] - p0[0]), p[1] - p0[1]);
    let a1 = Math.atan2(-(p1[0] - p[0]), p1[1] - p[1]);
    while (a1 < a0) a1 += 2 * Math.PI;
    const m = Math.max(1, Math.ceil(((a1 - a0) * r) / arcStep));
    for (let k = 0; k <= m; k++) {
      const a = a0 + ((a1 - a0) * k) / m;
      out.push([p[0] + r * Math.cos(a), p[1] + r * Math.sin(a)]);
    }
  }
  return out;
}

// the outline of the cell inset by t: core(d0 + max(0, t - r)) ⊕ disc(max(0, r - t))
function shape(s, nbrs, d0, r, t, R0) {
  const c = core(s, nbrs, d0 + Math.max(0, t - r), R0);
  return c ? rounded(c, Math.max(0, r - t)) : null;
}

function perimeter(poly) {
  let L = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    L += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return L;
}

// rings joined into one path: every ring starts next to where the previous one started
function spiral(rings) {
  const out = [];
  for (const ring of rings) {
    let k = 0;
    if (out.length) {
      const [x, y] = out[out.length - 1];
      let bd = Infinity;
      ring.forEach((p, i) => { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; k = i; } });
    }
    for (let i = 0; i <= ring.length; i++) out.push(ring[(k + i) % ring.length]);
  }
  return out;
}

// serpentine hatch inside a convex polygon
function hatchPoly(poly, sp, angle) {
  const c = Math.cos(angle * DEG), s = Math.sin(angle * DEG);
  const loc = poly.map(([x, y]) => [x * c + y * s, -x * s + y * c]);
  let y0 = Infinity, y1 = -Infinity;
  for (const p of loc) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  const out = [];
  let flip = false;
  for (let y = y0 + sp / 2; y < y1 - sp * 0.25; y += sp) {
    const xs = [];
    for (let i = 0; i < loc.length; i++) {
      const a = loc[i], b = loc[(i + 1) % loc.length];
      if ((a[1] <= y) !== (b[1] <= y)) xs.push(a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1]));
    }
    if (xs.length < 2) continue;
    let xa = Math.min(...xs), xb = Math.max(...xs);
    if (xb - xa < sp * 0.3) continue;
    if (flip) [xa, xb] = [xb, xa];
    flip = !flip;
    out.push([xa, y], [xb, y]);
  }
  return out.map(([u, v]) => [u * c - v * s, u * s + v * c]);
}

// ------------------------------------------------------------------ cell pattern
// p.spacing  cell size (lattice pitch)          p.jitter   0 = honeycomb, 1 = organic
// p.gap      gap between neighbouring cells     p.round    corner radius (mm)
// p.fill     "none" | "rings" | "hatch"         p.fillSpacing / p.fillAngle
// p.seed     shape seed
// Returns local polylines; all lines of one cell share .grp (the colour unit for the
// random colours and the unit that "whole cells only" keeps or drops).
export const MAX_CELLS = 12000;

export function cellsLocal(p, R) {
  const a = Math.max(0.6, +p.spacing || 3);
  const jit = Math.max(0, Math.min(1, +(p.jitter ?? 0.7)));
  const gap = Math.max(0, +p.gap || 0);
  const r = Math.max(0, +p.round || 0);
  const seed = (p.seed | 0) || 1;
  const fs = Math.max(0.15, +p.fillSpacing || 0.45);
  const fill = p.fill || "none";
  const rowH = a * Math.sqrt(3) / 2;
  const nj = Math.ceil(R / rowH) + 2, ni = Math.ceil(R / a) + 2;
  const out = [];
  const est = (2 * ni + 1) * (2 * nj + 1);
  if (est > MAX_CELLS * 1.6) return { lines: out, cells: 0, tooMany: est };
  const site = (i, j) => {
    const x = (i + (j & 1 ? 0.5 : 0)) * a, y = j * rowH;
    const ang = 2 * Math.PI * hash01(seed, i * 7919 + 13, j * 104729 + 7);
    const rad = jit * 0.42 * a * Math.sqrt(hash01(seed, i * 31 + 3, j * 977 + 11));
    return [x + rad * Math.cos(ang), y + rad * Math.sin(ang)];
  };
  let cells = 0;
  for (let j = -nj; j <= nj; j++) for (let i = -ni; i <= ni; i++) {
    const s = site(i, j);
    if (Math.hypot(s[0], s[1]) > R + a) continue;
    const nbrs = [];
    for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) if (di || dj) nbrs.push(site(i + di, j + dj));
    // the rounding a cell can take: never more than fits (small cells become round pebbles,
    // they do not vanish)
    let rr = r;
    while (rr > 0.03 && !core(s, nbrs, gap / 2 + rr, a * 2)) rr *= 0.6;
    if (rr <= 0.03) rr = 0;
    const d0 = gap / 2 + rr;
    const outer = shape(s, nbrs, d0, rr, 0, a * 2);
    if (!outer || perimeter(outer) < 0.8) continue;
    const grp = (i + 4096) * 8192 + (j + 4096);
    cells++;
    if (fill === "rings") {
      const rings = [outer];
      for (let k = 1; k < 200; k++) {
        const ring = shape(s, nbrs, d0, rr, k * fs, a * 2);
        if (!ring || perimeter(ring) < fs * 2.5) break;
        rings.push(ring);
      }
      const l = spiral(rings);
      l.grp = grp;
      out.push(l);
    } else {
      const l = outer.slice();
      l.closed = true; l.grp = grp;
      out.push(l);
      if (fill === "hatch") {
        const inner = shape(s, nbrs, d0, rr, fs, a * 2);
        if (inner) {
          const h = hatchPoly(inner, fs, +p.fillAngle || 0);
          if (h.length >= 2) { h.grp = grp; out.push(h); }
        }
      }
    }
  }
  return { lines: out, cells };
}

// ------------------------------------------------------------------ random colours
// lines: [{slot, pts, closed, grp?, src, piece}] (after clipping to the surface)
// o.mode   "off" | "group" (cell / stroke / object) | "piece" (every piece on the surface)
//          | "dash" (sections of about o.dash mm along every line, neighbours never alike)
// o.share  0..1 – share of the units that get a random colour, the rest keep their own
// o.slots  filament slots to draw from;  o.seed  integer
// The printer still changes colour as rarely as possible: the generator batches all sections
// of a colour (within one element nothing forces an order).
export function randomColours(lines, o) {
  const mode = o?.mode || "off";
  const slots = (o?.slots || []).filter(s => s >= 0);
  if (mode === "off" || !slots.length) return lines;
  const share = Math.max(0, Math.min(1, o.share ?? 1));
  const seed = (o.seed | 0) || 1;
  const pick = (id, base) => {
    if (hash01(seed, id, 1) >= share) return base;
    return slots[Math.min(slots.length - 1, Math.floor(hash01(seed, id, 2) * slots.length))];
  };
  if (mode === "group" || mode === "piece") {
    return lines.map(l => {
      const id = mode === "group" ? (l.grp ?? l.src) : (l.src * 1024 + (l.piece | 0));
      return { ...l, slot: pick(id, l.slot) };
    });
  }
  // dash: walk along every line, cut it every dash * (0.5..1.5) mm
  const dash = Math.max(0.5, +o.dash || 6);
  const out = [];
  for (const l of lines) {
    const P = l.closed ? l.pts.concat([l.pts[0]]) : l.pts;
    if (P.length < 2) continue;
    const id0 = (l.src * 1024 + (l.piece | 0)) * 4096;
    let n = 0, prev = null, cur = [P[0]];
    let want = dash * (0.5 + hash01(seed, id0, 3));
    let acc = 0;
    let slot = pick(id0, l.slot);
    const flush = () => {
      if (cur.length < 2) return;
      if (prev && prev.slot === slot) { prev.pts.push(...cur.slice(1)); return; }
      prev = { ...l, closed: false, slot, pts: cur };
      out.push(prev);
    };
    for (let i = 1; i < P.length; i++) {
      let [ax, ay] = P[i - 1];
      const [bx, by] = P[i];
      let seg = Math.hypot(bx - ax, by - ay);
      while (acc + seg >= want && seg > 1e-9) {
        const t = (want - acc) / seg;
        const cx = ax + (bx - ax) * t, cy = ay + (by - ay) * t;
        cur.push([cx, cy]);
        flush();
        n++;
        // next colour, never the same as the one before (when there is a choice)
        let ns = pick(id0 + n, l.slot);
        if (ns === slot && slots.length > 1 && share >= 1) ns = slots[(slots.indexOf(ns) + 1 + Math.floor(hash01(seed, id0 + n, 4) * (slots.length - 1))) % slots.length];
        slot = ns;
        cur = [[cx, cy]];
        seg -= Math.hypot(cx - ax, cy - ay);
        ax = cx; ay = cy; acc = 0;
        want = dash * (0.5 + hash01(seed, id0 + n, 3));
      }
      acc += seg;
      cur.push([bx, by]);
    }
    flush();
  }
  return out;
}
