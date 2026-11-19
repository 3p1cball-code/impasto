// Design elements: outline, pattern, freehand drawing, image, text, svg.
// Each element produces polylines in bed coordinates (mm) with a filament slot.
import { applyTf, simplify, chaikin, clipBy, hatch, circle, hexToRgb, rgbToHex, polyLength, DEG } from "./geom.js";
import { marchingSquares, isoLevels } from "./grid.js";
import { t } from "./i18n.js";

export const TYPES = {
  outline: { label: "type.outline", tf: false,
    defaults: { source: "mask", offset: 0.25, count: 1, spacing: 0.5, minLen: 3 } },
  contours: { label: "type.contours", tf: true,
    defaults: { mode: "iso", dz: 0.5, startZ: 0, minLen: 2, maxLines: 400,
                flowSpacing: 3, flowLen: 40, flowMinSlope: 2, flowStep: 0.3, margin: 0 } },
  pattern: { label: "type.pattern", tf: true,
    defaults: { kind: "wave", spacing: 2.0, amp: 0.6, wavelength: 4.0, frame: "surface", w: 40, h: 20, border: false, margin: 0 } },
  draw: { label: "type.draw", tf: true, defaults: { strokes: [], smooth: 2, margin: 0 } },
  image: { label: "type.image", tf: true,
    defaults: { src: null, name: "", w: 30, colors: 2, map: [], fill: true, contour: true, spacing: 0.45, angle: 45, smooth: 1, despeckle: true, margin: 0,
                frame: "rect", h: 0, crop: [0, 0, 1, 1], tile: 0 } },
  text: { label: "type.text", tf: true,
    defaults: { text: "Text", font: "sans-serif", weight: "700", italic: false, size: 6, fill: true, contour: true, spacing: 0.45, angle: 45, map: [], margin: 0,
                frame: "rect", w: 40, h: 20, crop: [0, 0, 1, 1], tile: 0 } },
  svg: { label: "type.svg", tf: true,
    defaults: { src: null, name: "", w: 30, stroke: true, fill: false, spacing: 0.45, angle: 45, margin: 0 } },
};

export const PATTERNS = {
  lines: "pattern.lines", grid: "pattern.grid", wave: "pattern.wave", zigzag: "pattern.zigzag",
  curls: "pattern.curls", circles: "pattern.circles", spiral: "pattern.spiral",
  dots: "pattern.dots", diamonds: "pattern.diamonds",
};

let uid = Date.now() % 100000;
export function newElement(type, at, slot) {
  const T = TYPES[type];
  const el = {
    id: "e" + (uid++).toString(36), type, name: t(T.label), slot, visible: true, layer: 0,
    tf: { x: at[0], y: at[1], rot: 0, s: 1 },
    props: structuredClone(T.defaults),
  };
  // height lines are computed in bed coordinates; remember where they were laid down so the
  // transform stays the identity at that point (props.ox/oy)
  if (type === "contours") { el.props.ox = at[0]; el.props.oy = at[1]; }
  return el;
}

// ------------------------------------------------------------------ caches
const rasters = new Map();       // key -> raster | "loading"
const isoCache = new Map();
let onAsync = () => {};
export function setAsyncCallback(fn) { onAsync = fn; }

function key(obj) { return JSON.stringify(obj); }

// ------------------------------------------------------------------ main entry
// returns { lines:[{slot, pts, closed}], box:[x0,y0,x1,y1] (local frame box), pending, info }
export function build(el, surf, ctx) {
  try {
    switch (el.type) {
      case "outline": return buildOutline(el, surf);
      case "contours": return buildContours(el, surf);
      case "pattern": return buildPattern(el, surf);
      case "draw": return buildDraw(el);
      case "image": return buildRaster(el, ctx, rasterImage);
      case "text": return buildRaster(el, ctx, rasterText);
      case "svg": return buildSvg(el, ctx);
    }
  } catch (e) {
    console.error(e);
    return { lines: [], box: null, info: t("msg.error", { msg: e.message }) };
  }
  return { lines: [], box: null };
}

// ------------------------------------------------------------------ outline
function buildOutline(el, surf) {
  const p = el.props;
  const out = [];
  const src = p.source === "object" ? "object" : "mask";
  const field = src === "object" ? surf.footDist() : surf.dist;
  const base = src === "object" ? 0 : surf.inset;
  for (let k = 0; k < Math.max(1, p.count | 0); k++) {
    const level = base + p.offset + k * p.spacing;
    const ck = `${src}|${src === "object" ? 0 : surf.version}|${level.toFixed(4)}`;
    let iso = isoCache.get(ck);
    if (!iso) {
      iso = surf.isolines(field, level).map(l => ({ closed: l.closed, pts: simplify(l.pts, 0.01) }));
      isoCache.set(ck, iso);
      if (isoCache.size > 80) isoCache.delete(isoCache.keys().next().value);
    }
    for (const l of iso) if (polyLength(l.pts) >= (p.minLen || 0)) out.push({ slot: el.slot, pts: l.pts, closed: l.closed });
  }
  return { lines: out, box: null };
}

// ------------------------------------------------------------------ height lines
// Height lines (iso Z) and slope lines (along the gradient, i.e. 90° to the height lines).
// The geometry does not depend on the element transform, so it is cached on the surface and
// only re-transformed when the element is moved – dragging stays smooth.
function surfCache(surf, key, make) {
  surf._el ??= new Map();
  const hit = surf._el.get(key);
  if (hit) return hit;
  const v = make();
  surf._el.set(key, v);
  if (surf._el.size > 32) surf._el.delete(surf._el.keys().next().value);
  return v;
}

function contourWorld(surf, startZ, step) {
  return surfCache(surf, `iso|${startZ.toFixed(3)}|${step.toFixed(4)}`, () => {
    const cells = surf.nx * surf.ny;
    const stride = cells > 2_000_000 ? 3 : cells > 700_000 ? 2 : 1;
    const out = [];
    for (const [, lines] of isoLevels(surf.HS, surf.nx, surf.ny, startZ, step, stride)) {
      for (const l of lines) {
        if (l.pts.length < 2) continue;
        const pts = simplify(l.pts.map(([i, j]) => [surf.x0 + i * surf.res, surf.y0 + j * surf.res]), 0.01);
        out.push({ pts, closed: l.closed });
      }
    }
    return out;
  });
}

// Evenly spaced slope lines (Jobard & Lefer, "Creating evenly-spaced streamlines", 1997):
// every line is traced as far as it goes – down to where the hill flattens or the usable
// surface ends, then up with what is left of the length budget – and stops when it comes
// closer than half the spacing to another line (real distances, not a coarse raster).
// New lines start one spacing beside the existing ones, so the gaps fill up evenly; a
// lattice of seeds picks up the regions no line has reached yet.
//   spacing  distance between neighbouring lines (new lines start at 0.9 x spacing)
//   maxLen   longest line, measured from its lower end
//   mask     usable surface incl. the element's margin – lines end on it, not clipped later
export function flowWorld(surf, mask, maskKey, spacing, maxLen, minSlopeDeg, stepLen, minLen = 0) {
  const key = `flow|${surf.version}|${maskKey}|${spacing.toFixed(2)}|${maxLen.toFixed(1)}|${minSlopeDeg.toFixed(2)}|${stepLen.toFixed(2)}|${(+minLen).toFixed(2)}`;
  return surfCache(surf, key, () => {
    const out = [];
    const b = surf.maskBox;
    if (!b) return out;
    const inside = (x, y) => surf.inMaskOf(mask, x, y);
    const minSlope = Math.max(Math.tan(Math.max(0, minSlopeDeg) * DEG), 1e-4);
    const dTest = spacing * 0.5;        // a growing line stops this close to another one
    const dSeed = spacing * 0.9;        // a new line needs this much room to start
    const keepLen = Math.max(+minLen || 0, stepLen * 3, Math.min(1.0, spacing * 0.5));
    // registered points of the accepted lines, hashed in cells of dSeed
    const cs = dSeed;
    const cells = new Map();
    const ck = (ix, iy) => ix * 65536 + iy;
    const addPts = (pts) => {
      for (const p of pts) {
        const k = ck(Math.floor((p[0] - surf.x0) / cs), Math.floor((p[1] - surf.y0) / cs));
        const l = cells.get(k);
        if (l) l.push(p[0], p[1]); else cells.set(k, [p[0], p[1]]);
      }
    };
    const near = (x, y, d) => {          // is any registered point closer than d?
      const ix = Math.floor((x - surf.x0) / cs), iy = Math.floor((y - surf.y0) / cs);
      const d2 = d * d;
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
        const l = cells.get(ck(ix + i, iy + j));
        if (!l) continue;
        for (let q = 0; q < l.length; q += 2) {
          const dx = l[q] - x, dy = l[q + 1] - y;
          if (dx * dx + dy * dy < d2) return true;
        }
      }
      return false;
    };
    const steep = (x, y) => {
      const g = surf.gradAt(x, y);
      return !!g && Math.hypot(g[0], g[1]) >= minSlope;
    };
    // one line through (sx, sy): down first, then up with the rest of the budget
    const trace = (sx, sy) => {
      const dn = traceSlope(surf, sx, sy, -1, stepLen, maxLen, minSlope, inside, near, dTest);
      const Ldn = dn.length * stepLen;
      const up = traceSlope(surf, sx, sy, 1, stepLen, Math.max(0, maxLen - Ldn), minSlope, inside, near, dTest);
      const pts = dn.reverse().concat([[sx, sy]], up);
      if (pts.length < 2 || (pts.length - 1) * stepLen < keepLen - 1e-9) return null;
      return pts;
    };
    const queue = [];
    const accept = (pts) => {
      addPts(pts);
      const sm = simplify(pts, 0.02);
      out.push({ pts: sm, closed: false });
      queue.push(pts);
    };
    const trySeed = (x, y) => {
      if (!inside(x, y) || !steep(x, y) || near(x, y, dSeed)) return false;
      const pts = trace(x, y);
      if (!pts) return false;
      accept(pts);
      return true;
    };
    // neighbours one spacing to the left and right of a line, every half spacing along it
    const grow = () => {
      while (queue.length && out.length < 20000) {
        const pts = queue.shift();
        const every = Math.max(1, Math.round((spacing * 0.5) / stepLen));
        for (let i = 0; i < pts.length; i += every) {
          const a = pts[Math.max(0, i - 1)], c = pts[Math.min(pts.length - 1, i + 1)];
          const tx = c[0] - a[0], ty = c[1] - a[1], tl = Math.hypot(tx, ty);
          if (tl < 1e-9) continue;
          const nx = -ty / tl, ny = tx / tl;
          for (const s of [1, -1]) trySeed(pts[i][0] + s * nx * spacing, pts[i][1] + s * ny * spacing);
        }
      }
    };
    // seed lattice (hexagonal, half a spacing), lowest points first: the first lines start at
    // the foot of the hill and run the whole way up
    const lat = [];
    const h = spacing * 0.5;
    let row = 0;
    for (let y = b[1] + h * 0.5; y <= b[3]; y += h * 0.87, row++) {
      for (let x = b[0] + h * 0.5 + (row % 2) * h * 0.5; x <= b[2]; x += h) {
        if (!inside(x, y)) continue;
        const z = surf.hsAt(x, y);
        if (z === z) lat.push([z, x, y]);
      }
    }
    lat.sort((p, q) => p[0] - q[0]);
    for (const [, x, y] of lat) {
      if (out.length >= 20000) break;
      if (trySeed(x, y)) grow();
    }
    return out;
  });
}

// steepest ascent (sgn = 1) or descent (sgn = -1), midpoint steps; stops where the hill gets
// flatter than minSlope, at the edge of the usable surface, where the direction turns back
// (ridge / summit), and dTest before another line
function traceSlope(surf, x0, y0, sgn, stepLen, maxLen, minSlope, inside, near, dTest) {
  const pts = [];
  let x = x0, y = y0, L = 0, px = 0, py = 0;
  for (let n = 0; n < 20000 && L + stepLen <= maxLen + 1e-9; n++) {
    const g = surf.gradAt(x, y);
    if (!g) break;
    const m = Math.hypot(g[0], g[1]);
    if (!(m >= minSlope)) break;
    const ux = (sgn * g[0]) / m, uy = (sgn * g[1]) / m;
    const g2 = surf.gradAt(x + ux * stepLen * 0.5, y + uy * stepLen * 0.5);
    let vx = ux, vy = uy;
    if (g2) {
      const m2 = Math.hypot(g2[0], g2[1]);
      if (m2 >= minSlope) { vx = (sgn * g2[0]) / m2; vy = (sgn * g2[1]) / m2; }
    }
    if (n > 0 && vx * px + vy * py < 0.5) break;      // turns by more than 60°: over the top
    const nx = x + vx * stepLen, ny = y + vy * stepLen;
    if (!inside(nx, ny) || near(nx, ny, dTest)) break;
    pts.push([nx, ny]);
    x = nx; y = ny; px = vx; py = vy; L += stepLen;
  }
  return pts;
}

// The contour geometry already lives in bed coordinates, so the element transform is applied
// relative to the point where the element was created (props.ox/oy) – at that point it is the
// identity, dragging shifts the lines, rotate/scale turn them around that anchor.
function contourMap(el) {
  const p = el.props, tf = el.tf, k = tf.s ?? 1, rot = tf.rot || 0;
  const ox = p.ox ?? tf.x, oy = p.oy ?? tf.y;
  const dx = tf.x - ox, dy = tf.y - oy;
  if (!dx && !dy && !rot && k === 1) return null;
  const c = Math.cos(rot * DEG), s = Math.sin(rot * DEG);
  return (pts) => pts.map(([x, y]) => {
    const u = x - ox, v = y - oy;
    return [ox + dx + k * (c * u - s * v), oy + dy + k * (s * u + c * v)];
  });
}

function buildContours(el, surf) {
  const p = el.props;
  const mode = p.mode || "iso";
  const map = contourMap(el);
  const tf = (pts) => (map ? map(pts) : pts);
  const out = [];
  const infos = [];
  if (mode !== "flow") {
    const [z0, z1] = surf.hsRange();
    const dz = Math.max(0.02, +p.dz || 0.5);
    const start = +p.startZ || 0;
    let n = Math.floor((z1 - start) / dz + 1e-9) - Math.ceil((z0 - start) / dz - 1e-9) + 1;
    n = Math.max(0, n);
    const maxLines = Math.max(2, Math.min(4000, +p.maxLines || 400));
    let eff = dz, skip = 1;
    if (n > maxLines) { skip = Math.ceil(n / maxLines); eff = dz * skip; }
    if (n > 0) {
      const before = out.length;
      for (const l of contourWorld(surf, start, eff)) {
        if (polyLength(l.pts) < (+p.minLen || 0)) continue;
        out.push({ slot: el.slot, pts: tf(l.pts), closed: l.closed });
      }
      infos.push(t("contours.iso", { n: out.length - before, dz: eff.toFixed(2), z0: z0.toFixed(1), z1: z1.toFixed(1) })
        + (skip > 1 ? t("contours.thin", { n: skip }) : ""));
    }
  }
  if (mode !== "iso") {
    const sp = Math.max(0.8, +p.flowSpacing || 3);
    const margin = Math.max(0, +p.margin || 0);
    const mask = margin > 0 ? surf.maskMargin(margin) : surf.mask;
    const lines = flowWorld(surf, mask, Math.round(margin * 100), sp, Math.max(3, +p.flowLen || 40),
      Math.max(0, +p.flowMinSlope || 0), Math.max(0.1, Math.min(1, +p.flowStep || 0.3)), +p.minLen || 0);
    let n = 0;
    for (const l of lines) {
      if (polyLength(l.pts) < (+p.minLen || 0)) continue;
      out.push({ slot: el.slot, pts: tf(l.pts), closed: false });
      n++;
    }
    infos.push(t("contours.flow", { n, sp: sp.toFixed(1) }));
  }
  return { lines: out, box: null, info: infos.join(" · ") };
}

// ------------------------------------------------------------------ pattern
function buildPattern(el, surf) {
  const p = el.props;
  const tf = el.tf;
  const s = tf.s || 1;
  let R;
  if (p.frame === "surface") {
    const b = surf.maskBox;
    R = 0;
    for (const [x, y] of [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]]) R = Math.max(R, Math.hypot(x - tf.x, y - tf.y));
    R = R / s + p.spacing;
  } else {
    R = Math.hypot(p.w, p.h) / 2 + p.spacing;
  }
  R = Math.min(R, 400);
  let lines = patternLocal(p, R);
  const step = 0.05 / s;
  if (p.frame === "rect") {
    const hw = p.w / 2, hh = p.h / 2;
    lines = lines.flatMap(l => clipBy(l, (x, y) => Math.abs(x) <= hw && Math.abs(y) <= hh, step));
    if (p.border) lines.push(Object.assign([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]], { closed: true }));
  } else if (p.frame === "ellipse") {
    const a = p.w / 2, b = p.h / 2;
    lines = lines.flatMap(l => clipBy(l, (x, y) => (x * x) / (a * a) + (y * y) / (b * b) <= 1, step));
    if (p.border) {
      const e = [];
      for (let i = 0; i <= 160; i++) { const t = (i / 160) * 2 * Math.PI; e.push([a * Math.cos(t), b * Math.sin(t)]); }
      lines.push(Object.assign(e, { closed: true }));
    }
  }
  const box = p.frame === "surface" ? null : [-p.w / 2, -p.h / 2, p.w / 2, p.h / 2];
  return { lines: lines.map(l => ({ slot: el.slot, pts: applyTf(l, tf), closed: !!l.closed })), box };
}

function patternLocal(p, R) {
  const sp = Math.max(0.2, p.spacing), A = p.amp, lam = Math.max(0.3, p.wavelength);
  const out = [];
  const rows = (fn) => {
    for (let y = -Math.floor(R / sp) * sp, k = 0; y <= R; y += sp, k++) out.push(fn(y, k));
  };
  const curve = (f, y, t0 = -R, t1 = R, st = 0.15) => {
    const l = [];
    for (let x = t0; x <= t1 + 1e-9; x += st) l.push(f(x, y));
    return l;
  };
  switch (p.kind) {
    case "lines": rows((y) => [[-R, y], [R, y]]); break;
    case "grid":
      rows((y) => [[-R, y], [R, y]]);
      rows((y) => [[y, -R], [y, R]]);
      break;
    case "wave": rows((y) => curve((x, yy) => [x, yy + A * Math.sin((2 * Math.PI * x) / lam)], y, -R, R, Math.min(0.2, lam / 24))); break;
    case "zigzag": rows((y) => {
      const l = [];
      for (let x = -R, i = 0; x <= R + lam; x += lam / 2, i++) l.push([x, y + (i % 2 ? A : -A)]);
      return l;
    }); break;
    case "curls": rows((y, k) => {
      // prolate cycloid: loops when A > lam / (2 pi)
      const l = [];
      const c = lam / (2 * Math.PI);
      const tMax = R / c + 2 * Math.PI;
      const st = Math.min(0.12, 0.2 / Math.max(A, 0.2));
      for (let t = -tMax; t <= tMax; t += st) l.push([c * t - A * Math.sin(t), y - A * Math.cos(t)]);
      return l;
    }); break;
    case "circles":
      for (let r = sp; r <= R; r += sp) out.push(Object.assign(circle(0, 0, r), { closed: true }));
      break;
    case "spiral": {
      const l = [];
      const tMax = (2 * Math.PI * R) / sp;
      for (let t = 0; t <= tMax; ) {
        const r = (sp * t) / (2 * Math.PI);
        l.push([r * Math.cos(t), r * Math.sin(t)]);
        t += Math.min(0.3, 0.2 / Math.max(r, 0.3));
      }
      out.push(l);
      break;
    }
    case "dots": {
      const r = Math.max(0.3, A);
      for (let y = -Math.floor(R / sp) * sp, k = 0; y <= R; y += sp, k++)
        for (let x = -Math.floor(R / sp) * sp + (k % 2 ? sp / 2 : 0); x <= R; x += sp)
          if (x * x + y * y <= R * R) out.push(Object.assign(circle(x, y, r), { closed: true }));
      break;
    }
    case "diamonds": {
      const a = 45;
      for (const ang of [a, -a]) {
        const c = Math.cos(ang * DEG), s = Math.sin(ang * DEG);
        for (let o = -Math.floor(R / sp) * sp; o <= R; o += sp)
          out.push([[-R * c - o * s, -R * s + o * c], [R * c - o * s, R * s + o * c]]);
      }
      break;
    }
  }
  return out;
}

// ------------------------------------------------------------------ freehand
function buildDraw(el) {
  const out = [];
  let b = [Infinity, Infinity, -Infinity, -Infinity];
  for (const st of el.props.strokes) {
    if (st.length < 2) continue;
    const sm = chaikin(simplify(st, 0.03), el.props.smooth | 0);
    for (const [x, y] of sm) { b = [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)]; }
    out.push({ slot: el.slot, pts: applyTf(sm, el.tf), closed: false });
  }
  const box = isFinite(b[0]) ? [b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1] : [-5, -5, 5, 5];
  return { lines: out, box };
}

// ------------------------------------------------------------------ raster based (image, text)
const contourCache = new Map();

function clampInt(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// local (element) box of a world box: inverse of applyTf
function localBox(b, tf) {
  const s = tf.s || 1, a = -(tf.rot || 0) * DEG, c = Math.cos(a), si = Math.sin(a);
  const xs = [], ys = [];
  for (const [x, y] of [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]]) {
    const u = (x - tf.x) / s, v = (y - tf.y) / s;
    xs.push(u * c - v * si); ys.push(u * si + v * c);
  }
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

// marching-squares contours of one colour, cropped, in mm around the crop centre
function contoursOf(r, key, c, mpp, smooth, s) {
  let v = contourCache.get(key.id);
  if (v) return v;
  const [c0, j0, c1, j1] = key.crop;
  const Wc = c1 - c0, Hc = j1 - j0;
  const field = new Float32Array((Wc + 2) * (Hc + 2));
  for (let j = 0; j < Hc; j++) for (let i = 0; i < Wc; i++) {
    if (r.lab[(j + j0) * r.W + i + c0] === c) field[(j + 1) * (Wc + 2) + i + 1] = 1;
  }
  const out = [];
  for (const l of marchingSquares(field, Wc + 2, Hc + 2, 0.5)) {
    let pts = l.pts.map(([i, j]) => [(i - 1 + 0.5 - Wc / 2) * mpp, -(j - 1 + 0.5 - Hc / 2) * mpp]);
    pts = simplify(chaikin(pts, (smooth ?? 1) + 1, l.closed), 0.015 / s);
    if (polyLength(pts) * s < 0.8) continue;
    out.push({ pts, closed: l.closed });
  }
  contourCache.set(key.id, out);
  if (contourCache.size > 80) contourCache.delete(contourCache.keys().next().value);
  return out;
}

function buildRaster(el, ctx, loader) {
  const p = el.props;
  const rk = loader.key(el);
  if (!rk) return { lines: [], box: [-p.w / 2 || -10, -5, p.w / 2 || 10, 5], info: t("msg.noFile") };
  let r = rasters.get(rk);
  if (!r) {
    rasters.set(rk, "loading");
    loader.load(el).then(res => { rasters.set(rk, res); onAsync(el, res); }).catch(e => { rasters.set(rk, { error: e.message }); onAsync(el); });
    return { lines: [], box: null, pending: true };
  }
  if (r === "loading") return { lines: [], box: null, pending: true };
  if (r.error) return { lines: [], box: null, info: r.error };

  const isText = el.type === "text";
  const s = el.tf.s || 1;
  const newCrop = p.crop;

  // ---- source crop -------------------------------------------------------
  const cr = Array.isArray(newCrop) && newCrop.length === 4 ? newCrop : [0, 0, 1, 1];
  const c0 = clampInt(Math.round(Math.min(cr[0], cr[2]) * r.W), 0, r.W - 2);
  const c1 = clampInt(Math.round(Math.max(cr[0], cr[2]) * r.W), c0 + 2, r.W);
  const j0 = clampInt(Math.round(Math.min(cr[1], cr[3]) * r.H), 0, r.H - 2);
  const j1 = clampInt(Math.round(Math.max(cr[1], cr[3]) * r.H), j0 + 2, r.H);
  const Wc = c1 - c0, Hc = j1 - j0;

  // ---- size: motif (tile) and the area it repeats in ---------------------
  const pitch = Math.max(0, +p.tile || 0);
  const tiled = pitch > 0;
  const mpp = tiled ? pitch / Wc : (isText ? 1 / r.ppm : Math.max(0.01, p.w || 30) / Wc);
  const tw = tiled ? pitch : Wc * mpp;
  const th = tw * (Hc / Wc);
  const fw = tiled ? Math.max(p.w || 30, tw) : Wc * mpp;
  const fh = tiled ? Math.max(p.h || th, th) : (p.h > 0 ? p.h : th);

  // ---- area: element frame or the usable surface -------------------------
  let box;
  if ((p.frame || "rect") === "surface" && ctx.surf?.maskBox) {
    const b = localBox(ctx.surf.maskBox, el.tf);
    box = [b[0] - tw, b[1] - th, b[2] + tw, b[3] + th];
  } else {
    box = [-fw / 2, -fh / 2, fw / 2, fh / 2];
  }
  let i0 = 0, i1 = 1, jt0 = 0, jt1 = 1;
  if (tiled) {
    i0 = Math.floor(box[0] / tw); i1 = Math.ceil(box[2] / tw);
    jt0 = Math.floor(box[1] / th); jt1 = Math.ceil(box[3] / th);
    while ((i1 - i0) * (jt1 - jt0) > 600) {          // never let a tiny pitch hang the UI
      i0++; i1--; jt0++; jt1--;
      if (i1 <= i0 || jt1 <= jt0) break;
    }
  }
  const cxOf = (i) => (tiled ? (i + 0.5) * tw : 0);
  const cyOf = (j) => (tiled ? (j + 0.5) * th : 0);
  const ntiles = (i1 - i0) * (jt1 - jt0);

  const lines = [];
  const nC = r.colors.length;
  const map = isText ? [el.slot] : clusterMap(el, r, ctx);
  for (let c = 0; c < nC; c++) {
    const slot = map[c];
    if (slot === undefined || slot < 0) continue;
    if (p.contour) {
      const ckey = {
        id: `${rk}|${c0},${j0},${Wc},${Hc}|${c}|${mpp.toFixed(4)}|${(p.smooth ?? 1)}|${s.toFixed(3)}`,
        crop: [c0, j0, c1, j1],
      };
      const iso = contoursOf(r, ckey, c, mpp, p.smooth, s);
      for (const l of iso) {
        for (let ti = i0; ti < i1; ti++) for (let tj = jt0; tj < jt1; tj++) {
          const ox = cxOf(ti), oy = cyOf(tj);
          const pts = l.pts.map(([x, y]) => [ox + x, oy + y]);
          lines.push({ slot, pts: applyTf(pts, el.tf), closed: l.closed });
        }
      }
    }
    if (p.fill) {
      const sp = p.spacing / s;
      const trim = p.contour ? (ctx.lineWidth * 0.5) / s : 0;
      const stepF = Math.max(Math.min(mpp * 0.7, 0.08 / s), sp * 0.3);
      const inside = (x, y) => {
        let ox = 0, oy = 0;
        if (tiled) { ox = (Math.floor(x / tw) + 0.5) * tw; oy = (Math.floor(y / th) + 0.5) * th; }
        const li = Math.floor((x - ox) / mpp + Wc / 2);
        const lj = Math.floor(-(y - oy) / mpp + Hc / 2);
        if (li < 0 || lj < 0 || li >= Wc || lj >= Hc) return false;
        return r.lab[(lj + j0) * r.W + (li + c0)] === c;
      };
      for (const l of hatch(inside, box, sp, p.angle, stepF, trim)) {
        lines.push({ slot, pts: applyTf(l, el.tf), closed: false });
      }
    }
  }
  const info = tiled ? t("props.tiles", { n: ntiles, mm: pitch.toFixed(1) }) : "";
  return { lines, box: [-fw / 2, -fh / 2, fw / 2, fh / 2], raster: r, info };
}

export function clusterMap(el, r, ctx) {
  const m = el.props.map || [];
  const out = [];
  for (let c = 0; c < r.colors.length; c++) out.push(m[c] !== undefined ? m[c] : r.defaultMap?.[c] ?? -1);
  return out;
}

export function rasterOf(el) {
  const loader = el.type === "text" ? rasterText : el.type === "svg" ? rasterSvg : rasterImage;
  const r = rasters.get(loader.key(el));
  return r && r !== "loading" && !r.error ? r : null;
}

const rasterImage = {
  key: (el) => (el.props.src ? `img|${el.id}|${el.props.src.length}|${hashStr(el.props.src)}|${el.props.colors}|${el.props.despeckle}` : null),
  async load(el) {
    const img = await loadImg(el.props.src);
    const maxPx = 700;
    const k = Math.min(1, maxPx / Math.max(img.width, img.height));
    const W = Math.max(2, Math.round(img.width * k)), H = Math.max(2, Math.round(img.height * k));
    const cv = new OffscreenCanvas(W, H);
    const g = cv.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0, W, H);
    const d = g.getImageData(0, 0, W, H).data;
    return quantize(d, W, H, Math.max(1, Math.min(6, el.props.colors | 0)), el.props.despeckle, el.ctxFilaments || null);
  },
};

const rasterText = {
  key: (el) => {
    const p = el.props;
    return p.text ? `txt|${p.text}|${p.font}|${p.weight}|${p.italic}|${p.size}` : null;
  },
  async load(el) {
    const p = el.props;
    const ppm = 24;    // px per mm
    const px = Math.max(8, p.size * ppm);
    const font = `${p.italic ? "italic " : ""}${p.weight} ${px}px ${p.font}`;
    try { await document.fonts.load(font); } catch {}
    const lines = String(p.text).split("\n");
    const probe = new OffscreenCanvas(10, 10).getContext("2d");
    probe.font = font;
    const lh = px * 1.15;
    const W = Math.ceil(Math.max(...lines.map(l => probe.measureText(l).width)) + px * 0.3);
    const H = Math.ceil(lh * lines.length + px * 0.2);
    const cv = new OffscreenCanvas(Math.max(4, W), Math.max(4, H));
    const g = cv.getContext("2d", { willReadFrequently: true });
    g.font = font; g.fillStyle = "#000"; g.textBaseline = "alphabetic"; g.textAlign = "center";
    lines.forEach((l, i) => g.fillText(l, cv.width / 2, px * 0.95 + i * lh));
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    const lab = new Int8Array(cv.width * cv.height).fill(-1);
    for (let i = 0; i < lab.length; i++) if (d[i * 4 + 3] > 110) lab[i] = 0;
    return { W: cv.width, H: cv.height, lab, colors: ["#000000"], ppm };
  },
};

// ------------------------------------------------------------------ svg
const rasterSvg = {
  key: (el) => (el.props.src ? `svg|${el.id}|${hashStr(el.props.src)}` : null),
  async load(el) {
    const doc = new DOMParser().parseFromString(el.props.src, "image/svg+xml");
    const root = doc.documentElement;
    if (!root || root.nodeName.toLowerCase() !== "svg") throw new Error(t("msg.badSvg"));
    const host = document.createElement("div");
    host.style.cssText = "position:absolute;left:-9999px;top:0;visibility:hidden;width:800px;height:800px";
    const svg = document.importNode(root, true);
    host.appendChild(svg);
    document.body.appendChild(host);
    try {
      let vb = (svg.getAttribute("viewBox") || "").split(/[\s,]+/).map(Number).filter(v => !isNaN(v));
      if (vb.length !== 4) {
        const bb = svg.getBBox();
        const w = parseFloat(svg.getAttribute("width")) || bb.x + bb.width, h = parseFloat(svg.getAttribute("height")) || bb.y + bb.height;
        vb = [0, 0, w || 100, h || 100];
        if (!svg.getAttribute("width")) vb = [bb.x, bb.y, bb.width || 100, bb.height || 100];
      }
      svg.setAttribute("viewBox", vb.join(" "));
      svg.setAttribute("width", 800); svg.setAttribute("height", (800 * vb[3]) / vb[2]);
      const rootM = svg.getScreenCTM().inverse();
      const shapes = svg.querySelectorAll("path,line,polyline,polygon,rect,circle,ellipse");
      const strokes = [];
      const unit = vb[2] / 800;
      for (const sh of shapes) {
        let L = 0;
        try { L = sh.getTotalLength(); } catch { continue; }
        if (!(L > 0)) continue;
        const n = Math.max(8, Math.min(20000, Math.ceil(L / (unit * 0.75))));
        const m = rootM.multiply(sh.getScreenCTM());
        let cur = [];
        let prev = null;
        const stepL = L / n;
        for (let i = 0; i <= n; i++) {
          const pt = sh.getPointAtLength((i / n) * L);
          const x = m.a * pt.x + m.c * pt.y + m.e, y = m.b * pt.x + m.d * pt.y + m.f;
          if (prev && Math.hypot(x - prev[0], y - prev[1]) > Math.max(stepL * 4, unit * 3)) { if (cur.length > 1) strokes.push(cur); cur = []; }
          cur.push([x, y]); prev = [x, y];
        }
        if (cur.length > 1) strokes.push(cur);
      }
      // raster for fills
      const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const img = await loadImg(url);
      URL.revokeObjectURL(url);
      const W = 800, H = Math.max(2, Math.round((800 * vb[3]) / vb[2]));
      const cv = new OffscreenCanvas(W, H);
      const g = cv.getContext("2d", { willReadFrequently: true });
      g.drawImage(img, 0, 0, W, H);
      const d = g.getImageData(0, 0, W, H).data;
      const lab = new Int8Array(W * H).fill(-1);
      for (let i = 0; i < lab.length; i++) if (d[i * 4 + 3] > 110) lab[i] = 0;
      return { vb, strokes, W, H, lab, colors: ["#000000"] };
    } finally {
      host.remove();
    }
  },
};

function buildSvg(el, ctx) {
  const p = el.props;
  const rk = rasterSvg.key(el);
  if (!rk) return { lines: [], box: [-10, -10, 10, 10], info: t("msg.noFile") };
  let r = rasters.get(rk);
  if (!r) {
    rasters.set(rk, "loading");
    rasterSvg.load(el).then(res => { rasters.set(rk, res); onAsync(el, res); }).catch(e => { rasters.set(rk, { error: e.message }); onAsync(el); });
    return { lines: [], box: null, pending: true };
  }
  if (r === "loading") return { lines: [], box: null, pending: true };
  if (r.error) return { lines: [], box: null, info: r.error };
  const [vx, vy, vw, vh] = r.vb;
  const k = p.w / vw;
  const hmm = vh * k;
  const s = el.tf.s || 1;
  const lines = [];
  if (p.stroke) {
    for (const st of r.strokes) {
      const pts = simplify(st.map(([x, y]) => [(x - vx - vw / 2) * k, -(y - vy - vh / 2) * k]), 0.01 / s);
      if (pts.length > 1) lines.push({ slot: el.slot, pts: applyTf(pts, el.tf), closed: false });
    }
  }
  if (p.fill) {
    const mpp = p.w / r.W;
    const inside = (x, y) => {
      const i = Math.floor(x / mpp + r.W / 2), j = Math.floor(-y / mpp + r.H / 2);
      return i >= 0 && j >= 0 && i < r.W && j < r.H && r.lab[j * r.W + i] === 0;
    };
    const hl = hatch(inside, [-p.w / 2, -hmm / 2, p.w / 2, hmm / 2], p.spacing / s, p.angle, Math.min(mpp * 0.7, 0.08 / s),
      p.stroke ? (ctx.lineWidth * 0.5) / s : 0);
    for (const l of hl) lines.push({ slot: el.slot, pts: applyTf(l, el.tf), closed: false });
  }
  return { lines, box: [-p.w / 2, -hmm / 2, p.w / 2, hmm / 2] };
}

// ------------------------------------------------------------------ helpers
function loadImg(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(t("msg.imageLoad")));
    im.src = src;
  });
}

export function hashStr(s) {
  let h = 2166136261 >>> 0;
  const step = Math.max(1, Math.floor(s.length / 4000));
  for (let i = 0; i < s.length; i += step) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h.toString(36) + "_" + s.length;
}

// k-means colour clustering; alpha < 128 = background (-1)
function quantize(d, W, H, K, despeckle) {
  const n = W * H;
  const px = [];
  for (let i = 0; i < n; i++) if (d[i * 4 + 3] >= 128) px.push(i);
  const lab = new Int8Array(n).fill(-1);
  if (!px.length) return { W, H, lab, colors: [] };
  // sample
  const stepS = Math.max(1, Math.floor(px.length / 30000));
  const S = [];
  for (let i = 0; i < px.length; i += stepS) { const q = px[i] * 4; S.push([d[q], d[q + 1], d[q + 2]]); }
  // k-means++ deterministic: start with darkest, then farthest
  const C = [];
  let dark = S.reduce((a, b) => (a[0] + a[1] + a[2] <= b[0] + b[1] + b[2] ? a : b));
  C.push(dark.slice());
  while (C.length < K) {
    let best = null, bd = -1;
    for (const s of S) {
      let md = Infinity;
      for (const c of C) { const dd = (s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2 + (s[2] - c[2]) ** 2; if (dd < md) md = dd; }
      if (md > bd) { bd = md; best = s; }
    }
    if (bd <= 1) break;
    C.push(best.slice());
  }
  for (let it = 0; it < 15; it++) {
    const acc = C.map(() => [0, 0, 0, 0]);
    for (const s of S) {
      let bi = 0, md = Infinity;
      C.forEach((c, i) => { const dd = (s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2 + (s[2] - c[2]) ** 2; if (dd < md) { md = dd; bi = i; } });
      const a = acc[bi]; a[0] += s[0]; a[1] += s[1]; a[2] += s[2]; a[3]++;
    }
    acc.forEach((a, i) => { if (a[3]) C[i] = [a[0] / a[3], a[1] / a[3], a[2] / a[3]]; });
  }
  // order by luminance (dark first)
  C.sort((a, b) => (a[0] * 0.3 + a[1] * 0.59 + a[2] * 0.11) - (b[0] * 0.3 + b[1] * 0.59 + b[2] * 0.11));
  for (const i of px) {
    const q = i * 4;
    let bi = 0, md = Infinity;
    for (let c = 0; c < C.length; c++) {
      const dd = (d[q] - C[c][0]) ** 2 + (d[q + 1] - C[c][1]) ** 2 + (d[q + 2] - C[c][2]) ** 2;
      if (dd < md) { md = dd; bi = c; }
    }
    lab[i] = bi;
  }
  if (despeckle) {
    const cp = lab.slice();
    for (let j = 1; j < H - 1; j++) for (let i = 1; i < W - 1; i++) {
      const k = j * W + i, v = cp[k];
      let same = 0;
      const cnt = {};
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const u = cp[k + dy * W + dx];
        if (u === v) same++;
        cnt[u] = (cnt[u] || 0) + 1;
      }
      if (same <= 2) { let bv = v, bc = 0; for (const u in cnt) if (cnt[u] > bc) { bc = cnt[u]; bv = +u; } lab[k] = bv; }
    }
  }
  // background guess: cluster touching the border most (if transparent border, none)
  const border = new Array(C.length).fill(0);
  let tb = 0;
  for (let i = 0; i < W; i++) for (const j of [0, H - 1]) { const v = lab[j * W + i]; if (v >= 0) border[v]++; else tb++; }
  for (let j = 0; j < H; j++) for (const i of [0, W - 1]) { const v = lab[j * W + i]; if (v >= 0) border[v]++; else tb++; }
  let bg = -1;
  const bmax = Math.max(...border);
  if (bmax > tb && C.length > 1) bg = border.indexOf(bmax);
  return { W, H, lab, colors: C.map(rgbToHex), bg };
}

// default mapping cluster -> filament slot (nearest colour), background skipped
export function defaultClusterMap(r, filaments, fallbackSlot) {
  return r.colors.map((hex, c) => {
    if (c === r.bg) return -1;
    if (r.colors.length - (r.bg >= 0 ? 1 : 0) <= 1) return fallbackSlot;
    const rgb = hexToRgb(hex);
    let best = fallbackSlot, bd = Infinity;
    for (const f of filaments) {
      const q = hexToRgb(f.color);
      const dd = (q[0] - rgb[0]) ** 2 + (q[1] - rgb[1]) ** 2 + (q[2] - rgb[2]) ** 2;
      if (dd < bd) { bd = dd; best = f.slot; }
    }
    return best;
  });
}
