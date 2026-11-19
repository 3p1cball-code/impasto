"""Slope lines (height-line element, mode "flow") on a synthetic 30° cone.

The lines are built in the browser (web/js/elements.js); the test runs the same module
in node. What the settings promise must hold for every line:
  - it lies on the usable surface (incl. the element margin),
  - it keeps half a spacing from every other line,
  - it is at most "max length" long and runs uphill,
  - lines start at the foot and run up as far as the length allows.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node missing")

SCRIPT = r"""
const W = new URL(process.argv[1]);
const { Surface } = await import(new URL("grid.js", W));
const { build } = await import(new URL("elements.js", W));
const { polyLength } = await import(new URL("geom.js", W));
const props = JSON.parse(process.argv[2]);
// cone: base radius 32 mm, 30° flanks, centre (40, 40), 0.1 mm grid, 0.12 mm layer steps
const res = 0.1, nx = 801, ny = 801, n = nx * ny, R = 32, k = Math.tan(30 * Math.PI / 180);
const meta = new TextEncoder().encode(JSON.stringify({ x0: 0, y0: 0, res, nx, ny, sigma: 0.6, tip_r: 0.3 }));
const hl = meta.length + ((4 - (meta.length % 4)) % 4);
const buf = new ArrayBuffer(4 + hl + n * 13);
new DataView(buf).setUint32(0, hl, true);
new Uint8Array(buf, 4, meta.length).set(meta);
new Uint8Array(buf, 4 + meta.length, hl - meta.length).fill(32);
const H = new Float32Array(n), ZN = new Float32Array(n), S = new Uint8Array(n), HSo = new Float32Array(n);
for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
  const r = Math.hypot(i * res - 40, j * res - 40), c = j * nx + i;
  if (r > R) { H[c] = ZN[c] = HSo[c] = -1; S[c] = 180; continue; }
  const h = (R - r) * k;
  H[c] = ZN[c] = Math.max(0.12, Math.ceil(h / 0.12) * 0.12);
  HSo[c] = h; S[c] = 60;
}
const u8 = new Uint8Array(buf);
let o = 4 + hl;
for (const a of [H, ZN, S, HSo]) { u8.set(new Uint8Array(a.buffer), o); o += a.byteLength; }
const surf = new Surface(buf);
surf.computeMask({ angle: 35, inset: 0.3, zmin: 0, minArea: 2, exclude: [] });
const el = { id: "t", type: "contours", slot: 0, tf: { x: 40, y: 40, rot: 0, s: 1 },
             props: Object.assign({ mode: "flow", dz: 0.5, startZ: 0, minLen: 0, maxLines: 400,
               flowSpacing: 3, flowLen: 40, flowMinSlope: 2, flowStep: 0.3, margin: 0, ox: 40, oy: 40 }, props) };
const b = build(el, surf, { lineWidth: 0.45 });
const mask = el.props.margin > 0 ? surf.maskMargin(el.props.margin) : surf.mask;
const out = b.lines.map(l => ({
  len: polyLength(l.pts),
  r0: Math.hypot(l.pts[0][0] - 40, l.pts[0][1] - 40),
  r1: Math.hypot(l.pts.at(-1)[0] - 40, l.pts.at(-1)[1] - 40),
  inMask: l.pts.every(([x, y]) => surf.inMaskOf(mask, x, y)),
  pts: l.pts,
}));
console.log(JSON.stringify(out));
"""


def run(**props):
    r = subprocess.run(["node", "--input-type=module", "-e", SCRIPT, JS.as_uri() + "/", json.dumps(props)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def min_gap(lines, step=0.1):
    import numpy as np
    from scipy.spatial import cKDTree
    pts, own = [], []
    for k, l in enumerate(lines):
        p = np.asarray(l["pts"])
        for a, b in zip(p[:-1], p[1:]):
            m = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
            t = np.arange(m)[:, None] / m
            pts.append(a + (b - a) * t)
            own.append(np.full(m, k))
        pts.append(p[-1:]); own.append([k])
    pts, own = np.vstack(pts), np.concatenate(own)
    tree = cKDTree(pts)
    d, i = tree.query(pts, k=40)
    other = own[i] != own[:, None]
    return float(np.where(other, d, np.inf).min())


def test_lines_cover_the_cone_evenly():
    lines = run(flowSpacing=3, flowLen=60)
    assert len(lines) >= 40
    assert all(l["inMask"] for l in lines)
    assert all(l["r1"] < l["r0"] for l in lines)               # bottom -> top
    # they start at the foot: the usable edge is 0.3 mm inside the 32 mm base
    feet = [l for l in lines if l["r0"] > 31.0]
    assert len(feet) >= 0.9 * len(lines)
    # ... and the foot is filled evenly: 2 pi 31.5 / 3 mm = 66 places, at least 0.9 x spacing apart
    assert 45 <= len(feet) <= 66
    # nothing closer than half a spacing (minus the simplify tolerance)
    assert min_gap(lines) >= 1.5 - 0.05
    # the first lines run all the way up to where the neighbours crowd them out
    assert min(l["r1"] for l in lines) < 1.5


def test_max_length_is_the_whole_line_from_the_foot():
    lines = run(flowSpacing=3, flowLen=12)
    assert all(l["len"] <= 12 + 1e-6 for l in lines)
    feet = [l for l in lines if l["r0"] > 31.0]
    # the old version gave each direction half the budget: 6 mm from a seed at the foot
    assert min(l["len"] for l in feet) > 11


def test_margin_is_kept_while_tracing():
    lines = run(flowSpacing=3, flowLen=60, margin=2)
    assert lines and all(l["inMask"] for l in lines)
    assert max(l["r0"] for l in lines) < 32 - 0.3 - 2 + 0.15


def test_min_slope_stops_the_lines():
    assert run(flowSpacing=3, flowMinSlope=31) == []          # the cone is 30°
