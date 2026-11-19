"""Cell pattern and random colours (web/js/cells.js), run in node.

  - cells are convex, keep the gap to their neighbours and are deterministic per seed,
  - the "rings" fill is one path per cell that stays inside the cell,
  - random colours: same seed = same colours, share 0 keeps the base colour,
    a cell keeps one colour for all its lines, dash sections alternate.
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
const { cellsLocal, randomColours, hash01 } = await import(new URL("cells.js", W));
const cmd = JSON.parse(process.argv[2]);
let out;
if (cmd.op === "cells") out = cellsLocal(cmd.p, cmd.R);
else if (cmd.op === "colours") out = randomColours(cmd.lines, cmd.o);
else out = [hash01(1, 2, 3), hash01(1, 2, 3), hash01(2, 2, 3)];
process.stdout.write(JSON.stringify(out, (k, v) => v));
"""


def run(cmd):
    r = subprocess.run(["node", "--input-type=module", "-e", SCRIPT, (JS / "x").as_uri(), json.dumps(cmd)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def cells(**p):
    base = {"spacing": 4, "jitter": 0.7, "gap": 0.6, "round": 0.8, "fill": "none", "fillSpacing": 0.45, "seed": 3}
    base.update(p)
    return run({"op": "cells", "p": base, "R": 12})


def seg_dist(p, a, b):
    import math
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def test_hash_is_deterministic():
    a, b, c = run({"op": "hash"})
    assert a == b and a != c and 0 <= a < 1


def test_cells_keep_the_gap_and_are_deterministic():
    r1, r2 = cells(), cells()
    assert r1 == r2
    assert r1["cells"] > 20
    lines = r1["lines"]
    # json drops the array properties (closed / grp) – geometry is enough here
    other = cells(seed=4)
    assert other["lines"] != lines
    # no point of one cell comes closer than the gap to another cell's rim
    import numpy as np
    pts = [np.array(l) for l in lines]
    from scipy.spatial import cKDTree
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)])
    allp = np.vstack(pts)
    tree = cKDTree(allp)
    pairs = tree.query_pairs(0.6 - 0.03, output_type="ndarray")
    assert not (owner[pairs[:, 0]] != owner[pairs[:, 1]]).any()


def test_rings_fill_is_one_path_inside_its_cell():
    outl = cells()["lines"]
    rings = cells(fill="rings")["lines"]
    assert len(rings) == len(outl)            # one path per cell
    import numpy as np
    for o, r in zip(outl, rings):
        o, r = np.array(o), np.array(r)
        assert len(r) > len(o)                # outline plus inner rings
        c = o.mean(0)
        # every ring point is inside the (convex) outline: no further from the centre than its rim
        ro = np.hypot(*(o - c).T).max()
        assert np.hypot(*(r - c).T).max() <= ro + 1e-6


def test_hatch_fill_adds_a_line_per_cell():
    h = cells(fill="hatch", spacing=6)["lines"]
    o = cells(spacing=6)["lines"]
    assert len(h) > len(o)


def test_random_colours():
    lines = [{"slot": 0, "pts": [[0, 0], [5, 0]], "closed": False, "src": i, "piece": 0, "grp": i // 2}
             for i in range(200)]
    o = {"mode": "group", "share": 1, "slots": [1, 2, 3], "seed": 7}
    a = run({"op": "colours", "lines": lines, "o": o})
    b = run({"op": "colours", "lines": lines, "o": o})
    assert [x["slot"] for x in a] == [x["slot"] for x in b]
    assert {x["slot"] for x in a} == {1, 2, 3}
    # both lines of a cell (grp) share one colour
    for i in range(0, 200, 2):
        assert a[i]["slot"] == a[i + 1]["slot"]
    keep = run({"op": "colours", "lines": lines, "o": {**o, "share": 0}})
    assert {x["slot"] for x in keep} == {0}
    some = run({"op": "colours", "lines": lines, "o": {**o, "share": 0.3}})
    n0 = sum(1 for x in some if x["slot"] == 0)
    assert 100 < n0 < 180


def test_dash_sections_alternate():
    line = {"slot": 0, "pts": [[x, 0] for x in range(0, 201)], "closed": False, "src": 0, "piece": 0}
    out = run({"op": "colours", "lines": [line], "o": {"mode": "dash", "share": 1, "slots": [0, 1, 2], "seed": 5, "dash": 6}})
    assert len(out) > 15
    for a, b in zip(out, out[1:]):
        assert a["slot"] != b["slot"]
        assert a["pts"][-1] == b["pts"][0]    # sections join up
    total = sum(sum(abs(q[0] - p[0]) for p, q in zip(o["pts"], o["pts"][1:])) for o in out)
    assert abs(total - 200) < 1e-6
