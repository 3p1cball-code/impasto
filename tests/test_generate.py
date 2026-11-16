import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from npg import bambu, generate, merge, parse, surface

ROOT = Path(__file__).resolve().parents[1]
SAMPLE3MF = ROOT / "Brille_RIG_MKV_2026_Master_plate_2.gcode.3mf"

pytestmark = pytest.mark.skipif(not SAMPLE3MF.exists(), reason="sample missing")


@pytest.fixture(scope="module")
def setup():
    job = bambu.load_path(SAMPLE3MF)
    tp = parse.parse(job.lines, job.exec_start, job.insert_at)
    g = surface.analyze(surface.heightmap(tp))
    return job, tp, g


def design():
    paths = []
    for y in np.arange(56, 61, 0.45):
        paths.append({"slot": 0, "pts": [[60, y], [120, y]]})
    for x in np.arange(80, 100, 0.9):           # crossing lines in another colour
        paths.append({"slot": 1, "pts": [[x, 55], [x, 62]]})
    t = np.linspace(0, 2 * np.pi, 120)
    paths.append({"slot": 3, "pts": np.c_[92 + 3 * np.cos(t), 59 + 3 * np.sin(t)].tolist(), "closed": True})
    return paths


def block_moves(text):
    x = y = z = None
    out = []
    for ln in text.split("\n"):
        if not ln.startswith("G1"):
            continue
        d = dict((w[0], float(w[1:])) for w in ln.split(";")[0].split()[1:] if w[0] in "XYZEF")
        nx, ny, nz = d.get("X", x), d.get("Y", y), d.get("Z", z)
        if "E" in d and d["E"] > 0 and ("X" in d or "Y" in d):
            out.append((x, y, z, nx, ny, nz, d["E"]))
        x, y, z = nx, ny, nz
    return out


def test_layer_header_and_colours(setup):
    job, tp, g = setup
    res = generate.generate(job, tp, g, design(), generate.Params())
    st = res.stats
    assert st["colors"] == [0, 1, 3]            # starts with the loaded filament (white, slot 0)
    assert st["toolchanges"] == 2
    assert "M620 S1A" in res.gcode and "M620 S3A" in res.gcode
    assert res.gcode.count("; CHANGE_LAYER") == 1
    assert "M73 L218" in res.gcode


def test_never_below_surface(setup):
    job, tp, g = setup
    p = generate.Params()
    res = generate.generate(job, tp, g, design(), p)
    mv = block_moves(res.gcode)
    assert len(mv) > 500
    arr = np.array(mv, dtype=float)
    zs = surface.sample_max(g.ZN, g, arr[:, 3], arr[:, 4])
    ok = np.isfinite(zs)
    gap = arr[ok, 5] - zs[ok]
    assert gap.min() >= p.z_offset - 1e-3          # nozzle never deeper than offset above the dilated surface
    assert (arr[:, 6] > 0).all()


def test_crossings_are_lifted(setup):
    job, tp, g = setup
    res = generate.generate(job, tp, g, design(), generate.Params(stack=True))
    res0 = generate.generate(job, tp, g, design(), generate.Params(stack=False))
    assert res.stats["max_z"] > res0.stats["max_z"] - 1e-6
    a = np.array(block_moves(res.gcode))
    b = np.array(block_moves(res0.gcode))
    assert a[:, 5].mean() > b[:, 5].mean()


def test_merge_and_3mf(setup):
    job, tp, g = setup
    res = generate.generate(job, tp, g, design(), generate.Params())
    text = merge.merged_text(job, res)
    lines = text.split("\n")
    assert "; total layer number: 218" in lines[:20]
    # original content preserved before the insertion point (except M73 progress values)
    for a, b in zip(job.lines[job.exec_start:job.insert_at], lines[job.exec_start:job.insert_at]):
        if a != b:
            assert a.startswith("M73 P") and b.startswith("M73 P")
    i = lines.index("; ===== NON-PLANAR LAYER (Non-Planar Gcoder) START =====")
    assert i == job.insert_at
    # end gcode lowered-z line raised above the decoration
    m = [l for l in lines if "lower z a little" in l][-1]
    assert float(re.search(r"Z([\d.]+)", m).group(1)) >= res.max_z
    data = merge.build_3mf(job, res, text)
    zf = zipfile.ZipFile(io.BytesIO(data))
    g2 = zf.read(f"Metadata/plate_{job.plate}.gcode")
    assert zf.read(f"Metadata/plate_{job.plate}.gcode.md5").decode() == hashlib.md5(g2).hexdigest().upper()
    si = zf.read("Metadata/slice_info.config").decode()
    for fid in ("1", "2", "3", "4"):
        assert f'<filament id="{fid}"' in si
    # the merged G-code parses and contains the extra layer
    j2 = bambu.load_bytes(data, "x.gcode.3mf")
    tp2 = parse.parse(j2.lines, j2.exec_start, j2.insert_at)
    assert len(tp2.layer_z) == len(tp.layer_z) + 1


def test_element_order_stacks(setup):
    job, tp, g = setup
    paths = [{"slot": 0, "pts": [[60, 56], [90, 56]], "order": 0},
             {"slot": 0, "pts": [[60, 56], [90, 56]], "order": 1}]      # same line, later element
    res = generate.generate(job, tp, g, paths, generate.Params(z_offset=0.15, dir_opt=False))
    assert res.stats["elements"] == 2
    assert res.gcode.count("; CHANGE_LAYER") == 1
    z = res.moves[res.moves[:, 0] == 0][:, 7]
    n = len(z) // 2
    assert z[n:].mean() - z[:n].mean() > 0.1            # second element lies on the first
    assert res.stats["stacked_mm"] > 10


def test_partly_overlapping_element_falls_back_to_the_surface(setup):
    job, tp, g = setup
    paths = [{"slot": 0, "pts": [[60, 56], [75, 56]], "order": 0},
             {"slot": 0, "pts": [[60, 56], [105, 56]], "order": 1}]     # overlaps on the left only
    res = generate.generate(job, tp, g, paths, generate.Params(z_offset=0.15, dir_opt=False))
    ex = res.moves[res.moves[:, 0] == 0]
    z = ex[:, 7]
    n = len(z) // 2
    z2 = z[n:]
    assert z2.max() - z2.min() > 0.1                    # part lifted, part on the surface
    assert res.stats["stacked_mm"] > 5


def test_single_element_header(setup):
    job, tp, g = setup
    paths = [{"slot": 0, "pts": [[60, 56], [90, 56]]}]
    res = generate.generate(job, tp, g, paths, generate.Params(z_offset=0.15))
    assert res.stats["elements"] == 1
    assert res.gcode.count("; CHANGE_LAYER") == 1
    head = res.gcode.split("\n")[:14]
    assert head[1] == "; CHANGE_LAYER"
    assert any(l.startswith("; Z_HEIGHT:") for l in head)
    assert any(l.startswith("; layer num/total_layer_count:") for l in head)


def travel_mm(res):
    mv = np.asarray(res.moves, dtype=float).reshape(-1, 9)
    d = np.hypot(mv[:, 5] - mv[:, 2], mv[:, 6] - mv[:, 3])
    return float(d[(mv[:, 0] == 1) & (d > 1e-3)].sum())


def test_planned_travel_matches_the_gcode(setup):
    job, tp, g = setup
    res = generate.generate(job, tp, g, design(), generate.Params())
    planned = res.stats["dir"]["travel_mm"]
    assert planned > 0
    assert abs(planned - travel_mm(res)) < 0.05 * travel_mm(res)


def test_travel_weight_shortens_the_travel(setup):
    job, tp, g = setup
    paths = [{"slot": 0, "pts": [[60.0, y], [92.0, y]]} for y in np.arange(55.0, 61.0, 0.5)]
    strict = generate.generate(job, tp, g, paths, generate.Params(dir_travel_weight=500.0))
    loose = generate.generate(job, tp, g, paths, generate.Params(dir_travel_weight=0.0))
    assert travel_mm(loose) < 0.8 * travel_mm(strict)       # short ways win
    assert strict.stats["dir"]["down_mm"] <= loose.stats["dir"]["down_mm"] + 1e-9


def test_closed_ring_stays_closed(setup):
    # a closed outline may be entered at any vertex (it is rolled to the cheapest
    # one); rolling must not lose or duplicate a segment
    job, tp, g = setup
    t = np.linspace(0, 2 * np.pi, 200)
    pts = np.c_[106 + 2.0 * np.cos(t), 24 + 2.0 * np.sin(t)]
    res = generate.generate(job, tp, g, [{"slot": 0, "pts": pts.tolist(), "closed": True}],
                            generate.Params(dir_min_rise=5.0))
    ex = res.moves[res.moves[:, 0] == 0]
    ring_len = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    assert res.stats["strokes"][0] == 1                      # not cut
    assert abs(res.stats["length_mm"][0] - ring_len) < 0.01   # nothing lost
    assert np.hypot(ex[0, 2] - ex[-1, 5], ex[0, 3] - ex[-1, 6]) < 0.1   # closes
