"""The smoothed height field the height lines are computed from.

It is part of the analysis (surface.analyze) and travels to the browser in the grid
buffer, so both ends must agree on its size.
"""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from impasto import bambu, parse, server, surface

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Brille_RIG_MKV_2026_Master_plate_2.gcode.3mf"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample missing")


@pytest.fixture(scope="module")
def grid():
    job = bambu.load_path(SAMPLE)
    tp = parse.parse(job.lines, job.exec_start, job.insert_at)
    return surface.analyze(surface.heightmap(tp))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path)
    with TestClient(server.app) as c:
        yield c


def test_analysis_keeps_the_smoothed_height(grid):
    assert grid.Hs is not None
    assert grid.Hs.shape == grid.H.shape
    valid = np.isfinite(grid.H)
    assert np.isfinite(grid.Hs[valid]).all()
    assert np.isnan(grid.Hs[~valid]).all()
    # smoothing removes the layer steps: less extreme than the raw map, same level
    assert grid.Hs[valid].std() <= grid.H[valid].std()
    assert abs(float(np.mean(grid.Hs[valid] - grid.H[valid]))) < 0.5


def test_grid_buffer_carries_the_field(client):
    data = SAMPLE.read_bytes()
    r = client.post("/api/upload", files={"file": ("plate.gcode.3mf", data, "application/octet-stream")})
    assert r.status_code == 200, r.text
    fid = r.json()["id"]
    r = client.get(f"/api/job/{fid}/grid?plate=2")
    assert r.status_code == 200, r.text
    meta_len = struct.unpack("<I", r.content[:4])[0]
    meta = json.loads(r.content[4:4 + meta_len].decode())
    n = meta["nx"] * meta["ny"]
    # H, ZN, HS as float32 plus one byte of slope and one byte of filament per cell
    assert meta["fmt"] == 2
    assert len(r.content) == 4 + meta_len + 4 * n * 3 + n * 2
    hs = np.frombuffer(r.content[4 + meta_len + 4 * n * 2 + n: 4 + meta_len + 4 * n * 3 + n], dtype=np.float32)
    assert hs.size == n
    assert (hs > -0.5).any()
    fil = np.frombuffer(r.content[4 + meta_len + 4 * n * 3 + n:], dtype=np.int8)
    assert fil.size == n
    assert (fil[hs > -0.5] >= 0).all()


def test_surface_keeps_the_filament_of_the_top_layer(grid):
    valid = np.isfinite(grid.H)
    assert grid.FIL is not None and grid.FIL.shape == grid.H.shape
    assert (grid.FIL[valid] >= 0).all()          # filled from the nearest known cell
    assert set(np.unique(grid.FIL[valid])) <= {0, 1, 2, 3}


def test_top_layer_wins_over_lower_filaments():
    """Each cell reports the filament of the bead that forms its visible top surface."""
    from impasto.parse import Toolpaths
    seg = np.array([
        [10.0, 10.0, 12.0, 10.0, 0.2],   # filament 0, low
        [10.0, 10.0, 12.0, 10.0, 0.9],   # filament 1, right above it
        [10.0, 11.0, 12.0, 11.0, 0.6],   # filament 2, next lane
    ], dtype=np.float32)
    tp = Toolpaths(seg=seg, width=np.full(3, 0.45, np.float32), feat=np.zeros(3, np.uint8),
                   fil=np.array([0, 1, 2], np.int8), layer=np.zeros(3, np.int16),
                   features=["Outer wall"], layer_z=[0.2, 0.9], last_filament=2, max_z=0.9,
                   end_pos=(12.0, 11.0, 0.6))
    g = surface.heightmap(tp, res=0.1, margin=1.0)

    def at(x, y):
        i = int(round((x - g.x0) / g.res)); j = int(round((y - g.y0) / g.res))
        return int(g.FIL[j, i]), float(g.H[j, i])

    assert at(10.5, 10.0) == (1, pytest.approx(0.9))     # top bead wins
    assert at(10.5, 11.0)[0] == 2                        # the other lane keeps its filament
