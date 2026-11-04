"""Shared pieces for the Nimbus glider simulations.

Everything geometric (areas, lengths, masses, moments of inertia) is computed
from the CAD STL exports in ./cad. Everything aerodynamic is a modelling
assumption and lives in the ASSUMPTIONS section of each sim, so it's easy to
see and change.
"""
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
CAD_DIR = HERE / "cad"
RESULTS_DIR = HERE / "results"

G = 9.80665              # m/s^2
FT = 0.3048              # m per ft
NU_AIR = 1.5e-5          # kinematic viscosity of air, m^2/s

# ---- Build assumptions (edit to match the real hardware) --------------------
PLA_DENSITY = 1240.0     # kg/m^3
WALLS_MM = 1.2           # 3 perimeters x 0.4 mm nozzle
INFILL = 0.15            # 15 % infill for bulky parts
ELECTRONICS_KG = 0.0     # sensors/battery inside the enclosure -- NOT in the CAD, set this!

# ---- Sweep / release conditions ----------------------------------------------
HEIGHTS_FT = np.arange(0, 5001, 100)   # drop heights to simulate, ft AGL
RELEASE_SPEED = 3.0                    # horizontal speed at release, m/s


@dataclass
class Part:
    name: str
    volume: float            # m^3 (solid)
    mass: float              # kg (as printed)
    centroid: np.ndarray     # m
    inertia: np.ndarray      # kg m^2 about centroid, 3x3 (uniform density)
    lo: np.ndarray           # bounding box minimum corner, m
    size: np.ndarray         # bounding box extents, m
    thickness_axis: int      # index of the thinnest bounding-box axis
    planform: float          # area projected along thickness axis, m^2 (one side)


def _read_stl(path):
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    tris = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                         count=count, offset=84)
    return tris["v"].reshape(-1, 3, 3).astype(float) / 1000.0   # mm -> m


def _mass_properties(tris):
    """Volume, centroid, inertia tensor (unit density) of a closed mesh (Eberly 2002)."""
    p0, p1, p2 = tris[:, 0], tris[:, 1], tris[:, 2]
    d = np.cross(p1 - p0, p2 - p0)
    t0 = p0 + p1
    f1 = t0 + p2
    t1 = p0 * p0
    t2 = t1 + p1 * t0
    f2 = t2 + p2 * f1
    f3 = p0 * t1 + p1 * t2 + p2 * f2
    g0, g1, g2 = f2 + p0 * (f1 + p0), f2 + p1 * (f1 + p1), f2 + p2 * (f1 + p2)
    x, y, z = 0, 1, 2
    vol = np.sum(d[:, x] * f1[:, x]) / 6
    first = np.sum(d * f2, axis=0) / 24
    second = np.sum(d * f3, axis=0) / 60
    xy = np.sum(d[:, x] * (p0[:, y] * g0[:, x] + p1[:, y] * g1[:, x] + p2[:, y] * g2[:, x])) / 120
    yz = np.sum(d[:, y] * (p0[:, z] * g0[:, y] + p1[:, z] * g1[:, y] + p2[:, z] * g2[:, y])) / 120
    zx = np.sum(d[:, z] * (p0[:, x] * g0[:, z] + p1[:, x] * g1[:, z] + p2[:, x] * g2[:, z])) / 120
    sign = np.sign(vol)
    vol, first, second, xy, yz, zx = (sign * v for v in (vol, first, second, xy, yz, zx))
    c = first / vol
    I = np.array([
        [second[y] + second[z], -xy, -zx],
        [-xy, second[x] + second[z], -yz],
        [-zx, -yz, second[x] + second[y]],
    ])
    # shift from origin to centroid
    I -= vol * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return vol, c, I, d


def load_part(stem, bulky=False):
    """Load cad/<stem>.stl. Bulky parts get shell+infill mass, thin parts print solid."""
    tris = _read_stl(CAD_DIR / f"{stem}.stl")
    vol, c, I_unit, d = _mass_properties(tris)
    pts = tris.reshape(-1, 3)
    lo = pts.min(axis=0)
    size = pts.max(axis=0) - lo
    t_axis = int(np.argmin(size))
    area = 0.5 * np.sum(np.linalg.norm(d, axis=1))
    planform = np.sum(np.abs(d[:, t_axis])) / 4      # top + bottom counted -> /2 more
    if bulky:
        shell = min(vol, area * WALLS_MM / 1000)
        printed_vol = shell + INFILL * (vol - shell)
    else:
        printed_vol = vol
    mass = printed_vol * PLA_DENSITY
    inertia = I_unit * (mass / vol)                   # uniform-density approximation
    return Part(stem, vol, mass, c, inertia, lo, size, t_axis, planform)


def centerline(stem, bins=30):
    """Mid-line of a thin strip part, traced from its top/bottom faces in the STL.

    Returns (n, 2) points in the strip's plane, in m: column 0 runs along the
    part's long axis from its low end (the root), column 1 is the sideways offset.
    """
    tris = _read_stl(CAD_DIR / f"{stem}.stl")
    pts = tris.reshape(-1, 3)
    lo, size = pts.min(axis=0), pts.max(axis=0) - pts.min(axis=0)
    thin, long = int(np.argmin(size)), int(np.argmax(size))
    side = 3 - thin - long
    d = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face = np.abs(d[:, thin]) > 0.9 * np.linalg.norm(d, axis=1)
    c, w = tris[face].mean(axis=1) - lo, np.abs(d[face, thin])
    edges = np.linspace(0, size[long], bins + 1)
    idx = np.clip(np.digitize(c[:, long], edges) - 1, 0, bins - 1)
    mid = [(np.average(c[idx == i, long], weights=w[idx == i]),
            np.average(c[idx == i, side], weights=w[idx == i]))
           for i in range(bins) if np.any(idx == i)]
    mid = np.array(mid)
    return np.vstack([[0.0, mid[0, 1]], mid, [size[long], mid[-1, 1]]])


def air_density(h_m):
    """ISA troposphere density, kg/m^3."""
    return 1.225 * (1 - 2.25577e-5 * np.maximum(h_m, 0.0)) ** 4.2559


def plate_coeffs(alpha, speed, chord, thickness):
    """Lift/drag coefficients of a thin flat plate, smooth over all angles.

    CL = 1.2 sin(2a), CD = CD0 + 2 sin^2(a) is the standard flat-plate
    approximation (peaks: CL ~1.2 at 45 deg, CD ~2 broadside). CD0 is laminar
    skin friction on both faces plus pressure drag on the blunt, laser-cut edges.
    """
    re = max(speed * chord / NU_AIR, 1e3)
    cd0 = 2 * 1.328 / np.sqrt(re) + 0.5 * thickness / chord
    return 1.2 * np.sin(2 * alpha), cd0 + 2 * np.sin(alpha) ** 2


def lift_unit(vx, vz, speed):
    """Unit vector perpendicular to the flight path, tilted toward +x/up."""
    return np.array([-vz, vx]) / speed
