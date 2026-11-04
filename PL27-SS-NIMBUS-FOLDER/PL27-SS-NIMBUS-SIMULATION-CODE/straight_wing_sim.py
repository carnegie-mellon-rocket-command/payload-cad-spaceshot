"""Straight-wing Nimbus glider: how far does it glide when dropped from 0-5000 ft?

Model: 2-D point mass (x = downrange, z = altitude) flying at a fixed trim
angle of attack, with lift and drag from the flat-plate wings plus bluff-body
drag from the sensor enclosure. Starts at RELEASE_SPEED and lands at z = 0.

Geometry and mass come from the CAD (see nimbus_common.load_part). The
aerodynamic behaviour comes from the ASSUMPTIONS block below -- change those
to explore the design.

Run:  python3 straight_wing_sim.py
Out:  results/straight_wing_results.csv, results/straight_wing_sim.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import nimbus_common as nc

# ---- ASSUMPTIONS ---------------------------------------------------------------
TRIM_AOA_DEG = 8.0     # angle of attack the glider trims at
OSWALD_E = 0.8         # span efficiency factor
BOX_CD = 1.05          # drag coefficient of the sensor enclosure (bluff box)

# ---- Geometry and mass from the CAD ---------------------------------------------
enclosure = nc.load_part("PL27-SS-NIMBUS-SENSOR-ENCLOSURE", bulky=True)
wing = nc.load_part("PL27-SS-NIMBUS-STRAIGHT-WING-HALF")

MASS = enclosure.mass + 2 * wing.mass + nc.ELECTRONICS_KG
SEMISPAN = wing.size.max()
CHORD = wing.planform / SEMISPAN
THICKNESS = wing.size.min()
WING_AREA = 2 * wing.planform
SPAN = 2 * SEMISPAN + enclosure.size.max()           # halves on opposite sides of the box
ASPECT = SPAN ** 2 / WING_AREA
BOX_FRONTAL = np.prod(np.sort(enclosure.size)[:2])   # smallest face leads
ALPHA = np.radians(TRIM_AOA_DEG)


def derivs(t, s, wind):
    x, z, vx, vz = s
    vax = vx - wind
    speed = max(np.hypot(vax, vz), 1e-6)
    q = 0.5 * nc.air_density(z) * speed ** 2
    cl, cd = nc.plate_coeffs(ALPHA, speed, CHORD, THICKNESS)
    cd += cl ** 2 / (np.pi * OSWALD_E * ASPECT)                  # induced drag
    lift = q * WING_AREA * cl
    drag = q * (WING_AREA * cd + BOX_CD * BOX_FRONTAL)
    fx, fz = lift * nc.lift_unit(vax, vz, speed) - drag * np.array([vax, vz]) / speed
    return [vx, vz, fx / MASS, fz / MASS - nc.G]


def _ground(t, s, wind):
    return s[1]


_ground.terminal, _ground.direction = True, -1


def simulate(height_m, wind=0.0):
    """Drop from height_m; return downrange distance, flight time and the path."""
    if height_m <= 0:
        return {"distance": 0.0, "time": 0.0, "t": np.zeros(1), "x": np.zeros(1), "z": np.zeros(1)}
    sol = solve_ivp(derivs, (0, 1e4), [0.0, height_m, nc.RELEASE_SPEED, 0.0], args=(wind,),
                    events=_ground, max_step=0.5, rtol=1e-7, atol=1e-9)
    return {"distance": sol.y[0, -1], "time": sol.t[-1], "t": sol.t, "x": sol.y[0], "z": sol.y[1],
            "speed": np.hypot(sol.y[2] - wind, sol.y[3])}


def sweep(heights_ft=nc.HEIGHTS_FT, wind=0.0):
    rows = []
    for h_ft in heights_ft:
        r = simulate(h_ft * nc.FT, wind)
        rows.append({"height_ft": h_ft, "distance_ft": r["distance"] / nc.FT,
                     "distance_m": r["distance"], "descent_time_s": r["time"]})
    return rows


def describe():
    print("Straight-wing glider (from CAD)")
    print(f"  mass {MASS * 1e3:.1f} g   wing area {WING_AREA * 1e4:.1f} cm^2   span {SPAN * 1e3:.0f} mm"
          f"   chord {CHORD * 1e3:.1f} mm   aspect ratio {ASPECT:.1f}")
    print(f"  wing loading {MASS * nc.G / WING_AREA:.0f} N/m^2   enclosure frontal area"
          f" {BOX_FRONTAL * 1e4:.1f} cm^2 (= {BOX_FRONTAL / WING_AREA:.0%} of wing area)")


def main():
    describe()
    rows = sweep()
    nc.RESULTS_DIR.mkdir(exist_ok=True)
    with open(nc.RESULTS_DIR / "straight_wing_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    top = simulate(nc.HEIGHTS_FT[-1] * nc.FT)
    n = len(top["t"]) // 2
    glide = (top["x"][-1] - top["x"][n]) / (top["z"][n] - top["z"][-1])
    print(f"  steady glide ratio ~{glide:.2f} : 1, steady speed ~{top['speed'][-1]:.1f} m/s")
    for r in rows:
        if r["height_ft"] in (1000, 2500, 5000):
            print(f"  from {r['height_ft']:>4} ft: {r['distance_ft']:7.0f} ft downrange,"
                  f" {r['descent_time_s']:5.0f} s in the air")

    h = np.array([r["height_ft"] for r in rows])
    d = np.array([r["distance_ft"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(h, d, color="tab:blue", lw=2)
    a1.set(xlabel="Drop height (ft)", ylabel="Horizontal distance (ft)",
           title="Glide distance vs drop height")
    a1.grid(alpha=0.3)
    for h_ft in (1000, 2500, 5000):
        r = simulate(h_ft * nc.FT)
        a2.plot(r["x"] / nc.FT, r["z"] / nc.FT, label=f"{h_ft} ft drop")
    a2.set(xlabel="Downrange (ft)", ylabel="Altitude (ft)", title="Flight paths")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.suptitle(f"Straight-wing Nimbus glider  |  {MASS * 1e3:.0f} g, wing {WING_AREA * 1e4:.0f} cm², "
                 f"trim AoA {TRIM_AOA_DEG:.0f}°, still air")
    fig.tight_layout()
    fig.savefig(nc.RESULTS_DIR / "straight_wing_sim.png", dpi=150)
    print(f"  saved results/straight_wing_results.csv and results/straight_wing_sim.png")


if __name__ == "__main__":
    main()
