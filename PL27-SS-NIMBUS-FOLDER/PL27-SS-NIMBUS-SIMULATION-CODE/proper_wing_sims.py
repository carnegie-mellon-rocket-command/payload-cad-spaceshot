"""What if the Nimbus gliders had proper wings?  |  vs  ()  vs  )  shaped wings.

The CAD wings are 1/16"-thick flat strips about 10 mm wide. Here they become
"proper" wings: a cambered low-Reynolds-number airfoil with a real chord
(PROPER_CHORD), printed hollow. Each wing keeps its CAD shape -- the centerline
is traced from the STL -- and the CAD sensor enclosure stays the same:

  |    straight halves                  (STRAIGHT-WING-HALF)      -> glides
  ()   curved halves as mirror images   (CURVED-WING-HALF, x2)    -> symmetric, glides
  )    curved halves facing the same    (CURVED-WING-HALF, x2, S) -> autorotates (spins)
       way, like the current assembly

Gliders (| and ()): 2-D point mass at the best-glide angle of attack, with
airfoil profile drag, induced drag (from span/aspect ratio of the CAD shape)
and enclosure drag. The curved wing's average sweep (span / arc length)
reduces its lift.

Spinner ()): same blade-element rotor as curved_wing_sim.py, but with airfoil
blades and the induced velocity through the rotor disk solved from momentum
theory (Glauert/Buhl), so thrust can't exceed what a real rotor can produce.
Spin rate and rotational kinetic energy are tracked; I comes from the blade
centerline traced from the CAD.

Run:  python3 proper_wing_sims.py
Out:  results/proper_wings/  (CSV + PNG per shape, comparison CSV + PNG)
"""
import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import curved_wing_sim as cad_curved
import nimbus_common as nc
import straight_wing_sim as cad_straight

# ---- ASSUMPTIONS: what a "proper wing" is ----------------------------------------
PROPER_CHORD = None           # m; None = match the enclosure width (63.5 mm)
WING_AREAL_DENSITY = 1.2      # kg/m^2 of wing: printed hollow airfoil (~0.5 mm skin + ribs)
CL0 = 0.35                    # airfoil lift at 0 deg from camber
LIFT_SLOPE = 0.9 * 2 * np.pi  # per rad, thin cambered airfoil at low Re
STALL_DEG = 12.0              # where the airfoil stalls (lift then follows a flat plate)
CL_MARGIN = 0.85              # gliders trim at best glide, but at no more than 85 % of max lift
CD_FORM = 0.008               # profile drag on top of laminar skin friction
CD_K = 0.012                  # profile drag rise away from CL_DESIGN
CL_DESIGN = 0.5
OSWALD = {"|": 0.80, "()": 0.82}   # span efficiency (crescent planforms do slightly better)
BOX_CD = 1.05                 # sensor enclosure, bluff box
BLADE_PITCH_DEG = 3.0         # ")" spinner: blade pitch
THRUST_TILT_DEG = 5.0         # ")" spinner: thrust tilt off the flight path (sideways glide)
R_STATION = 0.75              # ")" spinner: blade-element station, fraction of tip radius

HEIGHTS_FT = np.arange(0, 5001, 250)
WIND_MPH = 10.0
WIND = WIND_MPH * 0.44704
OUT = nc.RESULTS_DIR / "proper_wings"

# ---- Geometry from the CAD -------------------------------------------------------
STRAIGHT = "PL27-SS-NIMBUS-STRAIGHT-WING-HALF"
CURVED = "PL27-SS-NIMBUS-CURVED-WING-HALF"
enclosure = nc.load_part("PL27-SS-NIMBUS-SENSOR-ENCLOSURE", bulky=True)
CHORD = PROPER_CHORD or enclosure.size.max()
MOUNT = enclosure.size.max() / 2                                    # wing roots on the box sides
BOX_FRONTAL = np.prod(np.sort(enclosure.size)[:2])                  # gliders: small face leads
BOX_FLAT = np.prod(np.delete(enclosure.size, enclosure.thickness_axis))  # spinner: falls flat
SPIN_AXIS = nc.load_part(CURVED).thickness_axis
PITCH = np.radians(BLADE_PITCH_DEG)
TILT = np.radians(THRUST_TILT_DEG)
STALL = np.radians(STALL_DEG)


def cd_min(re):
    return CD_FORM + 2 * 1.328 / np.sqrt(max(re, 1e3))


def airfoil(alpha, re, cos_sweep=1.0):
    """Section CL, CD at any angle: attached airfoil blended into flat-plate post-stall."""
    cl_att = cos_sweep * (CL0 + LIFT_SLOPE * alpha)
    cd_att = cd_min(re) + CD_K * (cl_att - CL_DESIGN) ** 2
    cl_post = 1.2 * np.sin(2 * alpha)
    cd_post = cd_min(re) + 2 * np.sin(alpha) ** 2
    centre, half = (STALL - np.radians(8)) / 2, (STALL + np.radians(8)) / 2   # attached: -8 .. stall
    w = 1 / (1 + np.exp((abs(alpha - centre) - half) / np.radians(2)))
    return w * cl_att + (1 - w) * cl_post, w * cd_att + (1 - w) * cd_post


@dataclass
class Design:
    shape: str
    label: str
    spins: bool
    mass: float
    wing_area: float
    span: float
    cos_sweep: float
    oswald: float = 0.8
    alpha_trim: float = 0.0
    r_tip: float = 0.0
    inertia: float = 0.0

    @property
    def aspect(self):
        return self.span ** 2 / self.wing_area


def half_wing(stem):
    pts = nc.centerline(stem)
    ds = np.hypot(*np.diff(pts, axis=0).T)
    return pts, ds, ds.sum(), pts[-1, 0]


def glider_polar(d, alpha, speed):
    cl, cd = airfoil(alpha, speed * CHORD / nc.NU_AIR, d.cos_sweep)
    return cl, cd + cl ** 2 / (np.pi * d.oswald * d.aspect) + BOX_CD * BOX_FRONTAL / d.wing_area


def make_glider(shape, label, stem):
    pts, ds, arc, semispan = half_wing(stem)
    area = 2 * CHORD * arc
    d = Design(shape, label, False,
               mass=enclosure.mass + WING_AREAL_DENSITY * area + nc.ELECTRONICS_KG,
               wing_area=area, span=2 * semispan + enclosure.size.max(),
               cos_sweep=semispan / arc, oswald=OSWALD[shape])
    v_nom = np.sqrt(2 * d.mass * nc.G / (1.225 * area * 0.8))
    alphas = np.radians(np.linspace(-3, STALL_DEG + 3, 400))
    cl, cd = np.array([glider_polar(d, a, v_nom) for a in alphas]).T
    ld = np.where(cl <= CL_MARGIN * cl.max(), cl / cd, -np.inf)
    d.alpha_trim = alphas[int(np.argmax(ld))]
    return d


def make_spinner(shape, label, stem):
    pts, ds, arc, semispan = half_wing(stem)
    area = 2 * CHORD * arc
    mid = (pts[1:] + pts[:-1]) / 2
    r = np.hypot(MOUNT + mid[:, 0], mid[:, 1] - pts[0, 1])
    blade_I = np.sum(WING_AREAL_DENSITY * CHORD * ds * (r ** 2 + CHORD ** 2 / 12))
    return Design(shape, label, True,
                  mass=enclosure.mass + WING_AREAL_DENSITY * area + nc.ELECTRONICS_KG,
                  wing_area=area, span=2 * semispan + enclosure.size.max(),
                  cos_sweep=semispan / arc,
                  r_tip=np.hypot(MOUNT + pts[:, 0], pts[:, 1] - pts[0, 1]).max(),
                  inertia=enclosure.inertia[SPIN_AXIS, SPIN_AXIS] + 2 * blade_I)


DESIGNS = [
    make_glider("|", "| straight wings (glides)", STRAIGHT),
    make_glider("()", "() mirrored curved wings (glides)", CURVED),
    make_spinner(")", ") same-way curved wings (spins)", CURVED),
]


# ---- Equations of motion -----------------------------------------------------------
def ct_momentum(a):
    """Rotor thrust coefficient vs axial induction (momentum theory + Buhl's high-a fix)."""
    return 4 * a * (1 - a) if a <= 0.4 else 8 / 9 - 4 / 9 * a + 14 / 9 * a ** 2


def rotor(d, rho, speed, w):
    """Thrust, torque and induction of the spinning wings, blade element + momentum."""
    r = R_STATION * d.r_tip
    disk = np.pi * d.r_tip ** 2
    u = w * r

    def blade(a):
        va = speed * (1 - a)
        phi = np.arctan2(va, u)
        rel2 = va ** 2 + u ** 2
        cl, cd = airfoil(phi - PITCH, np.sqrt(rel2) * CHORD / nc.NU_AIR, d.cos_sweep)
        q = 0.5 * rho * rel2 * d.wing_area
        return q * (cl * np.cos(phi) + cd * np.sin(phi)), q * r * (cl * np.sin(phi) - cd * np.cos(phi))

    qdisk = 0.5 * rho * speed ** 2 * disk
    resid = lambda a: blade(a)[0] / qdisk - ct_momentum(a)
    a = 0.0 if resid(0.0) <= 0 else (1.0 if resid(1.0) >= 0 else brentq(resid, 0.0, 1.0, xtol=1e-6))
    thrust, torque = blade(a)
    return thrust, torque


def derivs(t, s, d, wind):
    x, z, vx, vz = s[:4]
    vax = vx - wind
    speed = max(np.hypot(vax, vz), 1e-6)
    rho = nc.air_density(z)
    q = 0.5 * rho * speed ** 2
    back = -np.array([vax, vz]) / speed
    side = nc.lift_unit(vax, vz, speed)
    if d.spins:
        thrust, torque = rotor(d, rho, speed, s[4])
        f = (thrust * np.cos(TILT) + q * BOX_CD * BOX_FLAT) * back + thrust * np.sin(TILT) * side
        extra = [torque / d.inertia]
    else:
        cl, cd = glider_polar(d, d.alpha_trim, speed)
        f = q * d.wing_area * (cd * back + cl * side)
        extra = []
    return [vx, vz, f[0] / d.mass, f[1] / d.mass - nc.G] + extra


def _ground(t, s, d, wind):
    return s[1]


_ground.terminal, _ground.direction = True, -1


def simulate(d, height_m, wind=0.0):
    if height_m <= 0:
        z0 = np.zeros(1)
        return {"distance": 0.0, "time": 0.0, "t": z0, "x": z0, "z": z0, "w": z0, "speed": z0}
    s0 = [0.0, height_m, nc.RELEASE_SPEED, 0.0] + ([0.0] if d.spins else [])
    sol = solve_ivp(derivs, (0, 1e5), s0, args=(d, wind), events=_ground, method="LSODA",
                    max_step=1.0, rtol=1e-6, atol=1e-8)
    return {"distance": sol.y[0, -1], "time": sol.t[-1], "t": sol.t, "x": sol.y[0], "z": sol.y[1],
            "w": sol.y[4] if d.spins else np.zeros_like(sol.t),
            "speed": np.hypot(sol.y[2] - wind, sol.y[3])}


def sweep(d, wind=0.0):
    rows = []
    for h_ft in HEIGHTS_FT:
        r = simulate(d, h_ft * nc.FT, wind)
        rows.append({"height_ft": h_ft, "distance_ft": r["distance"] / nc.FT, "descent_time_s": r["time"],
                     "landing_spin_rpm": r["w"][-1] * 60 / (2 * np.pi),
                     "landing_rot_ke_J": 0.5 * d.inertia * r["w"][-1] ** 2})
    return rows


# ---- Output ------------------------------------------------------------------------
SLUG = {"|": "straight", "()": "mirrored_curved", ")": "spinning_curved"}
COLOR = {"|": "tab:blue", "()": "tab:green", ")": "tab:orange"}


def describe(d):
    print(f"\n{d.label}")
    print(f"  mass {d.mass * 1e3:.0f} g  wing area {d.wing_area * 1e4:.0f} cm^2  span {d.span * 1e3:.0f} mm"
          f"  chord {CHORD * 1e3:.1f} mm  aspect ratio {d.aspect:.1f}  avg sweep"
          f" {np.degrees(np.arccos(d.cos_sweep)):.0f} deg")
    if d.spins:
        print(f"  tip radius {d.r_tip * 1e3:.0f} mm  spin inertia {d.inertia:.2e} kg m^2"
              f"  blade pitch {BLADE_PITCH_DEG:.0f} deg")
    else:
        cl, cd = glider_polar(d, d.alpha_trim, 8.0)
        print(f"  trims at {np.degrees(d.alpha_trim):.1f} deg AoA, CL {cl:.2f}, best glide ratio"
              f" ~{cl / cd:.1f} : 1 (enclosure is {BOX_CD * BOX_FRONTAL / d.wing_area / cd:.0%} of drag)")


def plot_design(d, still, windy):
    h = HEIGHTS_FT
    panels = 3 if d.spins else 2
    fig, ax = plt.subplots(1, panels, figsize=(5.3 * panels, 4.8))
    ax[0].plot(h, [r["distance_ft"] for r in still], color=COLOR[d.shape], lw=2, label="still air")
    ax[0].plot(h, [r["distance_ft"] for r in windy], color=COLOR[d.shape], lw=1.6, ls="--",
               label=f"{WIND_MPH:.0f} mph tailwind")
    ax[0].set(xlabel="Drop height (ft)", ylabel="Horizontal distance (ft)", title="Distance vs drop height")
    ax[0].legend()
    for h_ft in (1000, 2500, 5000):
        r = simulate(d, h_ft * nc.FT)
        ax[1].plot(r["x"] / nc.FT, r["z"] / nc.FT, label=f"{h_ft} ft drop")
    ax[1].set(xlabel="Downrange (ft)", ylabel="Altitude (ft)", title="Flight paths (still air)")
    ax[1].legend()
    if d.spins:
        r = simulate(d, HEIGHTS_FT[-1] * nc.FT)
        rpm = r["w"] * 60 / (2 * np.pi)
        early = r["t"] <= max(4 * r["t"][np.argmax(rpm >= 0.9 * rpm[-1])], 5)
        ax[2].plot(r["t"][early], rpm[early], color="tab:purple", lw=2)
        ax[2].set(xlabel="Time after release (s)", ylabel="Spin rate (rpm)", title="Spin-up (5000 ft drop)")
        tw = ax[2].twinx()
        tw.plot(r["t"][early], 0.5 * d.inertia * r["w"][early] ** 2, color="tab:green", ls="--")
        tw.set_ylabel("Rotational KE (J)", color="tab:green")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Proper wings: {d.label}  |  {d.mass * 1e3:.0f} g, chord {CHORD * 1e3:.0f} mm, "
                 f"AR {d.aspect:.1f}")
    fig.tight_layout()
    fig.savefig(OUT / f"{SLUG[d.shape]}_proper_wing.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for d in DESIGNS:
        describe(d)
        still, windy = sweep(d), sweep(d, WIND)
        results[d.shape] = (still, windy)
        with open(OUT / f"{SLUG[d.shape]}_proper_wing.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["height_ft", "distance_ft", f"distance_ft_{WIND_MPH:.0f}mph", "descent_time_s",
                        "landing_spin_rpm", "landing_rot_ke_J"])
            for a, b in zip(still, windy):
                w.writerow([a["height_ft"], round(a["distance_ft"], 1), round(b["distance_ft"], 1),
                            round(a["descent_time_s"], 1), round(a["landing_spin_rpm"]),
                            round(a["landing_rot_ke_J"], 3)])
        plot_design(d, still, windy)

    # current CAD (flat 1/16" strips) for reference
    cad = {"| CAD flat strips": cad_straight.sweep(HEIGHTS_FT), ") CAD flat strips": cad_curved.sweep(HEIGHTS_FT)}

    print(f"\n{'drop':>7} |" + "".join(f"{d.shape:>9}" for d in DESIGNS) + "   still air, ft downrange |"
          + "".join(f"{d.shape:>9}" for d in DESIGNS) + f"   {WIND_MPH:.0f} mph tailwind |"
          + "".join(f"{d.shape:>8}" for d in DESIGNS) + "   time aloft, s")
    for i, h_ft in enumerate(HEIGHTS_FT):
        if h_ft in (500, 1000, 2500, 5000):
            print(f"{h_ft:>4} ft |" + "".join(f"{results[d.shape][0][i]['distance_ft']:9.0f}" for d in DESIGNS)
                  + " " * 26 + "|" + "".join(f"{results[d.shape][1][i]['distance_ft']:9.0f}" for d in DESIGNS)
                  + " " * 20 + "|" + "".join(f"{results[d.shape][0][i]['descent_time_s']:8.0f}" for d in DESIGNS))

    with open(OUT / "proper_wing_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["height_ft"]
                   + [f"{SLUG[d.shape]}_distance_ft" for d in DESIGNS]
                   + [f"{SLUG[d.shape]}_distance_ft_{WIND_MPH:.0f}mph" for d in DESIGNS]
                   + [f"{SLUG[d.shape]}_time_s" for d in DESIGNS]
                   + ["cad_straight_distance_ft", "cad_curved_distance_ft"])
        for i, h_ft in enumerate(HEIGHTS_FT):
            w.writerow([h_ft]
                       + [round(results[d.shape][0][i]["distance_ft"], 1) for d in DESIGNS]
                       + [round(results[d.shape][1][i]["distance_ft"], 1) for d in DESIGNS]
                       + [round(results[d.shape][0][i]["descent_time_s"], 1) for d in DESIGNS]
                       + [round(rows[i]["distance_ft"], 1) for rows in cad.values()])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.2))
    for d in DESIGNS:
        still, windy = results[d.shape]
        for a in (a1, a2):
            a.plot(HEIGHTS_FT, [r["distance_ft"] for r in still], color=COLOR[d.shape], lw=2.2,
                   label=f"{d.label}, still air")
            a.plot(HEIGHTS_FT, [r["distance_ft"] for r in windy], color=COLOR[d.shape], lw=1.5, ls="--",
                   label=f"{d.label}, {WIND_MPH:.0f} mph wind")
        a3.plot(HEIGHTS_FT, [r["descent_time_s"] for r in still], color=COLOR[d.shape], lw=2.2, label=d.label)
    for name, rows in cad.items():
        for a in (a1, a2):
            a.plot(HEIGHTS_FT, [r["distance_ft"] for r in rows], color="gray", lw=1.2,
                   ls="-" if name.startswith("|") else ":", label=f"{name} (current CAD), still air")
        a3.plot(HEIGHTS_FT, [r["descent_time_s"] for r in rows], color="gray", lw=1.2,
                ls="-" if name.startswith("|") else ":", label=f"{name} (current CAD)")
    a1.set(title="Distance traveled (full scale)", xlabel="Drop height (ft)", ylabel="Horizontal distance (ft)")
    a2.set(title="Distance traveled (log scale)", xlabel="Drop height (ft)", ylabel="Horizontal distance (ft)",
           yscale="log", ylim=(10, None))
    a3.set(title="Time in the air", xlabel="Drop height (ft)", ylabel="Descent time (s)")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7.5)
    a3.legend(fontsize=7.5)
    fig.suptitle(f"Nimbus with proper wings: |  vs  ()  vs  )   (airfoil, {CHORD * 1e3:.0f} mm chord, "
                 f"{WING_AREAL_DENSITY:.1f} kg/m² wings; CAD enclosure, electronics = {nc.ELECTRONICS_KG * 1e3:.0f} g)")
    fig.tight_layout()
    fig.savefig(OUT / "proper_wing_comparison.png", dpi=150)
    print(f"\nsaved {len(DESIGNS) * 2 + 2} files to results/proper_wings/")


if __name__ == "__main__":
    main()
