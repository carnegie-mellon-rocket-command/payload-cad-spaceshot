"""Curved-wing Nimbus glider: spinning (autorotating) descent from 0-5000 ft.

Model: 2-D point mass (x = downrange, z = altitude) plus the spin rate w about
the axis normal to the blades. The two curved blades are handled with
blade-element theory at one representative station (R_STATION of tip radius):
the air a blade sees there combines the fall speed V with the spin speed w*r,

    inflow angle   phi = atan(V / (w r))
    thrust         T = N * q_rel * A_blade * (CL cos phi + CD sin phi)
    spin torque    Q = N * q_rel * A_blade * r * (CL sin phi - CD cos phi)
    spin-up        I dw/dt = Q

so the rotor starts as a flat plate falling broadside, spins up, and settles
where the torque balances. Rotational kinetic energy 0.5*I*w^2 is tracked and
reported in the energy budget. Thrust is tilted THRUST_TILT_DEG off the flight
path, which is what makes it travel sideways instead of falling straight down.

Blade area, radius, mass and moment of inertia come from the CAD (the spin axis
is the blades' thickness axis). The aerodynamic behaviour comes from the
ASSUMPTIONS block below.

Run:  python3 curved_wing_sim.py
Out:  results/curved_wing_results.csv, results/curved_wing_sim.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import nimbus_common as nc

# ---- ASSUMPTIONS ---------------------------------------------------------------
BLADE_PITCH_DEG = 8.0    # effective blade pitch from the curved planform (drives the spin)
THRUST_TILT_DEG = 5.0    # thrust tilt off the flight path (sets the sideways glide)
R_STATION = 0.75         # blade-element station, fraction of tip radius
BOX_CD = 1.05            # drag coefficient of the sensor enclosure (bluff box)

# ---- Geometry, mass and inertia from the CAD ------------------------------------
enclosure = nc.load_part("PL27-SS-NIMBUS-SENSOR-ENCLOSURE", bulky=True)
blade = nc.load_part("PL27-SS-NIMBUS-CURVED-WING-HALF")

N_BLADES = 2
MASS = enclosure.mass + N_BLADES * blade.mass + nc.ELECTRONICS_KG
AXIS = blade.thickness_axis                           # spin axis = normal to the blades
LONG = int(np.argmax(blade.size))                     # blade runs root -> tip along this axis
MOUNT = enclosure.size.max() / 2                      # blade roots on the enclosure sides
BLADE_LEN = blade.size[LONG]
R_TIP = MOUNT + BLADE_LEN
R_EFF = R_STATION * R_TIP
DISK_AREA = np.pi * R_TIP ** 2
CHORD = blade.planform / BLADE_LEN
THICKNESS = blade.size[AXIS]
CENTROID_R = MOUNT + (blade.centroid[LONG] - blade.lo[LONG])
I_SPIN = (enclosure.inertia[AXIS, AXIS]
          + N_BLADES * (blade.inertia[AXIS, AXIS] + blade.mass * CENTROID_R ** 2))
BOX_AREA = np.prod(np.delete(enclosure.size, enclosure.thickness_axis))  # falls flat
PITCH = np.radians(BLADE_PITCH_DEG)
TILT = np.radians(THRUST_TILT_DEG)


def rotor(rho, speed, w):
    """Blade-element thrust (N) and spin torque (N m) at fall speed `speed`, spin `w`."""
    u = w * R_EFF
    w_rel = np.hypot(speed, u)
    phi = np.arctan2(speed, u)
    cl, cd = nc.plate_coeffs(phi - PITCH, w_rel, CHORD, THICKNESS)
    q = 0.5 * rho * w_rel ** 2 * N_BLADES * blade.planform
    thrust = q * (cl * np.cos(phi) + cd * np.sin(phi))
    torque = q * R_EFF * (cl * np.sin(phi) - cd * np.cos(phi))
    return thrust, torque


def derivs(t, s, wind):
    x, z, vx, vz, w = s
    vax = vx - wind
    speed = max(np.hypot(vax, vz), 1e-6)
    rho = nc.air_density(z)
    thrust, torque = rotor(rho, speed, w)
    back = -np.array([vax, vz]) / speed
    box = 0.5 * rho * speed ** 2 * BOX_CD * BOX_AREA
    fx, fz = (thrust * np.cos(TILT) + box) * back + thrust * np.sin(TILT) * nc.lift_unit(vax, vz, speed)
    return [vx, vz, fx / MASS, fz / MASS - nc.G, torque / I_SPIN]


def _ground(t, s, wind):
    return s[1]


_ground.terminal, _ground.direction = True, -1


def simulate(height_m, wind=0.0):
    """Drop from height_m with no spin; return distance, time, path, spin history."""
    if height_m <= 0:
        z0 = np.zeros(1)
        return {"distance": 0.0, "time": 0.0, "t": z0, "x": z0, "z": z0, "w": z0, "speed": z0}
    sol = solve_ivp(derivs, (0, 1e4), [0.0, height_m, nc.RELEASE_SPEED, 0.0, 0.0], args=(wind,),
                    events=_ground, method="LSODA", max_step=0.5, rtol=1e-7, atol=1e-9)
    return {"distance": sol.y[0, -1], "time": sol.t[-1], "t": sol.t, "x": sol.y[0], "z": sol.y[1],
            "w": sol.y[4], "speed": np.hypot(sol.y[2] - wind, sol.y[3])}


def energy_budget(r, height_m):
    pe = MASS * nc.G * height_m
    ke = 0.5 * MASS * r["speed"][-1] ** 2
    rot = 0.5 * I_SPIN * r["w"][-1] ** 2
    return pe, ke, rot, pe + 0.5 * MASS * nc.RELEASE_SPEED ** 2 - ke - rot


def sweep(heights_ft=nc.HEIGHTS_FT, wind=0.0):
    rows = []
    for h_ft in heights_ft:
        r = simulate(h_ft * nc.FT, wind)
        rows.append({"height_ft": h_ft, "distance_ft": r["distance"] / nc.FT,
                     "distance_m": r["distance"], "descent_time_s": r["time"],
                     "landing_spin_rpm": r["w"][-1] * 60 / (2 * np.pi),
                     "landing_rot_ke_J": 0.5 * I_SPIN * r["w"][-1] ** 2})
    return rows


def describe():
    print("Curved-wing spinning glider (from CAD)")
    print(f"  mass {MASS * 1e3:.1f} g   tip radius {R_TIP * 1e3:.0f} mm   blade area"
          f" {N_BLADES * blade.planform * 1e4:.1f} cm^2   disk area {DISK_AREA * 1e4:.0f} cm^2")
    print(f"  spin moment of inertia {I_SPIN:.3e} kg m^2  (enclosure"
          f" {enclosure.inertia[AXIS, AXIS]:.2e}, blades {I_SPIN - enclosure.inertia[AXIS, AXIS]:.2e})")


def main():
    describe()
    rows = sweep()
    nc.RESULTS_DIR.mkdir(exist_ok=True)
    with open(nc.RESULTS_DIR / "curved_wing_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    h_top = nc.HEIGHTS_FT[-1] * nc.FT
    top = simulate(h_top)
    pe, ke, rot, lost = energy_budget(top, h_top)
    rpm = top["w"] * 60 / (2 * np.pi)
    spun = top["t"][np.argmax(rpm >= 0.9 * rpm[-1])]
    print(f"  settles at ~{rpm[-1]:.0f} rpm, ~{top['speed'][-1]:.1f} m/s (90% spin-up after {spun:.1f} s)")
    print(f"  energy from {nc.HEIGHTS_FT[-1]} ft: {pe:.0f} J potential -> {ke:.2f} J translational KE"
          f" + {rot:.2f} J rotational KE at landing, {lost:.0f} J lost to drag")
    for r in rows:
        if r["height_ft"] in (1000, 2500, 5000):
            print(f"  from {r['height_ft']:>4} ft: {r['distance_ft']:7.0f} ft downrange,"
                  f" {r['descent_time_s']:5.0f} s in the air")

    h = np.array([r["height_ft"] for r in rows])
    d = np.array([r["distance_ft"] for r in rows])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    a1.plot(h, d, color="tab:orange", lw=2)
    a1.set(xlabel="Drop height (ft)", ylabel="Horizontal distance (ft)",
           title="Glide distance vs drop height")
    a1.grid(alpha=0.3)
    for h_ft in (1000, 2500, 5000):
        r = simulate(h_ft * nc.FT)
        a2.plot(r["x"] / nc.FT, r["z"] / nc.FT, label=f"{h_ft} ft drop")
    a2.set(xlabel="Downrange (ft)", ylabel="Altitude (ft)", title="Flight paths")
    a2.legend()
    a2.grid(alpha=0.3)
    early = top["t"] <= max(4 * spun, 5)
    a3.plot(top["t"][early], rpm[early], color="tab:purple", lw=2)
    a3.set(xlabel="Time after release (s)", ylabel="Spin rate (rpm)",
           title=f"Spin-up ({nc.HEIGHTS_FT[-1]} ft drop)")
    a3.grid(alpha=0.3)
    a4 = a3.twinx()
    a4.plot(top["t"][early], 0.5 * I_SPIN * top["w"][early] ** 2, color="tab:green", ls="--")
    a4.set_ylabel("Rotational KE (J)", color="tab:green")
    fig.suptitle(f"Curved-wing spinning Nimbus glider  |  {MASS * 1e3:.0f} g, I = {I_SPIN:.2e} kg·m², "
                 f"pitch {BLADE_PITCH_DEG:.0f}°, tilt {THRUST_TILT_DEG:.0f}°, still air")
    fig.tight_layout()
    fig.savefig(nc.RESULTS_DIR / "curved_wing_sim.png", dpi=150)
    print(f"  saved results/curved_wing_results.csv and results/curved_wing_sim.png")


if __name__ == "__main__":
    main()
