"""Overlay the straight-wing and spinning curved-wing Nimbus gliders.

Runs both simulations over the same drop heights, in still air and with a
steady tailwind, and plots horizontal distance and time aloft on shared axes.

Run:  python3 compare_gliders.py
Out:  results/glider_comparison.csv, results/glider_comparison.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import curved_wing_sim as curved
import nimbus_common as nc
import straight_wing_sim as straight

WIND_MPH = 10.0          # tailwind for the dashed "windy day" curves
WIND = WIND_MPH * 0.44704


def main():
    straight.describe()
    curved.describe()
    runs = {
        ("straight", 0.0): straight.sweep(),
        ("straight", WIND): straight.sweep(wind=WIND),
        ("curved", 0.0): curved.sweep(),
        ("curved", WIND): curved.sweep(wind=WIND),
    }

    h = nc.HEIGHTS_FT
    col = lambda key, field: np.array([r[field] for r in runs[key]])
    nc.RESULTS_DIR.mkdir(exist_ok=True)
    with open(nc.RESULTS_DIR / "glider_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["height_ft",
                    "straight_distance_ft", "curved_distance_ft",
                    f"straight_distance_ft_{WIND_MPH:.0f}mph", f"curved_distance_ft_{WIND_MPH:.0f}mph",
                    "straight_time_s", "curved_time_s"])
        for i, h_ft in enumerate(h):
            w.writerow([h_ft] + [round(col(k, "distance_ft")[i], 1) for k in
                                 (("straight", 0.0), ("curved", 0.0), ("straight", WIND), ("curved", WIND))]
                       + [round(col((m, 0.0), "descent_time_s")[i], 1) for m in ("straight", "curved")])

    print(f"\n{'height':>8} | {'straight':>10} {'curved':>10}   still air (ft downrange)"
          f" | {'straight':>10} {'curved':>10}   {WIND_MPH:.0f} mph tailwind")
    for i, h_ft in enumerate(h):
        if h_ft in (500, 1000, 2000, 3000, 4000, 5000):
            print(f"{h_ft:>6} ft | {col(('straight', 0.0), 'distance_ft')[i]:10.0f}"
                  f" {col(('curved', 0.0), 'distance_ft')[i]:10.0f}                          "
                  f" | {col(('straight', WIND), 'distance_ft')[i]:10.0f}"
                  f" {col(('curved', WIND), 'distance_ft')[i]:10.0f}")

    colors = {"straight": "tab:blue", "curved": "tab:orange"}
    labels = {"straight": "Straight wing (glides)", "curved": "Curved wing (spins)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in ("straight", "curved"):
        a1.plot(h, col((m, 0.0), "distance_ft"), color=colors[m], lw=2.2, label=f"{labels[m]}, still air")
        a1.plot(h, col((m, WIND), "distance_ft"), color=colors[m], lw=1.6, ls="--",
                label=f"{labels[m]}, {WIND_MPH:.0f} mph tailwind")
        a2.plot(h, col((m, 0.0), "descent_time_s"), color=colors[m], lw=2.2, label=labels[m])
    a1.set(xlabel="Drop height (ft)", ylabel="Horizontal distance traveled (ft)",
           title="Distance traveled vs drop height")
    a2.set(xlabel="Drop height (ft)", ylabel="Time in the air (s)", title="Descent time vs drop height")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Nimbus gliders: straight wing vs spinning curved wing "
                 f"(geometry & mass from CAD, electronics = {nc.ELECTRONICS_KG * 1e3:.0f} g)")
    fig.tight_layout()
    fig.savefig(nc.RESULTS_DIR / "glider_comparison.png", dpi=150)
    print("\nsaved results/glider_comparison.csv and results/glider_comparison.png")


if __name__ == "__main__":
    main()
