#!/usr/bin/env python
"""Top-down 2D video of an evaluation, from the trajectories saved with eval_unity.py --eval_traj_path.

Every evaluation thread becomes one panel (one row per .npz file). Obstacle outlines are read from
layouts.json or the scene description next to the executable.

    python onpolicy/scripts/render_2d.py results/traj_iid.npz results/traj_ood.npz --out scenes.mp4

Sizes are to scale: robots are drawn with their 0.2 m collision radius, pedestrians with the 0.3 m
circle around their two leg colliders, and the shaded ring is the 0.6 m social zone used by the
intrusion metrics (darker when a robot is inside it).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection
from matplotlib.lines import Line2D

INK, INK2, SURF, WALL = "#0b0b0b", "#52514e", "#fcfcfb", "#8c8b86"
ROBOT, PED, GOOD, FAIL = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b"
R_ROBOT, R_PED, R_SOCIAL = 0.20, 0.30, 0.60
WALL_PAD = 0.06          # obstacles are drawn this much wider on each side, so that thin walls stay visible
T0 = 3                   # the first steps are observation padding
TRAIL = 8                # past positions drawn behind a robot
MARK = 5                 # steps a goal / collision marker stays on screen
LAYOUTS = json.load(open(Path(__file__).with_name("layouts.json")))
NAMES = {"iid": "In-distribution", "ood": "Unseen layout", "faster": "Rushing pedestrians", "denser": "Denser crowd", "rooms": "Unseen four-room layout"}

plt.rcParams.update({"font.size": 10, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})


def layout(scene):
    """(room size in metres, obstacle outlines, title) of the build a thread ran."""
    path = Path(str(scene))
    stem = re.sub(r"_n\d+$", "", path.stem)          # released builds end in their number of robots, _n<N>
    if stem == "ood_rooms":
        return LAYOUTS[stem]["size"], [np.array(p) for p in LAYOUTS[stem]["obstacles"]], NAMES["rooms"]
    parts = stem.split("_")                            # iid_1, ood_2, ood_3_faster, ...
    if stem in LAYOUTS or (len(parts) == 3 and "_".join(parts[:2]) in LAYOUTS):
        base = "_".join(parts[:2])
        key = parts[2] if len(parts) == 3 else parts[0]
        size, polys = LAYOUTS[base]["size"], [np.array(p) for p in LAYOUTS[base]["obstacles"]]
        if key in ("faster", "denser"):
            base = "iid_" + parts[1]                   # rushing / denser reuse the training layouts
            size, polys = LAYOUTS[base]["size"], [np.array(p) for p in LAYOUTS[base]["obstacles"]]
        return size, polys, f"{NAMES[key]} {parts[1]}"
    config = json.load(open(path.with_suffix(".json")))    # written by simulator/build.sh
    sx, sz = config["room_size"]
    polys = []
    for o in config["obstacles"] + walls(sx, sz):
        x, z = o["position"]
        if o.get("type", "box") == "cylinder":
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            polys.append(np.stack([x + o["size"][0] / 2 * np.cos(a), z + o["size"][0] / 2 * np.sin(a)], -1))
        else:
            hx, hz = o["size"][0] / 2, o["size"][1] / 2
            c, s = np.cos(np.radians(-o.get("rotation", 0))), np.sin(np.radians(-o.get("rotation", 0)))
            corners = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]])
            polys.append(corners @ np.array([[c, s], [-s, c]]) + [x, z])
    return max(sx, sz), polys, path.stem


def walls(sx, sz, t=0.1):
    return [{"position": [sx / 2, 0], "size": [t, sz]}, {"position": [-sx / 2, 0], "size": [t, sz]},
            {"position": [0, sz / 2], "size": [sx, t]}, {"position": [0, -sz / 2], "size": [sx, t]}]


def widen(polys):
    ring = WALL_PAD * np.stack([np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
                                np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))], -1)
    from scipy.spatial import ConvexHull
    out = []
    for q in polys:
        pts = (q[:, None, :] + ring[None]).reshape(-1, 2)
        out.append(pts[ConvexHull(pts).vertices])
    return out


def discs(ax, xy, radius, **kw):
    """Circles in metres (data units), so they stay to scale at any figure size."""
    xy = np.atleast_2d(xy)
    d = np.full(len(xy), 2 * radius)
    c = EllipseCollection(d, d, np.zeros(len(xy)), units="xy", offsets=xy, offset_transform=ax.transData, **kw)
    ax.add_collection(c)
    return c


def load(path, thread):
    z = np.load(path)
    r, g, h = z["robot"][thread], z["goal"][thread], z["human"][thread]
    yaw = np.radians(z["human_yaw"][thread])
    u = np.stack([np.sin(yaw), np.cos(yaw)], -1)
    # Within a control step the robots move before the pedestrians, so the robot position logged at
    # step t was tested for collisions against the pedestrians logged at step t-1: draw them together.
    h, u = np.concatenate([h[:1], h[:-1]]), np.concatenate([u[:1], u[:-1]])
    real = np.abs(h).max(-1) < 100                       # unused pedestrian slots are parked far away
    size, polys, title = layout(z["scene"][thread])
    return dict(r=r, g=g, h=np.where(real[..., None], h, 1e4), u=u, done=z["done"][thread],
                reward=z["reward"][thread], size=size, polys=widen(polys), title=title,
                n_ped=int(real[T0:].any(0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj", nargs="+", help=".npz files written with --eval_traj_path (one row each)")
    parser.add_argument("--out", required=True, help="output .mp4 or .gif")
    parser.add_argument("--start", type=float, default=0.0, help="first second to draw")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to draw (0: all)")
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed relative to real time")
    parser.add_argument("--panel", type=float, default=4.4, help="panel width in inches")
    args = parser.parse_args()

    rows = [[load(p, k) for k in range(len(np.load(p)["scene"]))] for p in args.traj]
    cols = max(len(r) for r in rows)
    T = min(len(d["r"]) for row in rows for d in row)
    t0 = T0 + int(round(args.start / 0.2))
    t1 = T if args.duration <= 0 else min(T, t0 + int(round(args.duration / 0.2)))

    legend_rows = 1 if cols > 1 else 2                  # the legend needs two lines under a single panel
    foot = 0.3 + 0.25 * legend_rows                     # inches below the panels
    fig, axes = plt.subplots(len(rows), cols, figsize=(args.panel * cols, args.panel * len(rows) + 0.35 + foot), squeeze=False)
    panels = []
    for i, row in enumerate(rows):
        for k in range(cols):
            ax = axes[i, k]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k >= len(row):
                continue
            d = row[k]
            half = d["size"] / 2 + 0.35
            ax.set_xlim(-half, half), ax.set_ylim(-half, half), ax.set_aspect("equal")
            ax.add_collection(PolyCollection(d["polys"], facecolors=WALL, edgecolors=WALL, linewidths=0.6, zorder=0.8))
            n = d["r"].shape[1]
            ax.set_title(f"{d['title']}  ·  {n} robots, {d['n_ped']} pedestrians", fontsize=10, loc="left", color=INK, pad=4)
            d["zone"] = discs(ax, d["h"][t0], R_SOCIAL, fc=PED, ec=PED, lw=0.8, alpha=0.2, zorder=2)
            d["links"] = LineCollection([], colors=ROBOT, lw=0.8, alpha=0.35, linestyles=(0, (2, 3)), zorder=1)
            ax.add_collection(d["links"])
            d["trail"] = discs(ax, np.full((n * TRAIL, 2), 1e4), 0.11, fc=ROBOT, ec="none", zorder=3)
            d["goals"] = ax.scatter(*d["g"][t0].T, s=95, marker="*", fc=SURF, ec=ROBOT, lw=1.1, zorder=4)
            d["peds"] = discs(ax, d["h"][t0], R_PED, fc=PED, ec="none", zorder=5)
            d["robots"] = discs(ax, d["r"][t0], R_ROBOT, fc=ROBOT, ec=SURF, lw=0.8, zorder=6)
            d["arrows"] = ax.quiver(*d["h"][t0].T, *d["u"][t0].T, color=SURF, angles="xy", scale_units="xy",
                                    scale=1 / 0.26, width=0.045, units="xy", headwidth=3.4, headlength=3.2,
                                    headaxislength=2.9, pivot="mid", zorder=7)
            d["reached"] = ax.scatter([], [], s=230, marker="*", fc=GOOD, ec=GOOD, zorder=7)
            d["failed"] = ax.scatter([], [], s=300, marker="X", fc=FAIL, ec=SURF, lw=1.4, zorder=8)
            panels.append(d)

    key = [Line2D([], [], marker="o", ls="", color=ROBOT, ms=7, label="Robot"),
           Line2D([], [], marker="*", ls="", mfc=SURF, mec=ROBOT, ms=11, label="Robot's goal"),
           Line2D([], [], marker="*", ls="", mfc=GOOD, mec=GOOD, ms=14, label="Goal reached"),
           Line2D([], [], marker="X", ls="", mfc=FAIL, mec=SURF, mew=1.2, ms=13, label="Collision"),
           Line2D([], [], marker="o", ls="", color=PED, ms=9, label="Pedestrian"),
           Line2D([], [], marker="o", ls="", mfc="#f6d3c4", mec=PED, mew=1, ms=14, label="Social zone")]
    fig.legend(handles=key, loc="lower center", ncol=6 // legend_rows, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.18 / fig.get_figheight()))
    clock = fig.text(0.99, 0.005, "", ha="right", fontsize=9, color=INK2)

    def step(i):
        t = t0 + i
        for d in panels:
            r, n = d["r"], d["r"].shape[1]
            # an episode that ended at step s respawns the robot at s+1: the trail starts after it
            ends = [np.flatnonzero(d["done"][T0:t, a]) + T0 for a in range(n)]
            start = np.array([max([T0] + [e + 1 for e in ends[a] if e < t]) for a in range(n)])
            pts, alpha = np.full((n, TRAIL, 2), 1e4), np.zeros((n, TRAIL))
            for a in range(n):
                s = max(start[a], t - TRAIL)
                if t > s:
                    pts[a, TRAIL - (t - s):] = r[s:t, a]
                    alpha[a, TRAIL - (t - s):] = 0.2 + 0.6 * np.arange(1, t - s + 1) / (t - s)
            d["trail"].set_offsets(pts.reshape(-1, 2))
            d["trail"].set_facecolor([matplotlib.colors.to_rgba(ROBOT, x) for x in alpha.ravel()])
            d["robots"].set_offsets(r[t])
            d["goals"].set_offsets(d["g"][t])
            d["links"].set_segments([[r[t, a], d["g"][t, a]] for a in range(n)])
            # episodes that ended in the last MARK steps: a green star on the goal or a red cross at the
            # position of the contact (reported one step after it)
            recent = [(a, e) for a in range(n) for e in ends[a] if t - MARK < e <= t]
            ok = [d["g"][e - 1, a] for a, e in recent if d["reward"][e, a] > 5]
            hit = [r[e - 1, a] for a, e in recent if d["reward"][e, a] < 0]
            d["reached"].set_offsets(np.array(ok) if ok else np.empty((0, 2)))
            d["failed"].set_offsets(np.array(hit) if hit else np.empty((0, 2)))
            h, u = d["h"][t], d["u"][t]
            inside = (np.linalg.norm(h[:, None] - r[t][None], axis=-1) < R_SOCIAL).any(1)
            d["peds"].set_offsets(h)
            d["zone"].set_offsets(h)
            d["zone"].set_alpha(None)
            d["zone"].set_facecolor([matplotlib.colors.to_rgba(PED, 0.5 if x else 0.16) for x in inside])
            d["zone"].set_edgecolor([INK if x else matplotlib.colors.to_rgba(PED, 0.55) for x in inside])
            d["arrows"].set_offsets(h)
            d["arrows"].set_UVC(u[:, 0], u[:, 1])
        clock.set_text(f"t = {(t - T0) * 0.2:5.1f} s   ({args.speed:g}x)")
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.35 / fig.get_figheight(), bottom=foot / fig.get_figheight(),
                        wspace=0.03, hspace=0.12)
    ani = animation.FuncAnimation(fig, step, frames=t1 - t0, interval=1000 / (5 * args.speed))
    fps = 5 * args.speed
    if args.out.endswith(".gif"):
        ani.save(args.out, writer=animation.PillowWriter(fps=fps), dpi=80)
    else:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg is needed for .mp4 output (or write a .gif)")
        ani.save(args.out, writer=animation.FFMpegWriter(fps=fps, bitrate=8000, extra_args=["-pix_fmt", "yuv420p"]), dpi=120)
    print(f"wrote {args.out}: {len(panels)} panels, {t1 - t0} steps ({(t1 - t0) * 0.2:.0f} s)")


if __name__ == "__main__":
    main()
