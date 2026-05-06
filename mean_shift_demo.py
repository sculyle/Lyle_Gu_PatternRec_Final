"""
Mean Shift Demo — Step-by-step teaching figures
ECE 5363 Pattern Recognition, Project 5

Generates 6 figures saved to the same directory as this script:
  fig_ms_1_data.png        — raw unlabeled data
  fig_ms_2_mechanism.png   — one step: window, mean, shift
  fig_ms_windows.png       — watch the window move across 4 iterations
  fig_ms_3_convergence.png — paths of 20 points converging to modes
  fig_ms_4_result.png      — final cluster assignments
  fig_ms_5_bandwidth.png   — effect of bandwidth (too small / good / too large)

Requirements: numpy, matplotlib, scikit-learn
  pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED    = 42
GREY    = 'lightgrey'
ORANGE  = '#DD8452'
COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def base_scatter(ax, X):
    ax.scatter(X[:, 0], X[:, 1], color=GREY, s=32, zorder=1, alpha=0.9)


def no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def ms_trajectory(X, start, bandwidth, max_iter=50, tol=1e-4):
    path = [start.copy()]
    pt = start.copy()
    for _ in range(max_iter):
        within = np.linalg.norm(X - pt, axis=1) < bandwidth
        if within.sum() == 0:
            break
        new_pt = X[within].mean(axis=0)
        path.append(new_pt.copy())
        if np.linalg.norm(new_pt - pt) < tol:
            break
        pt = new_pt
    return np.array(path)


X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=SEED)
bw_auto = estimate_bandwidth(X, quantile=0.2, random_state=SEED)
bw_good = bw_auto * 2.8   # gives 3 well-separated clusters

ms = MeanShift(bandwidth=bw_good, bin_seeding=True)
ms.fit(X)


# ── Figure 1: Raw data ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], color='steelblue', s=32, zorder=1)
ax.set_title("The Data\n(no labels — how many groups are there?)", fontsize=12)
no_ticks(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_1_data.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_1_data.png")


# ── Figure 2: One step ─────────────────────────────────────────

point   = X[np.argmax(X[:, 0])]
inside  = np.linalg.norm(X - point, axis=1) < bw_auto
mean_pt = X[inside].mean(axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_aspect('equal', adjustable='datalim')
base_scatter(ax, X)
ax.scatter(X[inside, 0], X[inside, 1], color=ORANGE, s=45, zorder=2, label='Inside window')
ax.scatter(*point,   color='black', s=120, zorder=4, label='Current point')
ax.scatter(*mean_pt, color='red',   s=200, marker='*', zorder=5, label='Mean  →  shift here')
circle = plt.Circle(point, bw_auto, fill=False, color='black',
                    linewidth=1.5, linestyle='--', zorder=3)
ax.add_patch(circle)
ax.annotate('', xy=mean_pt, xytext=point,
            arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax.set_title("One Step: Place Window → Find Mean → Shift", fontsize=12)
ax.legend(loc='upper left', fontsize=9)
no_ticks(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_2_mechanism.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_2_mechanism.png")


# ── Figure 3 (windows): Watch the window move ─────────────────
# Tracks the single point farthest from its final mode so the
# window visibly travels across 4 iterations before settling.

# One tracked point per cluster, each starting at the cluster's edge
# (farthest from its mode). Shows all three windows moving in parallel
# and converging to three distinct modes — that's how K is determined.

track_per_cluster = []
for lab in sorted(set(ms.labels_)):
    idx = np.where(ms.labels_ == lab)[0]
    dists = np.linalg.norm(X[idx] - ms.cluster_centers_[lab], axis=1)
    # 75th percentile: clearly inside the cluster but still far enough to show movement
    p75 = idx[np.argsort(dists)[int(len(idx) * 0.75)]]
    track_per_cluster.append(p75)

trajs = [ms_trajectory(X, X[t].copy(), bw_auto) for t in track_per_cluster]

# Consistent square axes so circles render round in every panel
margin = 2.0
x_c    = (X[:, 0].min() + X[:, 0].max()) / 2
y_c    = (X[:, 1].min() + X[:, 1].max()) / 2
half   = max(X[:, 0].max() - X[:, 0].min(),
             X[:, 1].max() - X[:, 1].min()) / 2 + margin
xlim   = (x_c - half, x_c + half)
ylim   = (y_c - half, y_c + half)

panel_snaps  = [0, 1, 2, None]   # None = show converged state
panel_titles = ['Iteration 0', 'Iteration 1', 'Iteration 2',
                f'Converged — {len(trajs)} modes found\n= {len(trajs)} clusters']

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Three Windows — Three Paths — Three Clusters",
             fontsize=12, fontweight='bold')

for ax, snap, title in zip(axes, panel_snaps, panel_titles):
    base_scatter(ax, X)

    for lab, traj in enumerate(trajs):
        color = COLORS[lab % len(COLORS)]

        if snap is None:                        # converged panel
            mode = ms.cluster_centers_[lab]     # authoritative mode position
            ax.scatter(*mode, color=color, s=140, zorder=4,
                       edgecolors='black', linewidths=0.8)
            ax.scatter(*mode, color='red', s=320, marker='*', zorder=5)
        else:
            it      = min(snap, len(traj) - 1)
            pt      = traj[it]
            inside  = np.linalg.norm(X - pt, axis=1) < bw_auto
            mean_pt = X[inside].mean(axis=0) if inside.any() else pt

            ax.add_patch(plt.Circle(pt, bw_auto, fill=False, color=color,
                                    linewidth=1.8, linestyle='--', zorder=3))
            ax.scatter(*pt,      color=color, s=130, zorder=4,
                       edgecolors='black', linewidths=0.8)
            ax.scatter(*mean_pt, color='red', s=220, marker='*', zorder=5)
            if np.linalg.norm(mean_pt - pt) > 0.05:
                ax.annotate('', xy=mean_pt, xytext=pt,
                            arrowprops=dict(arrowstyle='->', color=color, lw=2.2))

    ax.set_title(title, fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    no_ticks(ax)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_windows.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_windows.png")


# ── Figure: Same mode regardless of starting position ─────────
# Two panels — left: fresh single cluster, right: one of the 3 existing
# clusters zoomed in. Both show 8 starting points spread around the
# perimeter, each with an arrow to the mode, proving they all land
# in the same place regardless of where they began.

def angular_starts(X_pts, center, n=8):
    """Return one point per angular sector (farthest from center in each)."""
    angles = np.arctan2(X_pts[:, 1] - center[1], X_pts[:, 0] - center[0])
    pts = []
    for i in range(n):
        lo = -np.pi + i * (2 * np.pi / n)
        hi = lo + (2 * np.pi / n)
        mask = (angles >= lo) & (angles < hi)
        if mask.any():
            dists = np.linalg.norm(X_pts[mask] - center, axis=1)
            pts.append(X_pts[mask][np.argmax(dists)])
    return pts

PT_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
             '#8172B2', '#937860', '#DA8BC3', '#64B5CD']

# Fresh single-cluster dataset
X_one, _ = make_blobs(n_samples=60, centers=[[0, 0]],
                      cluster_std=1.0, random_state=SEED + 1)
bw_one  = estimate_bandwidth(X_one, quantile=0.35, random_state=SEED)
mode_one = MeanShift(bandwidth=bw_one).fit(X_one).cluster_centers_[0]

# Zoom into the most spread existing cluster
spread   = [X[ms.labels_ == lab].std() for lab in range(len(ms.cluster_centers_))]
zoom_lab = int(np.argmax(spread))
mode_z   = ms.cluster_centers_[zoom_lab]
pad_z    = 3.5
xlim_z   = (mode_z[0] - pad_z, mode_z[0] + pad_z)
ylim_z   = (mode_z[1] - pad_z, mode_z[1] + pad_z)

starts_one  = angular_starts(X_one, mode_one)
starts_zoom = angular_starts(X[ms.labels_ == zoom_lab], mode_z)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Different Starting Positions — Same Mode",
             fontsize=12, fontweight='bold')

panels = [
    dict(X_bg=X_one, X_traj=X_one, bw=bw_one,
         starts=starts_one, mode=mode_one,
         xlim=None, title="Fresh single cluster"),
    dict(X_bg=X,    X_traj=X,    bw=bw_auto,
         starts=starts_zoom, mode=mode_z,
         xlim=(xlim_z, ylim_z),
         title="One cluster from the 3-cluster data\n(zoomed in)"),
]

for ax, p in zip(axes, panels):
    ax.scatter(p['X_bg'][:, 0], p['X_bg'][:, 1],
               color=GREY, s=32, zorder=1, alpha=0.9)

    for i, start in enumerate(p['starts']):
        color = PT_COLORS[i % len(PT_COLORS)]
        traj  = ms_trajectory(p['X_traj'], start.copy(), p['bw'])
        ax.scatter(*start, color=color, s=90, zorder=4,
                   edgecolors='black', linewidths=0.5)
        if np.linalg.norm(traj[-1] - start) > 0.05:
            ax.annotate('', xy=traj[-1], xytext=start,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, alpha=0.85))

    ax.scatter(*p['mode'], color='red', s=380, marker='*', zorder=5)
    if p['xlim']:
        ax.set_xlim(p['xlim'][0])
        ax.set_ylim(p['xlim'][1])
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(p['title'], fontsize=10)
    no_ticks(ax)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_single_mode.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_single_mode.png")


# ── Figure parallel: All points shift at the same time ────────
# Shows 2 representative points per cluster simultaneously, each
# with its own window and arrow — answers "does it do one cluster
# at a time?" (no — all 150 points run independently in parallel).

bw_par = bw_auto   # same bandwidth as one-step figure for consistency

# For each cluster, pick the 2 points with the largest one-step shift —
# this guarantees the arrows will be long enough to see clearly.
tracked_par = []
tracked_par_labs = []
for lab in sorted(set(ms.labels_)):
    idx = np.where(ms.labels_ == lab)[0]
    shifts = []
    for i in idx:
        inside = np.linalg.norm(X - X[i], axis=1) < bw_par
        mean_i = X[inside].mean(axis=0) if inside.any() else X[i]
        shifts.append(np.linalg.norm(mean_i - X[i]))
    order = idx[np.argsort(shifts)[::-1]]   # largest shift first
    tracked_par.append(order[0])
    tracked_par.append(order[1])
    tracked_par_labs.extend([lab, lab])

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal', adjustable='datalim')
base_scatter(ax, X)

for pt_idx, lab in zip(tracked_par, tracked_par_labs):
    color  = COLORS[lab % len(COLORS)]
    pt     = X[pt_idx]
    inside = np.linalg.norm(X - pt, axis=1) < bw_par
    mean_pt = X[inside].mean(axis=0) if inside.any() else pt

    ax.add_patch(plt.Circle(pt, bw_par, fill=False, color=color,
                             linewidth=1.8, linestyle='--', zorder=3, alpha=0.85))
    ax.scatter(*pt,      color=color, s=130, zorder=4,
               edgecolors='black', linewidths=0.8)
    ax.scatter(*mean_pt, color='red', s=180, marker='*', zorder=5)
    ax.annotate('', xy=mean_pt, xytext=pt,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.2, alpha=0.9))

ax.set_title("All Points Shift at the Same Time\n"
             "(2 shown per cluster — all 150 run independently, no coordination)",
             fontsize=12)
no_ticks(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_parallel.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_parallel.png")


# ── Figure 4: Convergence paths (color = which cluster) ───────

rng        = np.random.RandomState(SEED)
sample_idx = rng.choice(len(X), size=20, replace=False)

fig, ax = plt.subplots(figsize=(6, 5))
base_scatter(ax, X)

for i in sample_idx:
    color  = COLORS[ms.labels_[i] % len(COLORS)]
    traj_i = ms_trajectory(X, X[i].copy(), bw_good)
    if len(traj_i) < 2:
        continue
    ax.plot(traj_i[:, 0], traj_i[:, 1], '-', color=color,
            alpha=0.55, linewidth=1.4, zorder=2)
    ax.annotate('', xy=traj_i[-1], xytext=traj_i[-2],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=3)

ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
           color='red', s=300, marker='*', zorder=5, label='Modes (density peaks)')
ax.set_title("Every Point Climbs to a Mode\n(color = which cluster it ends in)", fontsize=12)
ax.legend(fontsize=9)
no_ticks(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_3_convergence.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_3_convergence.png")


# ── Figure 5: Final clusters ──────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
for lab in sorted(set(ms.labels_)):
    mask = ms.labels_ == lab
    ax.scatter(X[mask, 0], X[mask, 1], color=COLORS[lab % len(COLORS)],
               s=40, label=f'Cluster {lab + 1}')
ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
           color='red', s=200, marker='x', linewidths=2.5, zorder=5, label='Modes')
n = len(ms.cluster_centers_)
ax.set_title(f"Result: {n} Clusters Found — No K Specified", fontsize=12)
ax.legend(fontsize=9)
no_ticks(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_4_result.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_4_result.png")


# ── Figure 6: Bandwidth sweep ─────────────────────────────────

configs = [
    (bw_auto * 0.35, "Too Small"),
    (bw_good,        "Well-Tuned"),
    (bw_auto * 7.0,  "Too Large"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (bw, desc) in zip(axes, configs):
    ms_b = MeanShift(bandwidth=bw, bin_seeding=True)
    ms_b.fit(X)
    n = len(ms_b.cluster_centers_)
    for lab in sorted(set(ms_b.labels_)):
        mask = ms_b.labels_ == lab
        ax.scatter(X[mask, 0], X[mask, 1], color=COLORS[lab % len(COLORS)], s=25, alpha=0.8)
    ax.scatter(ms_b.cluster_centers_[:, 0], ms_b.cluster_centers_[:, 1],
               color='red', s=100, marker='x', linewidths=2, zorder=3)
    ax.set_title(f"h = {bw:.2f}  ({desc})\n{n} cluster{'s' if n != 1 else ''} found", fontsize=11)
    no_ticks(ax)

fig.suptitle("The Only Parameter: Bandwidth (h)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_5_bandwidth.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_5_bandwidth.png")

print("\nDone. 6 figures saved.")
