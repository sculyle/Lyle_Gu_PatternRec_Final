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

labels  = ms.labels_
centers = ms.cluster_centers_

modes_per_point = centers[labels]
dists_to_mode   = np.linalg.norm(X - modes_per_point, axis=1)
track_idx       = np.argsort(dists_to_mode)[-3]

traj       = ms_trajectory(X, X[track_idx].copy(), bw_auto)
snap_iters = [0, 1, 2, min(3, len(traj) - 2)]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Repeat the Same Step — Watch the Window Move",
             fontsize=12, fontweight='bold')

for ax, it in zip(axes, snap_iters):
    pt      = traj[it]
    inside  = np.linalg.norm(X - pt, axis=1) < bw_auto
    mean_pt = X[inside].mean(axis=0) if inside.any() else pt

    base_scatter(ax, X)
    ax.scatter(X[inside, 0], X[inside, 1], color=ORANGE, s=45, zorder=2)
    ax.scatter(*pt,      color='black', s=130, zorder=4)
    ax.scatter(*mean_pt, color='red',   s=220, marker='*', zorder=5)
    ax.add_patch(plt.Circle(pt, bw_auto, fill=False, color='black',
                             linewidth=1.8, linestyle='--', zorder=3))
    if np.linalg.norm(mean_pt - pt) > 0.05:
        ax.annotate('', xy=mean_pt, xytext=pt,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.2))
    ax.set_title(f"Iteration {it}", fontsize=10)
    no_ticks(ax)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_windows.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_windows.png")


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
