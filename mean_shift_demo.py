"""
Mean Shift Demo — Step-by-step teaching figures
ECE 5363 Pattern Recognition, Project 5

Generates 5 figures saved to the same directory as this script:
  fig_ms_1_data.png        — raw unlabeled data
  fig_ms_2_mechanism.png   — one step: window, mean, shift
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
SEED = 42

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=SEED)
bw_auto = estimate_bandwidth(X, quantile=0.2, random_state=SEED)
bw_good = bw_auto * 2.8  # gives 3 well-separated clusters


# ── Figure 1: Raw data ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], color='steelblue', s=40)
ax.set_title("The Data\n(no labels — we don't know how many groups there are)", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_1_data.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_1_data.png")


# ── Figure 2: One step of Mean Shift ─────────────────────────
# Use bw_auto (smaller window) so the window is clearly visible

point = X[np.argmax(X[:, 0])]  # rightmost point — on the edge of a blob
dists = np.linalg.norm(X - point, axis=1)
inside = dists < bw_auto
mean_pt = X[inside].mean(axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_aspect('equal', adjustable='datalim')

ax.scatter(X[:, 0], X[:, 1], color='lightgrey', s=35, zorder=1, label='All points')
ax.scatter(X[inside, 0], X[inside, 1], color='#DD8452', s=45, zorder=2, label='Points inside window')
ax.scatter(*point, color='black', s=120, zorder=4, label='Current point')
ax.scatter(*mean_pt, color='red', s=200, marker='*', zorder=5, label='Mean  →  shift here')

circle = plt.Circle(point, bw_auto, fill=False, color='black', linewidth=1.5, linestyle='--')
ax.add_patch(circle)
ax.annotate('', xy=mean_pt, xytext=point,
            arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

ax.set_title("One Step: Place Window → Find Mean → Shift", fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_2_mechanism.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_2_mechanism.png")


# ── Figure 3: Convergence paths ───────────────────────────────

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


rng = np.random.RandomState(SEED)
sample_idx = rng.choice(len(X), size=20, replace=False)

ms = MeanShift(bandwidth=bw_good, bin_seeding=True)
ms.fit(X)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], color='lightgrey', s=30, zorder=1)

for i in sample_idx:
    traj = ms_trajectory(X, X[i].copy(), bw_good)
    if len(traj) < 2:
        continue
    ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3, linewidth=1.0, zorder=2)
    ax.annotate('', xy=traj[-1], xytext=traj[-2],
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=3)

ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
           color='red', s=300, marker='*', zorder=5, label='Modes (density peaks)')

ax.set_title("Repeat Until Convergence\n(every point climbs toward a density peak)", fontsize=12)
ax.legend(fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_3_convergence.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_3_convergence.png")


# ── Figure 4: Final clusters ──────────────────────────────────

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

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
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_4_result.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_4_result.png")


# ── Figure 5: Bandwidth sweep ─────────────────────────────────

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
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("The Only Parameter: Bandwidth (h)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_5_bandwidth.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_5_bandwidth.png")

print("\nDone. 5 figures saved.")
