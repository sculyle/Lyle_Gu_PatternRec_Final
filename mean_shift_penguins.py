"""
Mean Shift Real Data Demo — Palmer Penguins
ECE 5363 Pattern Recognition, Project 5

Features: bill_length_mm vs bill_depth_mm  (2D, no PCA needed)
These features cleanly separate all 3 species.
Clustering is done on standardized features; figures are plotted in original units.

Generates 3 figures saved to the same directory as this script:
  fig_ms_penguins_1_raw.png       — raw scatter of the two bill measurements
  fig_ms_penguins_2_result.png    — Mean Shift result vs ground-truth species
  fig_ms_penguins_3_bandwidth.png — bandwidth sweep (too small / good / too large)

Requirements: numpy, matplotlib, seaborn, scikit-learn
  pip install numpy matplotlib seaborn scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
CLUSTER_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
SPECIES_COLORS  = {'Adelie': '#4C72B0', 'Chinstrap': '#DD8452', 'Gentoo': '#55A868'}

# ── Load & prepare data ───────────────────────────────────────
df = sns.load_dataset('penguins').dropna()
X_raw = df[['bill_length_mm', 'bill_depth_mm']].values
species = df['species'].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

bw_auto = estimate_bandwidth(X, quantile=0.2, random_state=SEED)
bw_good = bw_auto * 0.9   # gives 3 well-separated clusters

ms = MeanShift(bandwidth=bw_good, bin_seeding=True)
ms.fit(X)
centers_orig = scaler.inverse_transform(ms.cluster_centers_)
n_found = len(centers_orig)


# ── Figure 1: Raw data ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_raw[:, 0], X_raw[:, 1], color='steelblue', s=30, alpha=0.7)
ax.set_xlabel("Bill Length (mm)", fontsize=11)
ax.set_ylabel("Bill Depth (mm)", fontsize=11)
ax.set_title("The Data: Palmer Penguins\n(333 penguins — no species labels given)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_penguins_1_raw.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_penguins_1_raw.png")


# ── Figure 2: Mean Shift result vs ground truth ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left — Mean Shift result
for lab in sorted(set(ms.labels_)):
    mask = ms.labels_ == lab
    axes[0].scatter(X_raw[mask, 0], X_raw[mask, 1],
                    color=CLUSTER_COLORS[lab % len(CLUSTER_COLORS)],
                    s=30, alpha=0.8, label=f'Cluster {lab + 1}')
axes[0].scatter(centers_orig[:, 0], centers_orig[:, 1],
                color='red', s=200, marker='x', linewidths=2.5, zorder=5, label='Modes')
axes[0].set_title(f"Mean Shift Result\n{n_found} clusters found  (h = {bw_good:.2f})", fontsize=11)
axes[0].set_xlabel("Bill Length (mm)", fontsize=10)
axes[0].set_ylabel("Bill Depth (mm)", fontsize=10)
axes[0].legend(fontsize=9)

# Right — ground truth species
for sp, color in SPECIES_COLORS.items():
    mask = species == sp
    axes[1].scatter(X_raw[mask, 0], X_raw[mask, 1],
                    color=color, s=30, alpha=0.8, label=sp)
axes[1].set_title("Ground Truth Species\n(for reference — not given to the algorithm)", fontsize=11)
axes[1].set_xlabel("Bill Length (mm)", fontsize=10)
axes[1].set_ylabel("Bill Depth (mm)", fontsize=10)
axes[1].legend(fontsize=9)

fig.suptitle("Mean Shift on Palmer Penguins", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_penguins_2_result.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_penguins_2_result.png")


# ── Figure 3: Bandwidth sweep ─────────────────────────────────
configs = [
    (bw_auto * 0.4, "Too Small"),
    (bw_good,       "Well-Tuned"),
    (bw_auto * 2.5, "Too Large"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (bw, desc) in zip(axes, configs):
    ms_b = MeanShift(bandwidth=bw, bin_seeding=True)
    ms_b.fit(X)
    centers_b = scaler.inverse_transform(ms_b.cluster_centers_)
    n = len(centers_b)
    for lab in sorted(set(ms_b.labels_)):
        mask = ms_b.labels_ == lab
        ax.scatter(X_raw[mask, 0], X_raw[mask, 1],
                   color=CLUSTER_COLORS[lab % len(CLUSTER_COLORS)], s=20, alpha=0.8)
    ax.scatter(centers_b[:, 0], centers_b[:, 1],
               color='red', s=100, marker='x', linewidths=2, zorder=3)
    ax.set_title(f"h = {bw:.2f}  ({desc})\n{n} cluster{'s' if n != 1 else ''} found", fontsize=10)
    ax.set_xlabel("Bill Length (mm)", fontsize=9)
    ax.set_ylabel("Bill Depth (mm)", fontsize=9)

fig.suptitle("Effect of Bandwidth on Penguin Clustering", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ms_penguins_3_bandwidth.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_ms_penguins_3_bandwidth.png")

print(f"\nDone.  bw_auto={bw_auto:.3f}  bw_good={bw_good:.3f}  clusters found={n_found}")
