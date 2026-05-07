"""
DBSCAN clustering demo for Pattern Recognition course.
All labels in English (for PPT use).

Design notes:
- Figs 1-4 share the SAME dataset (300 moons + 25 outliers) for fair comparison
- Every clustering plot explicitly distinguishes Core / Border / Noise points
  with different colors AND different markers
- Fig 5 uses different datasets to compare DBSCAN vs K-Means

Run from the same folder as this script:
    python dbscan_demo_unified.py
Outputs 5 PNG files into the same folder.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Shared dataset for figures 1-4: moons + outliers
# ============================================================
np.random.seed(42)
X_clean, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
np.random.seed(7)
outliers = np.random.uniform(low=-1.5, high=2.5, size=(25, 2))
X = np.vstack([X_clean, outliers])
print(f"Shared dataset for Figs 1-4: {X.shape[0]} points "
      f"({X_clean.shape[0]} moons + {outliers.shape[0]} outliers)")


# ============================================================
# Helper: classify every point as core / border / noise
# ============================================================
def classify_points(X, eps, minPts):
    """Return three boolean masks: core, border, noise."""
    nbrs = NearestNeighbors(radius=eps).fit(X)
    neighbors = nbrs.radius_neighbors(X, return_distance=False)
    n_neighbors = np.array([len(n) for n in neighbors])  # includes self
    is_core = n_neighbors >= minPts

    is_border = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if not is_core[i]:
            # Border = at least one core point in its neighborhood
            if any(is_core[j] for j in neighbors[i] if j != i):
                is_border[i] = True
    is_noise = ~is_core & ~is_border
    return is_core, is_border, is_noise


# ============================================================
# Helper: plot a clustering result with explicit point types
# ============================================================
def plot_clustering(ax, X, labels, eps, minPts, title,
                    show_legend=True, show_eps_circle=False):
    """
    Plot clustering result with:
      - Core points: filled circles, colored by cluster
      - Border points: triangles, colored by cluster but lighter edge
      - Noise points: gray X markers
    """
    is_core, is_border, is_noise = classify_points(X, eps, minPts)
    cluster_palette = ['#E63946', '#06A77D', '#1D3557', '#F4A261', '#9B5DE5']

    # Plot core points (filled circles, by cluster color)
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue  # noise handled separately
        mask = (labels == lbl) & is_core
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=cluster_palette[lbl % len(cluster_palette)],
                       s=45, marker='o', alpha=0.85,
                       edgecolor='black', linewidth=0.4,
                       label=f'Core (cluster {lbl})' if show_legend else None)

    # Plot border points (triangle marker, by cluster color)
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        mask = (labels == lbl) & is_border
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=cluster_palette[lbl % len(cluster_palette)],
                       s=70, marker='^', alpha=0.85,
                       edgecolor='black', linewidth=0.8,
                       label=f'Border (cluster {lbl})' if show_legend else None)

    # Plot noise points (gray X)
    if is_noise.any():
        ax.scatter(X[is_noise, 0], X[is_noise, 1],
                   c='lightgray', s=70, marker='X', alpha=0.95,
                   edgecolor='black', linewidth=0.8,
                   label='Noise' if show_legend else None)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3); ax.set_aspect('equal')
    if show_legend:
        ax.legend(loc='lower right', fontsize=8, framealpha=0.95)


# ============================================================
# FIGURE 1: Raw data only (no clustering applied)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(X[:, 0], X[:, 1], c='#4A4E69', s=35, alpha=0.7,
           edgecolor='white', linewidth=0.4)
ax.set_title(f"Raw Input Data\n({len(X)} points: {len(X_clean)} moons + {len(outliers)} outliers)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Feature 1", fontsize=12)
ax.set_ylabel("Feature 2", fontsize=12)
ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig1_raw_data.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved fig1_raw_data.png")

# ============================================================
# FIGURE 2: Core / Border / Noise illustration
# ============================================================
eps_demo = 0.15
minPts_demo = 5
is_core, is_border, is_noise = classify_points(X, eps_demo, minPts_demo)

fig, ax = plt.subplots(figsize=(9, 7))

# Core (green circles)
ax.scatter(X[is_core, 0], X[is_core, 1],
           c='#06A77D', s=45, marker='o', alpha=0.85,
           edgecolor='black', linewidth=0.4,
           label=f'Core ({is_core.sum()})')
# Border (orange triangles)
ax.scatter(X[is_border, 0], X[is_border, 1],
           c='#F4A261', s=85, marker='^', alpha=0.95,
           edgecolor='black', linewidth=0.8,
           label=f'Border ({is_border.sum()})')
# Noise (red X)
ax.scatter(X[is_noise, 0], X[is_noise, 1],
           c='#E63946', s=85, marker='X', alpha=0.95,
           edgecolor='black', linewidth=0.8,
           label=f'Noise ({is_noise.sum()})')

# Show one example eps-neighborhood circle around a core point
core_indices = np.where(is_core)[0]
example_idx = core_indices[len(core_indices) // 2]
circle = plt.Circle(X[example_idx], eps_demo, fill=False,
                    color='#1D3557', linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.annotate(f'ε-neighborhood\n(eps={eps_demo})',
            xy=X[example_idx],
            xytext=(X[example_idx, 0] - 0.7, X[example_idx, 1] + 0.5),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1D3557', lw=1.5))

ax.set_title(f"DBSCAN Point Types  (eps={eps_demo}, minPts={minPts_demo})",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig2_point_types.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved fig2_point_types.png  "
      f"(core={is_core.sum()}, border={is_border.sum()}, noise={is_noise.sum()})")


# ============================================================
# FIGURE 3: K-distance graph (how to choose eps)
# ============================================================
k = 5  # = minPts
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, k - 1])

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(k_distances, color='#1D3557', linewidth=2.5)
suggested_eps = 0.15
ax.axhline(suggested_eps, color='#06A77D', linestyle='--', linewidth=2,
           label=f'Suggested eps ≈ {suggested_eps}')
# Mark the knee
knee_x = np.argmax(k_distances >= suggested_eps)
ax.scatter([knee_x], [suggested_eps], color='#E63946', s=150, zorder=10,
           label='Knee point', edgecolor='black', linewidth=1.5)
ax.set_xlabel("Points sorted by k-distance", fontsize=12)
ax.set_ylabel(f"Distance to {k}-th nearest neighbor", fontsize=12)
ax.set_title(f"K-Distance Graph for Choosing eps  (k = minPts = {k})",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig3_kdistance.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved fig3_kdistance.png")


# ============================================================
# FIGURE 4: Parameter Sensitivity (vary eps and minPts)
# Each subplot uses explicit core/border/noise visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Effect of DBSCAN Hyperparameters on Same Dataset",
             fontsize=16, fontweight='bold')

# --- Row 1: vary eps with minPts=5 fixed ---
eps_values = [0.05, 0.15, 0.50]
eps_labels = [
    "(too small: most points are noise)",
    "(good: 2 clean clusters)",
    "(too large: clusters merge)"
]
for i, (eps, label) in enumerate(zip(eps_values, eps_labels)):
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    is_c, is_b, is_n = classify_points(X, eps, 5)
    title = (f"eps={eps}, minPts=5\n"
             f"{n_clusters} clusters | core={is_c.sum()}, "
             f"border={is_b.sum()}, noise={is_n.sum()}\n{label}")
    plot_clustering(axes[0, i], X, db.labels_, eps, 5, title,
                    show_legend=(i == 0))

# --- Row 2: vary minPts with eps=0.15 fixed ---
minPts_values = [2, 5, 30]
minPts_labels = [
    "(too small: noise treated as clusters)",
    "(good)",
    "(too large: most points become noise)"
]
for i, (mp, label) in enumerate(zip(minPts_values, minPts_labels)):
    db = DBSCAN(eps=0.15, min_samples=mp).fit(X)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    is_c, is_b, is_n = classify_points(X, 0.15, mp)
    title = (f"eps=0.15, minPts={mp}\n"
             f"{n_clusters} clusters | core={is_c.sum()}, "
             f"border={is_b.sum()}, noise={is_n.sum()}\n{label}")
    plot_clustering(axes[1, i], X, db.labels_, 0.15, mp, title,
                    show_legend=(i == 0))

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig4_param_sensitivity.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved fig4_param_sensitivity.png")


# ============================================================
# FIGURE 5: DBSCAN vs K-Means on three challenging datasets
# (Different datasets here — purpose is to compare algorithms)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("DBSCAN vs K-Means on Challenging Datasets",
             fontsize=16, fontweight='bold')

# Dataset A: Moons + outliers (same family as our shared data, but fresh sample)
np.random.seed(42)
X_A_moon, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_A_out = np.random.uniform(low=-1.5, high=2.5, size=(30, 2))
X_A = np.vstack([X_A_moon, X_A_out])

# Dataset B: Concentric circles
X_B, _ = make_circles(n_samples=400, noise=0.05, factor=0.4, random_state=42)

# Dataset C: Varying densities
X_C, _ = make_blobs(n_samples=[300, 50, 50],
                    centers=[[0, 0], [4, 4], [-4, 4]],
                    cluster_std=[1.5, 0.3, 0.3],
                    random_state=42)

datasets = [
    (X_A, 0.20, 5, 2, "Moons + 30 outliers"),
    (X_B, 0.15, 5, 2, "Concentric circles"),
    (X_C, 0.50, 8, 3, "Varying densities"),
]
cluster_palette = ['#E63946', '#06A77D', '#1D3557', '#F4A261', '#9B5DE5']

for col, (Xd, eps, mp, k_for_kmeans, name) in enumerate(datasets):
    # --- Top row: K-Means ---
    km = KMeans(n_clusters=k_for_kmeans, n_init=10, random_state=42).fit(Xd)
    ax = axes[0, col]
    for kk in range(k_for_kmeans):
        ax.scatter(Xd[km.labels_ == kk, 0], Xd[km.labels_ == kk, 1],
                   c=cluster_palette[kk], s=25, alpha=0.7,
                   edgecolor='white', linewidth=0.3)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', s=200, marker='X', edgecolor='white', linewidth=2,
               label='Centroids', zorder=10)
    ax.set_title(f"K-Means (K={k_for_kmeans}) on {name}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')

    # --- Bottom row: DBSCAN with explicit core/border/noise ---
    db = DBSCAN(eps=eps, min_samples=mp).fit(Xd)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    is_c, is_b, is_n = classify_points(Xd, eps, mp)
    title = (f"DBSCAN (eps={eps}, minPts={mp}) on {name}\n"
             f"{n_clusters} clusters | core={is_c.sum()}, "
             f"border={is_b.sum()}, noise={is_n.sum()}")
    plot_clustering(axes[1, col], Xd, db.labels_, eps, mp, title,
                    show_legend=(col == 0))

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig5_dbscan_vs_kmeans.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved fig5_dbscan_vs_kmeans.png")

print("\nAll 5 figures saved successfully.")
