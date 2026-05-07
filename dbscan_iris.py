"""
DBSCAN clustering on the Iris dataset (Project 1 data).
Features used: Petal Length (meas_3) and Petal Width (meas_4),
chosen because between-class variance >> within-class variance
on these features.

Reads Proj1DataSet.xlsx from the SAME folder as this script.
Outputs a single 2x2 figure: dbscan_iris_demo.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# ============================================================
# Load data from the same folder as this script
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Proj1DataSet.xlsx")
df = pd.read_excel(data_path, sheet_name="Sheet1")

# meas_3 = Petal Length, meas_4 = Petal Width
X = df[["meas_3", "meas_4"]].values
species = df["species"].values

print(f"Loaded {len(X)} samples")
print(f"Species: {pd.Series(species).value_counts().to_dict()}")
print(f"Petal Length range: [{X[:,0].min():.1f}, {X[:,0].max():.1f}]")
print(f"Petal Width  range: [{X[:,1].min():.1f}, {X[:,1].max():.1f}]")

# ============================================================
# Choose DBSCAN hyperparameters
# eps chosen via k-distance heuristic (k = minPts = 5)
# ============================================================
EPS = 0.30
MIN_PTS = 5

# ============================================================
# Helper: classify every point as core / border / noise
# ============================================================
def classify_points(X, eps, minPts):
    nbrs = NearestNeighbors(radius=eps).fit(X)
    neighbors = nbrs.radius_neighbors(X, return_distance=False)
    n_neighbors = np.array([len(n) for n in neighbors])  # includes self
    is_core = n_neighbors >= minPts

    is_border = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if not is_core[i]:
            if any(is_core[j] for j in neighbors[i] if j != i):
                is_border[i] = True
    is_noise = ~is_core & ~is_border
    return is_core, is_border, is_noise


# ============================================================
# Run DBSCAN and classify every point
# ============================================================
db = DBSCAN(eps=EPS, min_samples=MIN_PTS).fit(X)
labels = db.labels_
is_core, is_border, is_noise = classify_points(X, EPS, MIN_PTS)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"\nDBSCAN result (eps={EPS}, minPts={MIN_PTS}):")
print(f"  Clusters: {n_clusters}")
print(f"  Core: {is_core.sum()}, Border: {is_border.sum()}, Noise: {is_noise.sum()}")

# ============================================================
# Pick three example points for Figure 2
# ============================================================
# Core example: a setosa center point (safely inside the dense cluster)
setosa_mask = (species == 'setosa')
setosa_core = np.where(setosa_mask & is_core)[0]
# Pick the one closest to setosa centroid for visual clarity
setosa_center = X[setosa_mask].mean(axis=0)
core_idx = setosa_core[np.argmin(
    np.linalg.norm(X[setosa_core] - setosa_center, axis=1))]

# Border example: a border point on the edge of the right cluster
border_indices = np.where(is_border)[0]
# Pick a border point that's clearly on the cluster periphery (max petal length)
border_idx = border_indices[np.argmax(X[border_indices, 0])]

# Noise example: a noise point
noise_indices = np.where(is_noise)[0]
if len(noise_indices) > 0:
    # Pick the noise point most isolated from the rest
    noise_idx = noise_indices[0]
else:
    # Fallback: shouldn't happen with these params, but just in case
    noise_idx = None

print(f"\nExample points for Figure 2:")
print(f"  Core   (idx={core_idx}): petal=({X[core_idx,0]:.1f}, {X[core_idx,1]:.1f}), "
      f"species={species[core_idx]}")
print(f"  Border (idx={border_idx}): petal=({X[border_idx,0]:.1f}, {X[border_idx,1]:.1f}), "
      f"species={species[border_idx]}")
if noise_idx is not None:
    print(f"  Noise  (idx={noise_idx}): petal=({X[noise_idx,0]:.1f}, {X[noise_idx,1]:.1f}), "
          f"species={species[noise_idx]}")

# ============================================================
# Build the 2x2 figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f"DBSCAN on Iris Dataset (Petal Length vs Petal Width, "
             f"eps={EPS}, minPts={MIN_PTS})",
             fontsize=15, fontweight='bold')

XLIM = (0.5, 7.5)
YLIM = (-0.1, 2.8)

# ------------------------------------------------------------
# FIGURE 1 (top-left): Raw data, all gray
# ------------------------------------------------------------
ax = axes[0, 0]
ax.scatter(X[:, 0], X[:, 1], c='#4A4E69', s=50, alpha=0.7,
           edgecolor='white', linewidth=0.5)
ax.set_title("(1) Raw Data\n(150 Iris samples, no labels)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# FIGURE 2 (top-right): Three example points with eps-circles
# ------------------------------------------------------------
ax = axes[0, 1]
# All other points in light gray
ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=40, alpha=0.6,
           edgecolor='white', linewidth=0.4)

# Draw eps-circle + highlight for each example
def draw_example(ax, point, eps, color, label, marker='o'):
    circle = plt.Circle(point, eps, fill=False, color=color,
                        linewidth=2.2, linestyle='--', zorder=5)
    ax.add_patch(circle)
    ax.scatter([point[0]], [point[1]], c=color, s=180, marker=marker,
               edgecolor='black', linewidth=1.5, zorder=10, label=label)

draw_example(ax, X[core_idx], EPS, '#06A77D',
             f'Core point\n(≥{MIN_PTS} neighbors in ε)', marker='o')
draw_example(ax, X[border_idx], EPS, '#F4A261',
             f'Border point\n(<{MIN_PTS} neighbors,\n but in a core\'s ε)',
             marker='^')
if noise_idx is not None:
    draw_example(ax, X[noise_idx], EPS, '#E63946',
                 f'Noise point\n(<{MIN_PTS} neighbors,\n no core nearby)',
                 marker='X')

ax.set_title(f"(2) Three Point Types Defined by DBSCAN\n"
             f"(ε-neighborhoods shown as dashed circles)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# FIGURE 3 (bottom-left): DBSCAN result with core/border/noise
# ------------------------------------------------------------
ax = axes[1, 0]
cluster_palette = ['#E63946', '#06A77D', '#1D3557', '#F4A261', '#9B5DE5']

# Core points (filled circles, by cluster color)
for lbl in sorted(set(labels)):
    if lbl == -1:
        continue
    mask = (labels == lbl) & is_core
    if mask.any():
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=cluster_palette[lbl % len(cluster_palette)],
                   s=55, marker='o', alpha=0.85,
                   edgecolor='black', linewidth=0.4,
                   label=f'Cluster {lbl} - Core ({mask.sum()})')

# Border points (triangles, by cluster color)
for lbl in sorted(set(labels)):
    if lbl == -1:
        continue
    mask = (labels == lbl) & is_border
    if mask.any():
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=cluster_palette[lbl % len(cluster_palette)],
                   s=90, marker='^', alpha=0.95,
                   edgecolor='black', linewidth=0.8,
                   label=f'Cluster {lbl} - Border ({mask.sum()})')

# Noise points (gray X)
if is_noise.any():
    ax.scatter(X[is_noise, 0], X[is_noise, 1],
               c='lightgray', s=90, marker='X', alpha=0.95,
               edgecolor='black', linewidth=0.8,
               label=f'Noise ({is_noise.sum()})')

ax.set_title(f"(3) DBSCAN Result: {n_clusters} clusters discovered\n"
             f"(Core ●, Border ▲, Noise ✗)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# FIGURE 4 (bottom-right): Ground truth species labels
# ------------------------------------------------------------
ax = axes[1, 1]
species_colors = {'setosa': '#E63946', 'versicolor': '#06A77D',
                  'virginica': '#1D3557'}
species_markers = {'setosa': 'o', 'versicolor': 's', 'virginica': 'D'}
for sp in ['setosa', 'versicolor', 'virginica']:
    mask = species == sp
    ax.scatter(X[mask, 0], X[mask, 1],
               c=species_colors[sp], s=55,
               marker=species_markers[sp], alpha=0.85,
               edgecolor='black', linewidth=0.4,
               label=f'{sp} ({mask.sum()})')

ax.set_title("(4) Ground Truth Species Labels\n(for comparison with panel 3)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(script_dir, "dbscan_iris_demo.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved figure: {out_path}")

# ============================================================
# Final analysis: cluster vs species composition
# ============================================================
from collections import Counter
print("\nCluster composition vs ground-truth species:")
for lbl in sorted(set(labels)):
    if lbl == -1:
        members = species[labels == -1]
        print(f"  Noise: {dict(Counter(members))}")
    else:
        members = species[labels == lbl]
        print(f"  Cluster {lbl}: {dict(Counter(members))}")
