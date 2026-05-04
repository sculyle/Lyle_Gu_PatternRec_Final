# Mean Shift Clustering Demo

Teaching figures for Mean Shift clustering, built for ECE 5363 Pattern Recognition at Texas Tech University.

Two scripts — one on synthetic data, one on the Palmer Penguins dataset.

## Install

```bash
pip install numpy matplotlib scikit-learn seaborn
```

## Scripts

### `mean_shift_demo.py` — Synthetic data, step by step

Generates 5 figures that walk through how Mean Shift works:

| Figure | What it shows |
|--------|--------------|
| `fig_ms_1_data.png` | Raw unlabeled data — no group info given |
| `fig_ms_2_mechanism.png` | One step: place a window, find the mean, shift toward it |
| `fig_ms_3_convergence.png` | Repeat for every point until each one reaches a density peak |
| `fig_ms_4_result.png` | Final cluster assignments — 3 found, no K specified |
| `fig_ms_5_bandwidth.png` | Bandwidth sweep: too small / well-tuned / too large |

```bash
python mean_shift_demo.py
```

---

### `mean_shift_penguins.py` — Real data (Palmer Penguins)

Applies Mean Shift to 333 real penguin measurements (bill length vs bill depth). The algorithm finds the 3 species groups without being told how many to look for.

| Figure | What it shows |
|--------|--------------|
| `fig_ms_penguins_1_raw.png` | Raw bill measurements, no labels |
| `fig_ms_penguins_2_result.png` | Mean Shift clusters vs actual species (side by side) |
| `fig_ms_penguins_3_bandwidth.png` | Bandwidth sweep on real data |

```bash
python mean_shift_penguins.py
```

The penguins dataset is downloaded automatically by seaborn on first run.

## How Mean Shift works (in one paragraph)

Pick any point. Place a circular window of radius *h* (the bandwidth) around it. Compute the mean of all points inside the window. Move the point to that mean. Repeat for the same point until it stops moving — it has found a local density peak (a *mode*). Do this for every point in the dataset. Any two points that converge to the same mode belong to the same cluster. The number of clusters is determined by the data, not by a parameter you set.
