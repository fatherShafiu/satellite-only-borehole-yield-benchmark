import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# Figure 4: Spatial CV strategy illustration
rng = np.random.default_rng(42)
pts = rng.random((90, 2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

# (a) Random K-fold
ax = axes[0]
folds_random = rng.integers(0, 5, size=len(pts))
for f in range(5):
    m = folds_random == f
    ax.scatter(pts[m, 0], pts[m, 1], s=24, alpha=0.85, label=f"Fold {f+1}")
ax.set_title("(a) Random K-fold")
ax.set_xlabel("Longitude (normalised)")
ax.set_ylabel("Latitude (normalised)")
ax.grid(alpha=0.2)
ax.text(0.02, -0.15, "Spatial neighbours split across folds -> leakage", transform=ax.transAxes, fontsize=9)

# (b) Spatial block CV
ax = axes[1]
blocks = []
for i in range(3):
    for j in range(2):
        blocks.append((i / 3, j / 2, 1 / 3, 1 / 2))
for k, (x0, y0, w, h) in enumerate(blocks):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linestyle="--", linewidth=1.0, edgecolor="gray"))

folds_block = np.zeros(len(pts), dtype=int)
for i, (x, y) in enumerate(pts):
    bx = min(int(x * 3), 2)
    by = min(int(y * 2), 1)
    folds_block[i] = by * 3 + bx
for f in np.unique(folds_block):
    m = folds_block == f
    ax.scatter(pts[m, 0], pts[m, 1], s=24, alpha=0.85)
ax.set_title("(b) Spatial Block CV")
ax.set_xlabel("Longitude (normalised)")
ax.set_ylabel("Latitude (normalised)")
ax.grid(alpha=0.2)
ax.text(0.02, -0.15, "Reduced leakage; boundary effects remain", transform=ax.transAxes, fontsize=9)

# (c) Spatial Group-KFold by district
ax = axes[2]
# Simulated district polygons / zones
zones = [
    (0.00, 0.00, 0.45, 0.50),
    (0.45, 0.00, 0.55, 0.50),
    (0.00, 0.50, 0.40, 0.50),
    (0.40, 0.50, 0.30, 0.50),
    (0.70, 0.50, 0.30, 0.50),
]
for z in zones:
    ax.add_patch(Rectangle((z[0], z[1]), z[2], z[3], fill=False, linewidth=1.2, edgecolor="black"))

folds_group = np.zeros(len(pts), dtype=int)
for i, (x, y) in enumerate(pts):
    if y < 0.5 and x < 0.45:
        folds_group[i] = 0
    elif y < 0.5:
        folds_group[i] = 1
    elif x < 0.40:
        folds_group[i] = 2
    elif x < 0.70:
        folds_group[i] = 3
    else:
        folds_group[i] = 4
for f in range(5):
    m = folds_group == f
    ax.scatter(pts[m, 0], pts[m, 1], s=24, alpha=0.9, label=f"District fold {f+1}")

ax.set_title("(c) Spatial Group-KFold (District)")
ax.set_xlabel("Longitude (normalised)")
ax.set_ylabel("Latitude (normalised)")
ax.grid(alpha=0.2)
ax.text(0.02, -0.15, "No district history in test fold -> realistic deployment", transform=ax.transAxes, fontsize=9)

fig.suptitle(
    "Figure 4. Cross-validation strategies for geospatial ML and leakage behaviour",
    fontsize=12,
)
fig.savefig(os.path.join(OUT_DIR, "spatial_cv_illustration.png"), dpi=300)
plt.close(fig)


# Figure 5: CV leakage gap comparison
models = [
    "RF Base",
    "XGBoost Base",
    "LightGBM Base",
    "Stack (RF+XGB)",
    "Spatial Stack",
]

spatial_r2 = np.array([-0.050, -0.055, -0.060, -0.065, 0.025])
random_r2 = np.array([0.280, 0.265, 0.250, 0.270, 0.340])
gap = random_r2 - spatial_r2

x = np.arange(len(models))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
ax.bar(x - width / 2, random_r2, width, label="Random K-fold R2", color="#4C78A8")
ax.bar(x + width / 2, spatial_r2, width, label="District-disjoint Spatial Group-KFold R2", color="#F58518")

for i, g in enumerate(gap):
    ax.plot([x[i] - width / 2, x[i] + width / 2], [random_r2[i], spatial_r2[i]], color="black", linewidth=1)
    ax.text(x[i], max(random_r2[i], spatial_r2[i]) + 0.02, f"Delta={g:.3f}", ha="center", fontsize=8)

ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.set_ylabel("R2")
ax.set_title("Figure 5. Random vs district-disjoint spatial CV R2 across model configurations")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.25)

# Highlight leakage fraction for Spatial Stack
spatial_stack_idx = len(models) - 1
leakage_fraction = 100.0 * gap[spatial_stack_idx] / random_r2[spatial_stack_idx]
ax.text(
    0.99,
    0.02,
    f"Spatial Stack relative leakage fraction: {leakage_fraction:.0f}%",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

fig.savefig(os.path.join(OUT_DIR, "cv_gap_comparison.png"), dpi=300)
plt.close(fig)

print("Generated:")
print(os.path.join(OUT_DIR, "spatial_cv_illustration.png"))
print(os.path.join(OUT_DIR, "cv_gap_comparison.png"))
