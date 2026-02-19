#!/usr/bin/env python3
"""
Generate a conceptual schematic for the ISF hypothesis:
- Panel A: Single branch cosmic web (filaments, nodes, voids)
- Panel B: N=6 branches superimposed → smoother distribution
- Panel C: What we observe: our galaxy + DM from overlap
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ── Helper: generate a toy 2D cosmic web ──────────────────────────
def make_cosmic_web(n_nodes=15, n_filaments=20, seed=0):
    """Generate random nodes and filaments for a toy cosmic web."""
    rng = np.random.RandomState(seed)
    # Nodes (galaxy clusters)
    nodes_x = rng.uniform(0.05, 0.95, n_nodes)
    nodes_y = rng.uniform(0.05, 0.95, n_nodes)
    # Filaments: connect nearby nodes
    filaments = []
    for i in range(n_nodes):
        dists = np.sqrt((nodes_x - nodes_x[i])**2 + (nodes_y - nodes_y[i])**2)
        nearest = np.argsort(dists)[1:4]  # connect to 3 nearest
        for j in nearest:
            if (i, j) not in filaments and (j, i) not in filaments:
                filaments.append((i, j))
    # Field galaxies along filaments
    gal_x, gal_y = [], []
    for i, j in filaments:
        n_gal = rng.randint(3, 8)
        t = rng.uniform(0.1, 0.9, n_gal)
        scatter = rng.normal(0, 0.015, n_gal)
        gx = nodes_x[i] + t * (nodes_x[j] - nodes_x[i]) + scatter
        gy = nodes_y[i] + t * (nodes_y[j] - nodes_y[i]) + scatter
        gal_x.extend(gx)
        gal_y.extend(gy)
    return nodes_x, nodes_y, filaments, np.array(gal_x), np.array(gal_y)


# ── Panel A: Single branch ────────────────────────────────────────
ax = axes[0]
nx, ny, fils, gx, gy = make_cosmic_web(n_nodes=12, seed=42)

# Draw filaments
for i, j in fils:
    ax.plot([nx[i], nx[j]], [ny[i], ny[j]], '-', color='royalblue',
            lw=1.5, alpha=0.4)

# Draw galaxies
ax.scatter(gx, gy, s=8, color='royalblue', alpha=0.6, zorder=3)
ax.scatter(nx, ny, s=40, color='royalblue', edgecolors='navy',
           linewidths=0.8, zorder=4)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('(a) One $\\psi$-branch\n(one cosmic web)', fontsize=11,
             fontweight='bold')
ax.text(0.5, -0.08, 'Galaxies, filaments, voids\n(EM-confined to this branch)',
        ha='center', fontsize=8, transform=ax.transAxes, style='italic')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)


# ── Panel B: N branches superimposed ─────────────────────────────
ax = axes[1]
branch_colors = ['royalblue', 'crimson', 'forestgreen', 'darkorange',
                 'purple', 'brown']

all_gx, all_gy, all_c = [], [], []
for branch in range(6):
    nx, ny, fils, gx, gy = make_cosmic_web(n_nodes=10, seed=branch*7+13)
    # Draw filaments very faintly
    for i, j in fils:
        ax.plot([nx[i], nx[j]], [ny[i], ny[j]], '-',
                color=branch_colors[branch], lw=0.6, alpha=0.12)
    all_gx.extend(gx)
    all_gy.extend(gy)
    all_c.extend([branch_colors[branch]] * len(gx))
    ax.scatter(nx, ny, s=15, color=branch_colors[branch], alpha=0.25, zorder=3)

ax.scatter(all_gx, all_gy, s=4, c=all_c, alpha=0.35, zorder=2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('(b) $N = 6$ branches\n(superimposed)', fontsize=11,
             fontweight='bold')
ax.text(0.5, -0.08, 'Each color = independent branch\nVoids partially fill in → smoother DM',
        ha='center', fontsize=8, transform=ax.transAxes, style='italic')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)

# Branch legend
for i, (c, label) in enumerate(zip(branch_colors,
        ['Branch 1 (ours)', 'Branch 2', 'Branch 3',
         'Branch 4', 'Branch 5', 'Branch 6'])):
    ax.plot([], [], 'o', color=c, ms=5, label=label)
ax.legend(fontsize=5.5, loc='upper right', framealpha=0.8,
          handletextpad=0.3, borderpad=0.3)


# ── Panel C: What we observe ─────────────────────────────────────
ax = axes[2]

# Create a density map from superposition
grid = np.zeros((50, 50))
for branch in range(6):
    _, _, _, gx, gy = make_cosmic_web(n_nodes=10, seed=branch*7+13)
    for x, y in zip(gx, gy):
        ix = int(np.clip(x * 50, 0, 49))
        iy = int(np.clip(y * 50, 0, 49))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if 0 <= ix+di < 50 and 0 <= iy+dj < 50:
                    r2 = di**2 + dj**2
                    grid[iy+dj, ix+di] += np.exp(-r2 / 2.0)

# Our galaxy position
our_x, our_y = 0.45, 0.55
# Show DM density as background
im = ax.imshow(grid, extent=[0, 1, 0, 1], origin='lower',
               cmap='inferno', alpha=0.7, aspect='equal')
cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.set_label('DM density\n(overlap from\nother branches)', fontsize=7)
cb.ax.tick_params(labelsize=6)

# Mark "our galaxy"
ax.scatter([our_x], [our_y], s=120, marker='*', color='cyan',
           edgecolors='white', linewidths=1.0, zorder=5)
ax.annotate('Our\ngalaxy', xy=(our_x, our_y),
            xytext=(our_x + 0.15, our_y + 0.15),
            fontsize=8, color='cyan', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5),
            ha='left')

# Draw concentric circles to suggest rotation curve measurement
for r in [0.05, 0.10, 0.15]:
    circle = Circle((our_x, our_y), r, fill=False, color='cyan',
                     ls='--', lw=0.6, alpha=0.5)
    ax.add_patch(circle)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('(c) What we observe:\n$f_\\mathrm{DM}$ from overlap geometry', fontsize=11,
             fontweight='bold')
ax.text(0.5, -0.08, '$f_{DM}$ varies continuously depending on\nhow many branches have matter at our location',
        ha='center', fontsize=8, transform=ax.transAxes, style='italic')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('fig_isf_schematic.pdf', bbox_inches='tight')
plt.savefig('fig_isf_schematic.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig_isf_schematic")
