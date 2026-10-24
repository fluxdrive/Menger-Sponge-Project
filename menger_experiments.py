"""
Menger Sponge -- computational experiments for the CS 4302 project report.

Reproduces every numerical result in the report:
  1. cube counts N_n = 20^n
  2. the surface-area recurrence and its closed form  S_n = 2(20/9)^n + 4(8/9)^n
  3. the box-counting dimension  D = log 20 / log 3
  4. the sparse 27-tree node/memory table
  5. the voxel figure (menger_vox.png)

The sponge occupancy is built from the base-3 rule: a subcube is removed when
at least two of its three base-3 coordinates equal the middle value (1). By
self-similarity the level-k occupancy is the k-fold Kronecker power of the
3x3x3 base pattern.

Requires: numpy, matplotlib.  Run:  python3 menger_experiments.py
"""

import numpy as np
from functools import reduce

# --- base 3x3x3 pattern: remove a subcube if >= 2 of its coords are the middle
base = np.ones((3, 3, 3), dtype=np.uint8)
for i in range(3):
    for j in range(3):
        for k in range(3):
            if (i == 1) + (j == 1) + (k == 1) >= 2:
                base[i, j, k] = 0
assert base.sum() == 20


def sponge(k):
    """Level-k occupancy grid (side 3^k) as the k-fold Kronecker power of base."""
    return reduce(np.kron, [base] * k) if k >= 1 else np.ones((1, 1, 1), np.uint8)


def internal_adjacencies(occ):
    """Number of face-adjacent pairs of occupied cells, summed over the 3 axes."""
    o = occ.astype(int)
    return int((o[:-1] & o[1:]).sum()
               + (o[:, :-1] & o[:, 1:]).sum()
               + (o[:, :, :-1] & o[:, :, 1:]).sum())


# ===========================================================================
# 1. cube counts
# ===========================================================================
print("1. cube counts")
for k in range(6):
    assert sponge(k).sum() == 20 ** k
    print(f"   N_{k} = {20**k:,}")

# ===========================================================================
# 2. surface area: recurrence, closed form, and direct verification
# ===========================================================================
alpha = internal_adjacencies(sponge(1))     # = 24 (corner-edge adjacencies)
print(f"\n2. surface area   (alpha = {alpha} face-adjacencies among the 20 cells)")


def S_closed(n):
    # S_n = (6 - alpha/6)(20/9)^n + (alpha/6)(8/9)^n  ->  2(20/9)^n + 4(8/9)^n
    return (6 - alpha / 6) * (20 / 9) ** n + (alpha / 6) * (8 / 9) ** n


def S_direct(k):
    """Surface area by counting exposed unit-cube faces: 6*cubes - 2*adjacencies."""
    occ = sponge(k)
    exposed = 6 * int(occ.sum()) - 2 * internal_adjacencies(occ)
    return exposed * (3.0 ** (-k)) ** 2


print("   n |  S_n (direct) |  S_n (closed form)")
for k in range(6):
    d, c = S_direct(k), S_closed(k)
    print(f"   {k} | {d:12.5f}  | {c:12.5f}   match={abs(d-c) < 1e-9}")
print(f"   S_n / S_(n-1) -> 20/9 = {20/9:.4f}  (volume -> 0, surface area -> inf)")

# ===========================================================================
# 3. box-counting dimension at the natural scales eps = 3^-m
# ===========================================================================
print("\n3. box-counting dimension")
coords = np.argwhere(sponge(5) > 0)          # occupied voxels on the 243^3 grid
ms = list(range(6))
N = [len(np.unique(coords // (3 ** (5 - m)), axis=0)) for m in ms]   # measured N(3^-m)
x = np.log([3.0 ** m for m in ms])           # log(1/eps)
y = np.log(N)
slope, intercept = np.polyfit(x, y, 1)
r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
theoretical = np.log(20) / np.log(3)
print(f"   measured N(3^-m) = {N}   (= 20^m)")
print(f"   fitted D = {slope:.4f}   theoretical log20/log3 = {theoretical:.4f}   R^2 = {r2:.6f}")

# ===========================================================================
# 4. sparse 27-tree node counts and memory
# ===========================================================================
print("\n4. sparse 27-tree size vs. distance field (O(1) memory)")
print("   depth | leaves 20^n  | total nodes  | tree memory (4 B/node)")
for n in range(1, 8):
    leaves = 20 ** n
    nodes = (20 ** (n + 1) - 1) // 19
    mem = nodes * 4
    unit = (f"{mem/1e3:.1f} KB" if mem < 1e6 else
            f"{mem/1e6:.1f} MB" if mem < 1e9 else f"{mem/1e9:.2f} GB")
    print(f"     {n}   | {leaves:>12,} | {nodes:>12,} | {unit}")

# ===========================================================================
# 5. figures: box-counting plot and voxel model
# ===========================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # box-counting log-log plot
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    xx = np.linspace(-0.3, x.max() + 0.3, 100)
    ax.plot(xx, slope * xx + intercept, color="#c65b7c", lw=1.8,
            label=f"fit: slope = {slope:.4f}")
    ax.scatter(x, y, s=48, color="#6c7fb0", edgecolor="#2b3350", zorder=3,
               label=r"measured $N(3^{-m})=20^m$")
    ax.set_xlabel(r"$\log(1/\varepsilon)$")
    ax.set_ylabel(r"$\log N(\varepsilon)$")
    ax.set_title(f"Box-counting dimension = {slope:.4f}  (log20/log3 = {theoretical:.4f})",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("boxcount.png", dpi=170, bbox_inches="tight")

    # voxel model (level 3)
    vox = sponge(3).astype(bool)
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(vox, facecolors="#8b9ac4", edgecolor="#2b3350", linewidth=0.12, shade=True)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    plt.tight_layout(pad=0)
    plt.savefig("menger_vox.png", dpi=165, bbox_inches="tight", transparent=True)
    print("\n5. figures written: boxcount.png, menger_vox.png")
except ImportError:
    print("\n5. matplotlib not installed; skipped figures")
