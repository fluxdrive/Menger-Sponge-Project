# Menger Sponge Fractal Engine

A real-time, browser-based renderer for the **Menger sponge**, built with GPU ray marching, paired with a Python suite that validates the fractal's mathematical properties. The renderer never instantiates geometry: it evaluates a recursive signed-distance field in a GLSL shader, so detail equivalent to tens of millions of cubes costs constant memory.

Built as the course project for **CS 4302 Discrete Mathematics** at UT Dallas, where the sponge serves as a worked example of recurrence relations, induction, geometric series, and fractal dimension.



## Overview

The Menger sponge is the three-dimensional generalization of the Cantor set and the Sierpiński carpet. Starting from a cube, you divide it into a 3×3×3 grid of 27 subcubes, remove the 7 that touch the center, and recurse on the remaining 20 forever. The result is a solid whose volume collapses to zero while its surface area grows without bound, with a fractal dimension of log 20 / log 3 ≈ 2.7268.

This repository contains two independent implementations:

- **`menger_raymarch.html`** — an interactive WebGL renderer (GPU ray marching) you open in a browser.
- **`menger_experiments.py`** — a Python/NumPy suite that reproduces and verifies the sponge's mathematical results.

## The renderer

Open `menger_raymarch.html` in any modern browser. No build step, no install.

- **Drag** to orbit, **scroll** to zoom.
- Adjust **iteration depth**, **bloom**, **exposure**, **focus**, and **aperture** from the control panel.
- Toggle **cut** (cross-section), **glass** (refraction), **rays** (volumetric light shafts), and **2× SSAA**.

### How it works

Rather than building the 20ⁿ cubes (over 64 million at depth 6), the renderer casts a ray per pixel and advances it using a signed-distance estimator. The estimator is recursive and mirrors the construction directly: at each fold it maps the point into the repeating 3×3×3 cell and subtracts the cross-shaped tunnel of the seven removed cubes. The iteration depth is simply the shader loop bound, and because no geometry is stored, memory stays O(1) in the depth.

### Features

| Area | Techniques |
|------|-----------|
| Core | GPU ray marching with a signed-distance-field estimator |
| Shading | Fresnel, specular, orbit-trap coloring, hemisphere lighting |
| HDR pipeline | Bright-pass bloom, separable Gaussian blur, ACES tone mapping |
| Reflection & lens | Reflective floor with one reflection bounce, depth of field |
| Atmosphere | Volumetric light shafts, optional glass/refraction mode |
| Robustness | Distance-based level of detail to prevent float-precision artifacts at high depth |

### Implementation notes

The whole scene is a single full-screen quad; all geometry lives in the fragment shader. The pipeline runs in passes: the ray march writes linear HDR (plus depth in the alpha channel) to an off-screen float buffer, a depth-of-field pass blurs by circle of confusion, a bright-pass and two-iteration separable blur build the bloom, and a final composite applies ACES tone mapping and a vignette.

At high iteration depth the fold multiplies coordinates by 3ⁿ, which overruns 32-bit float precision (around n ≥ 7) and produces black-square artifacts. A distance-based level-of-detail cap limits the fold depth to the detail resolvable at a given distance, which removes the artifacts and the sub-pixel aliasing at no visual cost.

## The experiments

```bash
pip install numpy matplotlib
python3 menger_experiments.py
```

Running the script reproduces every numerical result and regenerates two figures (`boxcount.png`, `menger_vox.png`). It verifies:

1. **Cube counts** — Nₙ = 20ⁿ, by construction from the base-3 rule.
2. **Surface-area recurrence** — the closed form Sₙ = 2(20/9)ⁿ + 4(8/9)ⁿ, checked against a direct exposed-face count at every level; it diverges while the volume (20/27)ⁿ vanishes.
3. **Box-counting dimension** — measured covering numbers of exactly 20ᵐ at scales 3⁻ᵐ, giving a fitted dimension of 2.7268 (matching log 20 / log 3) with R² = 1.0.
4. **Representation comparison** — sparse 27-tree node counts and memory versus the O(1)-memory distance field.

The sponge occupancy is built from the base-3 membership rule (a cell is removed when at least two of its three base-3 coordinates equal the middle value), computed as a Kronecker power of the 3×3×3 base pattern, so the code mirrors the math.

## Repository layout

```
menger_raymarch.html        Interactive WebGL / GLSL renderer
menger_experiments.py       Python/NumPy validation suite
```

## Tech stack

**Three.js**, **WebGL**, **GLSL**, **Python**, **NumPy**, **Matplotlib**

## Background

The sponge was introduced by Karl Menger in 1926. For the mathematical details behind this project (the full induction proofs, the surface-area derivation, and the box-counting experiment), see the project report.

## License

MIT
