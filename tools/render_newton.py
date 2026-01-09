import numpy as np
import imageio.v2 as imageio
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Parameters
WIDTH = 432
HEIGHT = 432
XMIN, XMAX = -1.5, 0.5 # I used -2, 2 for newton{n}.png
YMIN, YMAX = -1.0, 1.0

ITERATION_FRAMES = [1, 3, 100]
EPS = 1e-6
MAX_RADIUS = 1e6

# Setup complex grid
x = np.linspace(XMIN, XMAX, WIDTH)
y = np.linspace(YMIN, YMAX, HEIGHT)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y

# Polynomial and derivative
def f(z):
    return z**3 - 2*z + 2

def df(z):
    return 3*z**2 - 2

# Find roots
def find_roots():
    guesses = [
        0.5 * np.exp(1j * a)
        for a in np.linspace(0, 2*np.pi, 16, endpoint=False)
    ]

    roots = []
    for z0 in guesses:
        z = z0
        for _ in range(100):
            dz = df(z)
            if abs(dz) < EPS:
                break
            z -= f(z) / dz
        if abs(f(z)) < 1e-5:
            if not any(abs(z - r) < 1e-3 for r in roots):
                roots.append(z)

    return np.array(roots)

# Coloring by weighted distances
def color_from_distances(z, roots):
    """
    Continuous coloring via inverse-distance weighting to all roots.
    """
    if len(roots) == 0:
        return np.zeros(z.shape + (3,), dtype=np.uint8)

    dists = np.stack([np.abs(z - r) for r in roots], axis=-1)
    dists = np.maximum(dists, EPS)

    weights = 1.0 / dists
    weights /= np.sum(weights, axis=-1, keepdims=True)

    # Fixed RGB colors per root
    base_colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    colors = base_colors[:len(roots)]

    rgb = np.zeros(z.shape + (3,), dtype=float)
    for i in range(len(roots)):
        rgb += weights[..., i:i+1] * colors[i]

    return (255 * rgb).astype(np.uint8)


# Newton iteration
def newton_iterations(z, steps):
    z = z.copy()
    for _ in range(steps):
        with np.errstate(divide="ignore", invalid="ignore"):
            z -= f(z) / df(z)
        z[np.abs(z) > MAX_RADIUS] = np.nan
    return z

# Main
roots = find_roots()

for n in ITERATION_FRAMES:
    Z = newton_iterations(Z0, n)

    invalid = np.isnan(Z.real) | np.isnan(Z.imag)
    rgb = color_from_distances(Z, roots)
    rgb[invalid] = 0

    output_path = IMAGES_DIR / f"newton_div{n}.png"
    imageio.imwrite(output_path, rgb)

    print(f"Saved {output_path}")
