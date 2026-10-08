import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in fm.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/lm"]):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm",
                     "font.size": 8})

def hexrgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

ROOT_COLS = [hexrgb("#2f5d93"), hexrgb("#3f9a8a"), hexrgb("#c98f2e")]  # blue, teal, ochre
CYC_COLS = [hexrgb("#f3c9d8"), hexrgb("#8c2457")]                       # pale lilac (near 0), deep violet (near 1)

def newton_iter(f, df, Z, n):
    Z = Z.copy()
    with np.errstate(all="ignore"):
        for _ in range(n):
            Z = Z - f(Z) / df(Z)
    return Z

def converge_info(f, df, Z0, roots, nmax=200, tol=1e-6):
    """index of root reached (or -1) and iteration count, plus final iterate parity info"""
    Z = Z0.copy()
    idx = -np.ones(Z.shape, int)
    cnt = np.full(Z.shape, nmax, float)
    with np.errstate(all="ignore"):
        for k in range(nmax):
            Z = Z - f(Z) / df(Z)
            for i, r in enumerate(roots):
                hit = (idx < 0) & (np.abs(Z - r) < tol)
                idx[hit] = i
                # smooth count
                cnt[hit] = k
    return idx, cnt, Z

def shade(col, t):
    # t in [0,1]: 0 = fast (full colour), 1 = slow (darker)
    t = np.clip(t, 0, 1)[..., None]
    return col * (1 - 0.55 * t) + 0.0 * t

def blend_image(Z, roots, p=3.0):
    d = np.stack([np.abs(Z - r) for r in roots], -1)
    d = np.where(np.isfinite(d), d, 1e9)
    w = 1.0 / np.maximum(d, 1e-9) ** p
    w = w / w.sum(-1, keepdims=True)
    img = np.tensordot(w, np.stack(ROOT_COLS), axes=([-1], [0]))
    return img

def grid(xmin, xmax, ymin, ymax, N):
    x = np.linspace(xmin, xmax, N); y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y

def roots_of(coeffs):
    r = np.roots(coeffs)
    # order: real root first, then upper, then lower
    r = sorted(r, key=lambda z: (abs(z.imag) > 1e-9, -z.imag))
    return np.array(r)

def limit_image(f, df, Z0, roots, n_even):
    idx, cnt, _ = converge_info(f, df, Z0, roots)
    img = np.zeros(Z0.shape + (3,))
    tmax = np.percentile(cnt[idx >= 0], 97) if (idx >= 0).any() else 1
    for i, c in enumerate(ROOT_COLS):
        m = idx == i
        img[m] = shade(c, cnt[m] / tmax)
    # remaining points: attracting cycle (if any)
    rest = idx < 0
    if rest.any():
        Zn = newton_iter(f, df, Z0, n_even)
        near1 = np.abs(Zn - 1) < np.abs(Zn - 0)
        img[rest & ~near1] = CYC_COLS[0]
        img[rest & near1] = CYC_COLS[1]
    return img

def panel(ax, img, ext, title, roots=None):
    ax.imshow(img, extent=ext, origin="lower", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#555555")
    ax.set_title(title, fontsize=8, pad=2)
    if roots is not None:
        for r in roots:
            if ext[0] <= r.real <= ext[1] and ext[2] <= r.imag <= ext[3]:
                ax.plot(r.real, r.imag, "o", ms=3.2, mfc="white", mec="black", mew=0.6)

N = 1100
# ---------- rho(z) = z^3 - 2z + 2
f1 = lambda z: z**3 - 2*z + 2
df1 = lambda z: 3*z**2 - 2
R1 = roots_of([1, 0, -2, 2])
extF = (-2.0, 1.4, -1.6, 1.6)
extZ = (0.8, 1.2, -0.19, 0.19)
ZF = grid(*extF, N)
ZZ = grid(*extZ, 700)
fig, axs = plt.subplots(1, 4, figsize=(5.0, 1.5))
panel(axs[0], limit_image(f1, df1, ZF, R1, 200), extF, r"even $n$", R1)
panel(axs[1], limit_image(f1, df1, ZF, R1, 201), extF, r"odd $n$", R1)
panel(axs[2], limit_image(f1, df1, ZZ, R1, 200), extZ, r"even $n$, near $1$", R1)
panel(axs[3], limit_image(f1, df1, ZZ, R1, 201), extZ, r"odd $n$, near $1$", R1)
for ax in axs[:2]:
    ax.add_patch(plt.Rectangle((extZ[0], extZ[2]), extZ[1]-extZ[0], extZ[3]-extZ[2],
                 fill=False, lw=0.6, ec="black"))
for ax, e in zip(axs, [extF, extF, extZ, extZ]):
    ax.plot([0, 1], [0, 0], "s", ms=2.4, mfc="white", mec="black", mew=0.5)
    ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
fig.savefig("newton_cubic.pdf", dpi=300)
fig.savefig("newton_cubic.png", dpi=110)

# ---------- kappa(z) = z^3 - 1
f2 = lambda z: z**3 - 1
df2 = lambda z: 3*z**2
R2 = roots_of([1, 0, 0, -1])
ext2 = (-1.6, 1.6, -1.6, 1.6)
Z0 = grid(ext2[0], ext2[1], ext2[2], ext2[3], N)
fig, axs = plt.subplots(1, 3, figsize=(5.0, 1.85))
panel(axs[0], blend_image(newton_iter(f2, df2, Z0, 1), R2), ext2, r"$n=1$", R2)
panel(axs[1], blend_image(newton_iter(f2, df2, Z0, 3), R2), ext2, r"$n=3$", R2)
panel(axs[2], limit_image(f2, df2, Z0, R2, 200), ext2, r"$n\to\infty$", R2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.04)
fig.savefig("newton_cyclotomic.pdf", dpi=300)
fig.savefig("newton_cyclotomic.png", dpi=110)
print(R1, R2)
