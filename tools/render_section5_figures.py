"""Figures of Section 5 (logistic family, Euclidean algorithm, doubling map).

Colours: the paper's palette (tools/palette.py).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction

# The paper's "bubblegum" palette (tools/palette.py of the repository).
DEEP = "#2BA9C9"      # MAIN: a deeper sky blue
PALE = "#FFFFFF"      # white background, as in the other figures
PLUM = "#9C1C6E"      # BERRY
ORCHID = "#F25CA8"    # PINK (ACCENT, for limits)
RAMP = ["#C4ECF5", "#8ED9EC", "#59C4DE", "#2BA9C9"]
SKYBAND = "#E3F6FB"; LEMONBAND = "#FFF6D6"; LAVBAND = "#EEE8FF"
GREY = (0.55, 0.55, 0.6)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "0.3",
                     "axes.linewidth": 0.6, "mathtext.fontset": "cm"})
PHI = (1 + 5 ** 0.5) / 2

# ---------------------------------------------------------------- Euclidean algorithm
def digits_S(choices, nblocks):
    """Partial quotients [0; 1^1 d_1 1^2 d_2 ...]."""
    a = [0]
    for n in range(1, nblocks + 1):
        a += [1] * n
        a.append(choices[(n - 1) % len(choices)])
    return a

def convergents(a, K):
    p = [1, a[0]]; q = [0, 1]          # p_{-1}, p_0
    for k in range(1, K + 1):
        p.append(a[k] * p[-1] + p[-2]); q.append(a[k] * q[-1] + q[-2])
    return p, q                         # p[k+1] = p_k

def value(a):
    x = Fraction(0)
    for d in reversed(a[1:]):
        x = 1 / (d + x)
    return float(a[0] + x)

def Minv(x, p, q, k):
    pk, pk1, qk, qk1 = p[k + 1], p[k], q[k + 1], q[k]
    den = pk - qk * x
    with np.errstate(divide="ignore", invalid="ignore"):
        return (qk1 * x - pk1) / den

def ang(x):            # projective line -> (-pi, pi]
    return 2 * np.arctan(x)

def fig_euclid(path):
    a = digits_S([2], 12)
    eta = value(a)
    p, q = convergents(a, len(a) - 1)
    # middle indices of the runs of ones
    mids = []
    start = 1
    for n in range(1, 13):
        mids.append(start + n // 2); start += n + 1
    ks = [mids[0], mids[1], mids[2], mids[3]]
    xs = np.tan(np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 40001) / 1)  # covers R
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    shades = RAMP
    for c, k in zip(shades, ks):
        y = Minv(xs, p, q, k)
        th, ty = ang(xs), ang(y)
        jump = np.abs(np.diff(ty)) > 3
        ty = ty.copy(); ty[1:][jump] = np.nan
        ax.plot(th, ty, color=c, lw=0.9, label=r"$k=%d$" % k)
    ax.axhline(ang(-1 / PHI), color=GREY, lw=0.6, ls="--")
    ax.plot([ang(eta)], [ang(PHI)], "o", color=PLUM, ms=4, zorder=5)
    ax.annotate(r"$\eta\mapsto\varphi$", (ang(eta), ang(PHI)), xytext=(8, 4), textcoords="offset points", color=PLUM)
    ax.text(-3.0, ang(-1 / PHI) - 0.55, r"every other point $\mapsto -1/\varphi$", color=GREY, fontsize=8)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    ax.set_xticks(ticks); ax.set_xticklabels([r"$\infty$", "$-1$", "$0$", "$1$", r"$\infty$"])
    ax.set_yticks(ticks); ax.set_yticklabels([r"$\infty$", "$-1$", "$0$", "$1$", r"$\infty$"])
    ax.set_title(r"the maps $M_k^{-1}$ applied by the algorithm to $\eta$", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_facecolor(PALE)
    # right panel: limits for several eta
    ax = axes[1]
    ax.set_facecolor(PALE)
    ax.axhline(ang(-1 / PHI), color=ORCHID, lw=1.4)
    rng = np.random.default_rng(3)
    for j, shift in enumerate([-3, -2, -1, 0, 1, 2]):
        ch = list(rng.integers(1, 3, size=12))
        e = value(digits_S(ch, 12)) + shift
        ax.plot([ang(e), ang(e)], [ang(-1 / PHI), ang(PHI)], color=ORCHID, lw=0.8)
        ax.plot([ang(e)], [ang(PHI)], "o", color=PLUM, ms=3)
    ax.text(0.02, 0.04, r"translates $\delta_{\eta+n}$ ($n\in\mathbb{Z}$) are also deep computations", transform=ax.transAxes, fontsize=7, color=GREY)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks(ticks); ax.set_xticklabels([r"$\infty$", "$-1$", "$0$", "$1$", r"$\infty$"])
    ax.set_yticks(ticks); ax.set_yticklabels([r"$\infty$", "$-1$", "$0$", "$1$", r"$\infty$"])
    ax.set_title(r"exact equality tests $\delta_\eta$ at several points", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ---------------------------------------------------------------- logistic map
def fig_logistic(path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rs = np.linspace(0.01, 4.0, 2400)
    x = np.full_like(rs, 0.3)
    for _ in range(600):
        x = rs * x * (1 - x)
    R, X = [], []
    for _ in range(160):
        x = rs * x * (1 - x)
        R.append(rs.copy()); X.append(x.copy())
    R = np.concatenate(R); X = np.concatenate(X)
    ax.axvspan(0, 1, color=SKYBAND)
    ax.axvspan(1, 3, color=LEMONBAND)
    ax.scatter(R, X, s=0.18, color=DEEP, alpha=0.75, linewidths=0, rasterized=True)
    ax.axvline(4, color=PLUM, lw=1.2)
    ax.text(0.5, 1.03, "stable", ha="center", fontsize=8, color=DEEP, transform=ax.get_xaxis_transform())
    ax.text(2.0, 1.03, r"NIP$_3$, not stable", ha="center", fontsize=8, color=PLUM, transform=ax.get_xaxis_transform())
    ax.text(3.97, 1.03, r"not NIP at $r=4$", ha="right", fontsize=8, color=PLUM, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 4.02); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$x$")
    ax = axes[1]
    r = 2.8
    f = lambda t: r * t * (1 - t)
    t = np.linspace(0, 1, 400)
    ax.set_facecolor(PALE)
    ax.plot(t, f(t), color=DEEP, lw=1)
    ax.plot(t, t, color=GREY, lw=0.6)
    for x0, c in [(0.05, ORCHID), (0.93, PLUM)]:
        x = x0; pts = [(x, 0)]
        for _ in range(30):
            y = f(x); pts += [(x, y), (y, y)]; x = y
        P = np.array(pts); ax.plot(P[:, 0], P[:, 1], color=c, lw=0.6)
    p = 1 - 1 / r
    ax.plot([p], [p], "o", color=PLUM, ms=3.5)
    ax.annotate(r"$1-1/r$", (p, p), xytext=(6, -12), textcoords="offset points", fontsize=8, color=PLUM)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(r"$r=2.8$: every orbit in $(0,1)$ converges to $1-1/r$", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------- doubling map: independence
def fig_doubling(path):
    """Iterates of the predicate x -> sin^2(pi x) under doubling, at times 3k."""
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_facecolor(PALE)
    patterns = {"high (block 011)": [1, 0, 1, 1], "low (block 000)": [0, 1, 0, 0]}
    K = 4
    xs = []
    for bits in ([1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 1]):
        b = []
        for e in bits:
            b += [0, 1, 1] if e else [0, 0, 0]
        x = sum(d / 2 ** (i + 1) for i, d in enumerate(b + [0, 1, 0, 1, 0, 1]))
        xs.append((bits, x))
    for j, (bits, x) in enumerate(xs):
        vals = [np.sin(np.pi * ((2 ** (3 * k)) * x % 1)) ** 2 for k in range(K)]
        ax.plot(range(K), vals, "-o", color=[DEEP, ORCHID, PLUM][j], ms=4, lw=0.9,
                label="pattern " + "".join("H" if e else "L" for e in bits))
    ax.axhspan(0.75, 1.0, color=SKYBAND)
    ax.axhspan(0, 0.25, color=LAVBAND)
    ax.set_xticks(range(K)); ax.set_xticklabels([r"$f^{0}$", r"$f^{3}$", r"$f^{6}$", r"$f^{9}$"])
    ax.set_ylim(0, 1); ax.set_ylabel(r"$\sin^2(\pi f^{3k}x)$")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

if __name__ == "__main__":
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    fig_euclid(os.path.join(here, "euclid_equality.pdf"))
    fig_logistic(os.path.join(here, "logistic_levels.pdf"))
    fig_doubling(os.path.join(here, "doubling_independence.pdf"))
    print("done")
