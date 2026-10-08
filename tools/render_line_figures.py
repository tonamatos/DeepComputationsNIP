import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

for f in fm.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/lm"]):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Latin Modern Roman", "mathtext.fontset": "cm",
                     "font.size": 8, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palette import MAIN as BLUE, ACCENT as CRIMSON, RAMP as BLUE_RAMP

def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ------------------------------------------------------------------ square roots
# RP^1 parametrized by theta in (-pi, pi], x = tan(theta/2); output angle 2*arctan(N^n(x)).
th = np.linspace(-np.pi, np.pi, 6001)
x = np.tan(th / 2)
def N(x):
    with np.errstate(all="ignore"):
        return (x**2 + 1) / (2 * x)
def ang(y):
    return 2 * np.arctan(y)  # infinity -> +-pi
fig, ax = plt.subplots(figsize=(3.9, 2.25))
cols = BLUE_RAMP
y = x.copy()
for k, n in enumerate([1, 2, 3, 6]):
    while True:
        break
    yy = x.copy()
    for _ in range(n):
        yy = N(yy)
    a = ang(yy)
    a[np.abs(np.diff(a, prepend=a[0])) > 2.5] = np.nan  # break at jumps through infinity
    ax.plot(th, a, color=cols[k], lw=1.0, label=fr"$n={n}$")
# limit
ax.plot([-np.pi, 0], [-np.pi/2, -np.pi/2], color=CRIMSON, lw=1.6)
ax.plot([0, np.pi], [np.pi/2, np.pi/2], color=CRIMSON, lw=1.6, label=r"$\mathfrak{f}=\lim_n N^n$")
for t0 in (0,):
    ax.plot([t0], [np.pi], "o", ms=3.5, color=CRIMSON)
ax.plot([-np.pi, np.pi], [np.pi, np.pi], "o", ms=3.5, color=CRIMSON, clip_on=False)
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
tk = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
ax.set_xticks(tk); ax.set_xticklabels([r"$\infty$", r"$-\sqrt{a}$", r"$0$", r"$\sqrt{a}$", r"$\infty$"])
ax.set_yticks(tk); ax.set_yticklabels([r"$\infty$", r"$-\sqrt{a}$", r"$0$", r"$\sqrt{a}$", r"$\infty$"])
ax.set_xlabel(r"initial point $x\in\mathbb{RP}^1$"); ax.set_ylabel(r"$N^n(x)$")
clean(ax)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig("sqrt_newton.pdf"); fig.savefig("sqrt_newton.png", dpi=140)

# ------------------------------------------------------------------ Bresenham
alpha = (np.sqrt(5) - 1) / 2
def draw(ax, rho, ncols, x0=0, highlight=None, title=None, eps_side=None):
    for n in range(x0, x0 + ncols):
        yv = np.floor(n * alpha + rho + 1e-12 * (eps_side or 0))
        c = BLUE
        if highlight is not None and n == highlight[0]:
            c = highlight[1]
        ax.add_patch(Rectangle((n, yv), 1, 1, fc=c, ec="white", lw=0.6, alpha=0.9))
    xs = np.linspace(x0, x0 + ncols, 50)
    ax.plot(xs, alpha * xs + rho, color="black", lw=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title: ax.set_title(title, fontsize=8, pad=2)

fig = plt.figure(figsize=(5.0, 1.85))
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1], wspace=0.35)
ax0 = fig.add_subplot(gs[0])
rho = 0.13
draw(ax0, rho, 16, title=r"pixels of $y=\alpha x+\rho$")
for n in range(16):
    b = int(np.floor((n + 1) * alpha + rho) - np.floor(n * alpha + rho))
    ax0.text(n + 0.5, -0.75, str(b), ha="center", va="center", fontsize=7, color="#444444")
ax0.text(-0.2, -0.75, r"$b_n$", ha="right", va="center", fontsize=8)
ax0.set_xlim(-1.2, 16.2); ax0.set_ylim(-1.4, 11)
# tie: line through lattice point at x = k: choose rho so that k*alpha + rho is an integer
k = 3
rho_t = np.ceil(k * alpha) - k * alpha       # k*alpha + rho_t = integer
for j, (side, ttl) in enumerate([(+1, r"$R^+_c$"), (-1, r"$R^-_c$")]):
    ax = fig.add_subplot(gs[j + 1])
    for n in range(k - 2, k + 3):
        v = n * alpha + rho_t
        if n == k:
            yv = v if side > 0 else v - 1   # tie broken upward or downward
            yv = np.round(yv)
            c = CRIMSON
        else:
            yv = np.floor(v); c = BLUE
        ax.add_patch(Rectangle((n, yv), 1, 1, fc=c, ec="white", lw=0.6, alpha=0.9))
    xs = np.linspace(k - 2, k + 3, 20)
    ax.plot(xs, alpha * xs + rho_t, color="black", lw=0.8)
    ax.plot([k], [k * alpha + rho_t], "o", ms=3, mfc="white", mec="black", mew=0.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(ttl, fontsize=8, pad=2)
    ax.set_xlim(k - 2.2, k + 3.2)
fig.savefig("bresenham.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("bresenham.png", dpi=140, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ tolerance tests
a = 0.55
xx = np.linspace(0, 1, 4001)
fig, ax = plt.subplots(figsize=(3.9, 1.8))
cols = BLUE_RAMP
for k, n in enumerate([4, 10, 30, 120]):
    q = a + 0.6 / n**1.5
    ax.plot(xx, np.maximum(0, 1 - n * np.abs(xx - q)), color=cols[k], lw=1.0, label=fr"$h_{{q,{n}}}$")
ax.plot([0, 1], [0, 0], color=CRIMSON, lw=1.6, label=r"$[\![x=a]\!]$", zorder=3)
ax.plot([a], [1], "o", ms=4, color=CRIMSON, zorder=4)
ax.plot([a], [0], "o", ms=4, mfc="white", mec=CRIMSON, mew=1.0, zorder=4)
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.12)
ax.set_xticks([0, a, 1]); ax.set_xticklabels([r"$0$", r"$a$", r"$1$"])
ax.set_yticks([0, 1])
ax.set_xlabel(r"input $x$")
clean(ax)
ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig("tolerance.pdf"); fig.savefig("tolerance.png", dpi=140)
print("ok")
