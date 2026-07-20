"""Figures for the 'Discriminatory uncertainty' note.

Two groups, identical calibration of point predictions, systematically
different prediction-interval widths. Light and dark SVG variants.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_du"
os.makedirs(OUT, exist_ok=True)

TEAL = "#009485"    # group A
INDIGO = "#526cfe"  # group B

MODES = {
    "light": dict(text="#212121", muted="#757575", grid="#e3e3e3", faint="#bdbdbd"),
    "dark": dict(text="#dee2ef", muted="#8a91a8", grid="#3a3e4d", faint="#565b6e"),
}


def style(mode):
    c = MODES[mode]
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "svg.fonttype": "none",
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": c["text"], "axes.edgecolor": c["faint"],
        "axes.labelcolor": c["muted"], "xtick.color": c["muted"],
        "ytick.color": c["muted"], "axes.grid": True,
        "grid.color": c["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titlecolor": c["text"],
    })
    return c


def save(fig, name, mode):
    fig.savefig(f"{OUT}/{name}-{mode}.svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ simulate
# True credit quality z; the bureau observes it with group-dependent noise.
# The score is the Bayes posterior mean, so BOTH groups are calibrated.
n = 40_000
group_b = rng.random(n) < 0.5                       # False = A, True = B
z = rng.normal(-2.5, 1.2, n)
noise_sd = np.where(group_b, 0.72, 0.14)
x = z + rng.normal(0, 1.0, n) * noise_sd

prior_var = 1.2 ** 2
post_var = 1 / (1 / prior_var + 1 / noise_sd ** 2)
post_mu = post_var * (-2.5 / prior_var + x / noise_sd ** 2)
post_sd = np.sqrt(post_var)

# point prediction = E[sigmoid(z) | x] via Gauss-Hermite quadrature
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(31)
p_hat = (sigmoid(post_mu[:, None] + post_sd[:, None] * gh_x) * gh_w).sum(1) / gh_w.sum()

# 90% prediction interval for the default probability
lo = sigmoid(post_mu - 1.645 * post_sd)
hi = sigmoid(post_mu + 1.645 * post_sd)
width = hi - lo
y = rng.random(n) < sigmoid(z)

results = {
    "mean_width_A": float(width[~group_b].mean()),
    "mean_width_B": float(width[group_b].mean()),
}

# ------------------------------------------------------------------ fig 1: same score, different intervals
def fig_intervals(mode):
    c = style(mode)
    band = (p_hat > 0.045) & (p_hat < 0.055)
    idx_a = rng.choice(np.where(band & ~group_b)[0], 9, replace=False)
    idx_b = rng.choice(np.where(band & group_b)[0], 9, replace=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for k, (idx, col, off) in enumerate([(idx_a, TEAL, 10), (idx_b, INDIGO, 0)]):
        order = np.argsort(width[idx])
        for j, i in enumerate(idx[order]):
            ypos = off + j
            ax.plot([lo[i] * 100, hi[i] * 100], [ypos, ypos], color=col, lw=2,
                    solid_capstyle="round", alpha=0.9)
            ax.plot(p_hat[i] * 100, ypos, "o", ms=5, color=col,
                    mec="white", mew=0.9, zorder=3)
    ax.axvline(5.0, color=c["faint"], lw=0.8, ls=":")
    ax.text(5.0, 19.6, "score ≈ 5%", ha="center", fontsize=9, color=c["muted"])
    ax.set_yticks([14, 4], ["group A", "group B"])
    ax.set_xlabel("predicted probability of default (%), with 90% interval")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.2, 21)
    save(fig, "du-intervals", mode)


# ------------------------------------------------------------------ fig 2: what the audit sees / misses
def fig_audit(mode):
    c = style(mode)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")
    bins = np.quantile(p_hat, np.linspace(0, 1, 11))
    for grp, col, lab in [(~group_b, TEAL, "group A"), (group_b, INDIGO, "group B")]:
        mid, obs = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = grp & (p_hat >= a) & (p_hat < b)
            if m.sum() > 50:
                mid.append(p_hat[m].mean() * 100)
                obs.append(y[m].mean() * 100)
        axes[0].plot(mid, obs, "o-", ms=4.5, lw=1.6, color=col, label=lab,
                     mec="white", mew=0.7)
    top = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, top], [0, top], color=c["faint"], lw=0.8, ls=":")
    axes[0].set_xlabel("predicted PD (%)")
    axes[0].set_ylabel("observed default rate (%)")
    axes[0].set_title("what the audit sees: calibration", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=c["text"])

    w_bins = np.linspace(0, 0.45, 40)
    for grp, col, lab in [(~group_b, TEAL, "group A"), (group_b, INDIGO, "group B")]:
        axes[1].hist(width[grp], bins=w_bins, density=True, histtype="stepfilled",
                     alpha=0.45, color=col, label=lab)
        axes[1].axvline(width[grp].mean(), color=col, lw=1.4, ls="--")
    axes[1].set_xlabel("90% interval width")
    axes[1].set_ylabel("density")
    axes[1].set_title("what it misses: certainty", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=c["text"])
    save(fig, "du-audit", mode)


# ------------------------------------------------------------------ fig 3: downstream consequence
def fig_review(mode):
    c = style(mode)
    rule = width > 0.10                      # "uncertain -> manual review"
    share_a = rule[~group_b].mean() * 100
    share_b = rule[group_b].mean() * 100
    results["review_share_A"] = float(share_a)
    results["review_share_B"] = float(share_b)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.barh([1, 0], [share_a, share_b], height=0.5, color=[TEAL, INDIGO])
    ax.set_yticks([1, 0], ["group A", "group B"])
    for ypos, v in [(1, share_a), (0, share_b)]:
        ax.text(v + 1.5, ypos, f"{v:.0f}%", va="center", fontsize=10,
                color=c["text"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("share routed to manual review (interval width > 0.10)")
    ax.grid(axis="y", visible=False)
    save(fig, "du-review", mode)


for mode in MODES:
    fig_intervals(mode)
    fig_audit(mode)
    fig_review(mode)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
