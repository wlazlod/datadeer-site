"""Generate figures for the 'Why SHAP values are not points' note.

Synthetic credit data -> XGBoost PD model -> real SHAP values -> SVG figures
in light and dark variants, plus results.json with the numbers quoted in prose.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

SEED = 42
rng = np.random.default_rng(SEED)
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)

TEAL = "#009485"    # risk down / safe
INDIGO = "#526cfe"  # neutral second series
CORAL = "#e6695b"   # risk up

MODES = {
    "light": dict(text="#212121", muted="#757575", grid="#e3e3e3", faint="#bdbdbd"),
    "dark": dict(text="#dee2ef", muted="#8a91a8", grid="#3a3e4d", faint="#565b6e"),
}

FEATURES = [
    "utilization", "tenure_m", "n_closed_ok", "age",
    "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "overdue_now",
    "short_episodes", "phone_reuse",
]
LABELS = {
    "utilization": "limit utilization", "tenure_m": "tenure (months)",
    "n_closed_ok": "products closed on time", "age": "age",
    "max_dpd_3m": "max delay 3m", "max_dpd_6m": "max delay 6m",
    "max_dpd_12m": "max delay 12m", "overdue_now": "overdue now",
    "short_episodes": "short episodes", "phone_reuse": "phone reuse",
}


def make_data(n, rng):
    tenure = np.minimum(rng.gamma(2.0, 24.0, n), 140.0)
    n_closed = rng.poisson(np.clip(tenure / 18.0, 0.1, 8.0))
    util = rng.beta(1.6, 2.4, n)
    age = np.clip(rng.normal(40, 12, n), 19, 80)

    sev = rng.exponential(1.0, n) * (rng.random(n) < 0.38)     # latent delinquency
    dpd3 = np.round(np.clip(sev * 28, 0, 120) * (rng.random(n) < 0.7))
    dpd6 = dpd3 + np.round(np.clip(sev * 12 * rng.random(n), 0, 60))
    dpd12 = dpd6 + np.round(np.clip(sev * 10 * rng.random(n), 0, 60))
    overdue = np.round(np.clip(sev * 900 * rng.random(n), 0, 5000))

    episodes = rng.poisson(0.8 + 1.4 * (sev > 0), n)
    closes_on_time = (n_closed >= 4) & (dpd12 < 30)            # "trusted" pattern
    reuse = np.where(rng.random(n) < 0.005, rng.integers(1, 8, n), 0)

    # True log-odds: utilization hurts new customers, is ~neutral for trusted
    # ones, and paying a trusted profile down to zero *raises* risk slightly
    # (loss of the safe interaction). Episodes flip sign by closes_on_time.
    trusted = (tenure > 60) & (n_closed >= 4)
    z = (
        -3.1
        + 0.80 * np.tanh(dpd3 / 30.0)
        + 0.60 * np.tanh((dpd6 - dpd3) / 25.0)
        + 0.50 * np.tanh((dpd12 - dpd6) / 25.0)
        + 0.45 * np.tanh(overdue / 800.0)
        + np.where(trusted, 0.55 * np.clip(0.15 - util, 0, None) / 0.15,
                   1.60 * util)
        - 0.030 * (age - 40)
        - 0.25 * np.tanh(tenure / 60.0)
        + np.where(closes_on_time, -0.28, 0.30) * np.clip(episodes, 0, 6)
        + 0.85 * np.minimum(reuse, 3)
    )
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([util, tenure, n_closed, age, dpd3, dpd6, dpd12,
                         overdue, episodes, reuse])
    return X, y, closes_on_time


X, y, closes_ok = make_data(60_000, rng)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.5, colsample_bylevel=0.7, random_state=SEED,
    eval_metric="logloss",
)
model.fit(X, y)

bg = X[rng.choice(len(X), 200, replace=False)]
explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
S = explainer.shap_values(X[:4000])
base = float(np.ravel(explainer.expected_value)[0])

results = {"auc_note": "synthetic", "base_value": base}


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


# ---------------------------------------------------------------- applicant
# A "trusted" applicant: high utilization, long tenure, clean delinquency.
x_row = np.array([[0.86, 96.0, 7.0, 44.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0]])
sv_row = explainer.shap_values(x_row)[0]
pd_row = float(model.predict_proba(x_row)[0, 1])
results["applicant_pd"] = pd_row
results["applicant_util_shap"] = float(sv_row[0])

# ---------------------------------------------------------------- fig 1: waterfall
def fig_waterfall(mode):
    c = style(mode)
    order = np.argsort(-np.abs(sv_row))
    fx = base + sv_row.sum()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    level = base
    ys, names = [], []
    shown = order[:7]
    rest = np.setdiff1d(order, shown)
    items = [(LABELS[FEATURES[i]], sv_row[i]) for i in shown]
    if len(rest):
        items.append((f"{len(rest)} other features", sv_row[rest].sum()))
    for k, (name, v) in enumerate(items):
        color = CORAL if v > 0 else TEAL
        ax.barh(len(items) - k, v, left=level, height=0.55, color=color)
        ax.plot([level, level], [len(items) - k - 0.6, len(items) - k + 0.28],
                color=c["faint"], lw=0.7, ls=":")
        if abs(v) >= 0.08:
            xpos = level + v + (0.015 if v >= 0 else -0.015)
            ax.text(xpos, len(items) - k, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=c["muted"])
        level += v
        ys.append(len(items) - k)
        names.append(name)
    ax.axvline(base, color=c["faint"], lw=0.8, ls=":")
    ax.set_yticks(ys, names)
    ax.text(base, len(items) + 0.85, f"base value {base:+.2f}",
            ha="center", fontsize=9, color=c["muted"])
    ax.text(level, 0.15, f"f(x) = {level:+.2f}", ha="center", fontsize=9,
            color=c["text"], fontweight="bold")
    ax.set_xlabel("model output (log-odds)")
    lo = min(base, base + np.cumsum([v for _, v in items]).min())
    hi = max(base, base + np.cumsum([v for _, v in items]).max())
    ax.set_xlim(lo - 0.22, hi + 0.10)
    ax.set_ylim(-0.4, len(items) + 1.3)
    ax.grid(axis="y", visible=False)
    save(fig, "waterfall", mode)


# ---------------------------------------------------------------- fig 2: what-if
util_grid = np.linspace(0.01, 0.9, 140)


def whatif_curve(row):
    rows = np.repeat(row, len(util_grid), axis=0)
    rows[:, 0] = util_grid
    return model.predict_proba(rows)[:, 1] * 100


def fig_whatif(mode):
    c = style(mode)
    pd_curve = whatif_curve(x_row.copy())
    # SHAP-implied straight-line reading: remove the whole utilization bar
    margin = np.log(pd_row / (1 - pd_row))
    implied_at_zero = sigmoid(margin - sv_row[0]) * 100
    # "followed the letter": utilization -> 0.02 AND closed products -> 2
    x_after = x_row.copy(); x_after[0, 0] = 0.02; x_after[0, 2] = 2.0
    pd_after = float(model.predict_proba(x_after)[0, 1]) * 100
    results["pd_before_pct"] = pd_row * 100
    results["pd_after_letter_pct"] = pd_after
    results["pd_implied_pct"] = float(implied_at_zero)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(util_grid, pd_curve, color=TEAL, lw=2,
            label="model, this applicant")
    ax.plot([x_row[0, 0], 0.02], [pd_row * 100, implied_at_zero],
            color=INDIGO, lw=1.6, ls="--", label="what the bar suggests")
    ax.plot(x_row[0, 0], pd_row * 100, "o", ms=7, color=TEAL, mec="white", mew=1.2)
    ax.plot(0.02, pd_after, "X", ms=9, color=CORAL, mec="white", mew=1.0)
    ax.annotate("today", (x_row[0, 0], pd_row * 100), textcoords="offset points",
                xytext=(6, 8), fontsize=9, color=c["text"])
    ax.annotate("after following the letter\n(paid to zero, closed products)",
                (0.02, pd_after), textcoords="offset points", xytext=(14, -6),
                fontsize=9, color=CORAL, va="top")
    ax.set_xlabel("limit utilization")
    ax.set_ylabel("predicted PD (%)")
    ax.set_ylim(top=max(pd_after, pd_row * 100) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              labelcolor=c["text"])
    save(fig, "whatif", mode)


# ---------------------------------------------------------------- fig 3: day over day
def fig_daydelta(mode):
    c = style(mode)
    x_mon = np.array([[0.55, 30.0, 2.0, 33.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    x_tue = x_mon.copy()
    x_tue[0, 4] = 31.0; x_tue[0, 5] = 31.0; x_tue[0, 6] = 31.0  # one new episode
    sv_mon = explainer.shap_values(x_mon)[0]
    sv_tue = explainer.shap_values(x_tue)[0]
    pd_mon = float(model.predict_proba(x_mon)[0, 1]) * 100
    pd_tue = float(model.predict_proba(x_tue)[0, 1]) * 100
    results["pd_mon_pct"], results["pd_tue_pct"] = pd_mon, pd_tue
    d_in = (x_tue - x_mon)[0]
    d_sv = sv_tue - sv_mon
    changed = d_in != 0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True, layout="constrained")
    ypos = np.arange(len(FEATURES))[::-1]
    axes[0].barh(ypos, np.where(changed, 1.0, 0.0), height=0.55,
                 color=[CORAL if ch else MODES[mode]["grid"] for ch in changed])
    axes[0].set_title("inputs that changed", fontsize=10)
    axes[0].set_xticks([0, 1], ["no", "yes"])
    axes[0].set_yticks(ypos, [LABELS[f] for f in FEATURES])
    colors = [CORAL if ch else INDIGO for ch in changed]
    axes[1].barh(ypos, d_sv, height=0.55, color=colors)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=CORAL, label="input changed"),
                            Patch(color=INDIGO, label="input unchanged")],
                   frameon=False, fontsize=8, loc="lower right",
                   labelcolor=MODES[mode]["text"])
    axes[1].axvline(0, color=c["faint"], lw=0.8)
    axes[1].set_title("change in SHAP attribution", fontsize=10)
    axes[1].set_xlabel("Δ SHAP (log-odds)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"one delinquency event · PD {pd_mon:.1f}% → {pd_tue:.1f}%",
                 fontsize=10, color=c["muted"])
    save(fig, "daydelta", mode)


# ---------------------------------------------------------------- fig 4: importance
mean_abs = np.abs(S).mean(axis=0)
results["rank_ungrouped"] = [FEATURES[i] for i in np.argsort(-mean_abs)]

FAMILY = ["max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "overdue_now"]


def fig_importance(mode):
    c = style(mode)
    fam_idx = [FEATURES.index(f) for f in FAMILY]
    single = [f for f in FEATURES if f not in FAMILY]
    # grouped: sum SHAP inside the family per row, then mean |.|
    grouped_vals = {LABELS[f]: np.abs(S[:, FEATURES.index(f)]).mean() for f in single}
    grouped_vals["delinquency (grouped)"] = np.abs(S[:, fam_idx].sum(axis=1)).mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), layout="constrained")
    order = np.argsort(-mean_abs)[:7]
    names = [LABELS[FEATURES[i]] for i in order]
    vals = mean_abs[order]
    cols = [CORAL if FEATURES[i] in FAMILY else
            (INDIGO if FEATURES[i] == "age" else MODES[mode]["faint"])
            for i in order]
    axes[0].barh(np.arange(len(vals))[::-1], vals, height=0.55, color=cols)
    axes[0].set_yticks(np.arange(len(vals))[::-1], names)
    axes[0].set_title("ungrouped mean |SHAP|", fontsize=10)

    g_sorted = sorted(grouped_vals.items(), key=lambda kv: -kv[1])[:7]
    g_names = [k for k, _ in g_sorted]
    g_vals = [v for _, v in g_sorted]
    g_cols = [CORAL if "delinquency" in k else (INDIGO if k == "age" else
              MODES[mode]["faint"]) for k in g_names]
    axes[1].barh(np.arange(len(g_vals))[::-1], g_vals, height=0.55, color=g_cols)
    axes[1].set_yticks(np.arange(len(g_vals))[::-1], g_names)
    axes[1].set_title("grouped mean |SHAP|", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean |SHAP| (log-odds)")
    results["rank_grouped"] = g_names
    save(fig, "importance", mode)


# ---------------------------------------------------------------- fig 5: impossible rows
def fig_impossible(mode):
    c = style(mode)
    idx = rng.choice(len(X), 700, replace=False)
    real3, real6 = X[idx, 4], X[idx, 5]
    # rows interventional SHAP actually queries: keep dpd_3m from the applicant,
    # draw dpd_6m from the background marginal
    q3 = np.full(len(bg), 90.0)
    q6 = bg[:, 5]
    bad = q6 < q3
    results["impossible_share_pct"] = float(bad.mean() * 100)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.scatter(real3, real6, s=9, color=TEAL, alpha=0.35, lw=0,
               label="real applicants")
    ax.scatter(q3[bad], q6[bad], s=22, color=CORAL, lw=0, zorder=3,
               label="rows SHAP queries (impossible)")
    ax.scatter(q3[~bad], q6[~bad], s=22, color=INDIGO, lw=0, zorder=3,
               label="rows SHAP queries (plausible)")
    lims = [0, 130]
    ax.plot(lims, lims, color=c["faint"], lw=0.8, ls=":")
    ax.text(108, 96, "6m ≥ 3m", fontsize=9, color=c["muted"], rotation=38)
    ax.set_xlabel("max delay over 3 months (days)")
    ax.set_ylabel("max delay over 6 months (days)")
    ax.set_xlim(-3, 130); ax.set_ylim(-3, 130)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["text"])
    save(fig, "impossible", mode)


# ---------------------------------------------------------------- fig 6: mean vs max
def fig_meanmax(mode):
    c = style(mode)
    mx = np.abs(S).max(axis=0)
    results["phone_mean_rank"] = int(np.argsort(-mean_abs).tolist().index(FEATURES.index("phone_reuse")) + 1)
    results["phone_max_rank"] = int(np.argsort(-mx).tolist().index(FEATURES.index("phone_reuse")) + 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for i, f in enumerate(FEATURES):
        is_phone = f == "phone_reuse"
        ax.scatter(mean_abs[i], mx[i], s=55 if is_phone else 30,
                   color=CORAL if is_phone else INDIGO, zorder=3)
        if is_phone or mean_abs[i] > np.percentile(mean_abs, 60):
            ax.annotate(LABELS[f], (mean_abs[i], mx[i]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=9, color=CORAL if is_phone else c["muted"])
    ax.set_xlim(right=float(mean_abs.max()) * 1.45)
    ax.set_xlabel("mean |SHAP|  (the usual ranking)")
    ax.set_ylabel("max |SHAP|  (when it fires)")
    save(fig, "meanmax", mode)


# ---------------------------------------------------------------- fig 7: dependence
model_1f = xgb.XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.15,
                             random_state=SEED, eval_metric="logloss")
model_1f.fit(X[:, [8]], y)


def fig_dependence(mode):
    c = style(mode)
    ep_idx = FEATURES.index("short_episodes")
    epi = X[:4000, ep_idx] + rng.normal(0, 0.09, 4000)  # jitter
    sv_epi = S[:, ep_idx]
    grp = closes_ok[:4000]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained")
    grid = np.linspace(0, 8, 60).reshape(-1, 1)
    axes[0].plot(grid, model_1f.predict_proba(grid)[:, 1] * 100, color=TEAL, lw=2)
    axes[0].set_title("a model of the feature alone", fontsize=10)
    axes[0].set_xlabel("short delinquency episodes")
    axes[0].set_ylabel("predicted PD (%)")

    axes[1].scatter(epi[grp], sv_epi[grp], s=8, alpha=0.35, lw=0, color=TEAL,
                    label="always closes on time")
    axes[1].scatter(epi[~grp], sv_epi[~grp], s=8, alpha=0.35, lw=0, color=CORAL,
                    label="missed closes")
    axes[1].axhline(0, color=c["faint"], lw=0.8)
    axes[1].set_title("the feature inside the full model", fontsize=10)
    axes[1].set_xlabel("short delinquency episodes")
    axes[1].set_ylabel("SHAP value (log-odds)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left",
                   labelcolor=c["text"], markerscale=2.2)
    save(fig, "dependence", mode)


for mode in MODES:
    fig_waterfall(mode)
    fig_whatif(mode)
    fig_daydelta(mode)
    fig_importance(mode)
    fig_impossible(mode)
    fig_meanmax(mode)
    fig_dependence(mode)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
