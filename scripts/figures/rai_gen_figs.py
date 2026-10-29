"""Generate figures for the 'Four questions before an algorithm decides about people' note.

Synthetic population -> self-selected applicants -> XGBoost PD model -> group,
selection, calibration and drift diagnostics, plus an airline-overbooking
example of asymmetric costs -> SVG figures in light and dark variants, plus
results.json with the numbers quoted in prose.
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
from scipy.stats import binom
from sklearn.metrics import roc_auc_score

SEED = 42
rng = np.random.default_rng(SEED)
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)

TEAL = "#009485"    # good / calibrated
INDIGO = "#526cfe"  # neutral second series
CORAL = "#e6695b"   # warning / miscalibrated

MODES = {
    "light": dict(text="#212121", muted="#757575", grid="#e3e3e3", faint="#bdbdbd"),
    "dark": dict(text="#dee2ef", muted="#8a91a8", grid="#3a3e4d", faint="#565b6e"),
}

# The model sees everything except sex. The last three columns are shopping
# behaviour: they carry no risk of their own, only correlation with sex.
FEATURES = [
    "age", "tenure_m", "utilization", "max_dpd_12m", "n_active_loans",
    "income_stability", "mobile_share", "auto_share", "night_share",
]
PROXIES = ["mobile_share", "auto_share", "night_share"]
PROXY_LABELS = {
    "mobile_share": "main shopping channel",
    "auto_share": "automotive purchases",
    "night_share": "time of day online",
}
AGE_BANDS = [(18, 25), (26, 35), (36, 50), (51, 80)]
CUTOFF = 0.08          # approve when the predicted PD is below 8%


def make_data(n, rng, active_loans_shift=0.0):
    sex = (rng.random(n) < 0.5).astype(int)
    age = np.clip(rng.gamma(6.0, 6.5, n) + 18, 18, 80)
    tenure = np.minimum(rng.gamma(1.5, 20.0, n), (age - 18) * 12)
    util = rng.beta(1.6, 2.4, n)
    dpd = np.round(rng.exponential(1.0, n) * (rng.random(n) < 0.3) * 25)
    loans = rng.poisson(1.2, n) + active_loans_shift
    stab = np.clip(rng.beta(4, 2, n) - 0.06 * sex + 0.002 * (age - 40), 0, 1)

    mobile = np.clip(rng.beta(2 + 2 * sex, 3, n), 0, 1)
    auto = np.clip(rng.beta(2, 6 - 3 * sex, n), 0, 1)
    night = np.clip(rng.beta(2 + 1.2 * sex, 4, n), 0, 1)

    z = (
        -3.6
        + 1.1 * np.tanh(dpd / 30.0)
        + 1.5 * util
        + 0.35 * np.minimum(loans, 6)
        - 1.6 * (stab - 0.6)
        - 0.3 * np.tanh(tenure / 48.0)
        - 0.003 * (age - 40)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    X = np.column_stack([age, tenure, util, dpd, loans, stab, mobile, auto, night])
    return X, y, sex


def applies(X, rng):
    """Who applies. Financial strain drives applications at every age, but far
    more strongly among the young: a young person with savings rarely needs a
    deferred payment, an older one uses it for convenience."""
    age, _, util, dpd, loans, stab = X[:, :6].T
    strain = util + 0.6 * np.tanh(dpd / 30.0) + 0.25 * np.minimum(loans, 6) - stab
    k = 0.2 + 10.0 * np.exp(-(age - 18) / 10.0)
    return rng.random(len(X)) < 1 / (1 + np.exp(-(-0.3 + k * strain)))


def make_applicants(n, rng, **kw):
    parts = []
    while sum(len(p[1]) for p in parts) < n:
        X_, y_, s_ = make_data(n, rng, **kw)
        m = applies(X_, rng)
        parts.append((X_[m], y_[m], s_[m]))
    return tuple(np.concatenate([p[i] for p in parts])[:n] for i in range(3))


X, y, sex = make_applicants(60_000, rng)
Xt, yt, sext = make_applicants(40_000, rng)
Xp, yp, _ = make_data(40_000, rng)           # everyone, applicant or not
params = dict(n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.9,
              colsample_bytree=0.7, random_state=SEED, eval_metric="logloss")
model = xgb.XGBClassifier(**params).fit(X, y)
pd_t = model.predict_proba(Xt)[:, 1]
approved = pd_t < CUTOFF

results = {"auc": roc_auc_score(yt, pd_t), "default_rate": float(yt.mean()),
           "approval_rate": float(approved.mean())}


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


# ---------------------------------------------------------------- reasons
# Interventional TreeSHAP for one declined applicant, summed into domains a
# person understands. Own generator, so the numbers below do not shift.
DOMAINS = {
    "repayment history": ["max_dpd_12m"],
    "current debt": ["utilization", "n_active_loans"],
    "customer tenure": ["tenure_m"],
    "income stability": ["income_stability"],
    "everything else": ["age"] + PROXIES,
}
bg = Xt[np.random.default_rng(SEED + 1).choice(len(Xt), 200, replace=False)]
explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
cand = np.where(~approved & (Xt[:, FEATURES.index("max_dpd_12m")] == 0))[0][:400]
S_c = explainer.shap_values(Xt[cand])
D_c = np.column_stack([S_c[:, [FEATURES.index(f) for f in fs]].sum(axis=1)
                       for fs in DOMAINS.values()])
# The applicant from the talk: clean history and long tenure help, debt decides.
ok = (D_c[:, 0] < 0) & (D_c[:, 2] < 0) & (D_c[:, 3] > 0) & (D_c.argmax(axis=1) == 1)
pick = int(np.argmax(np.where(ok, np.minimum.reduce([-D_c[:, 0], -D_c[:, 2], D_c[:, 3]]), -np.inf)))
reasons = D_c[pick]
results["reasons"] = {
    "pd": float(pd_t[cand[pick]]),
    "base_pd": float(1 / (1 + np.exp(-np.ravel(explainer.expected_value)[0]))),
    **{k: round(float(v), 3) for k, v in zip(DOMAINS, reasons)},
}


def fig_reasons(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(DOMAINS)[::-1]
    vals = reasons[::-1]
    ax.barh(names, vals, color=[TEAL if v < 0 else CORAL for v in vals], height=0.55)
    ax.axvline(0, color=c["faint"], lw=1)
    lim = np.abs(vals).max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.text(-lim, len(names) - 0.35, "◀ lowers risk", color=TEAL, fontsize=10, fontweight="bold")
    ax.text(lim, len(names) - 0.35, "raises risk ▶", color=CORAL, fontsize=10,
            fontweight="bold", ha="right")
    ax.set_xlabel("contribution to the risk score (log-odds)")
    ax.grid(axis="y", visible=False)
    save(fig, "reasons", mode)


# ---------------------------------------------------------------- approval by age band
age_t = Xt[:, 0]
band_of = np.digitize(np.floor(age_t), [lo for lo, _ in AGE_BANDS[1:]])
band_names = [f"{lo}–{hi}" if hi < 80 else f"{lo}+" for lo, hi in AGE_BANDS]
band_rates = np.array([approved[band_of == b].mean() for b in range(len(AGE_BANDS))])
results["approval_by_age"] = dict(zip(band_names, band_rates.round(3).tolist()))
results["age_impact_ratio"] = float(band_rates.min() / band_rates.max())


def fig_approval(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    warn = 0.8 * band_rates.max()
    colors = [CORAL if r < warn else TEAL for r in band_rates]
    ax.bar(band_names, band_rates * 100, color=colors, width=0.6)
    ax.axhline(warn * 100, color=c["muted"], ls="--", lw=1)
    ax.text(-0.3, warn * 100 + 1.5, "4/5 of the best group",
            ha="left", color=c["muted"], fontsize=9)
    for i, r in enumerate(band_rates):
        ax.text(i, r * 100 - 6, f"{r:.0%}", ha="center", fontsize=10, color="white")
    ax.set_ylabel("approval rate (%)")
    ax.set_xlabel("applicant age")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    save(fig, "approval", mode)


# ---------------------------------------------------------------- selection
# The same model, three populations: everyone in the age band, those who apply,
# and applicants reweighted so every band has the same financial profile
# (utilization, delinquency, active loans, income stability).
band_p = np.digitize(np.floor(Xp[:, 0]), [lo for lo, _ in AGE_BANDS[1:]])
appr_p = model.predict_proba(Xp)[:, 1] < CUTOFF
pop_rates = np.array([appr_p[band_p == b].mean() for b in range(len(AGE_BANDS))])

ALIGN = [FEATURES.index(f) for f in
         ("utilization", "max_dpd_12m", "n_active_loans", "income_stability")]
band_tr = np.digitize(np.floor(X[:, 0]), [lo for lo, _ in AGE_BANDS[1:]])
band_clf = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                             random_state=SEED, eval_metric="mlogloss")
band_clf.fit(X[:, ALIGN], band_tr)
p_band_x = band_clf.predict_proba(Xt[:, ALIGN])[np.arange(len(band_of)), band_of]
prior = np.bincount(band_of, minlength=len(AGE_BANDS)) / len(band_of)
w = np.clip(prior[band_of] / p_band_x, 0, 20)       # importance weights, clipped
aligned_rates = np.array([np.average(approved[band_of == b], weights=w[band_of == b])
                          for b in range(len(AGE_BANDS))])
apply_rate = [float(applies(Xp[band_p == b], rng).mean()) for b in range(len(AGE_BANDS))]
results["selection"] = {
    "population": dict(zip(band_names, pop_rates.round(3).tolist())),
    "applicants": dict(zip(band_names, band_rates.round(3).tolist())),
    "aligned": dict(zip(band_names, aligned_rates.round(3).tolist())),
    "ratio_population": float(pop_rates.min() / pop_rates.max()),
    "ratio_applicants": float(band_rates.min() / band_rates.max()),
    "ratio_aligned": float(aligned_rates.min() / aligned_rates.max()),
    "apply_rate_by_band": dict(zip(band_names, np.round(apply_rate, 3).tolist())),
}


def fig_selection(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    xs = np.arange(len(AGE_BANDS))
    wd = 0.26
    series = [(band_rates, CORAL, "applicants, as observed"),
              (aligned_rates, INDIGO, "applicants, same financial profile"),
              (pop_rates, TEAL, "whole population")]
    for k, (r, col, lab) in enumerate(series):
        ax.bar(xs + (k - 1) * wd, r * 100, wd, color=col, label=lab)
    ax.set_xticks(xs, band_names)
    ax.set_ylabel("approval rate (%)")
    ax.set_xlabel("applicant age")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "selection", mode)


# ---------------------------------------------------------------- proxies
# Sex is not a model input. How well do the proxies -- or the score itself -- recover it?
def recover_auc(cols):
    clf = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                            random_state=SEED, eval_metric="logloss")
    clf.fit(X[:, cols], sex)
    return roc_auc_score(sext, clf.predict_proba(Xt[:, cols])[:, 1])


proxy_idx = [FEATURES.index(p) for p in PROXIES]
proxy_auc = {PROXY_LABELS[p]: recover_auc([FEATURES.index(p)]) for p in PROXIES}
proxy_auc["all three together"] = recover_auc(proxy_idx)
results["proxy_auc"] = {k: round(v, 3) for k, v in proxy_auc.items()}
results["approval_by_sex"] = [float(approved[sext == s].mean()) for s in (0, 1)]

# Drop the proxies and refit: what does it cost, what does it buy?
keep = [i for i in range(len(FEATURES)) if i not in proxy_idx]
model_np = xgb.XGBClassifier(**params).fit(X[:, keep], y)
pd_np = model_np.predict_proba(Xt[:, keep])[:, 1]
results["auc_without_proxies"] = roc_auc_score(yt, pd_np)
results["approval_by_sex_without_proxies"] = [
    float((pd_np < CUTOFF)[sext == s].mean()) for s in (0, 1)]


def fig_proxy(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    names = list(proxy_auc)[::-1]
    vals = [proxy_auc[k] for k in names]
    colors = [CORAL if k == "all three together" else INDIGO for k in names]
    ax.barh(names, vals, color=colors, height=0.55)
    ax.axvline(0.5, color=c["muted"], ls="--", lw=1)
    ax.text(0.505, len(names) - 0.55, "coin flip", color=c["muted"], fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xlim(0.45, max(vals) + 0.08)
    ax.set_xlabel("AUC of recovering sex from the feature")
    ax.grid(axis="y", visible=False)
    save(fig, "proxy", mode)


# ---------------------------------------------------------------- fairness frontier
# Each approved good repays +1, each approved bad loses LOSS; LOSS is set so the
# 8% cutoff is exactly break-even. For each target impact ratio, lift every band
# below it to target x best rate by approving its next-lowest-PD applicants, and
# trace profit against the ratio achieved.
LOSS = (1 - CUTOFF) / CUTOFF
best = int(np.argmax(band_rates))
frontier = []
for target in np.linspace(band_rates.min() / band_rates.max(), 1.0, 200):
    appr = approved.copy()
    for b in range(len(AGE_BANDS)):
        m = band_of == b
        if b != best and appr[m].mean() < target * band_rates[best]:
            appr[m] = pd_t[m] <= np.quantile(pd_t[m], target * band_rates[best])
    rates = np.array([appr[band_of == b].mean() for b in range(len(AGE_BANDS))])
    profit = (appr & (yt == 0)).sum() - LOSS * (appr & (yt == 1)).sum()
    frontier.append((rates.min() / rates.max(), profit, (appr & (yt == 1)).sum() / (yt == 1).sum()))
frontier = np.array(frontier)
frontier = frontier[np.argsort(frontier[:, 0])]
frontier[:, 1] = frontier[:, 1] / frontier[0, 1]
i80 = int(np.argmax(frontier[:, 0] >= 0.8))
results["frontier"] = {
    "impact_ratio_unadjusted": float(frontier[0, 0]),
    "impact_ratio_at_4_5": float(frontier[i80, 0]),
    "profit_at_4_5_relative": float(frontier[i80, 1]),
    "profit_at_parity_relative": float(frontier[-1, 1]),
    "profit_min_relative": float(frontier[:, 1].min()),
    "defaulters_approved_unadjusted": float(frontier[0, 2]),
    "defaulters_approved_at_4_5": float(frontier[i80, 2]),
}


def fig_frontier(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(frontier[:, 0], frontier[:, 1] * 100, color=INDIGO, lw=2)
    ax.scatter(*frontier[0, :2] * [1, 100], color=CORAL, zorder=3, s=40)
    ax.annotate("accuracy only", frontier[0, :2] * [1, 100], xytext=(8, 6),
                textcoords="offset points", fontsize=9, color=c["text"])
    ax.scatter(*frontier[i80, :2] * [1, 100], color=TEAL, zorder=3, s=40)
    ax.annotate("meets the 4/5 rule", frontier[i80, :2] * [1, 100], xytext=(8, 6),
                textcoords="offset points", fontsize=9, color=c["text"])
    ax.axvline(0.8, color=c["muted"], ls="--", lw=1)
    ax.set_xlabel("age impact ratio (lowest / highest approval rate)")
    ax.set_ylabel("portfolio profit (% of accuracy-only)")
    ax.set_ylim(95, 101)
    save(fig, "frontier", mode)


# ---------------------------------------------------------------- calibration
# Same features, same algorithm, but trained on a 1:1 resampled set and never
# corrected back -- a common shortcut for "imbalanced" data.
bad, good = np.where(y == 1)[0], np.where(y == 0)[0]
bal = np.concatenate([bad, rng.choice(good, len(bad), replace=False)])
model_bal = xgb.XGBClassifier(**params).fit(X[bal], y[bal])
pd_bal = model_bal.predict_proba(Xt)[:, 1]
results["auc_balanced"] = roc_auc_score(yt, pd_bal)
results["mean_pd_calibrated"] = float(pd_t.mean())
results["mean_pd_balanced"] = float(pd_bal.mean())


def reliability(p, n_bins=10):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return (np.array([p[idx == b].mean() for b in range(n_bins)]),
            np.array([yt[idx == b].mean() for b in range(n_bins)]))


def fig_calibration(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = 0.85
    ax.plot([0, lim], [0, lim], color=c["faint"], ls="--", lw=1)
    for p, col, lab in [(pd_t, TEAL, "calibrated"), (pd_bal, CORAL, "trained on 1:1 sample")]:
        mp, fr = reliability(p)
        ax.plot(mp, fr, "o-", color=col, lw=1.8, ms=4, label=lab)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted PD")
    ax.set_ylabel("observed default rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "calibration", mode)


# ---------------------------------------------------------------- drift
# 120 days of applications. On day 80 the bureau starts counting closed-but-not-
# yet-deregistered loans as active. Nothing about the applicants changes.
DAYS, SWITCH = 120, 80
ref = Xt[:, FEATURES.index("n_active_loans")]
bins = np.arange(-0.5, 9.5, 1.0)
ref_hist = np.histogram(np.minimum(ref, 8), bins)[0] / len(ref) + 1e-4


def psi(x):
    h = np.histogram(np.minimum(x, 8), bins)[0] / len(x) + 1e-4
    return float(np.sum((h - ref_hist) * np.log(h / ref_hist)))


daily_psi, daily_pd = [], []
for d in range(DAYS):
    Xd, _, _ = make_applicants(1500, rng)
    if d >= SWITCH:
        Xd[:, FEATURES.index("n_active_loans")] += rng.poisson(0.8, len(Xd))
    daily_psi.append(psi(Xd[:, FEATURES.index("n_active_loans")]))
    daily_pd.append(float(model.predict_proba(Xd)[:, 1].mean()))
daily_psi, daily_pd = np.array(daily_psi), np.array(daily_pd)
results["drift"] = {
    "psi_before": float(daily_psi[:SWITCH].mean()),
    "psi_after": float(daily_psi[SWITCH:].mean()),
    "mean_pd_before": float(daily_pd[:SWITCH].mean()),
    "mean_pd_after": float(daily_pd[SWITCH:].mean()),
}


def fig_drift(mode):
    c = style(mode)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    days = np.arange(DAYS)
    a1.plot(days, daily_psi, color=INDIGO, lw=1.4)
    a1.axhline(0.25, color=c["muted"], ls="--", lw=1)
    a1.text(2, 0.27, "alert threshold (PSI 0.25)", color=c["muted"], fontsize=9)
    a1.set_ylabel("PSI, active loans")
    a2.plot(days, daily_pd * 100, color=CORAL, lw=1.4)
    a2.set_ylabel("mean PD (%)")
    a2.set_xlabel("day")
    for a in (a1, a2):
        a.axvline(SWITCH, color=c["faint"], lw=1)
    a1.text(SWITCH - 1.5, 0.36, "vendor changes reporting",
            ha="right", fontsize=9, color=c["text"])
    save(fig, "drift", mode)


# ---------------------------------------------------------------- overbooking
# Not credit: a 180-seat flight where each booking shows up independently. The
# airline sells the number of seats that minimises expected cost *under the
# predicted* no-show rate; the true rate is 10%. An empty seat loses EMPTY, a
# passenger denied boarding costs BUMP (EU261 compensation for a medium-haul
# flight plus rebooking and goodwill).
SEATS, TRUE_NOSHOW, EMPTY, BUMP = 180, 0.10, 90.0, 600.0


def expected_cost(sold, noshow):
    k = np.arange(sold + 1)
    pk = binom.pmf(k, sold, 1 - noshow)
    return float(np.sum(pk * (EMPTY * np.clip(SEATS - k, 0, None)
                              + BUMP * np.clip(k - SEATS, 0, None))))


def best_sold(noshow):
    return min(range(SEATS, SEATS + 40), key=lambda s: expected_cost(s, noshow))


bias = np.linspace(-0.05, 0.05, 41)
ob_cost = np.array([expected_cost(best_sold(TRUE_NOSHOW + b), TRUE_NOSHOW) for b in bias])
ob_opt = expected_cost(best_sold(TRUE_NOSHOW), TRUE_NOSHOW)
i_lo, i_hi = int(np.argmin(np.abs(bias + 0.03))), int(np.argmin(np.abs(bias - 0.03)))
results["overbooking"] = {
    "optimal_sold": best_sold(TRUE_NOSHOW), "optimal_cost": ob_opt,
    "sold_if_under_3pp": best_sold(TRUE_NOSHOW - 0.03),
    "extra_cost_if_under_3pp": float(ob_cost[i_lo] - ob_opt),
    "sold_if_over_3pp": best_sold(TRUE_NOSHOW + 0.03),
    "extra_cost_if_over_3pp": float(ob_cost[i_hi] - ob_opt),
}


def fig_overbooking(mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    extra = ob_cost - ob_opt
    ax.plot(bias * 100, extra, color=INDIGO, lw=2)
    ax.fill_between(bias[bias <= 0] * 100, 0, extra[bias <= 0], color=TEAL, alpha=0.18)
    ax.fill_between(bias[bias >= 0] * 100, 0, extra[bias >= 0], color=CORAL, alpha=0.18)
    ax.axvline(0, color=c["faint"], lw=1)
    ax.text(-4.8, extra.max() * 0.82, "predicts too few no-shows:\nempty seats",
            fontsize=9, color=c["text"])
    ax.text(4.8, extra.max() * 0.82, "predicts too many no-shows:\npassengers left at the gate",
            fontsize=9, color=c["text"], ha="right")
    ax.set_xlabel("error in the predicted no-show rate (percentage points)")
    ax.set_ylabel("extra cost per flight (€)")
    save(fig, "overbooking", mode)


for mode in MODES:
    fig_reasons(mode)
    fig_selection(mode)
    fig_overbooking(mode)
    fig_approval(mode)
    fig_proxy(mode)
    fig_frontier(mode)
    fig_calibration(mode)
    fig_drift(mode)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
