"""
Statistical analysis of item picking durations.

Consumes the processed dataset from pipeline.py and produces:
  * output/stats.json      — machine-readable results
  * output/by_author.csv   — per-operator productivity table
  * output/by_category.csv — per-category duration table
  * output/figs/*.png      — figures
and prints a text summary.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import process, BURST_GAP_S, BURST_MIN_RUN, BREAK_CAP_S, DUP_GAP_S

OUT = Path("output"); FIGS = OUT / "figs"
OUT.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3})
BLUE, ORANGE, GREY = "#2a6f9e", "#e08a1e", "#8a8a8a"

d = process()
live = d.loc[d["is_live_valid"], "duration_s"].to_numpy()
bulk = d.loc[d["is_bulk"], "gap_s"].dropna().to_numpy()  # bulk cadence for comparison
res: dict = {}


def q(a, p):  # percentile helper
    return float(np.percentile(a, p))


# ---------------------------------------------------------------------------
# 1. Dataset composition
# ---------------------------------------------------------------------------
comp = d["duration_reason"].value_counts()
res["composition"] = {
    "rows": int(len(d)),
    "orders": int(d["order"].nunique()),
    "operators": int(d["author"].nunique()),
    "skus": int(d["sku"].nunique()),
    "date_min": str(d["scan"].min()), "date_max": str(d["scan"].max()),
    "lines_by_reason": {k: int(v) for k, v in comp.items()},
    "pct_bulk_of_all": round(100 * d["is_bulk"].mean(), 1),
    "bulk_serial_share": round(100 * d.loc[d.is_bulk, "is_serial"].mean(), 1),
    "live_serial_share": round(100 * d.loc[d.is_live_valid, "is_serial"].mean(), 1),
    "orders_majority_bulk": int((d.groupby("order")["is_bulk"].mean() > .5).sum()),
}

# ---------------------------------------------------------------------------
# 2. Live picking duration distribution
# ---------------------------------------------------------------------------
res["live_duration_s"] = {
    "n": int(live.size),
    "mean": round(float(live.mean()), 1),
    "sd": round(float(live.std(ddof=1)), 1),
    "min": round(float(live.min()), 1),
    **{f"p{p}": round(q(live, p), 1) for p in (5, 10, 25, 50, 75, 90, 95, 99)},
    "max": round(float(live.max()), 1),
    "geometric_mean": round(float(np.exp(np.log(live).mean())), 1),
}
# Lognormal fit (durations are strongly right-skewed / multiplicative)
shape, loc, scale = stats.lognorm.fit(live, floc=0)
res["live_duration_s"]["lognormal"] = {
    "sigma": round(float(shape), 3), "median_est": round(float(scale), 1),
    "mu": round(float(np.log(scale)), 3)}
# Bootstrap 95% CI for the median
rng = np.random.default_rng(0)
boot = [np.median(rng.choice(live, live.size, replace=True)) for _ in range(2000)]
res["live_duration_s"]["median_ci95"] = [round(q(boot, 2.5), 1), round(q(boot, 97.5), 1)]

# ---------------------------------------------------------------------------
# 3. Bulk (mass-scan) cadence vs live — do the regimes really differ?
# ---------------------------------------------------------------------------
u, p = stats.mannwhitneyu(live, bulk, alternative="greater")
# rank-biserial effect size (positive => live gaps stochastically longer)
rbc = 2 * u / (live.size * bulk.size) - 1
res["live_vs_bulk"] = {
    "live_median_s": round(float(np.median(live)), 1),
    "bulk_median_s": round(float(np.median(bulk)), 1),
    "mannwhitney_U": float(u), "p_value": float(p),
    "rank_biserial": round(float(rbc), 3),
    "naive_all_gap_median_s": round(float(d["gap_s"].dropna().median()), 1),
}

# ---------------------------------------------------------------------------
# 4. Naive vs corrected mean — the cost of ignoring mass-scanning
# ---------------------------------------------------------------------------
naive = d["gap_s"].dropna()
naive = naive[(naive >= DUP_GAP_S) & (naive <= BREAK_CAP_S)]  # same bounds, no regime split
res["naive_bias"] = {
    "naive_mean_s": round(float(naive.mean()), 1),
    "naive_median_s": round(float(naive.median()), 1),
    "corrected_mean_s": round(float(live.mean()), 1),
    "corrected_median_s": round(float(np.median(live)), 1),
    "median_understated_pct": round(100 * (1 - naive.median() / np.median(live)), 1),
}

# ---------------------------------------------------------------------------
# 5. Per-operator productivity (live picks only, >=30 valid picks)
# ---------------------------------------------------------------------------
la = d[d["is_live_valid"]].groupby("author")["duration_s"]
by_auth = pd.DataFrame({
    "n_live": la.size(),
    "median_s": la.median(),
    "mean_s": la.mean(),
    "picks_per_hour": 3600 / la.median(),
})
by_auth["bulk_share_pct"] = (d.groupby("author")["is_bulk"].mean() * 100).round(1)
by_auth["total_lines"] = d.groupby("author").size()
by_auth = by_auth[by_auth["n_live"] >= 30].sort_values("median_s")
by_auth.round(1).to_csv(OUT / "by_author.csv")
# Kruskal-Wallis across operators (do they really differ?)
grps = [g.to_numpy() for _, g in la if g.size >= 30]
H, pk = stats.kruskal(*grps)
res["operators"] = {
    "n_qualified": int(len(by_auth)),
    "fastest": by_auth.index[0], "fastest_median_s": round(float(by_auth["median_s"].iloc[0]), 1),
    "slowest": by_auth.index[-1], "slowest_median_s": round(float(by_auth["median_s"].iloc[-1]), 1),
    "spread_ratio": round(float(by_auth["median_s"].iloc[-1] / by_auth["median_s"].iloc[0]), 1),
    "kruskal_H": round(float(H), 1), "kruskal_p": float(pk),
}

# 5b. Confound check + robustness: operator ranking on NON-SERIAL live picks.
#     Serial items (phones/electronics) scan much faster and dominate mass-
#     scanning, so an operator's raw pace is confounded by product mix and by
#     residual short bursts (<BURST_MIN_RUN) leaking into the live set.
conf_r, conf_p = stats.spearmanr(by_auth["bulk_share_pct"], by_auth["median_s"])
ns = d[d["is_live_valid"] & ~d["is_serial"]].groupby("author")["duration_s"]
by_auth_ns = pd.DataFrame({"n": ns.size(), "median_s": ns.median()})
by_auth_ns = by_auth_ns[by_auth_ns["n"] >= 30].sort_values("median_s")
res["operator_confound"] = {
    "spearman_bulkshare_vs_median": round(float(conf_r), 3), "p_value": float(conf_p),
    "note": "negative => operators who mass-scan more show artificially faster live medians",
    "nonserial_only": {
        "n_qualified": int(len(by_auth_ns)),
        "fastest": by_auth_ns.index[0], "fastest_median_s": round(float(by_auth_ns["median_s"].iloc[0]), 1),
        "slowest": by_auth_ns.index[-1], "slowest_median_s": round(float(by_auth_ns["median_s"].iloc[-1]), 1),
        "spread_ratio": round(float(by_auth_ns["median_s"].iloc[-1] / by_auth_ns["median_s"].iloc[0]), 1),
    },
}
by_auth_ns.round(1).to_csv(OUT / "by_author_nonserial.csv")

# ---------------------------------------------------------------------------
# 6. Per-category durations (top categories by volume)
# ---------------------------------------------------------------------------
lc = d[d["is_live_valid"]]
topcats = lc["category"].value_counts().head(12).index
cc = lc[lc["category"].isin(topcats)].groupby("category")["duration_s"]
by_cat = pd.DataFrame({"n": cc.size(), "median_s": cc.median(),
                       "mean_s": cc.mean(), "p90_s": cc.quantile(.9)}).sort_values("median_s")
by_cat.round(1).to_csv(OUT / "by_category.csv")
grpsc = [g.to_numpy() for _, g in cc if g.size >= 30]
Hc, pc = stats.kruskal(*grpsc)
res["categories"] = {"kruskal_H": round(float(Hc), 1), "kruskal_p": float(pc),
                     "fastest": by_cat.index[0], "slowest": by_cat.index[-1]}

# ---------------------------------------------------------------------------
# 7. Serial vs non-serial, and quantity effect (live picks)
# ---------------------------------------------------------------------------
ser = lc.loc[lc.is_serial, "duration_s"]; non = lc.loc[~lc.is_serial, "duration_s"]
us, ps = stats.mannwhitneyu(ser, non)
res["serial_effect"] = {"serial_median_s": round(float(ser.median()), 1),
                        "nonserial_median_s": round(float(non.median()), 1),
                        "p_value": float(ps)}
rho, prho = stats.spearmanr(lc["qty"], lc["duration_s"])
res["quantity_effect"] = {"spearman_rho": round(float(rho), 3), "p_value": float(prho)}

# ===========================================================================
# FIGURES
# ===========================================================================
# Fig 1: duration distribution live vs bulk cadence
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.logspace(0, np.log10(BREAK_CAP_S), 50)
ax.hist(live, bins=bins, color=BLUE, alpha=.75, label=f"Live picks (n={live.size:,})")
ax.hist(bulk[bulk <= BREAK_CAP_S], bins=bins, color=ORANGE, alpha=.7,
        label=f"Mass-scan cadence (n={bulk.size:,})")
ax.axvline(np.median(live), color=BLUE, ls="--", lw=1.5)
ax.axvline(np.median(bulk), color=ORANGE, ls="--", lw=1.5)
ax.set_xscale("log"); ax.set_xlabel("Inter-scan gap / picking duration (s, log)")
ax.set_ylabel("Lines"); ax.set_title("Live picking vs mass-scan cadence")
ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "01_live_vs_bulk.png"); plt.close(fig)

# Fig 2: live duration histogram with lognormal fit
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(live, bins=bins, density=True, color=GREY, alpha=.6, label="Live picks")
xs = np.logspace(0, np.log10(BREAK_CAP_S), 300)
ax.plot(xs, stats.lognorm.pdf(xs, shape, 0, scale), color=BLUE, lw=2,
        label=f"Lognormal (median≈{scale:.0f}s, σ={shape:.2f})")
ax.set_xscale("log"); ax.set_xlabel("Picking duration (s, log)")
ax.set_ylabel("Density"); ax.set_title("Live picking-duration distribution + lognormal fit")
ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "02_lognormal_fit.png"); plt.close(fig)

# Fig 3: per-operator median with pick-count (top 25 by volume)
top = by_auth.sort_values("n_live", ascending=False).head(25).sort_values("median_s")
fig, ax = plt.subplots(figsize=(8, 7))
ypos = np.arange(len(top))
ax.barh(ypos, top["median_s"], color=BLUE)
ax.set_yticks(ypos); ax.set_yticklabels([f"op{n}" for n in range(len(top))], fontsize=8)
for i, (m, n) in enumerate(zip(top["median_s"], top["n_live"])):
    ax.text(m + 2, i, f"{m:.0f}s  (n={n})", va="center", fontsize=7)
ax.set_xlabel("Median live picking duration (s)")
ax.set_title("Per-operator median pick time (top 25 by volume)\nfast → slow")
ax.invert_yaxis(); fig.tight_layout(); fig.savefig(FIGS / "03_operators.png"); plt.close(fig)

# Fig 4: per-category boxplot (log)
fig, ax = plt.subplots(figsize=(8, 6))
order = list(by_cat.index)
data = [lc.loc[lc.category == c, "duration_s"].to_numpy() for c in order]
ax.boxplot(data, orientation="horizontal", showfliers=False, widths=.6)
ax.set_yticklabels([f"{c[:22]} (n={len(v)})" for c, v in zip(order, data)], fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Live picking duration (s, log)")
ax.set_title("Picking duration by item category"); fig.tight_layout()
fig.savefig(FIGS / "04_categories.png"); plt.close(fig)

# Fig 5: worked example — one live order and one mass-scan order timeline
def timeline(order, author, ax, title, color):
    s = d[(d.order == order) & (d.author == author)].sort_values("scan")
    t = (s.scan - s.scan.min()).dt.total_seconds()
    ax.plot(t, np.arange(len(t)), "o-", color=color, ms=4)
    ax.set_title(title, fontsize=9); ax.set_xlabel("s since first scan")
    ax.set_ylabel("scan #")
ex_bulk = d[d.is_bulk].groupby(["order", "author"]).size().sort_values().index[-1]
ex_live = (d[d.is_live_valid].groupby(["order", "author"]).size()
           .sort_values().index[-1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
timeline(*ex_live, a1, "Live picking (irregular, travel-paced)", BLUE)
timeline(*ex_bulk, a2, "Mass-scan burst (rapid, regular)", ORANGE)
fig.tight_layout(); fig.savefig(FIGS / "05_timelines.png"); plt.close(fig)

# ---------------------------------------------------------------------------
(OUT / "stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

# Console summary
def sec(t): return f"{t:.0f}s ({t/60:.1f}m)"
print("="*70)
print("ORDER-PICKING DURATION ANALYSIS")
print("="*70)
c = res["composition"]
print(f"{c['rows']:,} lines | {c['orders']} orders | {c['operators']} operators |"
      f" {c['skus']:,} SKUs | {c['date_min'][:10]}..{c['date_max'][:10]}")
print(f"Lines by regime: {c['lines_by_reason']}")
print(f"Mass-scan share: {c['pct_bulk_of_all']}% of lines "
      f"({c['orders_majority_bulk']} orders majority-bulk); "
      f"serial share bulk={c['bulk_serial_share']}% vs live={c['live_serial_share']}%")
L = res["live_duration_s"]
print(f"\nLIVE picking duration (n={L['n']:,}):")
print(f"  median {sec(L['p50'])}  (95% CI {L['median_ci95'][0]:.0f}-{L['median_ci95'][1]:.0f}s)"
      f" | mean {sec(L['mean'])} | geo-mean {sec(L['geometric_mean'])}")
print(f"  IQR {L['p25']:.0f}-{L['p75']:.0f}s | p90 {sec(L['p90'])} | p95 {sec(L['p95'])}")
print(f"  lognormal: median≈{L['lognormal']['median_est']:.0f}s, sigma={L['lognormal']['sigma']}")
b = res["live_vs_bulk"]
print(f"\nLive vs mass-scan: live median {b['live_median_s']}s vs bulk {b['bulk_median_s']}s"
      f" | Mann-Whitney p={b['p_value']:.1e}, rank-biserial={b['rank_biserial']}")
nb = res["naive_bias"]
print(f"Naive (unsplit) median {nb['naive_median_s']}s understates true live median by "
      f"{nb['median_understated_pct']}%")
o = res["operators"]
print(f"\nOperators (n={o['n_qualified']} with >=30 picks): fastest {o['fastest']} "
      f"{o['fastest_median_s']}s vs slowest {o['slowest']} {o['slowest_median_s']}s "
      f"({o['spread_ratio']}x) | Kruskal p={o['kruskal_p']:.1e}")
oc = res["operator_confound"]
print(f"  confound: bulk-share vs live-median Spearman={oc['spearman_bulkshare_vs_median']} "
      f"(p={oc['p_value']:.1e}) -> product-mix confounds raw pace")
print(f"  non-serial-only ranking (n={oc['nonserial_only']['n_qualified']}): "
      f"{oc['nonserial_only']['fastest_median_s']}s..{oc['nonserial_only']['slowest_median_s']}s "
      f"({oc['nonserial_only']['spread_ratio']}x)")
print(f"Categories: Kruskal p={res['categories']['kruskal_p']:.1e} | "
      f"fastest={res['categories']['fastest']} slowest={res['categories']['slowest']}")
print(f"Serial vs non-serial live median: {res['serial_effect']['serial_median_s']}s vs "
      f"{res['serial_effect']['nonserial_median_s']}s (p={res['serial_effect']['p_value']:.1e})")
print(f"Quantity vs duration: Spearman rho={res['quantity_effect']['spearman_rho']} "
      f"(p={res['quantity_effect']['p_value']:.1e})")
print("\nWrote output/stats.json, by_author.csv, by_category.csv, figs/*.png")
