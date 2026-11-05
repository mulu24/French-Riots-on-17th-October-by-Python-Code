"""
01 - Baseline Monte Carlo: cost of the 2026 French unrest (hand-picked ranges)
==============================================================================
First-pass model. Each cost component (property damage, policing, firefighters, tear gas,
lost output...) is an uncertain input drawn from a triangular(low, mode, high) distribution.
20,000 simulated 'worlds' are summed to get a distribution of total cost (EUR million).
All ranges are the author's ASSUMPTIONS, loosely anchored on 2023 riot figures.
Output: figures/fig1_cost.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(42)

# ---------- 1. Cost of the unrest: Monte Carlo (EUR million) ----------
# (low, mode, high) -- ASSUMPTIONS anchored on public figures:
#   2023 riots: insurers EUR 650m (France Assureurs), Medef > EUR 1bn
#   2026: Auvergne-Rhone-Alpes alone EUR 25-30m, ~3,000 shops hit, 100+ schools damaged
GDP = 3_000e3            # ~EUR 3.0tn in millions (debt 3.536tn / 117.5% of GDP)
daily_gdp = GDP / 365

items = {
    "Property damage (shops, cars, public bldgs)": (300, 700, 1200),
    "Schools repair":                              (40, 80, 150),
    "Transport / bus shelters":                    (20, 50, 100),
}
N = 20_000
draw = {k: rng.triangular(*v, N) for k, v in items.items()}

# Policing: officer-days x EUR/day (overtime + logistics)
officer_days = rng.triangular(150e3, 300e3, 500e3, N)
draw["Extra policing"] = officer_days * 250 / 1e6
# Firefighters: interventions x EUR/intervention
interventions = rng.triangular(10e3, 20e3, 35e3, N)
draw["Firefighters"] = interventions * 1500 / 1e6
# Tear gas & munitions: units x EUR/unit
grenades = rng.triangular(100e3, 250e3, 400e3, N)
draw["Tear gas / munitions"] = grenades * 40 / 1e6
# Lost output: share of daily GDP lost x days of disruption
loss_share = rng.triangular(0.003, 0.0075, 0.015, N)
days = rng.triangular(10, 14, 20, N)
draw["Lost output (closures, retail, tourism)"] = loss_share * daily_gdp * days

total = sum(draw.values())
means = {k: v.mean() for k, v in draw.items()}
p5, p50, p95 = np.percentile(total, [5, 50, 95])
print(f"Total cost EUR m: P5={p5:,.0f} median={p50:,.0f} P95={p95:,.0f}")
for k, v in means.items():
    print(f"  {k:45s} {v:8,.0f}")

# ---------- Charts ----------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].hist(total, bins=60, color="#3b6ea5", alpha=.85)
for v, lab in [(p5, "P5"), (p50, "Median"), (p95, "P95")]:
    ax[0].axvline(v, color="k", ls="--", lw=1)
    ax[0].text(v, ax[0].get_ylim()[1] * .92, f"{lab}\n{v:,.0f}", ha="center", fontsize=8,
               bbox=dict(fc="white", ec="none"))
ax[0].set_title("Monte Carlo: total cost of the unrest (EUR m)")
ax[0].set_xlabel("EUR million")
names = list(means.keys())
vals = [means[k] for k in names]
order = np.argsort(vals)
ax[1].barh([names[j] for j in order], [vals[j] for j in order], color="#c0504d")
ax[1].set_title("Mean cost by component (EUR m)")
plt.tight_layout()
plt.savefig("figures/fig1_cost.png", dpi=150)
