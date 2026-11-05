"""
03 - Calibrated Monte Carlo using IFRAP and DBRS figures
========================================================
Re-calibrates the cost model on published estimates of the 17-18 Oct 2026 violence
(as reported by realtyinvestor.eu): IFRAP total bill low/central/high = 309/613/1,101 EUR m,
with a component breakdown; DBRS insured-loss ceiling of EUR 1.0-1.2bn.
One shared 'severity' factor drives all weekend damage components together.
Scenario A = IFRAP range, Scenario B = tail stretched toward DBRS. Includes an insured-loss
cross-check. Costs outside IFRAP scope (policing etc.) remain author assumptions.
Output: figures/fig1_cost_v2.png, figures/fig4_insured_check.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(42)
N = 50_000

# ---- ANCHORS (as reported by realtyinvestor.eu, citing IFRAP and DBRS, 20 Oct 2026) ----
# IFRAP total bill of 17-18 Oct riots: low 309 / central 613 / high 1,101 (EUR m)
# IFRAP high breakdown: businesses 666, public bldgs 200, vehicles/private 45, operating losses 190
# DBRS: insured losses "should not exceed" 1.0-1.2bn if violence not repeated; IFRAP insured = 804 of 1,101 (73%)
high = {"Businesses": 666, "Public buildings": 200, "Vehicles/private": 45, "Operating losses": 190}
H = sum(high.values())                      # ~1,101

def weekend_damage(upper_total):
    """Shared severity factor s ~ triangular(309,613,upper)/H applied to the IFRAP breakdown,
    plus independent component noise (+-15%)."""
    s = rng.triangular(309, 613, upper_total, N) / H
    out = {}
    for k, v in high.items():
        out[k] = v * s * rng.uniform(0.85, 1.15, N)
    return out

# ---- Costs NOT in IFRAP (IFRAP excludes policing/health/justice): my assumptions ----
def extras():
    d = {}
    d["Policing (overtime/logistics)"] = rng.triangular(150e3, 300e3, 500e3, N) * 250 / 1e6   # officer-days x EUR
    d["Firefighters"]                  = rng.triangular(10e3, 20e3, 35e3, N) * 1500 / 1e6    # call-outs x EUR
    d["Tear gas / munitions"]          = rng.triangular(100e3, 250e3, 400e3, N) * 40 / 1e6
    d["School & regional damage (Sept-Oct)"] = rng.triangular(40, 90, 150, N) + rng.triangular(25, 28, 30, N)  # regional EUR 25-30m reported
    d["Transport / bus shelters"]      = rng.triangular(20, 50, 100, N)
    GDP_day = 3_000e3 / 365
    d["Lost output (closures, tourism)"] = rng.triangular(0.002, 0.005, 0.010, N) * GDP_day * rng.triangular(10, 14, 20, N)
    return d

def run(upper):
    comp = {**weekend_damage(upper), **extras()}
    return comp, sum(comp.values())

# Scenario A: IFRAP range only (upper = 1,101).  Scenario B: stretched to DBRS-implied total (~1,500)
compA, totA = run(1101)
compB, totB = run(1500)

for name, t in [("A: IFRAP range", totA), ("B: IFRAP + DBRS tail", totB)]:
    p5, p50, p95 = np.percentile(t, [5, 50, 95])
    print(f"{name}: P5={p5:,.0f}  median={p50:,.0f}  P95={p95:,.0f}  mean={t.mean():,.0f}")

# Insured-loss cross-check vs DBRS (insured share ~73% of material damage + operating)
dmgB = sum(compB[k] for k in high)
ins = dmgB * (804 / 1101) * rng.uniform(0.9, 1.1, N)
print("Implied insured loss B (EUR m): P5/P50/P95 =", np.round(np.percentile(ins, [5, 50, 95])))
# DBRS says insured losses "should not exceed" 1.0-1.2bn -> a ceiling, not a central estimate
print("Share of draws with insured loss below DBRS ceiling of 1,200:", round(float((ins <= 1200).mean()), 3))
for k, v in compB.items():
    print(f"  {k:40s} mean {v.mean():8,.0f}")

# ---- Charts ----
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax[0].hist(totA, bins=70, alpha=.6, color="#3b6ea5", label="A: IFRAP range")
ax[0].hist(totB, bins=70, alpha=.6, color="#c0504d", label="B: IFRAP + DBRS tail")
for t, c in [(totA, "#3b6ea5"), (totB, "#c0504d")]:
    ax[0].axvline(np.median(t), color=c, ls="--", lw=1.2)
ax[0].set_title("Monte Carlo: total cost of the unrest (EUR m)")
ax[0].set_xlabel("EUR million"); ax[0].legend()
names = sorted(compB, key=lambda k: compB[k].mean())
ax[1].barh(names, [compB[k].mean() for k in names], color="#c0504d")
ax[1].set_title("Mean cost by component, scenario B (EUR m)")
plt.tight_layout()
plt.savefig("figures/fig1_cost_v2.png", dpi=150)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.hist(ins, bins=60, color="#2e8b57", alpha=.85)
ax.axvspan(1000, 1200, color="k", alpha=.12, label="DBRS ceiling: EUR 1.0-1.2bn")
ax.axvline(804, color="#e69f00", ls="--", label="IFRAP insured: 804")
ax.set_title("Implied insured losses vs published figures (EUR m)")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig4_insured_check.png", dpi=150)
