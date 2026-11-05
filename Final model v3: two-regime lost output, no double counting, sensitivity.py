"""
04 - Final model v3: two-regime lost output, no double counting, sensitivity analysis
=====================================================================================
Improves model 03:
  * lost output split into few peak days and many low-intensity days,
  * excludes business interruption already counted by IFRAP (no double counting),
  * subtracts the share of postponed activity recovered later,
  * links shock severity to intensity/duration,
  * adds Spearman rank-correlation sensitivity and a x0.5 / x2 stress test.
The v2 median (1,767) is hardcoded in the chart for comparison.
Output: figures/fig1_cost_v3.png, figures/fig5_sensitivity.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(42)
N = 50_000
GDP_M = 3_000e3                # EUR m, ~annual GDP
GDP_DAY = GDP_M / 365          # ~8,219 EUR m per day

# ---------------- A. Weekend 17-18 Oct: anchored on IFRAP (via realtyinvestor.eu) ----------------
# IFRAP total bill: low 309 / central 613 / high 1,101. High-case breakdown below.
# NOTE: "Operating losses" (190) = lost turnover of damaged firms -> already counts business interruption,
#       so the lost-output block below EXCLUDES retail/business interruption (no double counting).
high = {"Businesses (damage)": 666, "Public buildings": 200, "Vehicles/private": 45, "Business interruption": 190}
H = sum(high.values())
UPPER = 1500                   # tail stretched toward DBRS insured ceiling (1.0-1.2bn)

total_w = rng.triangular(309, 613, UPPER, N)
sev = total_w / H              # shared severity factor (one bad weekend raises all components)
sev_pct = (total_w - 309) / (UPPER - 309)   # 0..1 position within range, used for correlation
comp = {k: v * sev * rng.uniform(0.85, 1.15, N) for k, v in high.items()}

# ---------------- B. Costs outside IFRAP scope ----------------
comp["Policing (overtime/logistics)"] = rng.triangular(150e3, 300e3, 500e3, N) * 250 / 1e6
comp["Firefighters"] = rng.triangular(10e3, 20e3, 35e3, N) * 1500 / 1e6
comp["Tear gas / munitions"] = rng.triangular(100e3, 250e3, 400e3, N) * 40 / 1e6
comp["School & regional damage (Sept-Oct)"] = rng.triangular(40, 90, 150, N) + rng.triangular(25, 28, 30, N)
comp["Transport / bus shelters"] = rng.triangular(20, 50, 100, N)

# ---------------- C. Lost output v3: two regimes, net of recovery, correlated with severity ----------------
# Peak days (violent weekends): few days, higher loss, scales with weekend severity
n_peak = rng.triangular(2, 3, 5, N)
share_peak = rng.triangular(0.0005, 0.0015, 0.0030, N) * (0.6 + 0.8 * sev_pct)
# Low-intensity days (blockades, strikes, closed schools): many days, small loss.
# Negative link: a harsher shock tends to be shorter -> fewer low-intensity days.
n_low = rng.triangular(8, 15, 25, N) * (1.2 - 0.4 * sev_pct)
share_low = rng.triangular(0.0001, 0.0003, 0.0008, N)
gross_loss = GDP_DAY * (n_peak * share_peak + n_low * share_low)
recovered = rng.uniform(0.30, 0.60, N)          # share of postponed activity made up later
comp["Net lost output (schools, transport, tourism)"] = gross_loss * (1 - recovered)

total = sum(comp.values())
p5, p50, p95 = np.percentile(total, [5, 50, 95])
print(f"v3 total EUR m: P5={p5:,.0f} median={p50:,.0f} P95={p95:,.0f} mean={total.mean():,.0f}")
print(f"(v2 for reference: P5=1,275 median=1,767 P95=2,384)")
lo = comp["Net lost output (schools, transport, tourism)"]
print(f"Net lost output: median={np.median(lo):,.0f}  = {np.median(lo)/GDP_M*100:.3f}% of annual GDP")
for k, v in sorted(comp.items(), key=lambda kv: -kv[1].mean()):
    print(f"  {k:48s} mean {v.mean():7,.0f}")

# Insured-loss cross-check (IFRAP insured share 804/1101 applied to weekend material + interruption)
ins = sum(comp[k] for k in high) * (804 / 1101) * rng.uniform(0.9, 1.1, N)
print("Implied insured loss P5/P50/P95:", np.round(np.percentile(ins, [5, 50, 95])),
      "| share below DBRS 1,200 ceiling:", round(float((ins <= 1200).mean()), 3))

# ---------------- D. Sensitivity: Spearman rank correlation of each input with the total ----------------
def rank(x): return np.argsort(np.argsort(x)).astype(float)
rt = rank(total)
sens = {k: np.corrcoef(rank(v), rt)[0, 1] for k, v in comp.items()}
sens["Recovered share (assumption)"] = np.corrcoef(rank(recovered), rt)[0, 1]
sens["Peak days n"] = np.corrcoef(rank(n_peak), rt)[0, 1]
print("\nRank correlation with total:")
for k, v in sorted(sens.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:48s} {v:+.2f}")

# Stress test: halve / double the lost-output block
for f in (0.5, 2.0):
    t2 = total - lo + lo * f
    print(f"Lost output x{f}: median total = {np.median(t2):,.0f}")

# ---------------- Charts ----------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax[0].hist(total, bins=70, color="#3b6ea5", alpha=.85)
for v, lab in [(p5, "P5"), (p50, "Median"), (p95, "P95")]:
    ax[0].axvline(v, color="k", ls="--", lw=1)
    ax[0].text(v, ax[0].get_ylim()[1] * .9, f"{lab}\n{v:,.0f}", ha="center", fontsize=8, bbox=dict(fc="white", ec="none"))
ax[0].axvline(1767, color="#c0504d", ls=":", lw=1.5, label="v2 median (1,767)")
ax[0].set_title("v3: total cost of the unrest (EUR m)"); ax[0].set_xlabel("EUR million"); ax[0].legend(fontsize=8)
names = sorted(comp, key=lambda k: comp[k].mean())
ax[1].barh(names, [comp[k].mean() for k in names], color="#c0504d")
ax[1].set_title("Mean cost by component (EUR m)")
plt.tight_layout(); plt.savefig("figures/fig1_cost_v3.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
order = sorted(sens, key=lambda k: abs(sens[k]))
ax.barh(order, [sens[k] for k in order], color=["#2e8b57" if sens[k] > 0 else "#c0504d" for k in order])
ax.set_title("Sensitivity: rank correlation of each input with total cost")
plt.tight_layout(); plt.savefig("figures/fig5_sensitivity.png", dpi=150)
