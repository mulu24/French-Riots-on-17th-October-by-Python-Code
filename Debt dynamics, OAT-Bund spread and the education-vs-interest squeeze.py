"""
02 - Debt dynamics, OAT-Bund spread and the 'education vs interest' squeeze
===========================================================================
Deterministic (non-random) macro block:
  * extra interest cost from a wider OAT-Bund spread,
  * debt-to-GDP recursion d' = d(1+i)/(1+g) - pb under three scenarios
    (drift / government path / spending-led reform),
  * Domar stabilising primary balance pb* = d(i-g)/(1+g).
Inputs: debt 119.3% of GDP (2026), nominal growth 2.6%, effective rate ~2.45%.
Output: figures/fig2_debt.png, figures/fig3_spread_budget.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ---------- 2. Spread -> interest burden ----------
issuance = 300e3         # EUR m gross yearly issuance (assumption)
avg_maturity = 8.0       # years (assumption)
d_spread_total = (0.0176 - 0.0055)      # Feb 55bp -> 23 Oct 176bp
share_unrest = 0.15                      # assumed share attributable to unrest
extra_y1 = issuance * d_spread_total * share_unrest * 0.5
print(f"Extra interest, year 1 (EUR m): {extra_y1:,.0f}")

# ---------- 3. Debt dynamics  d' = d(1+i)/(1+g) - pb ----------
g = 0.026               # nominal growth (0.5% real + 2.1% inflation)
i0 = 0.0245             # effective rate ~ 74.5bn / 3.05tn
years = np.arange(2026, 2033)

def path(spread_shift, pb_path, rollover=1/avg_maturity, d0=1.193):
    d, i, out = d0, i0, [d0]
    for t in range(1, len(years)):
        i = i + rollover * spread_shift      # gradual repricing of the stock
        d = d * (1 + i) / (1 + g) - pb_path[t]
        out.append(d)
    return np.array(out)

pb_now = np.full(len(years), -0.029)                       # primary deficit ~2.9% GDP
pb_plan = np.array([-0.029, -0.022, -0.015, -0.007, 0.0, 0.004, 0.006])
pb_hard = np.array([-0.029, -0.020, -0.008, 0.003, 0.010, 0.012, 0.012])

drift   = path(0.0120, pb_now)    # unrest + spread stays +120bp, no consolidation
plan    = path(0.0120, pb_plan)   # gov't glide path, spread stays wide
reform  = path(0.0060, pb_hard)   # spending-led consolidation, spread halves

for n, p in [("drift", drift), ("plan", plan), ("reform", reform)]:
    print(n, np.round(p * 100, 1))

# Domar stabilising primary balance: pb* = d (i-g)/(1+g)
for s in (0.0, 0.006, 0.012):
    ii = i0 + s
    print(f"spread +{s*1e4:.0f}bp -> pb* = {1.193*(ii-g)/(1+g)*100:.2f}% of GDP")

# ---------- Charts ----------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(years, drift * 100, label="Drift: wide spread, no consolidation", color="#c0504d", lw=2)
ax.plot(years, plan * 100, label="Government path, spread stays wide", color="#e69f00", lw=2)
ax.plot(years, reform * 100, label="Spending-led reform, spread halves", color="#2e8b57", lw=2)
ax.set_ylabel("Public debt, % of GDP")
ax.set_title("Debt-to-GDP scenarios (author's model)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig2_debt.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
pts = [("Feb", 55), ("31 Jul", 74), ("9 Oct", 135), ("21 Oct", 159), ("22 Oct", 171), ("23 Oct", 176)]
ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#3b6ea5")
ax[0].axhline(191, color="grey", ls=":"); ax[0].text(0, 193, "2011 peak 191bp", fontsize=8)
ax[0].set_title("OAT-Bund 10y spread (bp), reported points")
ax[1].bar(["Education 2027\n(draft)", "Debt service 2027\n(draft)"], [65.53, 74.5], color=["#2e8b57", "#c0504d"])
ax[1].set_title("2027 draft budget (EUR bn)")
plt.tight_layout()
plt.savefig("figures/fig3_spread_budget.png", dpi=150)
