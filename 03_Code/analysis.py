"""
AdPulse AI - offline analysis
==============================
Reproduces every number and chart used in the report.

    python 03_Code/analysis.py

Writes CSV tables and PNG charts into 04_Results/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import adpulse_engine as E

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results"
OUT.mkdir(exist_ok=True)

# --- chart style (validated reference palette, light mode) ----------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
CHANNEL_COLOR = {"Google Search": BLUE, "Meta Ads": ORANGE, "TikTok Ads": AQUA,
                 "Snapchat Ads": YELLOW, "YouTube": MAGENTA, "Influencers": GREEN}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9,
})


def aed(x, pos=None):
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def hbar(ax, labels, values, color=BLUE, fmt="{:.2f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left", color=INK2, fontsize=9)


# --- load ----------------------------------------------------------------------
df = E.load_campaigns()
cust = E.load_customers()
k = E.headline_kpis(df)

# 1. Descriptive -----------------------------------------------------------------
ch = E.channel_summary(df)
ch.round(4).to_csv(OUT / "channel_performance.csv", index=False)
cat = E.group_summary(df, "category"); cat.round(4).to_csv(OUT / "category_performance.csv", index=False)
brand = E.group_summary(df, "brand"); brand.round(4).to_csv(OUT / "brand_performance.csv", index=False)
season = E.group_summary(df, "season"); season.round(4).to_csv(OUT / "season_performance.csv", index=False)
obj = df.groupby(["channel", "objective"]).agg(spend_aed=("spend_aed", "sum"), profitable_share=("profitable", "mean"),
                                               verified_roas=("roas", "median")).reset_index()
obj.round(4).to_csv(OUT / "channel_objective_profitability.csv", index=False)
weekly = E.weekly_summary(df); weekly.round({"spend_aed": 2, "verified_revenue_aed": 2, "platform_reported_revenue_aed": 2, "contribution_aed": 2, "verified_roas": 4}).to_csv(OUT / "weekly_performance.csv", index=False)
leaks = E.money_leaks(df, 10); leaks.round(2).to_csv(OUT / "top10_money_leaking_campaigns.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(weekly["week_start"], weekly["verified_revenue_aed"], color=BLUE, lw=2, label="Verified revenue")
ax.plot(weekly["week_start"], weekly["spend_aed"], color=ORANGE, lw=2, label="Ad spend")
for s, lab in [("Ramadan & Eid", "Ramadan"), ("White Friday & 11.11", "White Friday"), ("Summer", "Summer")]:
    w = weekly[weekly["season"] == s]["week_start"]
    ax.axvspan(w.min(), w.max() + pd.Timedelta(days=6), color=GRID, alpha=0.6, lw=0)
    ax.text(w.min(), weekly["verified_revenue_aed"].max() * 1.04, " " + lab, fontsize=8, color=INK2)
ax.yaxis.set_major_formatter(aed)
ax.set_ylabel("AED per week")
ax.set_title("Weekly ad spend and verified revenue, 6 brands, 2025")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
save(fig, "fig01_weekly_spend_revenue.png")

fig, ax = plt.subplots(figsize=(8, 3.6))
s2 = season.sort_values("verified_roas", ascending=False)
hbar(ax, s2["season"], s2["verified_roas"], fmt="{:.2f}x")
ax.set_xlabel("Verified ROAS (revenue / spend)")
ax.set_title("Return on ad spend by season")
save(fig, "fig02_roas_by_season.png")

fig, ax = plt.subplots(figsize=(8.5, 4))
o = obj.copy(); o["label"] = o["channel"] + " - " + o["objective"]
o = o.sort_values("profitable_share", ascending=False)
hbar(ax, o["label"], o["profitable_share"] * 100, fmt="{:.0f}%")
ax.set_xlim(0, 112)
ax.set_xlabel("% of campaign-weeks that made a profit (after product margin)")
ax.set_title("How often each campaign type actually makes money")
save(fig, "fig03_profitable_share_channel_objective.png")

# 2. Attribution truth check ------------------------------------------------------
att = E.attribution_gap(df); att.round(4).to_csv(OUT / "attribution_gap.csv", index=False)
fig, ax = plt.subplots(figsize=(8.5, 4))
a2 = att.sort_values("platform_roas", ascending=False)
y = np.arange(len(a2)); h = 0.36
ax.barh(y - h / 2, a2["platform_roas"], h, color=ORANGE, label="ROAS claimed by ad platform", edgecolor=SURFACE, lw=2)
ax.barh(y + h / 2, a2["verified_roas"], h, color=BLUE, label="ROAS verified in store orders", edgecolor=SURFACE, lw=2)
for yi, (p, v) in enumerate(zip(a2["platform_roas"], a2["verified_roas"])):
    ax.text(p, yi - h / 2, f"  {p:.2f}x", va="center", fontsize=8, color=INK2)
    ax.text(v, yi + h / 2, f"  {v:.2f}x", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, a2["channel"]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Return on ad spend")
ax.set_title("Platform dashboards vs what the store actually recorded")
ax.legend(loc="lower right")
save(fig, "fig04_attribution_gap.png")

# 3. Predictive model ----------------------------------------------------------
m = E.train_profit_model(df)
m.metrics.round(4).to_csv(OUT / "model_metrics.csv", index=False)
m.importance.round(4).to_csv(OUT / "feature_importance.csv", index=False)
pd.DataFrame(m.confusion, index=["actual_loss", "actual_profit"], columns=["pred_loss", "pred_profit"]).to_csv(
    OUT / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(8, 4))
hbar(ax, m.importance["feature"], m.importance["importance"] * 100, fmt="{:.1f}%")
ax.set_xlabel("Share of model importance (Random Forest, mean decrease in impurity)")
ax.set_title("What decides whether a campaign-week makes money")
save(fig, "fig05_feature_importance.png")

fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.plot(m.roc["fpr"], m.roc["tpr"], color=BLUE, lw=2, label=f"Random Forest (AUC {m.metrics.loc[0, 'roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random guess (AUC 0.50)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve, Oct-Dec hold-out")
ax.legend(loc="lower right")
save(fig, "fig06_roc_curve.png")

# 4. Response curves & optimiser --------------------------------------------------
detail, opt = E.optimise_all_brands(df)
detail.round(4).to_csv(OUT / "budget_optimisation_by_channel.csv", index=False)
opt.round(4).to_csv(OUT / "budget_optimisation_by_brand.csv", index=False)

demo = detail[detail["brand"] == "Oudh Lane"]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for _, r in demo.iterrows():
    xs = np.linspace(0, r["max_spend"] * 1.3, 120)
    ax.plot(xs, E.curve_revenue(r["k"], r["b"], xs), color=CHANNEL_COLOR[r["channel"]], lw=2, label=r["channel"])
    ax.scatter([r["current_spend"]], [r["current_revenue"]], s=36, color=CHANNEL_COLOR[r["channel"]],
               edgecolor=SURFACE, linewidth=2, zorder=3)
ax.xaxis.set_major_formatter(aed); ax.yaxis.set_major_formatter(aed)
ax.set_xlabel("Weekly spend on channel (AED)"); ax.set_ylabel("Predicted weekly revenue (AED)")
ax.set_title("Diminishing returns: Oudh Lane response curves (dots = current spend)")
ax.legend(ncol=2, loc="upper left")
save(fig, "fig07_response_curves_oudh_lane.png")

pc = detail.groupby("channel")[["current_spend", "optimised_spend"]].sum()
pc = pc.loc[pc["current_spend"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.5, 4))
y = np.arange(len(pc))
ax.barh(y - h / 2, pc["current_spend"], h, color=MUTED, label="Current weekly spend", edgecolor=SURFACE, lw=2)
ax.barh(y + h / 2, pc["optimised_spend"], h, color=BLUE, label="AdPulse recommended", edgecolor=SURFACE, lw=2)
for yi, (c_, o_) in enumerate(zip(pc["current_spend"], pc["optimised_spend"])):
    ax.text(o_, yi + h / 2, f"  {aed(o_)} ({(o_ / c_ - 1) * 100:+.0f}%)", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, pc.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(aed); ax.set_xlabel("AED per week, all 6 brands, same total budget")
ax.set_title("Same budget, different split: recommended reallocation")
ax.legend(loc="lower right")
save(fig, "fig08_budget_reallocation.png")

# 5. Segmentation & LTV:CAC --------------------------------------------------------
seg, prof, sil = E.segment_customers(cust)
prof.round(4).to_csv(OUT / "customer_segments.csv", index=False)
sil.round(4).to_csv(OUT / "kmeans_model_selection.csv", index=False)
mix = E.segment_mix_by_channel(seg); mix.round(4).to_csv(OUT / "segment_mix_by_channel.csv")
lc = E.ltv_cac(df, cust); lc.round(4).to_csv(OUT / "ltv_cac_by_channel.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5, 3.8))
p2 = prof.sort_values("profit_share", ascending=False)
y = np.arange(len(p2))
ax.barh(y - h / 2, p2["share"] * 100, h, color=MUTED, label="% of customers", edgecolor=SURFACE, lw=2)
ax.barh(y + h / 2, p2["profit_share"] * 100, h, color=BLUE, label="% of gross profit", edgecolor=SURFACE, lw=2)
for yi, (s_, p_) in enumerate(zip(p2["share"], p2["profit_share"])):
    ax.text(s_ * 100, yi - h / 2, f"  {s_ * 100:.0f}%", va="center", fontsize=8, color=INK2)
    ax.text(p_ * 100, yi + h / 2, f"  {p_ * 100:.0f}%", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, p2["segment"]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Share (%)"); ax.set_title("Customer segments (K-means, k = 4): size vs profit")
ax.legend(loc="lower right")
save(fig, "fig09_customer_segments.png")

fig, ax = plt.subplots(figsize=(8, 3.8))
l2 = lc.sort_values("ltv_to_cac", ascending=False)
hbar(ax, l2["channel"], l2["ltv_to_cac"], fmt="{:.2f}x")
ax.axvline(1, color=CRITICAL, lw=1.2, ls="--")
ax.text(1.02, len(l2) - 0.6, "break-even (1.0x)", color=CRITICAL, fontsize=8)
ax.set_xlabel("12-month gross profit per customer / cost to acquire that customer")
ax.set_title("LTV : CAC by acquisition channel")
save(fig, "fig10_ltv_cac.png")

# 6. Scenarios -------------------------------------------------------------------
scen_def = [
    ("Base case (current allocation)", 0, 0, 0, False),
    ("AdPulse optimised allocation", 0, 0, 0, True),
    ("Budget +20%, current allocation", 0.20, 0, 0, False),
    ("Budget +20%, optimised", 0.20, 0, 0, True),
    ("Budget -20%, optimised", -0.20, 0, 0, True),
    ("Ad costs +15% (peak-season CPMs)", 0, 0.15, 0, False),
    ("Ad costs +15%, optimised", 0, 0.15, 0, True),
    ("Demand -10% (soft market)", 0, 0, -0.10, False),
]
srows = []
for name, bc, ci, dc, op in scen_def:
    r = E.scenario(df, bc, ci, dc, op)
    srows.append({"scenario": name, **r})
scen = pd.DataFrame(srows)
scen["contribution_vs_base_aed_week"] = scen["weekly_contribution"] - scen.loc[0, "weekly_contribution"]
scen.round(2).to_csv(OUT / "scenario_results.csv", index=False)

# 7. Pricing & break-even ---------------------------------------------------------
C = E.COST_ASSUMPTIONS
arpu = E.blended_arpu()
be = E.breakeven(C["fixed_cost_aed_month"], arpu, C["variable_cost_aed_customer_month"])
ue = E.unit_economics(arpu, C["variable_cost_aed_customer_month"], C["cac_aed"], C["monthly_churn"])
E.PRICING_TIERS.to_csv(OUT / "pricing_tiers.csv", index=False)
sens = []
for fixed in [75_000, 95_000, 120_000]:
    for a in [1000, arpu, 1500]:
        sens.append({"fixed_cost_aed_month": fixed, "blended_arpu_aed": round(a, 0),
                     "breakeven_subscribers": np.ceil(E.breakeven(fixed, a, C["variable_cost_aed_customer_month"]))})
pd.DataFrame(sens).to_csv(OUT / "breakeven_sensitivity.csv", index=False)
proj = {name: E.subscriber_projection(36, start, g) for name, start, g in
        [("Conservative", 3, 0.05), ("Base", 4, 0.08), ("Optimistic", 6, 0.10)]}
proj_all = pd.concat([p.assign(case=n) for n, p in proj.items()])
proj_all.round(2).to_csv(OUT / "subscriber_projection.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5, 4))
for (n, p), col in zip(proj.items(), [MUTED, BLUE, AQUA]):
    ax.plot(p["month"], p["operating_profit_aed"], color=col, lw=2, label=n)
ax.axhline(0, color=AXIS, lw=1)
ax.yaxis.set_major_formatter(aed)
ax.set_xlabel("Month after launch"); ax.set_ylabel("Monthly operating profit (AED)")
ax.set_title("AdPulse SaaS: path to monthly break-even (illustrative assumptions)")
ax.legend(loc="upper left")
save(fig, "fig11_breakeven_projection.png")

def first_positive(p):
    pos = p[p["operating_profit_aed"] > 0]
    return int(pos["month"].iloc[0]) if len(pos) else None

# 8. Key results ------------------------------------------------------------------
value_per_brand_year = opt["contribution_uplift_aed_year"].mean()
kr = {
    "campaign_weeks": k["rows"], "campaigns": k["campaigns"], "brands": k["brands"], "weeks": k["weeks"],
    "customers": len(cust),
    "total_spend_aed": k["spend"], "verified_revenue_aed": k["verified_revenue"],
    "platform_reported_revenue_aed": k["platform_revenue"], "verified_roas": k["verified_roas"],
    "platform_roas": k["platform_roas"], "platform_overclaim_pct": k["overclaim_pct"],
    "total_contribution_aed": k["contribution"],
    "spend_in_loss_making_campaign_weeks_aed": k["losing_spend"], "share_spend_loss_making": k["losing_share"],
    "rf_accuracy": m.metrics.loc[0, "accuracy"], "rf_precision": m.metrics.loc[0, "precision"],
    "rf_recall": m.metrics.loc[0, "recall"], "rf_f1": m.metrics.loc[0, "f1"], "rf_auc": m.metrics.loc[0, "roc_auc"],
    "lr_auc": m.metrics.loc[1, "roc_auc"], "lr_accuracy": m.metrics.loc[1, "accuracy"],
    "naive_accuracy": m.metrics.loc[2, "accuracy"],
    "rf_cv_auc_mean": m.cv_auc_mean, "rf_cv_auc_std": m.cv_auc_std,
    "model_train_rows": m.n_train, "model_test_rows": m.n_test, "model_split_date": m.split_date.date().isoformat(),
    "optimiser_portfolio_revenue_uplift_pct": opt["optimised_revenue_aed"].sum() / opt["current_revenue_aed"].sum() - 1,
    "optimiser_contribution_uplift_aed_week": opt["contribution_uplift_aed_week"].sum(),
    "optimiser_contribution_uplift_aed_year": opt["contribution_uplift_aed_year"].sum(),
    "optimiser_value_per_brand_year_aed": value_per_brand_year,
    "kmeans_k": 4, "kmeans_silhouette_k4": float(sil.loc[sil["k"] == 4, "silhouette"].iloc[0]),
    "saas_blended_arpu_aed": arpu, "saas_breakeven_subscribers": be,
    "saas_ltv_aed": ue["ltv"], "saas_ltv_to_cac": ue["ltv_to_cac"], "saas_payback_months": ue["payback_months"],
    "saas_gross_margin_pct": ue["gross_margin_pct"],
    "breakeven_month_conservative": first_positive(proj["Conservative"]),
    "breakeven_month_base": first_positive(proj["Base"]),
    "breakeven_month_optimistic": first_positive(proj["Optimistic"]),
    "growth_tier_price_as_share_of_value": 1490 * 12 / value_per_brand_year,
}
pd.DataFrame(list(kr.items()), columns=["metric", "value"]).to_csv(OUT / "key_results.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    for kk, v in kr.items():
        print(f"{kk:45s} {v:,.4f}" if isinstance(v, (int, float, np.floating)) and v is not None else f"{kk:45s} {v}")
    print(scen.round(0))
    print(opt.round(3))
    print(prof.round(2))
    print(lc.round(2))
    print(att.round(3))
    print(leaks.round(0))
    print(cat.round(3)); print(brand.round(3))
    print(detail[["brand", "channel", "b_raw", "b", "b_clipped", "r2"]].round(2))
