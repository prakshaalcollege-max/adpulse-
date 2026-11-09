"""
AdPulse AI - analytics engine
==============================

All the analytics used by the Streamlit app (app.py) and the offline
analysis script (analysis.py) live here, so the dashboard and the report are
always produced by exactly the same code.

Modules
  1. Data loading & validation
  2. Descriptive analytics (KPIs, channel / weekly summaries)
  3. Attribution "truth check" (platform-claimed vs verified revenue)
  4. Predictive model - will a campaign-week be profitable? (Random Forest)
  5. Response curves & budget optimiser (diminishing returns)
  6. Customer segmentation (K-means) and LTV : CAC by channel
  7. Scenario modelling
  8. SaaS pricing & break-even
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix, silhouette_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_Data"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Data loading & validation
# ---------------------------------------------------------------------------

CAMPAIGN_REQUIRED = [
    "week_start", "season", "brand", "category", "gross_margin", "channel", "campaign_id",
    "objective", "creative_format", "ad_language", "target_emirate", "creative_age_weeks",
    "spend_aed", "orders", "new_customers", "verified_revenue_aed", "platform_reported_revenue_aed",
]
CUSTOMER_REQUIRED = [
    "customer_id", "brand", "acquisition_channel", "orders_12m", "avg_order_value_aed",
    "revenue_12m_aed", "gross_profit_12m_aed", "days_since_last_order", "discount_order_share",
]


def missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def prepare_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns; safe to call on uploaded data."""
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    for c in ["spend_aed", "verified_revenue_aed", "platform_reported_revenue_aed", "gross_margin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["spend_aed", "verified_revenue_aed", "gross_margin"])
    df = df[df["spend_aed"] > 0]
    df["roas"] = df["verified_revenue_aed"] / df["spend_aed"]
    df["contribution_aed"] = df["verified_revenue_aed"] * df["gross_margin"] - df["spend_aed"]
    df["profitable"] = (df["contribution_aed"] > 0).astype(int)
    df["breakeven_roas"] = 1 / df["gross_margin"]
    return df


def load_campaigns(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "campaign_performance.csv"
    return prepare_campaigns(pd.read_csv(path))


def load_customers(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "customers.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Descriptive analytics
# ---------------------------------------------------------------------------

def headline_kpis(df: pd.DataFrame) -> dict:
    spend = df["spend_aed"].sum()
    rev = df["verified_revenue_aed"].sum()
    claimed = df["platform_reported_revenue_aed"].sum()
    losing = df[df["contribution_aed"] < 0]
    return {
        "spend": spend,
        "verified_revenue": rev,
        "platform_revenue": claimed,
        "verified_roas": rev / spend,
        "platform_roas": claimed / spend,
        "overclaim_pct": claimed / rev - 1,
        "contribution": df["contribution_aed"].sum(),
        "losing_spend": losing["spend_aed"].sum(),
        "losing_share": losing["spend_aed"].sum() / spend,
        "losing_rows": len(losing),
        "rows": len(df),
        "campaigns": df["campaign_id"].nunique(),
        "brands": df["brand"].nunique(),
        "weeks": df["week_start"].nunique(),
        "orders": df["orders"].sum(),
        "new_customers": df["new_customers"].sum(),
    }


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("channel").agg(
        spend_aed=("spend_aed", "sum"),
        verified_revenue_aed=("verified_revenue_aed", "sum"),
        platform_reported_revenue_aed=("platform_reported_revenue_aed", "sum"),
        contribution_aed=("contribution_aed", "sum"),
        orders=("orders", "sum"),
        new_customers=("new_customers", "sum"),
        profitable_share=("profitable", "mean"),
        campaign_weeks=("campaign_id", "size"),
    )
    g["verified_roas"] = g["verified_revenue_aed"] / g["spend_aed"]
    g["platform_roas"] = g["platform_reported_revenue_aed"] / g["spend_aed"]
    g["overclaim_pct"] = g["platform_reported_revenue_aed"] / g["verified_revenue_aed"] - 1
    g["cac_aed"] = g["spend_aed"] / g["new_customers"].replace(0, np.nan)
    g["spend_share"] = g["spend_aed"] / g["spend_aed"].sum()
    return g.sort_values("spend_aed", ascending=False).reset_index()


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).agg(
        spend_aed=("spend_aed", "sum"),
        verified_revenue_aed=("verified_revenue_aed", "sum"),
        contribution_aed=("contribution_aed", "sum"),
        profitable_share=("profitable", "mean"),
        campaign_weeks=("campaign_id", "size"),
    )
    g["verified_roas"] = g["verified_revenue_aed"] / g["spend_aed"]
    return g.sort_values("spend_aed", ascending=False).reset_index()


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["week_start", "season"], as_index=False).agg(
        spend_aed=("spend_aed", "sum"),
        verified_revenue_aed=("verified_revenue_aed", "sum"),
        platform_reported_revenue_aed=("platform_reported_revenue_aed", "sum"),
        contribution_aed=("contribution_aed", "sum"),
    )
    g["verified_roas"] = g["verified_revenue_aed"] / g["spend_aed"]
    return g.sort_values("week_start")


def money_leaks(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Campaigns that lost the most money (negative contribution) over the period."""
    g = df.groupby(["campaign_id", "brand", "channel", "objective"], as_index=False).agg(
        spend_aed=("spend_aed", "sum"),
        verified_revenue_aed=("verified_revenue_aed", "sum"),
        contribution_aed=("contribution_aed", "sum"),
        gross_margin=("gross_margin", "first"),
        weeks=("week_start", "nunique"),
    )
    g["verified_roas"] = g["verified_revenue_aed"] / g["spend_aed"]
    g["breakeven_roas"] = 1 / g["gross_margin"]
    return g.sort_values("contribution_aed").head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Attribution truth check
# ---------------------------------------------------------------------------

def attribution_gap(df: pd.DataFrame) -> pd.DataFrame:
    s = channel_summary(df)[["channel", "spend_aed", "verified_revenue_aed", "platform_reported_revenue_aed",
                              "verified_roas", "platform_roas", "overclaim_pct"]]
    s["phantom_revenue_aed"] = s["platform_reported_revenue_aed"] - s["verified_revenue_aed"]
    return s.sort_values("overclaim_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Predictive model: will this campaign-week be profitable?
# ---------------------------------------------------------------------------

CAT_FEATURES = ["category", "channel", "objective", "creative_format", "ad_language", "target_emirate", "season"]
NUM_FEATURES = ["log_spend", "creative_age_weeks", "gross_margin"]


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CAT_FEATURES + ["creative_age_weeks", "gross_margin"]].copy()
    X["log_spend"] = np.log(df["spend_aed"])
    return X[CAT_FEATURES + NUM_FEATURES]


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", num, NUM_FEATURES),
    ])


@dataclass
class ModelResult:
    model: Pipeline
    baseline: Pipeline
    metrics: pd.DataFrame
    importance: pd.DataFrame
    roc: dict
    confusion: np.ndarray
    split_date: pd.Timestamp
    n_train: int
    n_test: int
    cv_auc_mean: float
    cv_auc_std: float


def train_profit_model(df: pd.DataFrame, test_share: float = 0.25) -> ModelResult:
    """Time-based hold-out: train on the earliest 75% of weeks, test on the latest 25%.
    Uses only information a marketer knows BEFORE the week runs (no clicks/orders)."""
    df = df.sort_values("week_start")
    weeks = np.sort(df["week_start"].unique())
    split = pd.Timestamp(weeks[int(len(weeks) * (1 - test_share))])
    train, test = df[df["week_start"] < split], df[df["week_start"] >= split]
    X_tr, y_tr = model_frame(train), train["profitable"]
    X_te, y_te = model_frame(test), test["profitable"]

    rf = Pipeline([
        ("prep", _preprocessor(scale=False)),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=4,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])
    lr = Pipeline([
        ("prep", _preprocessor(scale=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv = cross_val_score(rf, X_tr, y_tr, cv=5, scoring="roc_auc")
    rf.fit(X_tr, y_tr)
    lr.fit(X_tr, y_tr)

    rows = []
    for name, m in [("Random Forest", rf), ("Logistic Regression (baseline)", lr)]:
        p = m.predict_proba(X_te)[:, 1]
        yhat = (p >= 0.5).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_te, yhat),
            "precision": precision_score(y_te, yhat),
            "recall": recall_score(y_te, yhat),
            "f1": f1_score(y_te, yhat),
            "roc_auc": roc_auc_score(y_te, p),
        })
    rows.append({"model": "Naive rule (always predict majority class)",
                 "accuracy": max(y_te.mean(), 1 - y_te.mean()), "precision": np.nan,
                 "recall": np.nan, "f1": np.nan, "roc_auc": 0.5})
    metrics = pd.DataFrame(rows)

    p_rf = rf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p_rf)
    cm = confusion_matrix(y_te, (p_rf >= 0.5).astype(int))

    # Feature importance aggregated back to the original business variables
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = rf.named_steps["clf"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        base = n.split("__", 1)[1]
        for f in CAT_FEATURES + NUM_FEATURES:
            if base == f or base.startswith(f + "_"):
                agg[f] = agg.get(f, 0) + v
                break
    pretty = {"category": "Product category", "channel": "Ad channel", "objective": "Objective (prospecting / retargeting)",
              "creative_format": "Creative format", "ad_language": "Ad language", "target_emirate": "Target emirate",
              "season": "Season", "log_spend": "Weekly spend level", "creative_age_weeks": "Creative age (fatigue)",
              "gross_margin": "Gross margin"}
    importance = (pd.DataFrame({"feature": [pretty[k] for k in agg], "importance": list(agg.values())})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    return ModelResult(rf, lr, metrics, importance, {"fpr": fpr, "tpr": tpr}, cm, split,
                       len(train), len(test), float(cv.mean()), float(cv.std()))


def predict_one(model: Pipeline, **kw) -> float:
    row = pd.DataFrame([kw])
    row["log_spend"] = np.log(row.pop("spend_aed"))
    return float(model.predict_proba(row[CAT_FEATURES + NUM_FEATURES])[:, 1][0])


# ---------------------------------------------------------------------------
# 5. Response curves & budget optimiser
# ---------------------------------------------------------------------------

# Elasticity is constrained to a plausible diminishing-returns range. Unconstrained
# estimates on 40-50 noisy weeks occasionally drift to ~1 (no saturation at all),
# which would let the optimiser pour unlimited money into one channel.
B_MIN, B_MAX = 0.30, 0.90


def fit_response_curves(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """For one brand, fit  revenue = k * spend^b  per channel on weekly totals,
    controlling for season with dummy variables. k is reported for a 'Regular' week.
    If b falls outside [B_MIN, B_MAX] it is clipped and the intercept re-estimated."""
    d = df[df["brand"] == brand]
    wk = d.groupby(["channel", "week_start", "season"], as_index=False)[["spend_aed", "verified_revenue_aed"]].sum()
    wk = wk[wk["verified_revenue_aed"] > 0]
    seasons = [s for s in sorted(wk["season"].unique()) if s != "Regular"]
    out = []
    for ch, g in wk.groupby("channel"):
        if len(g) < 12:
            continue
        X = np.column_stack([np.ones(len(g)), np.log(g["spend_aed"])] +
                            [(g["season"] == s).astype(float) for s in seasons])
        y = np.log(g["verified_revenue_aed"])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - resid.var() / y.var()
        b = float(np.clip(coef[1], B_MIN, B_MAX))
        # re-centre the intercept (identical to OLS when b was not clipped)
        season_part = X[:, 2:] @ coef[2:] if len(seasons) else 0.0
        a = float(np.mean(y - b * np.log(g["spend_aed"]) - season_part))
        resid_final = y - (a + b * np.log(g["spend_aed"]) + season_part)
        smear = float(np.mean(np.exp(resid_final)))   # Duan smearing correction for log models
        out.append({
            "brand": brand, "channel": ch, "k": np.exp(a) * smear, "b": b, "b_raw": float(coef[1]),
            "b_clipped": bool(b != coef[1]), "r2": r2, "weeks": len(g),
            "avg_weekly_spend": g["spend_aed"].mean(), "min_spend": g["spend_aed"].min(),
            "max_spend": g["spend_aed"].max(),
        })
    return pd.DataFrame(out)


def curve_revenue(k: float, b: float, spend):
    return k * np.power(np.maximum(spend, 0), b)


def optimise_budget(curves: pd.DataFrame, total_budget: float, margin: float,
                    lower: float = 0.3, upper: float = 2.0, steps: int = 600) -> pd.DataFrame:
    """Greedy marginal-return allocation of a fixed weekly budget across channels.
    Each channel is kept between `lower` x and `upper` x its average historic spend
    so the optimiser never extrapolates far outside the observed data."""
    c = curves.copy()
    base = c["avg_weekly_spend"].to_numpy()
    current = base / base.sum() * total_budget
    lo, hi = base * lower, base * upper
    # start from lower bounds (scaled if the budget is very small)
    alloc = lo * min(1.0, total_budget / lo.sum())
    remaining = total_budget - alloc.sum()
    step = max(total_budget / steps, 1.0)
    k, b = c["k"].to_numpy(), c["b"].to_numpy()
    while remaining > 1e-6:
        inc = min(step, remaining)
        gain = curve_revenue(k, b, alloc + inc) - curve_revenue(k, b, alloc)
        gain[alloc + inc > hi] = -np.inf
        if np.all(~np.isfinite(gain)):
            break
        j = int(np.argmax(gain))
        alloc[j] += inc
        remaining -= inc
    c["current_spend"] = current
    c["optimised_spend"] = alloc
    c["current_revenue"] = curve_revenue(k, b, current)
    c["optimised_revenue"] = curve_revenue(k, b, alloc)
    c["current_contribution"] = c["current_revenue"] * margin - c["current_spend"]
    c["optimised_contribution"] = c["optimised_revenue"] * margin - c["optimised_spend"]
    c["change_aed"] = c["optimised_spend"] - c["current_spend"]
    c["marginal_roas_current"] = k * b * np.power(current, b - 1)
    return c


def optimise_all_brands(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail, summary = [], []
    for brand in sorted(df["brand"].unique()):
        curves = fit_response_curves(df, brand)
        if curves.empty:
            continue
        margin = float(df.loc[df["brand"] == brand, "gross_margin"].iloc[0])
        budget = curves["avg_weekly_spend"].sum()
        o = optimise_budget(curves, budget, margin)
        detail.append(o)
        summary.append({
            "brand": brand, "weekly_budget_aed": budget, "gross_margin": margin,
            "current_revenue_aed": o["current_revenue"].sum(), "optimised_revenue_aed": o["optimised_revenue"].sum(),
            "current_contribution_aed": o["current_contribution"].sum(),
            "optimised_contribution_aed": o["optimised_contribution"].sum(),
        })
    s = pd.DataFrame(summary)
    s["revenue_uplift_pct"] = s["optimised_revenue_aed"] / s["current_revenue_aed"] - 1
    s["contribution_uplift_aed_week"] = s["optimised_contribution_aed"] - s["current_contribution_aed"]
    s["contribution_uplift_aed_year"] = s["contribution_uplift_aed_week"] * 52
    return pd.concat(detail, ignore_index=True), s


# ---------------------------------------------------------------------------
# 6. Customer segmentation & LTV : CAC
# ---------------------------------------------------------------------------

SEG_FEATURES = ["orders_12m", "avg_order_value_aed", "days_since_last_order", "discount_order_share"]


def segment_customers(cust: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = cust[SEG_FEATURES].copy()
    X["orders_12m"] = np.log1p(X["orders_12m"])
    X["avg_order_value_aed"] = np.log(X["avg_order_value_aed"])
    Xs = StandardScaler().fit_transform(X)

    sil = []
    for kk in range(2, 8):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(Xs)
        sil.append({"k": kk, "silhouette": silhouette_score(Xs, km.labels_, sample_size=3000, random_state=SEED),
                    "inertia": km.inertia_})
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    out = cust.copy()
    out["cluster"] = km.labels_

    prof = out.groupby("cluster").agg(
        customers=("customer_id", "size"),
        orders_12m=("orders_12m", "mean"),
        avg_order_value_aed=("avg_order_value_aed", "mean"),
        days_since_last_order=("days_since_last_order", "mean"),
        discount_order_share=("discount_order_share", "mean"),
        gross_profit_12m_aed=("gross_profit_12m_aed", "mean"),
    )
    # Name the clusters from their profiles (rule-based so names are stable)
    names = {}
    remaining = list(prof.index)

    def take(col, largest=True):
        s = prof.loc[remaining, col]
        idx = s.idxmax() if largest else s.idxmin()
        remaining.remove(idx)
        return idx

    names[take("orders_12m")] = "Loyal repeat buyers"
    names[take("discount_order_share")] = "Discount-driven shoppers"
    names[take("days_since_last_order")] = "Lapsed one-timers"
    for r in remaining:
        names[r] = "Recent big-basket buyers" if prof.loc[r, "avg_order_value_aed"] >= prof["avg_order_value_aed"].median() \
            else "Recent first-timers"
    seen = {}
    for c_ in sorted(names):                      # keep names unique when k is large
        n_ = names[c_]
        seen[n_] = seen.get(n_, 0) + 1
        if seen[n_] > 1:
            names[c_] = f"{n_} ({seen[n_]})"
    out["segment"] = out["cluster"].map(names)
    prof["segment"] = prof.index.map(names)
    prof["share"] = prof["customers"] / prof["customers"].sum()
    prof["profit_share"] = (prof["customers"] * prof["gross_profit_12m_aed"]) / (out["gross_profit_12m_aed"].sum())
    return out, prof.reset_index(), pd.DataFrame(sil)


def segment_mix_by_channel(seg: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(seg["acquisition_channel"], seg["segment"], normalize="index")
    return t


def ltv_cac(df: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    ch = channel_summary(df)[["channel", "spend_aed", "new_customers", "cac_aed"]]
    ltv = cust.groupby("acquisition_channel").agg(
        customers=("customer_id", "size"),
        ltv_12m_gross_profit_aed=("gross_profit_12m_aed", "mean"),
        repeat_rate=("orders_12m", lambda s: (s > 1).mean()),
        avg_orders=("orders_12m", "mean"),
    ).reset_index().rename(columns={"acquisition_channel": "channel"})
    m = ch.merge(ltv, on="channel")
    m["ltv_to_cac"] = m["ltv_12m_gross_profit_aed"] / m["cac_aed"]
    return m.sort_values("ltv_to_cac", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Scenario modelling (portfolio level, using fitted curves)
# ---------------------------------------------------------------------------

def scenario(df: pd.DataFrame, budget_change: float = 0.0, cost_inflation: float = 0.0,
             demand_change: float = 0.0, optimise: bool = False) -> dict:
    """Projected weekly revenue & contribution for the whole portfolio.
    budget_change, cost_inflation, demand_change are fractions (0.10 = +10%)."""
    rev = contrib = spend = 0.0
    for brand in df["brand"].unique():
        curves = fit_response_curves(df, brand)
        if curves.empty:
            continue
        margin = float(df.loc[df["brand"] == brand, "gross_margin"].iloc[0])
        budget = curves["avg_weekly_spend"].sum() * (1 + budget_change)
        if optimise:
            o = optimise_budget(curves, budget, margin)
            s = o["optimised_spend"].to_numpy()
        else:
            s = curves["avg_weekly_spend"].to_numpy() * (1 + budget_change)
        effective = s / (1 + cost_inflation)              # higher CPMs buy less media
        r = curve_revenue(curves["k"].to_numpy(), curves["b"].to_numpy(), effective) * (1 + demand_change)
        rev += r.sum()
        spend += s.sum()
        contrib += (r * margin).sum() - s.sum()
    return {"weekly_spend": spend, "weekly_revenue": rev, "weekly_contribution": contrib,
            "roas": rev / spend if spend else np.nan}


# ---------------------------------------------------------------------------
# 8. SaaS pricing & break-even (illustrative assumptions)
# ---------------------------------------------------------------------------

PRICING_TIERS = pd.DataFrame([
    {"tier": "Starter", "price_aed_month": 590, "for": "1 brand, up to AED 40k monthly ad spend, 3 channels",
     "mix": 0.50},
    {"tier": "Growth", "price_aed_month": 1490, "for": "1-2 brands, up to AED 200k monthly ad spend, all channels, optimiser",
     "mix": 0.40},
    {"tier": "Agency", "price_aed_month": 3900, "for": "Up to 10 brands, white-label reports, API access",
     "mix": 0.10},
])

COST_ASSUMPTIONS = {
    "fixed_cost_aed_month": 95_000,      # 4-person team, office/visa share, tools
    "variable_cost_aed_customer_month": 140,   # cloud, data connectors, support
    "cac_aed": 6_000,                    # sales + marketing cost per new subscriber
    "monthly_churn": 0.04,
}


def blended_arpu(tiers: pd.DataFrame = PRICING_TIERS) -> float:
    return float((tiers["price_aed_month"] * tiers["mix"]).sum() / tiers["mix"].sum())


def breakeven(fixed: float, arpu: float, var_cost: float) -> float:
    margin = arpu - var_cost
    return fixed / margin if margin > 0 else np.inf


def unit_economics(arpu: float, var_cost: float, cac: float, churn: float) -> dict:
    gm = arpu - var_cost
    lifetime = 1 / churn
    ltv = gm * lifetime
    return {"arpu": arpu, "gross_margin_per_customer": gm, "gross_margin_pct": gm / arpu,
            "lifetime_months": lifetime, "ltv": ltv, "ltv_to_cac": ltv / cac, "payback_months": cac / gm}


def subscriber_projection(months: int = 36, new_per_month_start: int = 4, growth: float = 0.08,
                          churn: float = 0.04, arpu: float | None = None, var_cost: float = 140,
                          fixed: float = 95_000, cac: float = 6_000) -> pd.DataFrame:
    arpu = arpu or blended_arpu()
    subs, cum = 0.0, 0.0
    rows = []
    for m in range(1, months + 1):
        new = new_per_month_start * (1 + growth) ** (m - 1)
        subs = subs * (1 - churn) + new
        revenue = subs * arpu
        cost = fixed + subs * var_cost + new * cac
        profit = revenue - cost
        cum += profit
        rows.append({"month": m, "new_subscribers": new, "subscribers": subs, "revenue_aed": revenue,
                     "total_cost_aed": cost, "operating_profit_aed": profit, "cumulative_profit_aed": cum})
    return pd.DataFrame(rows)
