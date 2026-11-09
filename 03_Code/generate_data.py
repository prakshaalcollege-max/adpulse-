"""
AdPulse AI - synthetic demonstration data generator
====================================================

Creates the two demonstration datasets used by the AdPulse prototype:

  02_Data/campaign_performance.csv  - one row per campaign per week (2025)
  02_Data/customers.csv             - one row per customer acquired in 2025

IMPORTANT: every brand, campaign, customer and number in these files is
SYNTHETIC. The data is generated from stated assumptions (below) so that the
analytics in the prototype can be demonstrated end-to-end. It is NOT real
market data and must not be cited as evidence about the UAE market.

Run from the repository root:   python 03_Code/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_Data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Assumptions
# ---------------------------------------------------------------------------

# Six fictional UAE direct-to-consumer brands
BRANDS = {
    #  name             category                    gross margin  size  AOV (AED)
    "Oudh Lane":      ("Fragrance & Beauty",         0.65,        1.30, 310),
    "Sahra Active":   ("Fashion & Activewear",       0.55,        1.00, 240),
    "Voltbay":        ("Electronics Accessories",    0.32,        1.40, 185),
    "Dar Living":     ("Home & Living",              0.45,        0.90, 360),
    "Greenbite":      ("Health Food & Snacks",       0.40,        0.60, 120),
    "Little Nakhla":  ("Baby & Kids",                0.48,        0.70, 210),
}

# Channel economics at a "typical" weekly spend S0:
#   ROAS0  = verified revenue / spend at S0
#   b      = spend elasticity (b < 1 means diminishing returns)
#   infl   = how much the ad platform's own dashboard over-claims revenue
#   cpc    = cost per click (AED), ctr = click-through rate
CHANNELS = {
    #                  S0    ROAS0  b     infl  noise  cpc   ctr
    "Google Search":  (3000, 4.6,  0.62, 1.15, 0.30, 2.40, 0.055),
    "Meta Ads":       (5000, 2.9,  0.75, 1.45, 0.33, 1.60, 0.012),
    "TikTok Ads":     (3500, 2.0,  0.80, 1.70, 0.38, 0.90, 0.009),
    "Snapchat Ads":   (3000, 2.3,  0.78, 1.60, 0.36, 1.00, 0.007),
    "YouTube":        (2500, 1.9,  0.85, 1.90, 0.35, 1.80, 0.004),
    "Influencers":    (4000, 2.4,  0.70, 1.05, 0.55, 1.20, 0.015),
}

# Which channels each brand uses
BRAND_CHANNELS = {
    "Oudh Lane":     ["Google Search", "Meta Ads", "Snapchat Ads", "TikTok Ads", "Influencers", "YouTube"],
    "Sahra Active":  ["Google Search", "Meta Ads", "TikTok Ads", "Snapchat Ads", "Influencers"],
    "Voltbay":       ["Google Search", "Meta Ads", "TikTok Ads", "YouTube", "Snapchat Ads"],
    "Dar Living":    ["Google Search", "Meta Ads", "Influencers", "YouTube"],
    "Greenbite":     ["Google Search", "Meta Ads", "Influencers", "TikTok Ads"],
    "Little Nakhla": ["Google Search", "Meta Ads", "Snapchat Ads", "Influencers"],
}

# Category x channel fit (multiplier on revenue)
FIT = {
    ("Electronics Accessories", "Google Search"): 1.15,
    ("Electronics Accessories", "TikTok Ads"): 0.80,
    ("Fragrance & Beauty", "Snapchat Ads"): 1.15,
    ("Fragrance & Beauty", "Influencers"): 1.10,
    ("Fashion & Activewear", "TikTok Ads"): 1.15,
    ("Home & Living", "Meta Ads"): 1.10,
    ("Home & Living", "TikTok Ads"): 0.85,
    ("Health Food & Snacks", "Influencers"): 1.10,
    ("Baby & Kids", "Meta Ads"): 1.10,
}

# Campaign objective: revenue multiplier, spend multiplier, elasticity shift
OBJECTIVES = {
    "Prospecting": (0.80, 1.00, 0.00),
    "Retargeting": (1.60, 0.40, -0.12),
}

# Creative formats allowed per channel and their effect
CREATIVE = {
    "Google Search": {"Search text ad": 1.00},
    "Meta Ads":      {"Static image": 0.90, "Carousel": 1.05, "Short video": 1.05, "UGC video": 1.15},
    "TikTok Ads":    {"Short video": 1.00, "UGC video": 1.25, "Static image": 0.75},
    "Snapchat Ads":  {"Short video": 1.00, "UGC video": 1.20, "Static image": 0.80},
    "YouTube":       {"Short video": 1.00, "Long video": 0.90},
    "Influencers":   {"UGC video": 1.00},
}

# Ad language effect per channel
LANGUAGE = {
    "Snapchat Ads": {"Arabic": 1.20, "Bilingual": 1.10, "English": 0.90},
    "TikTok Ads":   {"Arabic": 1.05, "Bilingual": 1.10, "English": 0.95},
    "Google Search": {"Arabic": 0.95, "Bilingual": 1.00, "English": 1.00},
}
LANG_WEIGHTS = {"English": 0.45, "Arabic": 0.25, "Bilingual": 0.30}

EMIRATES = {"All UAE": 1.00, "Dubai": 1.05, "Abu Dhabi": 1.08, "Sharjah & Northern": 0.85}
EMIRATE_WEIGHTS = [0.45, 0.25, 0.18, 0.12]

# Seasons (2025 calendar). demand multiplier by category, ad-cost inflation,
# and how much brands typically raise spend.
SEASON_DEMAND = {
    "Ramadan & Eid":        {"Fragrance & Beauty": 1.45, "Fashion & Activewear": 1.35, "Electronics Accessories": 1.05,
                             "Home & Living": 1.25, "Health Food & Snacks": 1.30, "Baby & Kids": 1.10},
    "White Friday & 11.11": {"Fragrance & Beauty": 1.30, "Fashion & Activewear": 1.35, "Electronics Accessories": 1.60,
                             "Home & Living": 1.35, "Health Food & Snacks": 1.05, "Baby & Kids": 1.25},
    "DSF & National Day":   {"Fragrance & Beauty": 1.20, "Fashion & Activewear": 1.25, "Electronics Accessories": 1.25,
                             "Home & Living": 1.20, "Health Food & Snacks": 1.05, "Baby & Kids": 1.15},
    "Back to School":       {"Fragrance & Beauty": 1.00, "Fashion & Activewear": 1.15, "Electronics Accessories": 1.25,
                             "Home & Living": 1.00, "Health Food & Snacks": 1.00, "Baby & Kids": 1.30},
    "Summer":               {c: 0.80 for c in ["Fragrance & Beauty", "Fashion & Activewear", "Home & Living",
                                               "Health Food & Snacks", "Baby & Kids"]} | {"Electronics Accessories": 0.90},
    "Regular":              {},
}
SEASON_COST = {"Ramadan & Eid": 1.25, "White Friday & 11.11": 1.35, "DSF & National Day": 1.15,
               "Back to School": 1.05, "Summer": 0.90, "Regular": 1.00}
SEASON_SPEND = {"Ramadan & Eid": 1.45, "White Friday & 11.11": 1.60, "DSF & National Day": 1.25,
                "Back to School": 1.10, "Summer": 0.75, "Regular": 1.00}

NEW_CUSTOMER_SHARE = {"Prospecting": 0.70, "Retargeting": 0.22}


def season_of(d: pd.Timestamp) -> str:
    """Approximate 2025 UAE retail calendar (week start date)."""
    md = (d.month, d.day)
    if d <= pd.Timestamp("2025-01-12") or d >= pd.Timestamp("2025-12-01"):
        return "DSF & National Day"
    if pd.Timestamp("2025-02-24") <= d <= pd.Timestamp("2025-03-31"):
        return "Ramadan & Eid"
    if pd.Timestamp("2025-11-03") <= d <= pd.Timestamp("2025-11-30"):
        return "White Friday & 11.11"
    if pd.Timestamp("2025-08-18") <= d <= pd.Timestamp("2025-09-07"):
        return "Back to School"
    if pd.Timestamp("2025-06-30") <= d <= pd.Timestamp("2025-08-17"):
        return "Summer"
    return "Regular"


# ---------------------------------------------------------------------------
# 2. Campaign-week data
# ---------------------------------------------------------------------------

weeks = pd.date_range("2025-01-06", periods=52, freq="7D")
rows = []
camp_counter = 0

for brand, (category, margin, size, aov) in BRANDS.items():
    for channel in BRAND_CHANNELS[brand]:
        S0, roas0, b0, infl, sigma, cpc0, ctr0 = CHANNELS[channel]
        objectives = ["Prospecting"] if channel in ("YouTube", "Influencers") else ["Prospecting", "Retargeting"]
        for objective in objectives:
            camp_counter += 1
            camp_id = f"CMP-{camp_counter:03d}"
            obj_rev, obj_spend, obj_b = OBJECTIVES[objective]
            b = b0 + obj_b
            # k so that revenue = roas0 * S0 at the typical spend
            k = roas0 * S0 ** (1 - b)
            fit = FIT.get((category, channel), 1.0)
            emirate = rng.choice(list(EMIRATES), p=EMIRATE_WEIGHTS)

            # creative is refreshed every 6-14 weeks; language/format may change on refresh
            creative_age = 0
            next_refresh = int(rng.integers(6, 15))
            fmt_options = list(CREATIVE[channel])
            fmt = rng.choice(fmt_options)
            lang = rng.choice(list(LANG_WEIGHTS), p=list(LANG_WEIGHTS.values()))

            for w in weeks:
                creative_age += 1
                if creative_age > next_refresh:
                    creative_age = 1
                    next_refresh = int(rng.integers(6, 15))
                    fmt = rng.choice(fmt_options)
                    lang = rng.choice(list(LANG_WEIGHTS), p=list(LANG_WEIGHTS.values()))
                if rng.random() > 0.86:          # campaign paused this week
                    continue

                season = season_of(w)
                demand = SEASON_DEMAND[season].get(category, 1.0)
                cost_infl = SEASON_COST[season]

                spend = S0 * size * obj_spend * SEASON_SPEND[season] * rng.lognormal(0, 0.35)
                spend = float(np.clip(spend, 300, 60000))

                fatigue = max(0.70, 1 - 0.022 * (creative_age - 1))
                mult = (obj_rev * fit * CREATIVE[channel][fmt] * LANGUAGE.get(channel, {}).get(lang, 1.0)
                        * (1.15 if (category == "Fragrance & Beauty" and lang != "English") else 1.0)
                        * EMIRATES[emirate] * demand / cost_infl * fatigue)
                revenue = k * spend ** b * mult * rng.lognormal(-sigma ** 2 / 2, sigma)

                cpc = cpc0 * cost_infl * rng.lognormal(0, 0.12)
                clicks = max(1, int(spend / cpc))
                ctr = ctr0 * (1.0 if objective == "Prospecting" else 1.6) * rng.lognormal(0, 0.15)
                impressions = int(clicks / ctr)
                order_aov = aov * rng.lognormal(0, 0.08)
                orders = max(0, int(round(revenue / order_aov)))
                revenue = orders * order_aov            # keep revenue consistent with whole orders
                new_share = 0.50 if channel == "Google Search" else NEW_CUSTOMER_SHARE[objective]
                new_customers = int(rng.binomial(orders, new_share)) if orders else 0
                platform_rev = revenue * infl * rng.lognormal(0, 0.10)

                rows.append({
                    "week_start": w.date().isoformat(),
                    "week_of_year": int(w.isocalendar().week),
                    "season": season,
                    "brand": brand,
                    "category": category,
                    "gross_margin": margin,
                    "channel": channel,
                    "campaign_id": camp_id,
                    "objective": objective,
                    "creative_format": fmt,
                    "ad_language": lang,
                    "target_emirate": emirate,
                    "creative_age_weeks": creative_age,
                    "spend_aed": round(spend, 2),
                    "impressions": impressions,
                    "clicks": clicks,
                    "orders": orders,
                    "new_customers": new_customers,
                    "verified_revenue_aed": round(revenue, 2),
                    "platform_reported_revenue_aed": round(platform_rev, 2),
                })

camp = pd.DataFrame(rows)
camp["roas"] = (camp["verified_revenue_aed"] / camp["spend_aed"]).round(3)
camp["contribution_aed"] = (camp["verified_revenue_aed"] * camp["gross_margin"] - camp["spend_aed"]).round(2)
camp["profitable"] = (camp["contribution_aed"] > 0).astype(int)
camp.to_csv(DATA_DIR / "campaign_performance.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Customer data (customers acquired during 2025)
# ---------------------------------------------------------------------------

# Repeat behaviour by acquisition channel (assumption)
CUST = {
    #                 extra orders (mean), discount-code share, AOV multiplier
    "Google Search": (1.55, 0.18, 1.05),
    "Meta Ads":      (1.05, 0.33, 1.00),
    "TikTok Ads":    (0.55, 0.58, 0.85),
    "Snapchat Ads":  (0.75, 0.50, 0.92),
    "YouTube":       (0.95, 0.25, 1.02),
    "Influencers":   (1.25, 0.55, 0.97),
}

N_CUST = 4000
# Sample brand/channel in proportion to new customers actually acquired in the campaign data
acq = camp.groupby(["brand", "channel"], as_index=False)["new_customers"].sum()
acq["p"] = acq["new_customers"] / acq["new_customers"].sum()
pick = rng.choice(len(acq), size=N_CUST, p=acq["p"].to_numpy())

cust_rows = []
for i, idx in enumerate(pick):
    brand, channel = acq.loc[idx, "brand"], acq.loc[idx, "channel"]
    category, margin, _, aov = BRANDS[brand]
    extra_mean, disc_share, aov_mult = CUST[channel]
    first = pd.Timestamp("2025-01-06") + pd.Timedelta(days=int(rng.integers(0, 330)))
    tenure = (pd.Timestamp("2026-01-04") - first).days
    heterogeneity = rng.gamma(1.4, 1 / 1.4)
    orders = 1 + rng.poisson(extra_mean * heterogeneity * min(1.0, tenure / 365 + 0.3))
    cust_aov = aov * aov_mult * rng.lognormal(0, 0.25)
    revenue = orders * cust_aov
    discount = float(np.clip(rng.beta(2, 2) * 2 * disc_share, 0, 1))
    # more frequent buyers tend to have bought more recently
    recency = int(np.clip(rng.exponential(tenure / (orders + 0.5)), 1, tenure))
    gross_profit = revenue * margin * (1 - 0.15 * discount)
    cust_rows.append({
        "customer_id": f"CUS-{i + 1:05d}",
        "brand": brand,
        "category": category,
        "acquisition_channel": channel,
        "first_order_date": first.date().isoformat(),
        "tenure_days": tenure,
        "orders_12m": int(orders),
        "avg_order_value_aed": round(cust_aov, 2),
        "revenue_12m_aed": round(revenue, 2),
        "gross_profit_12m_aed": round(gross_profit, 2),
        "days_since_last_order": recency,
        "discount_order_share": round(discount, 3),
        "home_emirate": rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"],
                                   p=[0.42, 0.24, 0.17, 0.07, 0.05, 0.03, 0.02]),
    })

cust = pd.DataFrame(cust_rows)
cust.to_csv(DATA_DIR / "customers.csv", index=False)

if __name__ == "__main__":
    print(f"campaign_performance.csv: {len(camp):,} rows, {camp['campaign_id'].nunique()} campaigns")
    print(f"customers.csv: {len(cust):,} rows")
    print(f"share of profitable campaign-weeks: {camp['profitable'].mean():.1%}")
    print(camp.groupby("channel")[["spend_aed", "verified_revenue_aed"]].sum().assign(
        roas=lambda d: d.verified_revenue_aed / d.spend_aed).round(2))
    print(camp.groupby("brand")["profitable"].mean().round(2))
