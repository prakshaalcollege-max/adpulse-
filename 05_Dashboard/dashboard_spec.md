# AdPulse AI: Dashboard Specification

**App:** `03_Code/app.py` (Streamlit). **Analytics:** `03_Code/adpulse_engine.py`. **Screenshots:** `05_Dashboard/screenshots/`.

## Design principles

1. **Lead with the money.** Every page opens with an AED figure or a verdict, not a chart.
2. **Say what to do.** Recommendation cards say CUT / INCREASE with an amount and the reason.
3. **Show the evidence underneath.** Charts, model metrics and method sit on the same page.
4. **Profit, not ROAS.** Every judgement uses gross margin: contribution = verified revenue x margin - spend.
5. **Honest about data.** The sidebar and the Data & Method page state that the demo data is synthetic.

## Layout

- Dark navy sidebar: logo, page navigation (8 pages), brand filter (applies to Command Centre and Truth Check), data-source label.
- Main area: max width 1300 px; KPI tiles in rows of 3-4; cards for alerts and recommendations; Plotly charts with hover tooltips and legends below the plot.
- Colour: one accent blue (#2a78d6); fixed colour per channel (Google Search blue, Meta orange, TikTok aqua, Snapchat yellow, YouTube magenta, Influencers green); red / amber / green pills only for status, always with a text label.

## Pages

| # | Page | Inputs | Outputs | Analytics |
|---|---|---|---|---|
| 1 | Command Centre | Brand filter | Claimed vs verified revenue headline; KPI tiles (spend, verified revenue, claimed ROAS, spend that lost money); top 5 money-losing campaigns; 4 recommended moves; annual profit from re-allocation; weekly trend | KPIs, loss ranking, optimiser |
| 2 | Truth Check | Brand filter | Claimed, verified and phantom revenue; claimed vs verified ROAS chart; reconciliation table | Attribution gap |
| 3 | Profit Predictor | Brand, channel, objective, creative, language, target, season, spend, creative age | Probability of profit with verdict; break-even ROAS; top 6 alternatives at same budget; accuracy, AUC, precision, recall; feature importance; ROC; model comparison table | Random Forest + logistic baseline, time-based hold-out |
| 4 | Budget Optimiser | Brand, weekly budget slider (50-150%) | Revenue current vs optimised; annual extra gross profit; spend by channel (current vs recommended); response curves; channel table with elasticity, R², marginal ROAS | Log-log response curves, greedy allocation |
| 5 | Customer Value | k slider (3-6) | Segment cards (share of customers and profit); segment mix by channel; LTV : CAC by channel; silhouette chart | K-means, LTV : CAC |
| 6 | Scenario Lab | Budget change, ad-cost change, demand change, optimised toggle | Weekly spend, revenue, gross profit, ROAS vs base; profit curve across budget levels; insight | Response-curve simulation |
| 7 | Business Case | Fixed cost, variable cost, CAC, churn, starting subscribers, growth, tier mix | Tier cards; ARPU; break-even subscribers; LTV : CAC; payback; 36-month revenue vs cost chart | Pricing and unit economics |
| 8 | Data & Method | CSV uploads (campaigns, customers) | Validation messages; template downloads; data preview; methodology table | Data validation |

## Data contract for uploads

**Campaign CSV (one row per campaign per week):** week_start, season, brand, category, gross_margin, channel, campaign_id, objective, creative_format, ad_language, target_emirate, creative_age_weeks, spend_aed, orders, new_customers, verified_revenue_aed, platform_reported_revenue_aed

**Customer CSV (one row per customer):** customer_id, brand, acquisition_channel, orders_12m, avg_order_value_aed, revenue_12m_aed, gross_profit_12m_aed, days_since_last_order, discount_order_share

Missing columns are reported by name; the app keeps using the demo data until a valid file is loaded. "Reset to demo data" restores the original dataset.
