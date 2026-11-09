"""
AdPulse AI - Marketing Spend Intelligence for UAE online brands
================================================================
Streamlit prototype.  Run locally:   streamlit run 03_Code/app.py
On Streamlit Community Cloud set the main file path to:  03_Code/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adpulse_engine as E  # noqa: E402

st.set_page_config(page_title="AdPulse AI - Marketing Spend Intelligence", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
CHANNEL_COLOR = {"Google Search": BLUE, "Meta Ads": ORANGE, "TikTok Ads": AQUA,
                 "Snapchat Ads": YELLOW, "YouTube": MAGENTA, "Influencers": GREEN}
SEGMENT_COLOR = {"Loyal repeat buyers": BLUE, "Recent big-basket buyers": AQUA,
                 "Discount-driven shoppers": ORANGE, "Lapsed one-timers": MUTED,
                 "Recent first-timers": VIOLET}

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px;}
  [data-testid="stSidebar"] {background: #0f1b2d;}
  [data-testid="stSidebar"] * {color: #e8edf5 !important;}
  [data-testid="stSidebar"] .stRadio label p {font-size: 0.98rem;}
  [data-testid="stSidebar"] [data-baseweb="select"] > div {background: #1c2b44 !important; border-color: #3a5179 !important;}
  [data-testid="stSidebar"] [data-baseweb="select"] * {color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important;}
  [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {background: #1c2b44 !important; border-color: #3a5179 !important;}
  [data-testid="stSidebar"] [data-testid="stSelectbox"] input {color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important;}
  .ap-logo {font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
  .ap-logo span {color: #5ea2ef !important;}
  .ap-tag {font-size: 0.82rem; opacity: 0.75; margin-top: -2px;}
  .hero {background: linear-gradient(120deg, #0f1b2d 0%, #1c3a63 100%); color: #fff; border-radius: 18px;
         padding: 26px 30px; margin-bottom: 18px;}
  .hero h1 {color: #fff; font-size: 1.85rem; line-height: 1.25; margin: 0 0 6px 0; font-weight: 750;}
  .hero p {color: #c9d6ea; margin: 0; font-size: 1.02rem;}
  .hero b {color: #fff;}
  .kpi {background: #fff; border: 1px solid rgba(11,11,11,0.09); border-radius: 14px; padding: 16px 18px; height: 100%;}
  .kpi .lbl {color: #52514e; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;}
  .kpi .val {color: #0b0b0b; font-size: 1.75rem; font-weight: 750; margin-top: 2px;}
  .kpi .sub {color: #52514e; font-size: 0.85rem; margin-top: 2px;}
  .card {background: #fff; border: 1px solid rgba(11,11,11,0.09); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;}
  .card h4 {margin: 0 0 4px 0; font-size: 1.02rem;}
  .card p {margin: 0; color: #52514e; font-size: 0.92rem;}
  .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; margin-right: 6px;}
  .pill.crit {background: #fbe6e6; color: #a32a2a;}
  .pill.good {background: #e3f5e3; color: #006300;}
  .pill.warn {background: #fff3d6; color: #7a5300;}
  .pill.info {background: #e3eefb; color: #1c5cab;}
  .move {border-left: 4px solid #2a78d6;}
  .leak {border-left: 4px solid #d03b3b;}
  .note {font-size: 0.82rem; color: #898781;}
  h2, h3 {letter-spacing: -0.01em;}
</style>
""", unsafe_allow_html=True)


def aed(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}AED {x / 1e6:.2f}M"
    if x >= 1e4:
        return f"{sign}AED {x / 1e3:.0f}k"
    return f"{sign}AED {x:,.{decimals}f}"


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=44, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
        title_font=dict(size=15, color=INK), hoverlabel=dict(bgcolor="#fff", font_size=13),
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title=None),
        bargap=0.3,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Data (demo or uploaded) - cached
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def demo_data():
    return E.load_campaigns(), E.load_customers()


def get_data():
    camp, cust = demo_data()
    if st.session_state.get("upload_campaigns") is not None:
        camp = st.session_state["upload_campaigns"]
    if st.session_state.get("upload_customers") is not None:
        cust = st.session_state["upload_customers"]
    return camp, cust


@st.cache_resource(show_spinner="Training the profitability model...")
def get_model(df: pd.DataFrame):
    return E.train_profit_model(df)


@st.cache_data(show_spinner=False)
def get_optimisation(df: pd.DataFrame):
    return E.optimise_all_brands(df)


@st.cache_data(show_spinner=False)
def get_curves(df: pd.DataFrame, brand: str):
    return E.fit_response_curves(df, brand)


@st.cache_data(show_spinner=False)
def get_segments(cust: pd.DataFrame, k: int):
    return E.segment_customers(cust, k)


@st.cache_data(show_spinner=False)
def get_scenario(df: pd.DataFrame, b, c, d, o):
    return E.scenario(df, b, c, d, o)


camp_all, cust_all = get_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="ap-logo">Ad<span>Pulse</span> AI</p>'
                '<p class="ap-tag">Marketing spend intelligence for UAE online brands</p>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate", [
        "Command Centre", "Truth Check", "Profit Predictor", "Budget Optimiser",
        "Customer Value", "Scenario Lab", "Business Case", "Data & Method"], label_visibility="collapsed")
    st.divider()
    brands = ["All brands"] + sorted(camp_all["brand"].unique())
    brand_sel = st.selectbox("Brand filter (Command Centre & Truth Check)", brands)
    src = "Uploaded data" if st.session_state.get("upload_campaigns") is not None else "Synthetic demo data"
    st.caption(f"Data source: **{src}**  \n{len(camp_all):,} campaign-weeks · {camp_all['brand'].nunique()} brands")
    st.caption("All demo brands and numbers are synthetic and for demonstration only.")

camp = camp_all if brand_sel == "All brands" else camp_all[camp_all["brand"] == brand_sel]
scope = "your 6 demo brands" if brand_sel == "All brands" else brand_sel

# ===========================================================================
# PAGE: Command Centre
# ===========================================================================
if page == "Command Centre":
    k = E.headline_kpis(camp)
    st.markdown(f"""
    <div class="hero">
      <h1>Your ad platforms say you made {aed(k['platform_revenue'])}.<br>Your store recorded {aed(k['verified_revenue'])}.</h1>
      <p>AdPulse checks every dirham of ad spend for <b>{scope}</b> against real orders, finds the campaigns that lose
      money, and tells you where next week's budget should go.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Ad spend (2025)", aed(k["spend"]), f"{k['campaigns']} campaigns · {k['weeks']} weeks")
    kpi(c2, "Verified revenue", aed(k["verified_revenue"]), f"Real ROAS {k['verified_roas']:.2f}x")
    kpi(c3, "Platforms claimed", f"{k['platform_roas']:.2f}x ROAS",
        f"{k['overclaim_pct']:.0%} more revenue than really happened")
    kpi(c4, "Spend that lost money", aed(k["losing_spend"]),
        f"{k['losing_share']:.0%} of spend went to loss-making campaign-weeks")
    st.write("")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Where the money is leaking")
        leaks = E.money_leaks(camp, 5)
        for _, r in leaks.iterrows():
            if r["contribution_aed"] >= 0:
                continue
            st.markdown(f"""
            <div class="card leak"><span class="pill crit">LOSING MONEY</span>
              <span class="pill info">{r['channel']} · {r['objective']}</span>
              <h4 style="margin-top:6px">{r['brand']} - {r['campaign_id']}</h4>
              <p>Spent {aed(r['spend_aed'])}, real ROAS <b>{r['verified_roas']:.2f}x</b> vs break-even
              <b>{r['breakeven_roas']:.2f}x</b> (at {r['gross_margin']:.0%} margin).
              Net loss after product cost: <b>{aed(r['contribution_aed'])}</b>.</p>
            </div>""", unsafe_allow_html=True)
    with right:
        st.subheader("Recommended moves")
        detail, opt = get_optimisation(camp_all)
        d = detail if brand_sel == "All brands" else detail[detail["brand"] == brand_sel]
        moves = pd.concat([d.nsmallest(2, "change_aed"), d.nlargest(2, "change_aed")])
        for _, r in moves.iterrows():
            up = r["change_aed"] > 0
            st.markdown(f"""
            <div class="card move"><span class="pill {'good' if up else 'warn'}">{'INCREASE' if up else 'CUT'}</span>
              <h4 style="margin-top:6px">{r['brand']}: {'add' if up else 'cut'} {aed(abs(r['change_aed']))}/week on {r['channel']}</h4>
              <p>Each extra dirham here currently returns about <b>{r['marginal_roas_current']:.2f}</b> in revenue.</p>
            </div>""", unsafe_allow_html=True)
        o = opt if brand_sel == "All brands" else opt[opt["brand"] == brand_sel]
        st.markdown(f"""<div class="card"><span class="pill good">SAME BUDGET</span>
            <h4 style="margin-top:6px">+{aed(o['contribution_uplift_aed_year'].sum())} profit a year</h4>
            <p>Estimated extra gross profit if the recommended split is followed, with no extra spend
            (revenue +{(o['optimised_revenue_aed'].sum() / o['current_revenue_aed'].sum() - 1):.1%}).</p></div>""",
                    unsafe_allow_html=True)

    st.subheader("Weekly spend vs verified revenue")
    w = E.weekly_summary(camp)
    fig = go.Figure()
    fig.add_scatter(x=w["week_start"], y=w["platform_reported_revenue_aed"], name="Revenue claimed by platforms",
                    line=dict(color=ORANGE, width=2, dash="dot"),
                    hovertemplate="%{x|%d %b}<br>Claimed: AED %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=w["week_start"], y=w["verified_revenue_aed"], name="Verified revenue (store orders)",
                    line=dict(color=BLUE, width=2), customdata=w["season"],
                    hovertemplate="%{x|%d %b} · %{customdata}<br>Verified: AED %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=w["week_start"], y=w["spend_aed"], name="Ad spend", line=dict(color=INK2, width=2),
                    hovertemplate="%{x|%d %b}<br>Spend: AED %{y:,.0f}<extra></extra>")
    fig.update_layout(hovermode="x unified", yaxis_title="AED per week")
    show(style_fig(fig, 380))

# ===========================================================================
# PAGE: Truth Check
# ===========================================================================
elif page == "Truth Check":
    st.title("Truth Check")
    st.markdown("Every ad platform grades its own homework. Meta, TikTok, Snapchat and Google each count a sale as "
                "theirs if the shopper touched their ad, so the same order is often claimed several times. "
                "AdPulse matches spend to **real store orders** (UTM tags, promo codes and order IDs), "
                "then compares the two.")
    att = E.attribution_gap(camp)
    k = E.headline_kpis(camp)
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Revenue claimed by platforms", aed(k["platform_revenue"]))
    kpi(c2, "Revenue that really happened", aed(k["verified_revenue"]))
    kpi(c3, "Phantom revenue", aed(k["platform_revenue"] - k["verified_revenue"]),
        f"{k['overclaim_pct']:.0%} over-claim overall")
    st.write("")
    a = att.sort_values("platform_roas")
    fig = go.Figure()
    fig.add_bar(y=a["channel"], x=a["platform_roas"], orientation="h", name="ROAS claimed by the platform",
                marker_color=ORANGE, hovertemplate="%{y}: %{x:.2f}x claimed<extra></extra>")
    fig.add_bar(y=a["channel"], x=a["verified_roas"], orientation="h", name="ROAS verified in store orders",
                marker_color=BLUE, hovertemplate="%{y}: %{x:.2f}x verified<extra></extra>",
                text=[f"{v:.2f}x" for v in a["verified_roas"]], textposition="outside")
    fig.update_layout(barmode="group", xaxis_title="Return on ad spend (revenue / spend)",
                      title="Claimed vs verified return on ad spend", xaxis_range=[0, a["platform_roas"].max() * 1.15])
    show(style_fig(fig, 420))
    tbl = att.copy()
    tbl["Spend"] = tbl["spend_aed"].map(aed)
    tbl["Claimed revenue"] = tbl["platform_reported_revenue_aed"].map(aed)
    tbl["Verified revenue"] = tbl["verified_revenue_aed"].map(aed)
    tbl["Over-claim"] = tbl["overclaim_pct"].map(lambda v: f"{v:+.0%}")
    st.dataframe(tbl[["channel", "Spend", "Claimed revenue", "Verified revenue", "Over-claim"]]
                 .rename(columns={"channel": "Channel"}), hide_index=True, width="stretch")
    st.info("**Why it matters:** a marketer who trusts the platform dashboards will move budget toward the channels "
            "that over-claim the most. In this data those are YouTube, TikTok and Snapchat, which are also the channels "
            "with the weakest real returns.")

# ===========================================================================
# PAGE: Profit Predictor
# ===========================================================================
elif page == "Profit Predictor":
    st.title("Profit Predictor")
    st.markdown("Before you launch or renew a campaign, AdPulse estimates the probability that it will "
                "**make money after product costs** in a given week. The model is a Random Forest trained only on "
                "facts you know *before* the week runs: channel, objective, creative, language, targeting, season, "
                "creative age and planned spend.")
    res = get_model(camp_all)
    brands_meta = camp_all.groupby("brand").agg(category=("category", "first"), margin=("gross_margin", "first"))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        b = c1.selectbox("Brand", brands_meta.index)
        channel = c2.selectbox("Channel", sorted(camp_all["channel"].unique()))
        fmt_opts = sorted(camp_all.loc[camp_all["channel"] == channel, "creative_format"].unique()) or ["Short video"]
        obj_opts = sorted(camp_all.loc[camp_all["channel"] == channel, "objective"].unique())
        objective = c3.selectbox("Objective", obj_opts)
        c4, c5, c6 = st.columns(3)
        fmt = c4.selectbox("Creative format", fmt_opts)
        lang = c5.selectbox("Ad language", ["English", "Arabic", "Bilingual"])
        emirate = c6.selectbox("Target", ["All UAE", "Dubai", "Abu Dhabi", "Sharjah & Northern"])
        c7, c8, c9 = st.columns(3)
        season = c7.selectbox("Season", ["Regular", "Ramadan & Eid", "White Friday & 11.11", "DSF & National Day",
                                         "Back to School", "Summer"])
        spend = c8.number_input("Planned weekly spend (AED)", 300, 60000, 4000, step=500)
        age = c9.slider("Creative age (weeks running)", 1, 14, 2)

    cat, margin = brands_meta.loc[b, "category"], float(brands_meta.loc[b, "margin"])
    base = dict(category=cat, channel=channel, objective=objective, creative_format=fmt, ad_language=lang,
                target_emirate=emirate, season=season, creative_age_weeks=age, gross_margin=margin, spend_aed=spend)
    p = E.predict_one(res.model, **base)
    verdict, cls = (("Likely profitable", "good") if p >= 0.65 else
                    ("Uncertain - test small first", "warn") if p >= 0.45 else ("Likely to lose money", "crit"))
    l, r = st.columns([1, 1.3])
    with l:
        st.markdown(f"""<div class="card"><span class="pill {cls}">{verdict.upper()}</span>
            <div class="kpi" style="border:none;padding:6px 0 0 0"><div class="lbl">Probability of profit</div>
            <div class="val">{p:.0%}</div>
            <div class="sub">{b} needs a real ROAS above <b>{1 / margin:.2f}x</b> to cover its {margin:.0%} gross margin.</div></div>
            </div>""", unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0))
    with r:
        st.markdown("**Better options for the same budget**")
        alts = []
        for ch in sorted(camp_all["channel"].unique()):
            sub = camp_all[camp_all["channel"] == ch]
            for ob in sub["objective"].unique():
                for f in sub[sub["objective"] == ob]["creative_format"].unique():
                    q = E.predict_one(res.model, **{**base, "channel": ch, "objective": ob, "creative_format": f,
                                                    "creative_age_weeks": 1})
                    alts.append({"Channel": ch, "Objective": ob, "Creative": f, "Probability": q})
        alts = pd.DataFrame(alts).sort_values("Probability", ascending=False).head(6)
        alts["Probability"] = alts["Probability"].map(lambda v: f"{v:.0%}")
        st.dataframe(alts, hide_index=True, width="stretch")
        st.caption("Same brand, season, language, target and spend, with fresh creative.")

    st.divider()
    st.subheader("How good is the model?")
    mt = res.metrics.copy()
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Accuracy (hold-out)", f"{mt.loc[0, 'accuracy']:.1%}", f"vs {mt.loc[2, 'accuracy']:.1%} naive guess")
    kpi(c2, "ROC AUC", f"{mt.loc[0, 'roc_auc']:.3f}", f"5-fold CV {res.cv_auc_mean:.3f} ± {res.cv_auc_std:.3f}")
    kpi(c3, "Precision", f"{mt.loc[0, 'precision']:.1%}", "when it says 'profitable', how often right")
    kpi(c4, "Recall", f"{mt.loc[0, 'recall']:.1%}", "share of profitable weeks it finds")
    st.caption(f"Trained on {res.n_train:,} campaign-weeks before {res.split_date:%d %b %Y}; tested on the "
               f"{res.n_test:,} later weeks it never saw (time-based hold-out).")
    l, r = st.columns(2)
    with l:
        imp = res.importance.sort_values("importance")
        fig = go.Figure(go.Bar(y=imp["feature"], x=imp["importance"] * 100, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title="What drives profitability", xaxis_title="Importance (%)")
        show(style_fig(fig, 380, legend=False))
    with r:
        fig = go.Figure()
        fig.add_scatter(x=res.roc["fpr"], y=res.roc["tpr"], name=f"Random Forest (AUC {mt.loc[0, 'roc_auc']:.2f})",
                        line=dict(color=BLUE, width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color=MUTED, width=1, dash="dash"))
        fig.update_layout(title="ROC curve (hold-out)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        show(style_fig(fig, 380))
    disp = mt.copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        disp[c] = disp[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(disp, hide_index=True, width="stretch")

# ===========================================================================
# PAGE: Budget Optimiser
# ===========================================================================
elif page == "Budget Optimiser":
    st.title("Budget Optimiser")
    st.markdown("Every channel has **diminishing returns**: the first AED 1,000 on Google Search earns far more "
                "than the tenth. AdPulse fits a response curve `revenue = k × spend^b` for each channel of each brand "
                "(controlling for season), then moves money from the flattest curves to the steepest until no "
                "further move improves revenue.")
    b = st.selectbox("Brand", sorted(camp_all["brand"].unique()))
    curves = get_curves(camp_all, b)
    margin = float(camp_all.loc[camp_all["brand"] == b, "gross_margin"].iloc[0])
    cur_budget = float(curves["avg_weekly_spend"].sum())
    budget = st.slider("Weekly ad budget (AED)", int(cur_budget * 0.5 // 500 * 500), int(cur_budget * 1.5 // 500 * 500),
                       int(round(cur_budget / 500) * 500), step=500)
    o = E.optimise_budget(curves, budget, margin)
    cur_rev, new_rev = o["current_revenue"].sum(), o["optimised_revenue"].sum()
    cur_c, new_c = o["current_contribution"].sum(), o["optimised_contribution"].sum()
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Weekly budget", aed(budget), f"current average {aed(cur_budget)}")
    kpi(c2, "Revenue, current split", aed(cur_rev), f"ROAS {cur_rev / budget:.2f}x")
    kpi(c3, "Revenue, AdPulse split", aed(new_rev), f"{new_rev / cur_rev - 1:+.1%} · ROAS {new_rev / budget:.2f}x")
    kpi(c4, "Extra gross profit", aed((new_c - cur_c) * 52), "per year at this budget")
    st.write("")
    l, r = st.columns([1, 1.15])
    with l:
        oo = o.sort_values("current_spend")
        fig = go.Figure()
        fig.add_bar(y=oo["channel"], x=oo["current_spend"], orientation="h", name="Current split", marker_color=MUTED,
                    hovertemplate="%{y}: AED %{x:,.0f}<extra>current</extra>")
        fig.add_bar(y=oo["channel"], x=oo["optimised_spend"], orientation="h", name="AdPulse recommended",
                    marker_color=BLUE, hovertemplate="%{y}: AED %{x:,.0f}<extra>recommended</extra>")
        fig.update_layout(barmode="group", title="Weekly spend by channel", xaxis_title="AED per week")
        show(style_fig(fig, 400))
    with r:
        fig = go.Figure()
        for _, row in o.iterrows():
            xs = np.linspace(0, row["max_spend"] * 1.4, 100)
            col = CHANNEL_COLOR.get(row["channel"], VIOLET)
            fig.add_scatter(x=xs, y=E.curve_revenue(row["k"], row["b"], xs), name=row["channel"],
                            line=dict(color=col, width=2),
                            hovertemplate=row["channel"] + "<br>Spend AED %{x:,.0f} → revenue AED %{y:,.0f}<extra></extra>")
            fig.add_scatter(x=[row["optimised_spend"]], y=[row["optimised_revenue"]], mode="markers", showlegend=False,
                            marker=dict(color=col, size=10, line=dict(color="#fff", width=2)),
                            hovertemplate=row["channel"] + " recommended<br>AED %{x:,.0f}<extra></extra>")
        fig.update_layout(title="Response curves (dots = recommended spend)", xaxis_title="Weekly spend (AED)",
                          yaxis_title="Predicted weekly revenue (AED)")
        show(style_fig(fig, 400))
    t = o[["channel", "current_spend", "optimised_spend", "change_aed", "b", "r2", "marginal_roas_current"]].copy()
    t = t.sort_values("change_aed", ascending=False)
    t.columns = ["Channel", "Current", "Recommended", "Change", "Elasticity b", "Curve R²", "Marginal ROAS now"]
    for c in ["Current", "Recommended"]:
        t[c] = t[c].map(aed)
    t["Change"] = o.loc[t.index, "change_aed"].map(lambda v: f"{'+' if v >= 0 else '-'}{aed(abs(v))[4:]}")
    t["Elasticity b"] = t["Elasticity b"].map(lambda v: f"{v:.2f}")
    t["Curve R²"] = t["Curve R²"].map(lambda v: f"{v:.2f}")
    t["Marginal ROAS now"] = t["Marginal ROAS now"].map(lambda v: f"{v:.2f}")
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Guard-rails: each channel stays between 0.3x and 2x its historical average so the optimiser never "
               "extrapolates far beyond what the data has seen. Elasticity is constrained to 0.30-0.90. "
               "Recommendations should be rolled out gradually and checked with a hold-out test.")

# ===========================================================================
# PAGE: Customer Value
# ===========================================================================
elif page == "Customer Value":
    st.title("Customer Value")
    st.markdown("A cheap customer is not a good customer if they never come back. AdPulse groups customers by "
                "buying behaviour (K-means clustering) and compares the **12-month gross profit** each channel's "
                "customers generate with what it **cost to acquire** them.")
    k_choice = st.slider("Number of segments (k)", 3, 6, 4)
    seg, prof, sil = get_segments(cust_all, k_choice)
    cols = st.columns(len(prof))
    for c, (_, r) in zip(cols, prof.sort_values("profit_share", ascending=False).iterrows()):
        kpi(c, r["segment"], f"{r['share']:.0%}",
            f"of customers · {r['profit_share']:.0%} of profit<br>{r['orders_12m']:.1f} orders · "
            f"AOV AED {r['avg_order_value_aed']:.0f}<br>{r['discount_order_share']:.0%} orders on discount")
    st.write("")
    l, r = st.columns(2)
    with l:
        mix = E.segment_mix_by_channel(seg)
        order = prof.sort_values("profit_share", ascending=False)["segment"].tolist()
        fig = go.Figure()
        palette = [BLUE, AQUA, ORANGE, MUTED, VIOLET, MAGENTA]
        for i, s in enumerate(order):
            if s in mix.columns:
                fig.add_bar(y=mix.index, x=mix[s] * 100, name=s, orientation="h",
                            marker=dict(color=SEGMENT_COLOR.get(s, palette[i % len(palette)]),
                                        line=dict(color="#fff", width=2)),
                            hovertemplate="%{y}: %{x:.0f}% " + s + "<extra></extra>")
        fig.update_layout(barmode="stack", title="Which customers each channel brings", xaxis_title="% of customers")
        show(style_fig(fig, 420))
    with r:
        lc = E.ltv_cac(camp_all, cust_all).sort_values("ltv_to_cac")
        fig = go.Figure(go.Bar(y=lc["channel"], x=lc["ltv_to_cac"], orientation="h", marker_color=BLUE,
                               text=[f"{v:.2f}x" for v in lc["ltv_to_cac"]], textposition="outside",
                               customdata=np.stack([lc["cac_aed"], lc["ltv_12m_gross_profit_aed"]], axis=1),
                               hovertemplate="%{y}<br>CAC AED %{customdata[0]:.0f} · 12m profit AED %{customdata[1]:.0f}"
                                             "<br>LTV:CAC %{x:.2f}x<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=CRIT, dash="dash", width=1.5), annotation_text="break-even 1.0x",
                      annotation_position="bottom right", annotation_font_color=CRIT)
        fig.update_layout(title="LTV : CAC by acquisition channel (12 months)", xaxis_title="Ratio",
                          xaxis_range=[0, lc["ltv_to_cac"].max() * 1.2])
        show(style_fig(fig, 420, legend=False))
    st.info("**Reading this:** a TikTok customer is cheap to see but expensive to acquire relative to what they are "
            "worth. Two-thirds are discount-driven and few come back. Google Search customers cost the least and "
            "repeat the most. Channels below 1.0x do not pay back their acquisition cost within 12 months.")
    with st.expander("Model selection: silhouette score by k"):
        fig = go.Figure(go.Scatter(x=sil["k"], y=sil["silhouette"], mode="lines+markers", line=dict(color=BLUE, width=2),
                                   marker=dict(size=9)))
        fig.update_layout(xaxis_title="k (number of clusters)", yaxis_title="Silhouette score")
        show(style_fig(fig, 300, legend=False))
        st.caption("Higher is better. The score peaks around k = 4-5; k = 4 is used because the segments are easier "
                   "to act on. Scores of about 0.25 mean the segments overlap; they are useful groupings rather than "
                   "sharply separated types.")

# ===========================================================================
# PAGE: Scenario Lab
# ===========================================================================
elif page == "Scenario Lab":
    st.title("Scenario Lab")
    st.markdown("What happens to weekly revenue and profit across the portfolio if conditions change? Move the "
                "sliders. Projections come from the fitted response curves.")
    c1, c2, c3, c4 = st.columns(4)
    bc = c1.slider("Change in total ad budget", -40, 40, 0, step=5, format="%d%%") / 100
    ci = c2.slider("Change in ad costs (CPM / CPC)", -20, 40, 0, step=5, format="%d%%") / 100
    dc = c3.slider("Change in consumer demand", -30, 30, 0, step=5, format="%d%%") / 100
    op = c4.toggle("Use AdPulse optimised split", value=True)
    base = get_scenario(camp_all, 0.0, 0.0, 0.0, False)
    sc = get_scenario(camp_all, bc, ci, dc, op)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Weekly spend", aed(sc["weekly_spend"]), f"base {aed(base['weekly_spend'])}")
    kpi(k2, "Weekly revenue", aed(sc["weekly_revenue"]), f"{sc['weekly_revenue'] / base['weekly_revenue'] - 1:+.1%} vs base")
    kpi(k3, "Weekly gross profit after ads", aed(sc["weekly_contribution"]),
        f"{sc['weekly_contribution'] - base['weekly_contribution']:+,.0f} AED vs base")
    kpi(k4, "Blended ROAS", f"{sc['roas']:.2f}x", f"base {base['roas']:.2f}x")
    st.write("")
    grid = []
    for b_ in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
        cur = get_scenario(camp_all, b_, ci, dc, False)
        opt_ = get_scenario(camp_all, b_, ci, dc, True)
        grid.append({"budget": f"{b_:+.0%}", "Current split": cur["weekly_contribution"],
                     "AdPulse split": opt_["weekly_contribution"]})
    g = pd.DataFrame(grid)
    fig = go.Figure()
    fig.add_scatter(x=g["budget"], y=g["Current split"], name="Current split", line=dict(color=MUTED, width=2),
                    mode="lines+markers", hovertemplate="Budget %{x}: AED %{y:,.0f}/week<extra>current</extra>")
    fig.add_scatter(x=g["budget"], y=g["AdPulse split"], name="AdPulse split", line=dict(color=BLUE, width=2),
                    mode="lines+markers", hovertemplate="Budget %{x}: AED %{y:,.0f}/week<extra>optimised</extra>")
    fig.update_layout(title="Weekly gross profit after ad spend, by budget level (with your cost & demand settings)",
                      xaxis_title="Change in total ad budget", yaxis_title="AED per week")
    show(style_fig(fig, 400))
    best = g.loc[g["AdPulse split"].idxmax(), "budget"]
    st.info(f"**Insight:** with these settings, profit peaks at a budget change of **{best}** when the money is "
            "allocated well. Spending more on the current split mostly buys expensive, low-return impressions.")

# ===========================================================================
# PAGE: Business Case
# ===========================================================================
elif page == "Business Case":
    st.title("Business Case: AdPulse as a SaaS business")
    st.markdown("How AdPulse makes money, and what it takes to break even. All cost figures are **illustrative "
                "planning assumptions** that you can change below.")
    tiers = E.PRICING_TIERS
    cols = st.columns(3)
    for c, (_, t) in zip(cols, tiers.iterrows()):
        c.markdown(f"""<div class="card"><span class="pill info">{t['tier'].upper()}</span>
            <div class="kpi" style="border:none;padding:6px 0 0 0"><div class="val">AED {t['price_aed_month']:,}</div>
            <div class="sub">per month · {t['for']}</div></div></div>""", unsafe_allow_html=True)
    _, opt = get_optimisation(camp_all)
    v = opt["contribution_uplift_aed_year"].mean()
    st.caption(f"Value check: in the demo data the optimiser alone is worth about {aed(v)} a year in extra gross profit "
               f"per brand. The Growth tier costs {aed(1490 * 12)} a year, about {1490 * 12 / v:.0%} of that value.")
    st.divider()
    c1, c2, c3 = st.columns(3)
    fixed = c1.number_input("Fixed costs (AED / month)", 30000, 300000, 95000, step=5000)
    var = c2.number_input("Variable cost per subscriber (AED / month)", 0, 800, 140, step=10)
    cac = c3.number_input("Cost to acquire a subscriber (AED)", 500, 20000, 6000, step=500)
    c4, c5, c6 = st.columns(3)
    churn = c4.slider("Monthly churn", 1.0, 10.0, 4.0, step=0.5, format="%.1f%%") / 100
    start = c5.slider("New subscribers in month 1", 1, 15, 4)
    growth = c6.slider("Monthly growth in new subscribers", 0, 15, 8, format="%d%%") / 100
    mix_s = st.slider("Tier mix: % Starter / % Growth (rest Agency)", 0, 100, (50, 90))
    mix = [mix_s[0], mix_s[1] - mix_s[0], 100 - mix_s[1]]
    t2 = tiers.copy(); t2["mix"] = mix
    arpu = E.blended_arpu(t2) if sum(mix) else 0
    be = E.breakeven(fixed, arpu, var)
    ue = E.unit_economics(arpu, var, cac, churn)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Blended price (ARPU)", f"AED {arpu:,.0f}", f"mix {mix[0]}/{mix[1]}/{mix[2]}")
    kpi(k2, "Break-even subscribers", f"{np.ceil(be):,.0f}" if np.isfinite(be) else "never",
        "to cover fixed + variable costs")
    kpi(k3, "LTV : CAC", f"{ue['ltv_to_cac']:.1f}x", f"LTV AED {ue['ltv']:,.0f}")
    kpi(k4, "CAC payback", f"{ue['payback_months']:.1f} months", f"gross margin {ue['gross_margin_pct']:.0%}")
    proj = E.subscriber_projection(36, start, growth, churn, arpu, var, fixed, cac)
    pos = proj[proj["operating_profit_aed"] > 0]
    st.write("")
    fig = go.Figure()
    fig.add_scatter(x=proj["month"], y=proj["revenue_aed"], name="Monthly revenue", line=dict(color=BLUE, width=2),
                    hovertemplate="Month %{x}: AED %{y:,.0f}<extra>revenue</extra>")
    fig.add_scatter(x=proj["month"], y=proj["total_cost_aed"], name="Monthly total cost (incl. acquisition)",
                    line=dict(color=ORANGE, width=2), hovertemplate="Month %{x}: AED %{y:,.0f}<extra>cost</extra>")
    if len(pos):
        fig.add_vline(x=int(pos["month"].iloc[0]), line=dict(color=GOOD, dash="dash"),
                      annotation_text=f"operating break-even, month {int(pos['month'].iloc[0])}")
    fig.update_layout(title="36-month projection", xaxis_title="Month after launch", yaxis_title="AED per month",
                      hovermode="x unified")
    show(style_fig(fig, 400))
    st.caption(f"Subscribers at month 36: {proj['subscribers'].iloc[-1]:,.0f}. Cumulative profit at month 36: "
               f"{aed(proj['cumulative_profit_aed'].iloc[-1])}. Acquisition spend is counted as a monthly cost, "
               "so operating break-even comes later than the steady-state subscriber break-even.")

# ===========================================================================
# PAGE: Data & Method
# ===========================================================================
elif page == "Data & Method":
    st.title("Data & Method")
    st.warning("**The demo data is synthetic.** It was generated by `03_Code/generate_data.py` from stated assumptions "
               "about six fictional UAE brands so that every feature can be demonstrated. It is not real market data. "
               "Upload your own exports below to run AdPulse on real numbers.")
    tab1, tab2, tab3 = st.tabs(["Upload your data", "Preview", "Methodology"])
    with tab1:
        st.markdown("**Campaign file**: one row per campaign per week. Required columns:")
        st.code(", ".join(E.CAMPAIGN_REQUIRED))
        up = st.file_uploader("Upload campaign CSV", type="csv", key="u1")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                miss = E.missing_columns(raw, E.CAMPAIGN_REQUIRED)
                if miss:
                    st.error(f"Missing columns: {', '.join(miss)}")
                else:
                    st.session_state["upload_campaigns"] = E.prepare_campaigns(raw)
                    st.success(f"Loaded {len(raw):,} rows. Every page now uses your campaign data.")
            except Exception as ex:  # noqa: BLE001
                st.error(f"Could not read file: {ex}")
        st.markdown("**Customer file**: one row per customer. Required columns:")
        st.code(", ".join(E.CUSTOMER_REQUIRED))
        up2 = st.file_uploader("Upload customer CSV", type="csv", key="u2")
        if up2 is not None:
            try:
                raw2 = pd.read_csv(up2)
                miss = E.missing_columns(raw2, E.CUSTOMER_REQUIRED)
                if miss:
                    st.error(f"Missing columns: {', '.join(miss)}")
                else:
                    st.session_state["upload_customers"] = raw2
                    st.success(f"Loaded {len(raw2):,} customers.")
            except Exception as ex:  # noqa: BLE001
                st.error(f"Could not read file: {ex}")
        if st.button("Reset to demo data"):
            st.session_state["upload_campaigns"] = None
            st.session_state["upload_customers"] = None
            st.rerun()
        demo_c, demo_u = demo_data()
        c1, c2 = st.columns(2)
        c1.download_button("Download campaign template (demo data)", demo_c.to_csv(index=False).encode(),
                           "campaign_performance.csv", "text/csv")
        c2.download_button("Download customer template (demo data)", demo_u.to_csv(index=False).encode(),
                           "customers.csv", "text/csv")
    with tab2:
        st.markdown(f"**Campaign data**: {len(camp_all):,} rows")
        st.dataframe(camp_all.head(200), width="stretch", hide_index=True)
        st.markdown(f"**Customer data**: {len(cust_all):,} rows")
        st.dataframe(cust_all.head(200), width="stretch", hide_index=True)
    with tab3:
        st.markdown("""
| Module | Question it answers | Method |
|---|---|---|
| Truth Check | How much revenue did each channel *really* drive? | Platform-reported vs store-verified revenue (UTM / promo-code / order-ID matching) |
| Profit Predictor | Will this campaign-week make money? | Random Forest classifier (400 trees), logistic regression baseline, time-based hold-out (Oct-Dec), 5-fold CV |
| Budget Optimiser | How should a fixed budget be split? | Log-log response curves per channel with season controls; greedy marginal-return allocation with guard-rails |
| Customer Value | Which channels bring customers worth keeping? | K-means (k chosen by silhouette) on orders, AOV, recency, discount use; LTV:CAC by channel |
| Scenario Lab | What if budgets, costs or demand change? | Response-curve simulation across all brands |
| Business Case | Can AdPulse itself be a profitable business? | Tiered SaaS pricing, break-even, LTV:CAC, 36-month projection |

**Profit definition.** Contribution = verified revenue × gross margin − ad spend. A campaign-week is
"profitable" when contribution > 0, i.e. real ROAS > 1 / gross margin (the break-even ROAS).

**Main limitations.** Synthetic data; first-order profit only in the classifier (repeat purchases are handled
in Customer Value); response curves are correlational, not causal, so they should be validated with geo or
hold-out tests before large budget moves.
""")
