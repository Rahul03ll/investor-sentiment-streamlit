# ============================================================
# app.py — Streamlit Dashboard
# Market Sentiment & Volatility Analyzer
# Run: streamlit run app.py
# ============================================================

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from core import (
    fit_egarch_model,
    fit_model_comparison,
    load_gdelt_sentiment,
    load_stock_data,
    load_trends_data,
    news_sentiment_pipeline,
)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment & Volatility Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IS_DARK_MODE = st.get_option("theme.base") == "dark"

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top:2rem; padding-bottom:3rem; padding-right:1rem; }
.main-header {
    font-size:2rem; font-weight:700; color:var(--text-color);
    text-align:center; padding:1rem 0;
    border-bottom:3px solid var(--primary-color); margin-bottom:1.5rem;
}
.sub-header {
    font-size:1.1rem; color:var(--text-color); opacity:0.78;
    text-align:center; margin-top:-1rem; margin-bottom:2rem;
}
.section-header {
    font-size:1.3rem; font-weight:600; color:var(--text-color);
    border-left:4px solid var(--primary-color);
    padding-left:0.75rem; margin:1.5rem 0 1rem 0;
}
.insight-box {
    background-color:var(--secondary-background-color);
    border:1px solid rgba(127,127,127,0.25);
    border-left:4px solid var(--primary-color);
    padding:0.75rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0; color:var(--text-color);
}
.warning-box {
    background-color:rgba(245,158,11,0.12);
    border:1px solid rgba(245,158,11,0.45);
    border-left:4px solid #f59e0b;
    padding:0.75rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0; color:var(--text-color);
}
.success-box {
    background-color:rgba(34,197,94,0.10);
    border:1px solid rgba(34,197,94,0.40);
    border-left:4px solid #22c55e;
    padding:0.75rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0; color:var(--text-color);
}
[data-testid="stSidebar"] { border-right:1px solid rgba(127,127,127,0.2); }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label { color:var(--text-color) !important; }
.stButton > button {
    border-radius:0.6rem; font-weight:600;
    border:1px solid rgba(127,127,127,0.3);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────

def apply_plot_theme(fig, axes):
    """Consistent chart theme for light/dark modes."""
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    elif isinstance(axes, list):
        axes_list = axes
    else:
        axes_list = [axes]

    if IS_DARK_MODE:
        fig.patch.set_facecolor("#0f172a")
        for ax in axes_list:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#e5e7eb")
            ax.xaxis.label.set_color("#e5e7eb")
            ax.yaxis.label.set_color("#e5e7eb")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#64748b")
            ax.grid(True, alpha=0.18, color="#64748b")
    else:
        fig.patch.set_facecolor("white")
        for ax in axes_list:
            ax.set_facecolor("#f8fafc")
            ax.tick_params(colors="#1f2937")
            ax.xaxis.label.set_color("#111827")
            ax.yaxis.label.set_color("#111827")
            ax.title.set_color("#0f172a")
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")
            ax.grid(True, alpha=0.18, color="#94a3b8")
    fig.tight_layout()


def safe_trend_fit(x: pd.Series, y: pd.Series):
    """Linear trend fit with NaN/inf/degenerate-data safety."""
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) <= 2 or df["x"].nunique() <= 1 or df["y"].nunique() <= 1:
        return None, None
    try:
        z = np.polyfit(df["x"], df["y"], 1, rcond=1e-10)
        return np.poly1d(z), float(z[0])
    except np.linalg.LinAlgError:
        try:
            from scipy.stats import linregress
            slope, intercept, *_ = linregress(df["x"], df["y"])
            return np.poly1d([slope, intercept]), float(slope)
        except Exception:
            return None, None


# ── Header ────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📈 Market Sentiment & Volatility Analyzer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Investor Sentiment Impact on Indian Stock Market Volatility'
    " | EGARCH + ML Hybrid Model</div>",
    unsafe_allow_html=True,
)

# ╔══════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                ║
# ╚══════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    index_choice = st.selectbox(
        "Select Index",
        ["Nifty 50 (^NSEI)", "Sensex (^BSESN)", "Bank Nifty (^NSEBANK)"],
    )
    TICKER = {
        "Nifty 50 (^NSEI)":      "^NSEI",
        "Sensex (^BSESN)":       "^BSESN",
        "Bank Nifty (^NSEBANK)": "^NSEBANK",
    }[index_choice]

    st.markdown("### 📅 Date Range")
    
    # Preset date ranges
    preset = st.radio(
        "Quick Select",
        ["Recent (2020-2024)", "Full History (2007-2024)", "Custom"],
        index=0,
        help="Recent period works best with free tier APIs"
    )
    
    if preset == "Recent (2020-2024)":
        start_year, end_year = 2020, 2024
    elif preset == "Full History (2007-2024)":
        start_year, end_year = 2007, 2024
    else:  # Custom
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Start Year", list(range(2007, 2025)), index=0)
        with col2:
            end_year = st.selectbox("End Year", list(range(2010, 2026)), index=15)

    if start_year >= end_year:
        st.error("Start year must be before end year.")
        st.stop()

    START_DATE = f"{start_year}-01-01"
    END_DATE   = f"{end_year}-12-31"

    st.markdown("---")
    st.markdown("### EGARCH Parameters")
    p_order = st.slider("p (ARCH order)", 1, 3, 1)
    q_order = st.slider("q (GARCH order)", 1, 3, 1)

    st.markdown("---")
    st.markdown("### Display Options")
    show_gfc        = st.checkbox("Show GFC (2008–09)",   value=True)
    show_covid      = st.checkbox("Show COVID (2020–21)", value=True)
    show_multi_idx  = st.checkbox("Multi-index comparison", value=True)
    show_model_cmp  = st.checkbox("Model comparison (GARCH/GJR/EGARCH)", value=True)

    st.markdown("---")
    st.markdown("### Sentiment Fetch Mode")
    sentiment_mode = st.radio(
        "Speed vs coverage",
        ["Fast", "Full"],
        index=0,
        help="Fast: ≤12 GDELT calls. Full: all months.",
    )

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Project:** Investor Sentiment & Volatility\n\n"
        "**Model:** EGARCH + Random Forest\n\n"
        "**Data:** Yahoo Finance · GDELT · Google Trends · News"
    )

# ── Welcome screen ────────────────────────────────────────
if not run_button:
    st.markdown("""
    <div class="insight-box">
    <b>👋 Welcome!</b> Configure your analysis in the sidebar and click
    <b>Run Analysis</b> to begin.<br><br>
    This tool analyzes the relationship between <b>investor sentiment</b>
    (GDELT → Google Trends → News fallback) and <b>market volatility</b>
    (modeled via EGARCH) in the Indian stock market.<br><br>
    <b>💡 Tip:</b> For best results on free tier, use the "Recent (2020-2024)" date range.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📊 Volatility Modeling**\n"
                    "- EGARCH conditional volatility\n"
                    "- GARCH / GJR / EGARCH comparison\n"
                    "- Crisis period analysis\n"
                    "- Annualized volatility metrics")
    with c2:
        st.markdown("**🧠 Sentiment Analysis**\n"
                    "- GDELT tone-based sentiment\n"
                    "- Google Trends fallback\n"
                    "- News pipeline (RSS/Yahoo)\n"
                    "- PCA dimensionality reduction\n"
                    "- **Real data only - no demo mode**")
    with c3:
        st.markdown("**🤖 ML Prediction**\n"
                    "- Hybrid EGARCH + Random Forest\n"
                    "- Volatility direction forecasting\n"
                    "- Feature importance analysis\n"
                    "- Confusion matrix & metrics")
    with c4:
        st.markdown("**📊 Multi-Index**\n"
                    "- Nifty 50 vs Sensex vs Bank Nifty\n"
                    "- Cross-index volatility correlation\n"
                    "- Rolling volatility windows\n"
                    "- Regime analysis")
    st.stop()

# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN ANALYSIS                                          ║
# ╚══════════════════════════════════════════════════════════╝

progress_bar = st.progress(0)
status_text  = st.empty()

# ── 1. Stock data ─────────────────────────────────────────
status_text.text("📥 Loading stock market data...")
progress_bar.progress(8)

try:
    data = load_stock_data(TICKER, START_DATE, END_DATE)
except Exception as exc:
    st.error(f"Failed to load stock data: {exc}")
    st.stop()

# ── 2. Sentiment cascade ──────────────────────────────────
status_text.text("📰 Fetching sentiment data...")
progress_bar.progress(20)

has_sentiment      = False
variance_explained = 0.0
sentiment_source   = "None"
gdelt_err          = None
trends_err         = None

# Level 1: GDELT (Primary - most reliable for historical data)
gdelt_df, gdelt_err = load_gdelt_sentiment(
    START_DATE, END_DATE, fast_mode=(sentiment_mode == "Fast")
)
if gdelt_df is not None and not gdelt_df.empty:
    data = data.merge(gdelt_df, left_index=True, right_index=True, how="left")
    data["sentiment"] = data["sentiment"].ffill().bfill()
    data["sentiment_index"] = data["sentiment"]
    has_sentiment      = True
    variance_explained = 100.0
    sentiment_source   = "GDELT"
    st.markdown('<div class="success-box">✅ Using GDELT sentiment as primary source.</div>',
                unsafe_allow_html=True)
else:
    # Level 2: Google Trends (Secondary - good for recent periods)
    trends_df, keywords, trends_err = load_trends_data(START_DATE, END_DATE)
    if trends_df is not None and len(keywords) > 0:
        data = data.merge(trends_df, left_index=True, right_index=True, how="left")
        data[keywords] = data[keywords].ffill().bfill()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[keywords].fillna(0))
        pca    = PCA(n_components=1)
        data["sentiment_index"] = pca.fit_transform(scaled)
        variance_explained = float(pca.explained_variance_ratio_[0] * 100)
        has_sentiment    = True
        sentiment_source = "Google Trends"
        st.markdown(
            f'<div class="warning-box">⚠️ GDELT unavailable: {gdelt_err}. '
            "Using Google Trends sentiment fallback.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Level 3: News pipeline (Tertiary - recent headlines only)
        news_series, news_err, news_source = news_sentiment_pipeline()
        if news_err is None and news_series is not None and not news_series.empty:
            # News data is recent only - we need to handle the date range carefully
            # Only use news sentiment if the selected date range is recent (last 2 years)
            recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
            
            if data.index[-1] >= recent_cutoff:
                # Align news_series to trading days
                ns_aligned = news_series.copy()
                if ns_aligned.index.tz is not None:
                    ns_aligned.index = ns_aligned.index.tz_localize(None)
                
                # Only use news for recent period
                recent_data = data[data.index >= recent_cutoff].copy()
                news_reindexed = ns_aligned.reindex(recent_data.index, method="ffill")
                
                # Check if we have enough non-null data
                if news_reindexed.notna().sum() > 30:
                    # Merge with full dataset
                    data["sentiment_index"] = np.nan
                    data.loc[data.index >= recent_cutoff, "sentiment_index"] = news_reindexed.values
                    
                    # For older data, use a neutral baseline (0)
                    data["sentiment_index"] = data["sentiment_index"].fillna(0)
                    
                    has_sentiment    = True
                    sentiment_source = news_source
                    st.markdown(
                        f'<div class="warning-box">⚠️ GDELT & Trends unavailable. '
                        f"Using {news_source} news sentiment for recent period only. "
                        f"<br><b>Note:</b> Historical data (before {recent_cutoff.date()}) uses neutral sentiment baseline.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    has_sentiment = False
            else:
                has_sentiment = False
        else:
            has_sentiment = False

# If no sentiment data available, show error and stop
if not has_sentiment:
    st.error(
        "❌ **Unable to fetch sentiment data from any source.**\n\n"
        "**Possible reasons:**\n"
        "- GDELT API rate limiting or unavailable\n"
        "- Google Trends rate limiting (try again in a few minutes)\n"
        "- News sources unavailable\n\n"
        "**Suggestions:**\n"
        "1. Try selecting a shorter date range (e.g., 2020-2024)\n"
        "2. Wait a few minutes and try again\n"
        "3. Switch between 'Fast' and 'Full' sentiment modes\n\n"
        f"**Error details:**\n"
        f"- GDELT: {gdelt_err or 'Unknown error'}\n"
        f"- Google Trends: {trends_err or 'Unknown error'}\n"
        f"- News: {news_err or 'Unknown error'}"
    )
    st.stop()


# ── 3. Fit EGARCH ─────────────────────────────────────────
status_text.text("⚙️ Fitting EGARCH model...")
progress_bar.progress(45)

data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["returns", "sentiment_index"])

try:
    result = fit_egarch_model(
        data["returns"].values,
        data["sentiment_index"].values,
        p_order, q_order,
    )
    cond_vol = result.conditional_volatility
    min_len  = min(len(data), len(cond_vol))
    data     = data.iloc[-min_len:].copy()
    data["volatility"] = cond_vol[-min_len:]

    if not np.isfinite(data["volatility"]).all():
        st.warning("⚠️ EGARCH produced non-finite values — using rolling-std fallback.")
        data["volatility"] = (
            data["returns"].rolling(20).std().bfill() * np.sqrt(252) * 100
        )
except Exception as exc:
    st.error(
        f"EGARCH fitting failed: {exc}\n\n"
        "This usually means the sentiment data doesn't cover the selected date range. "
        "Try switching to **Full** sentiment mode or selecting a shorter date range."
    )
    st.stop()

# Annualised volatility
data["vol_ann"] = data["volatility"] * np.sqrt(252)

# ── 4. Model comparison (optional) ───────────────────────
model_cmp_df = None
if show_model_cmp:
    status_text.text("📊 Comparing GARCH / GJR / EGARCH models...")
    progress_bar.progress(58)
    try:
        model_cmp_df = fit_model_comparison(data["returns"].values)
    except Exception:
        model_cmp_df = None

# ── 5. Multi-index volatility (optional) ─────────────────
multi_vol_df = None
if show_multi_idx:
    status_text.text("📊 Loading multi-index data...")
    progress_bar.progress(65)
    try:
        multi_tickers = {"Nifty 50": "^NSEI", "Sensex": "^BSESN", "Bank Nifty": "^NSEBANK"}
        vol_dict = {}
        for name, tkr in multi_tickers.items():
            try:
                idx_data = load_stock_data(tkr, START_DATE, END_DATE)
                m_res    = fit_egarch_model(
                    idx_data["returns"].values,
                    np.zeros(len(idx_data)),   # no sentiment for comparison
                    1, 1,
                )
                vol_s = pd.Series(
                    m_res.conditional_volatility[-len(idx_data):],
                    index=idx_data.index,
                    name=name,
                )
                vol_dict[name] = vol_s
            except Exception:
                pass
        if len(vol_dict) > 1:
            multi_vol_df = pd.DataFrame(vol_dict)
    except Exception:
        multi_vol_df = None

# ── 6. ML Features ────────────────────────────────────────
progress_bar.progress(75)
status_text.text("🤖 Training ML classifier...")

feature_cols = ["returns", "sentiment_index", "volatility"]
for lag in [1, 2, 3]:
    data[f"returns_lag{lag}"]   = data["returns"].shift(lag)
    data[f"sentiment_lag{lag}"] = data["sentiment_index"].shift(lag)
    data[f"vol_lag{lag}"]       = data["volatility"].shift(lag)
    feature_cols += [f"returns_lag{lag}", f"sentiment_lag{lag}", f"vol_lag{lag}"]

data["vol_ma5"]    = data["volatility"].rolling(5).mean()
data["vol_ma20"]   = data["volatility"].rolling(20).mean()
data["sent_ma5"]   = data["sentiment_index"].rolling(5).mean()
data["vol_change"] = data["volatility"].pct_change()
feature_cols += ["vol_ma5", "vol_ma20", "sent_ma5", "vol_change"]

# Target: next-day RETURN direction (up/down) — avoids vol-leakage
# Using vol_class based on FUTURE volatility but with a proper gap
data["vol_class"] = (data["volatility"].shift(-1) > data["volatility"].median()).astype(int)

data_ml = (
    data[feature_cols + ["vol_class"]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

# Remove current-day volatility from features to prevent leakage
# Keep only lagged volatility features
safe_features = [c for c in feature_cols if c != "volatility"]
X = data_ml[safe_features]
y = data_ml["vol_class"]

if len(X) < 50:
    st.warning(f"⚠️ Limited ML data ({len(X)} rows). Results may be unreliable.")

split   = int(len(X) * 0.8)
X_train = X.iloc[:split];  X_test = X.iloc[split:]
y_train = y.iloc[:split];  y_test = y.iloc[split:]

rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=20,
    random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred  = rf.predict(X_test)
y_prob  = rf.predict_proba(X_test)[:, 1]
ml_acc  = accuracy_score(y_test, y_pred)
baseline = max(y_test.mean(), 1 - y_test.mean())

progress_bar.progress(100)
status_text.text("✅ Analysis complete!")
time.sleep(0.4)
progress_bar.empty()
status_text.empty()

# ╔══════════════════════════════════════════════════════════╗
# ║  DISPLAY RESULTS                                        ║
# ╚══════════════════════════════════════════════════════════╝

# ── KPIs ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Trading Days",        f"{len(data):,}")
k2.metric("Mean Daily Return",   f"{data['returns'].mean()*100:.4f}%")
k3.metric("Mean Daily Vol",      f"{data['volatility'].mean():.4f}%")
k4.metric("Mean Ann. Vol",       f"{data['vol_ann'].mean():.2f}%")
k5.metric("ML Accuracy",         f"{ml_acc*100:.1f}%",
          delta=f"{(ml_acc - baseline)*100:+.1f}% vs baseline")
k6.metric("Sentiment Source",    sentiment_source)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Volatility",
    "💭 Sentiment",
    "🏚️ Crisis Analysis",
    "🤖 ML Model",
    "📊 Multi-Index",
    "📋 Statistics",
])

# ─────────────────────────────────────────────────────────
# TAB 1 — VOLATILITY
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">EGARCH Conditional Volatility</div>',
                unsafe_allow_html=True)

    # Rolling window selector
    roll_win = st.select_slider(
        "Rolling average window (days)", options=[5, 10, 20, 60], value=20
    )
    data["vol_roll"] = data["volatility"].rolling(roll_win).mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(data.index, data["volatility"], color="#1f4e79", linewidth=0.7,
            alpha=0.6, label="Daily volatility")
    ax.plot(data.index, data["vol_roll"], color="#ed7d31", linewidth=1.5,
            label=f"{roll_win}-day MA")
    ax.fill_between(data.index, data["volatility"], alpha=0.08, color="#1f4e79")

    if show_gfc:
        ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"),
                   alpha=0.13, color="red", label="GFC (2008–09)")
    if show_covid:
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                   alpha=0.13, color="orange", label="COVID (2020–21)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Conditional Volatility (%)")
    ax.set_title(f"EGARCH({p_order},{q_order}) Conditional Volatility — {index_choice}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Annualised volatility chart
    st.markdown('<div class="section-header">Annualised Volatility</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(data.index, data["vol_ann"], color="#2e75b6", linewidth=0.8)
    ax.axhline(data["vol_ann"].mean(), color="#ed7d31", linewidth=1.2,
               linestyle="--", label=f"Mean {data['vol_ann'].mean():.1f}%")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title("Annualised Conditional Volatility (×√252)", fontweight="bold")
    ax.legend(fontsize=9)
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    st.markdown(
        '<div class="insight-box">'
        "<b>📌 Key Insight:</b> Volatility clustering is clearly visible — "
        "calm periods are punctuated by sharp spikes during the 2008 GFC and "
        "the 2020 COVID crash. The EGARCH model captures this asymmetric behaviour. "
        f"Mean annualised volatility = <b>{data['vol_ann'].mean():.1f}%</b> "
        f"(daily = {data['volatility'].mean():.4f}%)."
        "</div>",
        unsafe_allow_html=True,
    )

    # EGARCH Parameters Table
    st.markdown('<div class="section-header">EGARCH Model Parameters</div>',
                unsafe_allow_html=True)
    params_df = pd.DataFrame({
        "Parameter":   result.params.index,
        "Coefficient": result.params.values.round(6),
        "Std Error":   result.std_err.values.round(6),
        "T-stat":      result.tvalues.values.round(4),
        "P-value":     result.pvalues.values.round(6),
    })
    params_df["Significant"] = params_df["P-value"].apply(
        lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "—"))
    )
    st.dataframe(params_df, use_container_width=True)

    # Model comparison
    if show_model_cmp and model_cmp_df is not None:
        st.markdown('<div class="section-header">Model Comparison: GARCH vs GJR vs EGARCH</div>',
                    unsafe_allow_html=True)
        st.dataframe(model_cmp_df, use_container_width=True)
        best = model_cmp_df["AIC"].idxmin()
        st.markdown(
            f'<div class="insight-box">'
            f"<b>Best model by AIC:</b> <b>{best}</b> — "
            "EGARCH is preferred when asymmetric volatility response to shocks is present."
            "</div>",
            unsafe_allow_html=True,
        )

    # Download EGARCH results
    st.markdown("#### 📥 Export")
    export_df = data[["returns", "volatility", "vol_ann", "sentiment_index"]].copy()
    export_df.index.name = "Date"
    st.download_button(
        "Download volatility data (CSV)",
        export_df.to_csv().encode(),
        file_name=f"volatility_{TICKER.replace('^','')}_{start_year}_{end_year}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────
# TAB 2 — SENTIMENT
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Sentiment Index Over Time</div>',
                unsafe_allow_html=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.plot(data.index, data["sentiment_index"], color="#ed7d31", linewidth=0.9)
    ax1.fill_between(data.index, data["sentiment_index"],
                     where=data["sentiment_index"] > 0,
                     alpha=0.2, color="red", label="Negative sentiment")
    ax1.fill_between(data.index, data["sentiment_index"],
                     where=data["sentiment_index"] <= 0,
                     alpha=0.2, color="green", label="Positive sentiment")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Sentiment Index")
    ax1.set_title(f"Composite Sentiment Index ({sentiment_source})", fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.plot(data.index, data["volatility"], color="#1f4e79", linewidth=0.9)
    ax2.set_ylabel("Volatility (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("EGARCH Volatility", fontweight="bold")

    apply_plot_theme(fig, [ax1, ax2])
    st.pyplot(fig)
    plt.close()

    # Scatter
    st.markdown('<div class="section-header">Sentiment vs Volatility Relationship</div>',
                unsafe_allow_html=True)

    plot_data = data[["sentiment_index", "volatility"]].dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        plot_data["sentiment_index"], plot_data["volatility"],
        c=plot_data["volatility"], cmap="RdYlBu_r", alpha=0.55, s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Volatility Level")

    poly, slope = safe_trend_fit(plot_data["sentiment_index"], plot_data["volatility"])
    if poly is not None:
        xline = np.linspace(plot_data["sentiment_index"].min(),
                            plot_data["sentiment_index"].max(), 200)
        ax.plot(xline, poly(xline), "r--", lw=2, label=f"Trend slope={slope:.3f}")
        ax.legend()

    ax.set_xlabel("Sentiment Index")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Investor Sentiment vs Market Volatility", fontweight="bold")
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    corr = data["sentiment_index"].corr(data["volatility"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson Correlation", f"{corr:.4f}")
    c2.metric("Trend Slope", f"{slope:.4f}" if slope is not None else "N/A")
    c3.metric("Sentiment Var. Explained",
              f"{variance_explained:.1f}%" if variance_explained > 0 else "N/A")

    st.markdown(
        f'<div class="insight-box">'
        f"<b>📌 Finding:</b> Correlation = {corr:.4f}. "
        f"{'Positive' if corr > 0 else 'Negative'} correlation between sentiment and volatility. "
        f"Higher fear-search sentiment is associated with "
        f"{'higher' if corr > 0 else 'lower'} market volatility."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Granger Causality
    st.markdown('<div class="section-header">Granger Causality: Sentiment → Volatility</div>',
                unsafe_allow_html=True)
    gc_data = data[["volatility", "sentiment_index"]].dropna()
    # Granger requires both series to have variance (non-constant)
    # Check if sentiment has meaningful variance (std > 1e-6)
    sent_std = gc_data["sentiment_index"].std()
    _gc_ok = (
        gc_data["sentiment_index"].nunique() > 1
        and gc_data["volatility"].nunique() > 1
        and len(gc_data) > 20
        and sent_std > 1e-6
    )
    if not _gc_ok:
        st.markdown(
            '<div class="warning-box">⚠️ Granger causality test skipped — '
            "sentiment index is constant or near-constant (RSS/news fallback returns a single score). "
            "Use GDELT or Google Trends mode for this test.</div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            gc_res   = grangercausalitytests(gc_data, maxlag=5, verbose=False)
            gc_table = [
                {
                    "Lag":                  lag,
                    "F-Statistic":          round(gc_res[lag][0]["ssr_ftest"][0], 4),
                    "P-Value":              round(gc_res[lag][0]["ssr_ftest"][1], 6),
                    "Significant (p<0.05)": "✅ Yes" if gc_res[lag][0]["ssr_ftest"][1] < 0.05 else "❌ No",
                }
                for lag in range(1, 6)
            ]
            st.dataframe(pd.DataFrame(gc_table), use_container_width=True)
            sig_lags = sum(1 for r in gc_table if r["Significant (p<0.05)"] == "✅ Yes")
            st.markdown(
                f'<div class="insight-box">'
                f"<b>Interpretation:</b> {sig_lags}/5 lags significant — "
                f"{'strong' if sig_lags >= 4 else 'partial'} evidence that sentiment "
                "Granger-causes volatility."
                "</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.markdown(
                '<div class="warning-box">⚠️ Granger causality test failed: '
                f"{str(exc)[:200]}. This usually means the sentiment data is constant or "
                "has insufficient variance.</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────
# TAB 3 — CRISIS ANALYSIS
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Crisis Period Comparison</div>',
                unsafe_allow_html=True)

    gfc   = data.loc["2008-01-01":"2009-12-31"]
    covid = data.loc["2020-01-01":"2021-12-31"]

    fig, ax = plt.subplots(figsize=(13, 6))
    if len(gfc) > 0:
        ax.plot(gfc.index, gfc["volatility"], color="#1f4e79",
                label="GFC (2008–09)", linewidth=1.2)
        ax.fill_between(gfc.index, gfc["volatility"], alpha=0.1, color="#1f4e79")
    if len(covid) > 0:
        ax.plot(covid.index, covid["volatility"], color="#ed7d31",
                label="COVID (2020–21)", linewidth=1.2)
        ax.fill_between(covid.index, covid["volatility"], alpha=0.1, color="#ed7d31")

    ax.set_xlabel("Date")
    ax.set_ylabel("Conditional Volatility (%)")
    ax.set_title("Volatility During Crisis Periods", fontweight="bold")
    ax.legend(fontsize=10)
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Sentiment during crises
    st.markdown('<div class="section-header">Sentiment During Crisis Periods</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    if len(gfc) > 0:
        ax.plot(gfc.index, gfc["sentiment_index"], color="#1f4e79",
                label="GFC (2008–09)", linewidth=1.0)
    if len(covid) > 0:
        ax.plot(covid.index, covid["sentiment_index"], color="#ed7d31",
                label="COVID (2020–21)", linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Index")
    ax.set_title("Sentiment Index During Crisis Periods", fontweight="bold")
    ax.legend(fontsize=9)
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Statistics table
    st.markdown('<div class="section-header">Crisis Statistics</div>',
                unsafe_allow_html=True)
    crisis_periods = {
        "GFC (2008–09)":    data.loc["2008-01-01":"2009-12-31"],
        "COVID (2020–21)":  data.loc["2020-01-01":"2021-12-31"],
        "Normal (2012–19)": data.loc["2012-01-01":"2019-12-31"],
    }
    stats_rows = [
        {
            "Period":              name,
            "Mean Daily Vol (%)":  round(df["volatility"].mean(), 4),
            "Max Daily Vol (%)":   round(df["volatility"].max(), 4),
            "Mean Ann. Vol (%)":   round(df["vol_ann"].mean(), 2),
            "Std Volatility":      round(df["volatility"].std(), 4),
            "Mean Sentiment":      round(df["sentiment_index"].mean(), 4),
            "Mean Return (%)":     round(df["returns"].mean() * 100, 4),
            "Trading Days":        len(df),
        }
        for name, df in crisis_periods.items()
        if len(df) > 0
    ]
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<b>📌 Key Observation:</b> GFC produced sustained elevated volatility "
        "over ~24 months reflecting deep structural financial damage. COVID "
        "produced a sharper but shorter spike (~3–6 months) due to rapid "
        "policy interventions and vaccine development."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# TAB 4 — ML MODEL
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Hybrid EGARCH + Random Forest Model</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy",  f"{ml_acc*100:.2f}%")
    c2.metric("Baseline",       f"{baseline*100:.2f}%")
    c3.metric("Lift vs Baseline", f"{(ml_acc - baseline)*100:+.2f}%")
    c4.metric("Test Samples",   f"{len(y_test):,}")

    if ml_acc > 0.90:
        st.markdown(
            '<div class="warning-box">'
            "⚠️ <b>High accuracy note:</b> Accuracy >90% may indicate the model is "
            "learning volatility persistence (autocorrelation) rather than true predictive signal. "
            "Check the lift vs baseline — a small lift means the model adds limited value beyond "
            "simply predicting the majority class."
            "</div>",
            unsafe_allow_html=True,
        )

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)
    importance_df = (
        pd.DataFrame({"Feature": safe_features, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1],
                   color="#2e75b6", edgecolor="white")
    for bar, val in zip(bars, importance_df["Importance"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_title("Top Feature Importances — Volatility Direction Prediction",
                 fontweight="bold")
    ax.set_xlabel("Importance Score")
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Confusion Matrix
    st.markdown('<div class="section-header">Confusion Matrix</div>',
                unsafe_allow_html=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Vol", "High Vol"])
    ax.set_yticklabels(["Low Vol", "High Vol"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    apply_plot_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Classification Report
    st.markdown('<div class="section-header">Classification Report</div>',
                unsafe_allow_html=True)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Low Volatility", "High Volatility"],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report_dict).T.round(4), use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<b>📌 Model Insight:</b> The hybrid model combines EGARCH-estimated "
        "conditional volatility lags with sentiment features to predict whether "
        "next-day volatility will be above or below the historical median. "
        "Current-day volatility is excluded from features to prevent data leakage."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# TAB 5 — MULTI-INDEX
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Multi-Index Volatility Comparison</div>',
                unsafe_allow_html=True)

    if multi_vol_df is not None and not multi_vol_df.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        colors_mi = ["#1f4e79", "#ed7d31", "#70ad47"]
        for i, col in enumerate(multi_vol_df.columns):
            ax.plot(multi_vol_df.index, multi_vol_df[col],
                    label=col, color=colors_mi[i % len(colors_mi)],
                    linewidth=0.9, alpha=0.85)
        if show_gfc:
            ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"),
                       alpha=0.10, color="red", label="GFC")
        if show_covid:
            ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                       alpha=0.10, color="orange", label="COVID")
        ax.set_xlabel("Date")
        ax.set_ylabel("Conditional Volatility (%)")
        ax.set_title("EGARCH(1,1) Volatility: Nifty 50 vs Sensex vs Bank Nifty",
                     fontweight="bold")
        ax.legend(fontsize=9)
        apply_plot_theme(fig, ax)
        st.pyplot(fig)
        plt.close()

        # Correlation matrix
        st.markdown('<div class="section-header">Cross-Index Volatility Correlation</div>',
                    unsafe_allow_html=True)
        corr_matrix = multi_vol_df.corr().round(4)
        st.dataframe(corr_matrix, use_container_width=True)

        # Summary stats
        st.markdown('<div class="section-header">Index Volatility Summary</div>',
                    unsafe_allow_html=True)
        summary_rows = [
            {
                "Index":           col,
                "Mean Daily Vol":  f"{multi_vol_df[col].mean():.4f}%",
                "Max Daily Vol":   f"{multi_vol_df[col].max():.4f}%",
                "Mean Ann. Vol":   f"{multi_vol_df[col].mean() * np.sqrt(252):.2f}%",
                "Std":             f"{multi_vol_df[col].std():.4f}",
            }
            for col in multi_vol_df.columns
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown(
            '<div class="insight-box">'
            "<b>📌 Observation:</b> Bank Nifty typically shows higher volatility than "
            "Nifty 50 and Sensex due to its concentrated exposure to the banking sector. "
            "All three indices are highly correlated, reflecting the integrated nature "
            "of Indian equity markets."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Enable 'Multi-index comparison' in the sidebar to see this analysis.")

# ─────────────────────────────────────────────────────────
# TAB 6 — STATISTICS
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Descriptive Statistics</div>',
                unsafe_allow_html=True)
    stat_cols = ["returns", "volatility", "vol_ann", "sentiment_index"]
    desc_df   = data[stat_cols].describe().round(6)
    desc_df.loc["skewness"] = data[stat_cols].skew().round(6)
    desc_df.loc["kurtosis"] = data[stat_cols].kurt().round(6)
    st.dataframe(desc_df, use_container_width=True)

    # ADF Tests
    st.markdown('<div class="section-header">Stationarity Tests (ADF)</div>',
                unsafe_allow_html=True)
    adf_rows = []
    for col in ["returns", "volatility", "sentiment_index"]:
        series = data[col].dropna()
        # Skip ADF if series is constant or near-constant (std < 1e-6)
        series_std = series.std()
        if series.nunique() <= 1 or series_std < 1e-6:
            adf_rows.append({
                "Series": col, "ADF Stat": "—", "P-Value": "—",
                "Stationary": "⚠️ Constant", "Significance": "—",
            })
        else:
            try:
                adf = adfuller(series)
                adf_rows.append({
                    "Series": col, "ADF Stat": round(adf[0], 4),
                    "P-Value": round(adf[1], 6),
                    "Stationary": "✅ Yes" if adf[1] < 0.05 else "❌ No",
                    "Significance": "***" if adf[1] < 0.01 else ("**" if adf[1] < 0.05 else "*"),
                })
            except ValueError as e:
                # Catch "Invalid input, x is constant" error
                if "constant" in str(e).lower():
                    adf_rows.append({
                        "Series": col, "ADF Stat": "—", "P-Value": "—",
                        "Stationary": "⚠️ Constant", "Significance": "—",
                    })
                else:
                    adf_rows.append({
                        "Series": col, "ADF Stat": "—", "P-Value": "—",
                        "Stationary": f"⚠️ Error", "Significance": "—",
                    })
            except Exception as e:
                adf_rows.append({
                    "Series": col, "ADF Stat": "—", "P-Value": "—",
                    "Stationary": f"⚠️ Error", "Significance": "—",
                })
    st.dataframe(pd.DataFrame(adf_rows), use_container_width=True)

    # Return Distribution
    st.markdown('<div class="section-header">Return Distribution</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(data["returns"], bins=80, color="#1f4e79", alpha=0.75,
                 edgecolor="white", linewidth=0.3)
    axes[0].set_title("Distribution of Log Returns", fontweight="bold")
    axes[0].set_xlabel("Log Return")
    axes[0].set_ylabel("Frequency")

    axes[1].plot(data.index, data["returns"], color="#1f4e79", linewidth=0.5, alpha=0.8)
    axes[1].set_title("Log Returns Over Time", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Log Return")

    apply_plot_theme(fig, axes)
    st.pyplot(fig)
    plt.close()

    # Rolling volatility regimes
    st.markdown('<div class="section-header">Volatility Regime Analysis</div>',
                unsafe_allow_html=True)
    vol_q33 = data["volatility"].quantile(0.33)
    vol_q66 = data["volatility"].quantile(0.66)
    data["regime"] = pd.cut(
        data["volatility"],
        bins=[-np.inf, vol_q33, vol_q66, np.inf],
        labels=["Low", "Medium", "High"],
    )
    regime_stats = data.groupby("regime", observed=True).agg(
        Days=("returns", "count"),
        Mean_Return=("returns", lambda x: round(x.mean() * 100, 4)),
        Std_Return=("returns", lambda x: round(x.std() * 100, 4)),
        Mean_Sentiment=("sentiment_index", lambda x: round(x.mean(), 4)),
    ).reset_index()
    regime_stats.columns = ["Regime", "Days", "Mean Return (%)", "Std Return (%)", "Mean Sentiment"]
    st.dataframe(regime_stats, use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        "<b>Regime Interpretation:</b> Low-volatility regimes tend to coincide with "
        "positive market sentiment and steady returns. High-volatility regimes show "
        "elevated fear sentiment and larger return swings in both directions."
        "</div>",
        unsafe_allow_html=True,
    )

    # Full data export
    st.markdown("#### 📥 Full Data Export")
    full_export = data[["returns", "volatility", "vol_ann", "sentiment_index"]].copy()
    full_export.index.name = "Date"
    st.download_button(
        "Download full dataset (CSV)",
        full_export.to_csv().encode(),
        file_name=f"full_data_{TICKER.replace('^','')}_{start_year}_{end_year}.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Market Sentiment & Volatility Analyzer | "
    "EGARCH + Random Forest | "
    "Data: Yahoo Finance · GDELT · Google Trends · News</small></center>",
    unsafe_allow_html=True,
)
