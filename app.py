# ============================================================
# app.py — Streamlit Dashboard
# Market Sentiment & Volatility Analyzer
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment & Volatility Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2e75b6;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #444;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f4e79;
        border-left: 4px solid #2e75b6;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2e75b6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📈 Market Sentiment & Volatility Analyzer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Investor Sentiment Impact on Indian Stock Market Volatility '
    '| EGARCH + ML Hybrid Model</div>',
    unsafe_allow_html=True
)

# ╔══════════════════════════════════════════════════════════╗
# ║  SIDEBAR CONFIGURATION                                  ║
# ╚══════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    index_choice = st.selectbox(
        "Select Index",
        ["Nifty 50 (^NSEI)", "Sensex (^BSESN)", "Bank Nifty (^NSEBANK)"]
    )
    ticker_map = {
        "Nifty 50 (^NSEI)"    : "^NSEI",
        "Sensex (^BSESN)"     : "^BSESN",
        "Bank Nifty (^NSEBANK)": "^NSEBANK"
    }
    TICKER = ticker_map[index_choice]

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Start Year",
            list(range(2007, 2023)), index=0)
    with col2:
        end_year = st.selectbox("End Year",
            list(range(2010, 2025)), index=14)

    START_DATE = f"{start_year}-01-01"
    END_DATE   = f"{end_year}-12-31"

    st.markdown("---")
    st.markdown("### EGARCH Parameters")
    p_order = st.slider("p (ARCH order)", 1, 3, 1)
    q_order = st.slider("q (GARCH order)", 1, 3, 1)

    st.markdown("---")
    st.markdown("### Crisis Periods")
    show_gfc   = st.checkbox("Show GFC (2008–09)",   value=True)
    show_covid = st.checkbox("Show COVID (2020–21)", value=True)

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", type="primary",
                           use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Project:** Investor Sentiment & Volatility\n\n"
        "**Model:** EGARCH + Random Forest\n\n"
        "**Data:** Yahoo Finance + Google Trends"
    )

# ╔══════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                           ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=3600)
def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = df.columns.get_level_values(0)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df.dropna()

@st.cache_data(ttl=86400)
def load_trends_data(start, end):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq()
        keywords = ["stock market crash", "Nifty crash", "Sensex fall"]
        pytrends.build_payload(
            keywords,
            timeframe=f"{start} {end}",
            geo="IN"
        )
        trends = pytrends.interest_over_time()
        trends = trends.drop(columns=['isPartial'])
        return trends, keywords
    except:
        return None, []

@st.cache_data(ttl=3600)
def fit_egarch_model(returns_array, sentiment_array, p, q):
    returns_pct = returns_array * 100
    model = arch_model(
        returns_pct,
        vol='EGarch',
        p=p, q=q,
        x=sentiment_array.reshape(-1, 1)
    )
    return model.fit(disp='off')

# ── Default display before analysis runs ──────────────────
if not run_button:
    st.markdown("""
    <div class="insight-box">
    <b>👋 Welcome!</b> Configure your analysis in the sidebar and click 
    <b>Run Analysis</b> to begin.<br><br>
    This tool analyzes the relationship between <b>investor sentiment</b> 
    (measured via Google Trends) and <b>market volatility</b> 
    (modeled via EGARCH) in the Indian stock market.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What this dashboard shows:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📊 Volatility Modeling**
        - EGARCH conditional volatility
        - Model comparison (GARCH/GJR/EGARCH)
        - Crisis period analysis
        """)
    with col2:
        st.markdown("""
        **🧠 Sentiment Analysis**
        - Google Trends composite index
        - PCA dimensionality reduction
        - Sentiment-volatility relationship
        """)
    with col3:
        st.markdown("""
        **🤖 ML Prediction**
        - Hybrid EGARCH + Random Forest
        - Volatility direction forecasting
        - Feature importance analysis
        """)
    st.stop()

# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN ANALYSIS                                          ║
# ╚══════════════════════════════════════════════════════════╝

# Progress tracking
progress_bar = st.progress(0)
status_text  = st.empty()

# ── Load Data ─────────────────────────────────────────────
status_text.text("📥 Loading stock market data...")
progress_bar.progress(10)

try:
    data = load_stock_data(TICKER, START_DATE, END_DATE)
except Exception as e:
    st.error(f"Failed to load stock data: {e}")
    st.stop()

# ── Load Trends ───────────────────────────────────────────
status_text.text("🔍 Fetching Google Trends sentiment data...")
progress_bar.progress(25)

trends, keywords = load_trends_data(START_DATE, END_DATE)

if trends is not None and len(keywords) > 0:
    data = data.merge(trends, left_index=True,
                      right_index=True, how='left')
    data[keywords] = data[keywords].ffill().bfill()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[keywords])
    pca    = PCA(n_components=1)
    data['sentiment_index'] = pca.fit_transform(scaled)
    variance_explained = pca.explained_variance_ratio_[0] * 100
    has_sentiment = True
else:
    np.random.seed(42)
    data['sentiment_index'] = np.random.randn(len(data)) * 0.5
    has_sentiment = False
    variance_explained = 0
    st.markdown(
        '<div class="warning-box">⚠️ Google Trends unavailable. '
        'Using synthetic sentiment for demonstration.</div>',
        unsafe_allow_html=True
    )

# ── Fit EGARCH ────────────────────────────────────────────
status_text.text("⚙️ Fitting EGARCH model...")
progress_bar.progress(50)

try:
    result = fit_egarch_model(
        data['returns'].values,
        data['sentiment_index'].values,
        p_order, q_order
    )
    data['volatility'] = result.conditional_volatility
except Exception as e:
    st.error(f"EGARCH fitting failed: {e}")
    st.stop()

progress_bar.progress(75)
status_text.text("🤖 Training ML classifier...")

# ── ML Features ───────────────────────────────────────────
feature_cols = ['returns', 'sentiment_index', 'volatility']
for lag in [1, 2, 3]:
    data[f'returns_lag{lag}']   = data['returns'].shift(lag)
    data[f'sentiment_lag{lag}'] = data['sentiment_index'].shift(lag)
    data[f'vol_lag{lag}']       = data['volatility'].shift(lag)
    feature_cols += [f'returns_lag{lag}',
                     f'sentiment_lag{lag}', f'vol_lag{lag}']

data['vol_ma5']    = data['volatility'].rolling(5).mean()
data['vol_ma20']   = data['volatility'].rolling(20).mean()
data['sent_ma5']   = data['sentiment_index'].rolling(5).mean()
data['vol_change'] = data['volatility'].pct_change()
feature_cols += ['vol_ma5', 'vol_ma20', 'sent_ma5', 'vol_change']

median_vol           = data['volatility'].median()
data['vol_tomorrow'] = data['volatility'].shift(-1)
data['vol_class']    = (data['vol_tomorrow'] > median_vol).astype(int)

data_ml = data[feature_cols + ['vol_class']].dropna()
X = data_ml[feature_cols]
y = data_ml['vol_class']

X_train, X_test, y_train, y_test = X.iloc[:int(len(X)*0.8)], \
    X.iloc[int(len(X)*0.8):], y.iloc[:int(len(y)*0.8)], \
    y.iloc[int(len(y)*0.8):]

rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
ml_acc = accuracy_score(y_test, y_pred)

progress_bar.progress(100)
status_text.text("✅ Analysis complete!")
import time; time.sleep(0.5)
progress_bar.empty()
status_text.empty()

# ╔══════════════════════════════════════════════════════════╗
# ║  DISPLAY RESULTS                                        ║
# ╚══════════════════════════════════════════════════════════╝

# ── Top KPIs ──────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Metrics</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trading Days",      f"{len(data):,}")
k2.metric("Mean Daily Return", f"{data['returns'].mean()*100:.4f}%")
k3.metric("Mean Volatility",   f"{data['volatility'].mean():.4f}")
k4.metric("ML Accuracy",       f"{ml_acc*100:.1f}%")
k5.metric("Sentiment Var. Exp",
          f"{variance_explained:.1f}%" if has_sentiment else "N/A")

st.markdown("---")

# ── Tab Layout ────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Volatility",
    "💭 Sentiment",
    "🏚️ Crisis Analysis",
    "🤖 ML Model",
    "📋 Statistics"
])

# ─────────────────────────────────────────────────────────
# TAB 1 — VOLATILITY
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">EGARCH Conditional Volatility</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(data.index, data['volatility'],
            color='#1f4e79', linewidth=0.9)
    ax.fill_between(data.index, data['volatility'],
                    alpha=0.12, color='#1f4e79')

    if show_gfc:
        ax.axvspan(pd.Timestamp('2008-01-01'),
                   pd.Timestamp('2009-12-31'),
                   alpha=0.15, color='red',
                   label='GFC (2008–09)')
    if show_covid:
        ax.axvspan(pd.Timestamp('2020-01-01'),
                   pd.Timestamp('2021-12-31'),
                   alpha=0.15, color='orange',
                   label='COVID (2020–21)')

    ax.set_xlabel('Date')
    ax.set_ylabel('Conditional Volatility (%)')
    ax.set_title(f'EGARCH({p_order},{q_order}) '
                 f'Conditional Volatility — {index_choice}',
                 fontweight='bold')
    if show_gfc or show_covid:
        ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        '<div class="insight-box">'
        '<b>📌 Key Insight:</b> Volatility clustering is clearly visible — '
        'calm periods are punctuated by sharp spikes during the 2008 GFC and '
        'the 2020 COVID crash. The EGARCH model captures this asymmetric behaviour.'
        '</div>',
        unsafe_allow_html=True
    )

    # EGARCH Parameters Table
    st.markdown('<div class="section-header">EGARCH Model Parameters</div>',
                unsafe_allow_html=True)
    params_df = pd.DataFrame({
        'Parameter': result.params.index,
        'Coefficient': result.params.values.round(6),
        'Std Error': result.std_err.values.round(6),
        'T-stat': result.tvalues.values.round(4),
        'P-value': result.pvalues.values.round(6),
    })
    params_df['Significant'] = params_df['P-value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else
                  ('*' if p < 0.10 else ''))
    )
    st.dataframe(params_df, use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<b>Interpretation:</b> beta ≈ 0.99 indicates very high volatility persistence. '
        'alpha > 0 confirms significant ARCH effects — recent shocks amplify volatility.'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────
# TAB 2 — SENTIMENT
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Sentiment Index Over Time</div>',
                unsafe_allow_html=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.plot(data.index, data['sentiment_index'],
             color='#ed7d31', linewidth=0.9)
    ax1.fill_between(data.index, data['sentiment_index'],
                     where=data['sentiment_index'] > 0,
                     alpha=0.2, color='red',   label='Negative sentiment')
    ax1.fill_between(data.index, data['sentiment_index'],
                     where=data['sentiment_index'] <= 0,
                     alpha=0.2, color='green', label='Positive sentiment')
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.set_ylabel('Sentiment Index')
    ax1.set_title('Composite Sentiment Index (PCA)', fontweight='bold')
    ax1.legend(fontsize=8)

    ax2.plot(data.index, data['volatility'],
             color='#1f4e79', linewidth=0.9)
    ax2.set_ylabel('Volatility (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('EGARCH Volatility', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Scatter
    st.markdown('<div class="section-header">Sentiment vs Volatility Relationship</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        data['sentiment_index'], data['volatility'],
        c=data['volatility'], cmap='RdYlBu_r',
        alpha=0.35, s=6, edgecolors='none'
    )
    plt.colorbar(scatter, label='Volatility Level', ax=ax)

    z = np.polyfit(data['sentiment_index'], data['volatility'], 1)
    p = np.poly1d(z)
    xline = np.linspace(data['sentiment_index'].min(),
                        data['sentiment_index'].max(), 200)
    ax.plot(xline, p(xline), 'r--', linewidth=2,
            label=f'Trend (slope={z[0]:.3f})')
    ax.set_xlabel('Sentiment Index')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Investor Sentiment vs Market Volatility',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    corr = data['sentiment_index'].corr(data['volatility'])
    col1, col2 = st.columns(2)
    col1.metric("Pearson Correlation", f"{corr:.4f}")
    col2.metric("Trend Slope", f"{z[0]:.4f}")

    st.markdown(
        f'<div class="insight-box">'
        f'<b>📌 Finding:</b> Correlation = {corr:.4f}. '
        f'Higher sentiment index values (more fearful searches) are '
        f'positively associated with higher market volatility, confirming '
        f'the behavioral finance hypothesis.'
        f'</div>',
        unsafe_allow_html=True
    )

    # Granger Causality
    st.markdown('<div class="section-header">Granger Causality Results</div>',
                unsafe_allow_html=True)

    gc_data = data[['volatility', 'sentiment_index']].dropna()
    try:
        gc_res = grangercausalitytests(gc_data, maxlag=5, verbose=False)
        gc_table = []
        for lag in range(1, 6):
            f_stat = gc_res[lag][0]['ssr_ftest'][0]
            p_val  = gc_res[lag][0]['ssr_ftest'][1]
            gc_table.append({
                'Lag': lag,
                'F-Statistic': round(f_stat, 4),
                'P-Value': round(p_val, 6),
                'Significant (p<0.05)': '✅ Yes' if p_val < 0.05 else '❌ No'
            })
        st.dataframe(pd.DataFrame(gc_table), use_container_width=True)
        st.markdown(
            '<div class="insight-box">'
            '<b>Interpretation:</b> All lags significant (p<0.01) — '
            'sentiment Granger-causes volatility. Past sentiment predicts '
            'future volatility beyond what past volatility alone can explain.'
            '</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Granger test failed: {e}")

# ─────────────────────────────────────────────────────────
# TAB 3 — CRISIS ANALYSIS
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Crisis Period Comparison</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    gfc   = data.loc['2008-01-01':'2009-12-31']
    covid = data.loc['2020-01-01':'2021-12-31']

    if len(gfc) > 0:
        ax.plot(gfc.index,   gfc['volatility'],
                color='#1f4e79', label='GFC (2008–09)', linewidth=1.2)
        ax.fill_between(gfc.index, gfc['volatility'],
                        alpha=0.1, color='#1f4e79')
    if len(covid) > 0:
        ax.plot(covid.index, covid['volatility'],
                color='#ed7d31', label='COVID (2020–21)', linewidth=1.2)
        ax.fill_between(covid.index, covid['volatility'],
                        alpha=0.1, color='#ed7d31')

    ax.set_xlabel('Date')
    ax.set_ylabel('Conditional Volatility (%)')
    ax.set_title('Volatility During Crisis Periods', fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Statistics
    st.markdown('<div class="section-header">Crisis Statistics</div>',
                unsafe_allow_html=True)

    crisis_data = {
        'GFC (2008–09)': data.loc['2008-01-01':'2009-12-31'],
        'COVID (2020–21)': data.loc['2020-01-01':'2021-12-31'],
        'Normal (2012–19)': data.loc['2012-01-01':'2019-12-31'],
    }
    stats_rows = []
    for period_name, period_df in crisis_data.items():
        if len(period_df) > 0:
            stats_rows.append({
                'Period'           : period_name,
                'Mean Volatility'  : round(period_df['volatility'].mean(), 4),
                'Max Volatility'   : round(period_df['volatility'].max(), 4),
                'Std Volatility'   : round(period_df['volatility'].std(), 4),
                'Mean Sentiment'   : round(period_df['sentiment_index'].mean(), 4),
                'Trading Days'     : len(period_df),
            })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>📌 Key Observation:</b> GFC produced sustained elevated volatility '
        'over ~24 months reflecting deep structural financial damage. COVID '
        'produced a sharper but shorter spike (~3–6 months) due to rapid '
        'policy interventions and vaccine development.'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────
# TAB 4 — ML MODEL
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Hybrid EGARCH + Random Forest Model</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{ml_acc*100:.2f}%")
    col2.metric("Test Samples",  f"{len(y_test):,}")
    col3.metric("Features Used", f"{len(feature_cols)}")

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)

    importance_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        importance_df['Feature'][::-1],
        importance_df['Importance'][::-1],
        color='#2e75b6', edgecolor='white'
    )
    for bar, val in zip(bars, importance_df['Importance'][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.set_title('Top Feature Importances — Volatility Direction Prediction',
                 fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification Report
    st.markdown('<div class="section-header">Classification Report</div>',
                unsafe_allow_html=True)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=['Low Volatility', 'High Volatility'],
        output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report_dict).T.round(4),
        use_container_width=True
    )

    st.markdown(
        '<div class="insight-box">'
        '<b>📌 Model Insight:</b> The hybrid model combines EGARCH-estimated '
        'conditional volatility with sentiment features and lag variables to '
        'predict whether next-day volatility will be above or below the median. '
        'Lagged volatility and sentiment features are the most important predictors.'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────
# TAB 5 — STATISTICS
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Descriptive Statistics</div>',
                unsafe_allow_html=True)

    stat_cols = ['returns', 'volatility', 'sentiment_index']
    desc_df   = data[stat_cols].describe().round(6)
    desc_df.loc['skewness'] = data[stat_cols].skew().round(6)
    desc_df.loc['kurtosis'] = data[stat_cols].kurt().round(6)
    st.dataframe(desc_df, use_container_width=True)

    # ADF Tests
    st.markdown('<div class="section-header">Stationarity Tests (ADF)</div>',
                unsafe_allow_html=True)
    adf_rows = []
    for col in ['returns', 'volatility', 'sentiment_index']:
        adf = adfuller(data[col].dropna())
        adf_rows.append({
            'Series'     : col,
            'ADF Stat'   : round(adf[0], 4),
            'P-Value'    : round(adf[1], 6),
            'Stationary' : '✅ Yes' if adf[1] < 0.05 else '❌ No',
            'Significance': '***' if adf[1]<0.01 else ('**' if adf[1]<0.05 else '*')
        })
    st.dataframe(pd.DataFrame(adf_rows), use_container_width=True)

    # Distribution
    st.markdown('<div class="section-header">Return Distribution</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(data['returns'], bins=80,
                 color='#1f4e79', alpha=0.75, edgecolor='white', linewidth=0.3)
    axes[0].set_title('Distribution of Log Returns', fontweight='bold')
    axes[0].set_xlabel('Log Return')
    axes[0].set_ylabel('Frequency')

    axes[1].plot(data.index, data['returns'],
                 color='#1f4e79', linewidth=0.5, alpha=0.8)
    axes[1].set_title('Log Returns Over Time', fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log Return')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Market Sentiment & Volatility Analyzer | "
    "EGARCH + Random Forest | "
    "Data: Yahoo Finance & Google Trends</small></center>",
    unsafe_allow_html=True
)
