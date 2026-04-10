# 📈 Market Sentiment & Volatility Analyzer

> **Investor Sentiment Impact on Indian Stock Market Volatility**  
> EGARCH + Random Forest Hybrid Model | Live Streamlit Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investor-sentiment-streamlit.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 Overview

This project quantifies the relationship between **investor sentiment** and **stock market volatility** in the Indian equity market. It combines:

- **EGARCH** (Exponential GARCH) for asymmetric volatility modelling
- **Google Trends / GDELT / News** for real-time sentiment signals
- **Random Forest** classifier for next-day volatility direction prediction
- An interactive **Streamlit dashboard** for live exploration

Indices covered: **Nifty 50**, **Sensex**, **Bank Nifty** — from 2007 to 2024.

---

## 🚀 Live Demo

👉 **[Open the App](https://(https://rahul03ll-investor-sentiment-streamlit-app-wyuzma.streamlit.app))**

---

## 📸 Features

| Tab | What you get |
|-----|-------------|
| 📈 **Volatility** | EGARCH conditional volatility, annualised vol, rolling MA, model comparison (GARCH / GJR / EGARCH), parameter table, CSV export |
| 💭 **Sentiment** | Sentiment index over time, scatter vs volatility, Granger causality tests |
| 🏚️ **Crisis Analysis** | GFC 2008–09 vs COVID 2020–21 comparison, crisis statistics table |
| 🤖 **ML Model** | Random Forest accuracy, feature importance, confusion matrix, classification report |
| 📊 **Multi-Index** | Nifty 50 vs Sensex vs Bank Nifty volatility comparison, cross-index correlation |
| 📋 **Statistics** | Descriptive stats, ADF stationarity tests, return distribution, volatility regime analysis |

---

## 🏗️ Project Structure

```
├── app.py                          ← Streamlit dashboard (all tabs & UI)
├── core.py                         ← Data loading, sentiment pipeline, EGARCH fitting
├── egarch_sentiment_analysis.py    ← Standalone analysis script (notebook-style)
├── requirements.txt                ← Python dependencies
├── tests/
│   ├── conftest.py                 ← Shared pytest fixtures
│   ├── test_data.py                ← Stock data loading tests
│   ├── test_ml.py                  ← ML training tests
│   ├── test_model.py               ← EGARCH model tests
│   ├── test_pipeline.py            ← Full pipeline smoke test
│   └── test_sentiment.py           ← Sentiment source tests
└── .streamlit/
    └── secrets.toml                ← API keys (not committed)
```

---

## 🧠 Methodology

### 1. Data Collection
| Source | Data | Library |
|--------|------|---------|
| Yahoo Finance | Daily OHLCV prices (2007–2024) | `yfinance` |
| GDELT Project | News tone / sentiment scores | `requests` |
| Google Trends | Fear keyword search volumes | `pytrends` |
| NewsAPI / RSS | Headline sentiment (fallback) | `vaderSentiment` |

### 2. Sentiment Pipeline (4-level cascade)
```
GDELT (primary)
  └─► Google Trends (fallback 1)
        └─► NewsAPI / Yahoo Finance / RSS (fallback 2)
              └─► Random demo data (offline fallback)
```
- Keywords: `"stock market crash"`, `"Nifty crash"`, `"Sensex fall"`
- PCA reduces multi-keyword Trends data to a single composite index

### 3. Stationarity Testing
- **Augmented Dickey-Fuller (ADF)** test on log returns and sentiment index
- Both series confirmed stationary → EGARCH modelling valid

### 4. Model Comparison

| Model | Description | Selected |
|-------|-------------|----------|
| GARCH(1,1) | Symmetric volatility | |
| GJR-GARCH(1,1) | Captures leverage effects | |
| **EGARCH(1,1)** | **Asymmetric, log-variance** | ✅ Lowest AIC/BIC |

### 5. EGARCH with Sentiment Regressor
- Sentiment index included as external regressor `x`
- Granger causality tests confirm sentiment → volatility predictive relationship

### 6. Hybrid ML Classifier
- **Features**: Lagged returns, lagged sentiment, lagged volatility, rolling MAs
- **Target**: Will tomorrow's volatility be above/below the historical median?
- **Model**: Random Forest (200 trees, `max_depth=5`, `min_samples_leaf=20`)
- **Split**: Chronological 80/20 — no data leakage

---

## 📊 Key Results (Nifty 50, 2007–2024)

| Metric | Value |
|--------|-------|
| Trading days | 4,236 |
| Mean daily return | ~0.039% (~10% annualised) |
| Mean daily volatility | ~1.16% (~18.4% annualised) |
| EGARCH beta (persistence) | 0.987 *** |
| EGARCH alpha (ARCH effect) | 0.206 *** |
| Granger causality (lag 1) | p < 0.001 |
| ML classifier accuracy | ~60–70% (above baseline) |

> **Note on ML accuracy:** Very high accuracy (>90%) is a sign of overfitting to volatility autocorrelation, not genuine predictive power. The app shows lift vs baseline to give a fair picture.

---

## ⚙️ Running Locally

```bash
# 1. Clone
git clone https://github.com/Rahul03ll/investor-sentiment-streamlit.git
cd investor-sentiment-streamlit

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py

# 5. (Optional) Run the standalone analysis script
python egarch_sentiment_analysis.py
```

### Optional: NewsAPI key
Create `.streamlit/secrets.toml` for live news sentiment:
```toml
NEWSAPI_KEY = "your_key_here"
```
Get a free key at [newsapi.org](https://newsapi.org).

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover: stock data loading, EGARCH fitting, ML training, full pipeline smoke test, and all sentiment sources.

---

## 📦 Dependencies

```
yfinance · pandas · numpy · scipy · matplotlib · seaborn
scikit-learn · statsmodels · arch · pytrends · streamlit
vaderSentiment · feedparser · plotly · requests
```

Full list in [`requirements.txt`](requirements.txt).

---

## 📚 References

- Nelson (1991) — *Conditional Heteroskedasticity in Asset Returns: A New Approach* (EGARCH)
- Baker & Wurgler (2007) — *Investor Sentiment in the Stock Market*
- Preis, Moat & Stanley (2013) — *Quantifying Trading Behavior in Financial Markets Using Google Trends*
- Haritha & Rishad (2020) — *A study on investor sentiment and its effect on Indian stock market*

---

## 👤 Author

**Rahul** — [github.com/Rahul03ll](https://github.com/Rahul03ll)
