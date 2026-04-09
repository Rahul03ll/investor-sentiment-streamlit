# Investor Sentiment & Stock Market Volatility Analysis

## Project Overview
This project analyzes the impact of **investor sentiment** on **stock market volatility**
in the Indian equity market using an EGARCH model augmented with Google Trends
sentiment data and a hybrid Random Forest classifier.

---

## Project Structure
```
├── egarch_sentiment_analysis.py   ← Main analysis script (full pipeline)
├── app.py                         ← Streamlit interactive dashboard
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## Methodology

### 1. Data Collection
- **Stock Data**: Nifty 50 (^NSEI) daily prices via Yahoo Finance (2007–2024)
- **Sentiment Data**: Google Trends for crisis-related keywords in India

### 2. Sentiment Index Construction
- Keywords: "stock market crash", "Nifty crash", "Sensex fall"
- Dimensionality reduction via **Principal Component Analysis (PCA)**
- First PC captures shared fear/negative sentiment variation

### 3. Stationarity Testing
- **Augmented Dickey-Fuller (ADF)** test on returns and sentiment

### 4. Model Comparison
| Model | Description |
|-------|-------------|
| GARCH(1,1) | Baseline symmetric volatility model |
| GJR-GARCH(1,1) | Captures leverage effects |
| **EGARCH(1,1)** | **Selected — lowest AIC/BIC, asymmetric** |

### 5. EGARCH with Sentiment
- Sentiment index included as external regressor
- Granger causality tests confirm predictive relationship

### 6. Out-of-Sample Forecasting
- 80/20 chronological train/test split
- Evaluated on RMSE and MAE

### 7. Hybrid ML Model
- **Features**: EGARCH volatility, sentiment index, lagged variables, moving averages
- **Target**: Binary — will tomorrow's volatility be above/below median?
- **Model**: Random Forest Classifier (200 trees)

---

## How to Run

### Full Analysis Script
```bash
pip install -r requirements.txt
python egarch_sentiment_analysis.py
```

### Streamlit Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| EGARCH beta (persistence) | ~0.987 |
| EGARCH alpha (ARCH effect) | ~0.206 |
| Granger Causality (lag 1) | F=25.01, p<0.001 |
| ML Classifier Accuracy | ~65–70% |

---

## Dependencies
- Python 3.9+
- See `requirements.txt`

---

## References
- Nelson (1991) — EGARCH
- Baker & Wurgler (2007) — Investor Sentiment
- Preis et al. (2013) — Google Trends in Finance
- Haritha & Rishad (2020) — Indian Market Sentiment
