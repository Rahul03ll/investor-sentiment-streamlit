# ============================================================
# core.py — Data loading, sentiment, and model fitting
# All heavy logic lives here so app.py stays clean and
# automated tests can import functions without Streamlit.
# ============================================================

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ── Optional imports (graceful degradation) ───────────────
try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False

try:
    from pytrends.request import TrendReq
    _HAS_PYTRENDS = True
except ImportError:
    _HAS_PYTRENDS = False


# ╔══════════════════════════════════════════════════════════╗
# ║  STOCK DATA                                             ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and compute log returns."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["returns"])


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — PRIMARY: GDELT                             ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=7200)
def load_gdelt_sentiment(
    start: str, end: str, fast_mode: bool = True
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Fetch GDELT GKG tone data for Indian market news.
    Returns (DataFrame with 'sentiment' column indexed by date, error_string).
    """
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")

        # In fast mode sample at most 12 months; full mode uses all months
        months: list[datetime] = []
        cur = start_dt.replace(day=1)
        while cur <= end_dt:
            months.append(cur)
            # next month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)

        if fast_mode and len(months) > 12:
            step = max(1, len(months) // 12)
            months = months[::step]

        records: list[dict] = []
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

        for month in months:
            query = (
                "India stock market Nifty Sensex"
                " sourcelang:english"
                " sourcecountry:IN"
            )
            params = {
                "query":  query,
                "mode":   "ArtList",
                "maxrecords": "25",
                "format": "json",
                "startdatetime": month.strftime("%Y%m%d000000"),
                "enddatetime":   (
                    month.replace(
                        day=28 if month.month == 2 else 30
                    )
                ).strftime("%Y%m%d235959"),
            }
            try:
                resp = requests.get(base_url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                articles = data.get("articles", [])
                for art in articles:
                    tone_str = art.get("tone", "")
                    if tone_str:
                        try:
                            tone_val = float(tone_str.split(",")[0])
                            date_str = art.get("seendate", "")[:8]
                            if len(date_str) == 8:
                                dt = datetime.strptime(date_str, "%Y%m%d")
                                records.append({"date": dt, "tone": tone_val})
                        except (ValueError, IndexError):
                            continue
                time.sleep(0.3)
            except requests.RequestException:
                continue

        if not records:
            return None, "GDELT returned no records for the selected period."

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("date")["tone"].mean().reset_index()
        df = df.rename(columns={"tone": "sentiment"})
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df.resample("D").mean()
        return df, None

    except Exception as exc:
        return None, f"GDELT error: {exc}"


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — SECONDARY: GOOGLE TRENDS                  ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=7200)
def load_trends_data(
    start: str, end: str
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """
    Fetch Google Trends data for fear/sentiment keywords.
    Returns (DataFrame, keywords_list, error_string).
    """
    if not _HAS_PYTRENDS:
        return None, [], "pytrends not installed."

    keywords = ["stock market crash", "Nifty crash", "Sensex fall"]
    try:
        pytrends = TrendReq(hl="en-US", tz=330, retries=2, backoff_factor=0.5, timeout=(10, 25))
        pytrends.build_payload(keywords, timeframe=f"{start} {end}", geo="IN")
        trends = pytrends.interest_over_time()

        if trends is None or trends.empty:
            return None, [], "Google Trends returned empty data."

        if "isPartial" in trends.columns:
            trends = trends.drop(columns=["isPartial"])

        # Strip timezone so merge with stock data works
        if trends.index.tz is not None:
            trends.index = trends.index.tz_localize(None)

        # Validate — timezone mismatch can produce all-NaN columns
        nan_cols = [k for k in keywords if k in trends.columns and trends[k].isna().all()]
        if nan_cols:
            return None, [], f"Trends merge produced all-NaN for: {nan_cols}."

        present_kw = [k for k in keywords if k in trends.columns]
        if not present_kw:
            return None, [], "No keyword columns found in Trends response."

        return trends[present_kw], present_kw, None

    except Exception as exc:
        return None, [], f"Google Trends error: {exc}"


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — TERTIARY: NEWS PIPELINE                   ║
# ╚══════════════════════════════════════════════════════════╝

def _load_newsapi_texts(api_key: str) -> tuple[list[str], str | None]:
    """Fetch article texts from NewsAPI."""
    if not api_key:
        return [], "NewsAPI key missing."
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        "Indian stock market Nifty Sensex",
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 50,
                "apiKey":   api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        texts = [
            f"{a.get('title', '')}. {a.get('description', '')}"
            for a in articles
            if a.get("title") or a.get("description")
        ]
        return texts, None if texts else "NewsAPI returned no articles."
    except Exception as exc:
        return [], f"NewsAPI error: {exc}"


def _load_yfinance_news_texts(ticker: str = "^NSEI") -> tuple[list[str], str | None]:
    """Fetch news titles from Yahoo Finance."""
    try:
        news = yf.Ticker(ticker).news
        texts = [item.get("title", "") for item in (news or [])[:50] if item.get("title")]
        return texts, None if texts else "Yahoo Finance news empty."
    except Exception as exc:
        return [], f"Yahoo Finance news error: {exc}"


def _load_rss_texts() -> tuple[list[str], str | None]:
    """Fetch headlines from Economic Times RSS (no API key needed)."""
    if not _HAS_FEEDPARSER:
        return [], "feedparser not installed."
    try:
        feed = feedparser.parse(
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        )
        texts = [e.title for e in feed.entries[:50] if getattr(e, "title", "")]
        return texts, None if texts else "RSS feed empty."
    except Exception as exc:
        return [], f"RSS error: {exc}"


def news_sentiment_pipeline(
    api_key: str | None = None,
) -> tuple[pd.Series, str | None, str]:
    """
    Robust news-sentiment pipeline with three fallback levels:
      1. NewsAPI  (requires key)
      2. Yahoo Finance news
      3. Economic Times RSS
    Returns (sentiment_series, error_or_None, source_name).
    """
    if not _HAS_VADER:
        dummy = pd.Series(
            [0.0] * 30,
            index=pd.date_range(end=datetime.now().date(), periods=30),
        )
        return dummy, "vaderSentiment not installed.", "None"

    analyzer = SentimentIntensityAnalyzer()

    for loader, source in [
        (lambda: _load_newsapi_texts(api_key or ""), "NewsAPI"),
        (lambda: _load_yfinance_news_texts(),         "Yahoo Finance"),
        (lambda: _load_rss_texts(),                   "RSS Feed"),
    ]:
        texts, err = loader()
        if texts:
            scores = [analyzer.polarity_scores(t)["compound"] for t in texts[:100]]
            idx    = pd.date_range(end=datetime.now().date(), periods=len(scores))
            series = pd.Series(scores, index=idx, name="sentiment")
            series = series.resample("D").mean().ffill()
            return series, None, source

    dummy = pd.Series(
        [0.0] * 30,
        index=pd.date_range(end=datetime.now().date(), periods=30),
    )
    return dummy, "All news sources failed.", "None"


# ╔══════════════════════════════════════════════════════════╗
# ║  EGARCH MODEL                                           ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=1800)
def fit_egarch_model(
    returns_array: np.ndarray,
    sentiment_array: np.ndarray,
    p: int,
    q: int,
):
    """
    Fit an EGARCH(p,q) model with sentiment as an external regressor.
    Returns the arch ARCHModelResult object.
    Raises ValueError if inputs are too short or contain non-finite values.
    """
    from arch import arch_model

    returns_pct = returns_array * 100

    # Align lengths
    min_len = min(len(returns_pct), len(sentiment_array))
    returns_pct    = returns_pct[:min_len]
    sentiment_trim = sentiment_array[:min_len]

    # Guard: need at least 100 observations for EGARCH to converge
    if min_len < 100:
        raise ValueError(
            f"Insufficient data: {min_len} observations after alignment "
            "(need ≥ 100). Check that sentiment data covers the selected date range."
        )

    # Guard: no NaN / inf in either array
    if not np.isfinite(returns_pct).all():
        raise ValueError("returns array contains NaN or inf values.")
    if not np.isfinite(sentiment_trim).all():
        # Replace any remaining NaN/inf in sentiment with column mean
        mean_s = np.nanmean(sentiment_trim)
        sentiment_trim = np.where(np.isfinite(sentiment_trim), sentiment_trim, mean_s)

    model = arch_model(
        returns_pct,
        vol="EGARCH",
        p=p,
        q=q,
        x=sentiment_trim.reshape(-1, 1),
    )
    return model.fit(disp="off")


# ╔══════════════════════════════════════════════════════════╗
# ║  MODEL COMPARISON                                       ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=1800)
def fit_model_comparison(returns_array: np.ndarray) -> pd.DataFrame:
    """
    Fit GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) on the same returns
    and return a comparison DataFrame with AIC, BIC, Log-Likelihood.
    """
    from arch import arch_model

    returns_pct = returns_array * 100
    specs = {
        "GARCH(1,1)":     arch_model(returns_pct, vol="Garch",  p=1,      q=1),
        "GJR-GARCH(1,1)": arch_model(returns_pct, vol="Garch",  p=1, o=1, q=1),
        "EGARCH(1,1)":    arch_model(returns_pct, vol="EGARCH", p=1,      q=1),
    }
    rows = []
    for name, spec in specs.items():
        try:
            res = spec.fit(disp="off")
            rows.append({
                "Model":            name,
                "Log-Likelihood":   round(res.loglikelihood, 2),
                "AIC":              round(res.aic, 2),
                "BIC":              round(res.bic, 2),
                "Best by AIC":      "",
                "Best by BIC":      "",
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    best_aic = df.loc[df["AIC"].idxmin(), "Model"]
    best_bic = df.loc[df["BIC"].idxmin(), "Model"]
    df["Best by AIC"] = df["Model"].apply(lambda m: "✅" if m == best_aic else "")
    df["Best by BIC"] = df["Model"].apply(lambda m: "✅" if m == best_bic else "")
    return df
