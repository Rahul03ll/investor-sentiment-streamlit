# ============================================================
# core.py — Data loading, sentiment pipeline, model fitting
# ============================================================

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

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


# ── helpers ───────────────────────────────────────────────

def _add_jitter(series: pd.Series, seed: int = 0) -> pd.Series:
    """Add tiny noise so a near-constant series has variance for ADF/Granger."""
    rng = np.random.default_rng(seed)
    std = series.std()
    if std == 0 or not np.isfinite(std):
        std = 1e-4
    return series + rng.normal(0, std * 1e-6, len(series))


# ╔══════════════════════════════════════════════════════════╗
# ║  STOCK DATA                                             ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV and compute log returns."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}' between {start} and {end}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["returns"])


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — PRIMARY: GDELT                             ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=7200)
def load_gdelt_sentiment(start: str, end: str, fast_mode: bool = True):
    """
    Fetch GDELT article tone for Indian market news.
    Returns (DataFrame['sentiment'], error_str | None).
    
    Optimized for Streamlit Cloud free tier with aggressive sampling.
    """
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")

        # Generate monthly intervals
        months = []
        cur = start_dt.replace(day=1)
        while cur <= end_dt:
            months.append(cur)
            cur = cur.replace(month=cur.month + 1) if cur.month < 12 \
                else cur.replace(year=cur.year + 1, month=1)

        # Aggressive sampling for free tier: max 8 API calls (reduced from 10)
        if len(months) > 8:
            step = max(1, len(months) // 8)
            months = months[::step][:8]
        
        # If still too many, prioritize recent months
        if len(months) > 8:
            months = months[-8:]  # Take most recent 8 months

        records = []
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Try multiple query variations
        queries = [
            "India Nifty stock market",
            "India Sensex BSE",
            "Indian stock exchange"
        ]
        
        for i, month in enumerate(months):
            # Rotate through queries to get better coverage
            query = queries[i % len(queries)]
            
            params = {
                "query": query,
                "mode": "ArtList",
                "maxrecords": "75",  # Increased from 50
                "format": "json",
                "startdatetime": month.strftime("%Y%m%d000000"),
                "enddatetime": month.replace(
                    day=28 if month.month == 2 else 30
                ).strftime("%Y%m%d235959"),
            }
            
            try:
                resp = requests.get(base_url, params=params, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])
                    
                    for art in articles:
                        tone_str = art.get("tone", "")
                        date_str = art.get("seendate", "")[:8]
                        
                        if tone_str and len(date_str) == 8:
                            try:
                                tone_val = float(tone_str.split(",")[0])
                                # Filter extreme outliers (GDELT can have bad data)
                                if -10 <= tone_val <= 10:
                                    records.append({
                                        "date": datetime.strptime(date_str, "%Y%m%d"),
                                        "tone": tone_val,
                                    })
                            except (ValueError, IndexError):
                                pass
                
                # Longer delay to avoid rate limiting
                time.sleep(0.7)  # Increased from 0.5s
                
            except requests.RequestException as e:
                # Continue on error, don't fail entire fetch
                continue

        if not records:
            return None, f"GDELT returned no records for the selected period. Try a shorter date range or wait a few minutes."

        # Process records
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        
        # Group by date and take mean
        df = df.groupby("date")["tone"].mean().rename("sentiment")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        
        # Resample to daily and interpolate gaps (more aggressive)
        df = df.resample("D").mean().interpolate(method="linear", limit=60)  # Increased from 30
        
        # If we have at least some data, return it
        if df.notna().sum() > 5:  # Reduced threshold from 10
            return df.to_frame(), None
        else:
            return None, f"GDELT returned insufficient data ({df.notna().sum()} days). Try 'Full' mode or a different date range."

    except Exception as exc:
        return None, f"GDELT error: {str(exc)}"


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — SECONDARY: GOOGLE TRENDS                  ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=7200)
def load_trends_data(start: str, end: str):
    """
    Fetch Google Trends for fear keywords.
    Returns (DataFrame, keywords_list, error_str | None).
    
    Optimized for reliability with retry logic.
    """
    if not _HAS_PYTRENDS:
        return None, [], "pytrends not installed."

    keywords = ["stock market crash", "Nifty crash", "Sensex fall"]
    
    # Try multiple times with different configurations
    for attempt in range(3):
        try:
            # Create TrendReq without retry parameters (let pytrends handle it internally)
            pt = TrendReq(
                hl="en-US",
                tz=330,
                timeout=(10, 30)
            )
            
            # Build payload
            pt.build_payload(keywords, timeframe=f"{start} {end}", geo="IN")
            trends = pt.interest_over_time()

            if trends is None or trends.empty:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None, [], "Google Trends returned empty data after retries."
            
            # Clean up the data
            if "isPartial" in trends.columns:
                trends = trends.drop(columns=["isPartial"])
            
            # Fix timezone issues
            if trends.index.tz is not None:
                trends.index = trends.index.tz_localize(None)

            # Check for valid keywords
            present = [k for k in keywords if k in trends.columns]
            if not present:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None, [], "No keyword columns in Trends response."

            # Drop all-NaN keywords
            present = [k for k in present if not trends[k].isna().all()]
            if not present:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None, [], "All Trends columns are NaN."

            # Success! Return the data
            return trends[present], present, None

        except Exception as exc:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return None, [], f"Google Trends error after {attempt + 1} attempts: {str(exc)}"
    
    return None, [], "Google Trends failed after all retry attempts."


# ╔══════════════════════════════════════════════════════════╗
# ║  SENTIMENT — TERTIARY: NEWS PIPELINE                   ║
# ╚══════════════════════════════════════════════════════════╝

def _newsapi_texts(api_key: str):
    if not api_key:
        return [], "No API key."
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "Indian stock market Nifty Sensex", "language": "en",
                    "sortBy": "publishedAt", "pageSize": 50, "apiKey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        arts = r.json().get("articles", [])
        texts = [f"{a.get('title','')}. {a.get('description','')}"
                 for a in arts if a.get("title") or a.get("description")]
        return texts, None if texts else "No articles."
    except Exception as e:
        return [], str(e)


def _yfinance_texts(ticker: str = "^NSEI"):
    try:
        news = yf.Ticker(ticker).news or []
        texts = [n.get("title", "") for n in news[:50] if n.get("title")]
        return texts, None if texts else "No Yahoo news."
    except Exception as e:
        return [], str(e)


def _rss_texts():
    if not _HAS_FEEDPARSER:
        return [], "feedparser not installed."
    try:
        feed = feedparser.parse(
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        )
        texts = [e.title for e in feed.entries[:50] if getattr(e, "title", "")]
        return texts, None if texts else "RSS empty."
    except Exception as e:
        return [], str(e)


def news_sentiment_pipeline(api_key=None):
    """
    3-level news sentiment fallback with improved data collection.
    Returns (pd.Series, error | None, source_name).
    """
    if not _HAS_VADER:
        return None, "vaderSentiment not installed.", "None"

    analyzer = SentimentIntensityAnalyzer()
    
    # Try each source with better error handling
    errors = []
    
    for loader, source in [
        (lambda: _newsapi_texts(api_key or ""), "NewsAPI"),
        (_yfinance_texts,                        "Yahoo Finance"),
        (_rss_texts,                             "RSS Feed"),
    ]:
        try:
            texts, err = loader()
            if err:
                errors.append(f"{source}: {err}")
                continue
                
            if texts and len(texts) >= 10:  # Need at least 10 headlines
                # Calculate sentiment scores
                scores = []
                
                # Get scores for all texts
                for t in texts[:100]:
                    try:
                        score = analyzer.polarity_scores(t)["compound"]
                        scores.append(score)
                    except Exception:
                        continue
                
                if len(scores) < 10:
                    errors.append(f"{source}: Insufficient valid scores ({len(scores)})")
                    continue
                
                # Create date range (recent data only)
                end_date = datetime.now().date()
                start_date = end_date - pd.Timedelta(days=len(scores))
                idx = pd.date_range(start=start_date, end=end_date, periods=len(scores))
                
                s = pd.Series(scores, index=idx, name="sentiment")
                
                # Only return if we have meaningful variance
                if s.std() > 0.01:  # Check for non-constant data
                    return s.resample("D").mean().ffill(), None, source
                else:
                    errors.append(f"{source}: Data has insufficient variance (std={s.std():.4f})")
                    continue
            else:
                errors.append(f"{source}: Insufficient headlines ({len(texts) if texts else 0})")
                    
        except Exception as e:
            errors.append(f"{source}: {str(e)}")
            continue
    
    # All sources failed
    error_msg = "; ".join(errors) if errors else "All news sources failed"
    return None, error_msg, "None"


# ╔══════════════════════════════════════════════════════════╗
# ║  EGARCH MODEL                                           ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=1800)
def fit_egarch_model(returns_array: np.ndarray, sentiment_array: np.ndarray,
                     p: int, q: int):
    """
    Fit EGARCH(p,q) with sentiment as external regressor.
    Handles alignment, NaN sanitisation, and minimum-length guard.
    """
    from arch import arch_model

    returns_pct = np.asarray(returns_array, dtype=float) * 100
    sentiment   = np.asarray(sentiment_array, dtype=float)

    # Align lengths
    n = min(len(returns_pct), len(sentiment))
    returns_pct = returns_pct[:n]
    sentiment   = sentiment[:n]

    if n < 100:
        raise ValueError(
            f"Only {n} observations after alignment — need ≥100. "
            "Check that sentiment covers the selected date range."
        )

    # Sanitise returns
    if not np.isfinite(returns_pct).all():
        raise ValueError("Returns contain NaN/inf.")

    # Sanitise sentiment: replace NaN/inf with column mean
    bad = ~np.isfinite(sentiment)
    if bad.any():
        sentiment[bad] = np.nanmean(sentiment[~bad]) if (~bad).any() else 0.0

    # If sentiment is constant (e.g. all-zeros from demo), add tiny jitter
    # so arch doesn't produce a degenerate model
    if np.std(sentiment) < 1e-10:
        rng = np.random.default_rng(42)
        sentiment = sentiment + rng.normal(0, 1e-6, n)

    model = arch_model(returns_pct, vol="EGARCH", p=p, q=q,
                       x=sentiment.reshape(-1, 1))
    return model.fit(disp="off", show_warning=False)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODEL COMPARISON                                       ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=1800)
def fit_model_comparison(returns_array: np.ndarray) -> pd.DataFrame:
    """Fit GARCH / GJR-GARCH / EGARCH and return AIC/BIC comparison."""
    from arch import arch_model

    rp = np.asarray(returns_array, dtype=float) * 100
    specs = {
        "GARCH(1,1)":     arch_model(rp, vol="Garch",  p=1,      q=1),
        "GJR-GARCH(1,1)": arch_model(rp, vol="Garch",  p=1, o=1, q=1),
        "EGARCH(1,1)":    arch_model(rp, vol="EGARCH", p=1,      q=1),
    }
    rows = []
    for name, spec in specs.items():
        try:
            res = spec.fit(disp="off", show_warning=False)
            rows.append({"Model": name,
                         "Log-Likelihood": round(res.loglikelihood, 2),
                         "AIC": round(res.aic, 2),
                         "BIC": round(res.bic, 2),
                         "Best AIC": "", "Best BIC": ""})
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    best_aic = df.loc[df["AIC"].idxmin(), "Model"]
    best_bic = df.loc[df["BIC"].idxmin(), "Model"]
    df["Best AIC"] = df["Model"].apply(lambda m: "✅" if m == best_aic else "")
    df["Best BIC"] = df["Model"].apply(lambda m: "✅" if m == best_bic else "")
    return df
