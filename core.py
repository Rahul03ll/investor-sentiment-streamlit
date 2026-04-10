import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


import streamlit as st

@st.cache_data(ttl=3600)  # Cache 1 hour
def load_stock_data(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


@st.cache_data(ttl=7200)  # Cache 2 hours for expensive API
def load_gdelt_sentiment(start, end, fast_mode=True):
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        span_days = (end_dt - start_dt).days
        if span_days > 3650:
            chunk_days = 120
        elif span_days > 1825:
            chunk_days = 60
        elif span_days > 730:
            chunk_days = 30
        else:
            chunk_days = 7

        if fast_mode:
            max_requests = 10
            max_duration_s = 10
        else:
            max_requests = 24
            max_duration_s = 20

        t0 = time.monotonic()
        all_data = []
        current = start_dt
        request_count = 0

        while current < end_dt and request_count < max_requests:
            if (time.monotonic() - t0) > max_duration_s:
                break
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            params = {
                "query": "(stock market OR nifty OR sensex) AND sourcecountry:IN",
                "mode": "ArtList",
                "format": "json",
                "maxrecords": 250,
                "startdatetime": current.strftime("%Y%m%d%H%M%S"),
                "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
            }
            try:
                response = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=params,
                    timeout=8,
                )
                response.raise_for_status()
                articles = response.json().get("articles", [])
                for art in articles:
                    try:
                        tone = float(art.get("tone", 0))
                        dt = pd.to_datetime(art["seendate"], errors="coerce")
                        if pd.isna(dt):
                            continue
                        all_data.append((dt.date(), tone))
                    except Exception:
                        continue
            except Exception:
                pass
            request_count += 1
            current = chunk_end

        if not all_data:
            return None, "GDELT timed out or returned no usable sentiment rows."
        df = pd.DataFrame(all_data, columns=["date", "sentiment"])
        df = df.groupby("date").mean()
        df.index = pd.to_datetime(df.index)
        return df, None
    except Exception as exc:
        return None, str(exc)


def load_trends_data(start, end):
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(
            hl="en-US",
            tz=330,
            retries=3,
            backoff_factor=0.5,
            timeout=(10, 25),
        )
        keywords = ["stock market crash", "Nifty crash", "Sensex fall"]
        time.sleep(1)  # Avoid immediate 429 rate-limit from Google
        pytrends.build_payload(keywords, timeframe=f"{start} {end}", geo="IN")
        time.sleep(1)  # Buffer before fetching
        trends = pytrends.interest_over_time()
        if trends is None or trends.empty:
            return None, [], "Google Trends returned no data."
        if "isPartial" in trends.columns:
            trends = trends.drop(columns=["isPartial"])
        return trends, keywords, None
    except Exception as exc:
        return None, [], str(exc)


@st.cache_data(ttl=1800)  # Cache 30 min for expensive fitting
def fit_egarch_model(returns_array, sentiment_array, p, q):
    from arch import arch_model

    returns_pct = returns_array * 100
    model = arch_model(
        returns_pct,
        vol="EGARCH",  # Fixed case sensitivity
        p=p,
        q=q,
        x=sentiment_array.reshape(-1, 1),
    )
    return model.fit(disp="off")
