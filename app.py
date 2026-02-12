import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = "https://financialmodelingprep.com/stable"
HTTP_TIMEOUT_SEC = 12  # fixed timeout (no UI)


# ----------------------------
# HTTP + JSON helpers
# ----------------------------
def safe_get(url: str, params: Dict[str, Any]) -> Tuple[Optional[int], int, str]:
    start = time.perf_counter()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SEC)
        ms = int((time.perf_counter() - start) * 1000)
        return r.status_code, ms, r.text
    except requests.exceptions.RequestException as e:
        ms = int((time.perf_counter() - start) * 1000)
        return None, ms, f"Request error: {e}"


def try_json(text: str):
    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)


def get_fmp_key() -> Optional[str]:
    # Secrets only
    try:
        key = st.secrets["FMP_API_KEY"]
        if isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    return None


# ----------------------------
# Finance helpers
# ----------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, dates: pd.Series) -> Optional[float]:
    if equity.empty:
        return None
    start = equity.iloc[0]
    end = equity.iloc[-1]
    if start <= 0:
        return None
    dt0 = pd.to_datetime(dates.iloc[0])
    dt1 = pd.to_datetime(dates.iloc[-1])
    years = (dt1 - dt0).days / 365.25
    if years <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)


def sharpe_simple(daily_ret: pd.Series) -> Optional[float]:
    # No risk-free rate, simple annualized Sharpe using 252
    r = daily_ret.dropna()
    if r.empty:
        return None
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return None
    return float((mu / sd) * np.sqrt(252))


# ----------------------------
# FMP fetchers (stable)
# ----------------------------
def fmp_quote(symbol: str, api_key: str):
    url = f"{BASE}/quote"
    params = {"symbol": symbol, "apikey": api_key}
    return safe_get(url, params)


def fmp_profile(symbol: str, api_key: str):
    url = f"{BASE}/profile"
    params = {"symbol": symbol, "apikey": api_key}
    return safe_get(url, params)


def fmp_historical_eod_full(symbol: str, api_key: str):
    url = f"{BASE}/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": api_key}
    return safe_get(url, params)


def parse_historical_eod(payload: Any) -> pd.DataFrame:
    """
    Expect a dict with a list field, or list directly depending on FMP response.
    We handle both safely.
    """
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        # Common patterns: {"symbol": "...", "historical": [...]}
        for k in ["historical", "data", "prices"]:
            if k in payload and isinstance(payload[k], list):
                rows = payload[k]
                break
        # Sometimes "historicalStockList" or others; keep minimal robust
        if not rows:
            # fallback: try any list value
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows = v
                    break
    elif isinstance(payload, list):
        rows = payload

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    # Normalize column names to common set if present
    # typical: date, open, high, low, close, volume
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="SmartTrader Lab — Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 SmartTrader Lab — Stock Analysis (FMP)")

api_key = get_fmp_key()
if not api_key:
    st.error(
        "ยังไม่พบ FMP_API_KEY ใน Streamlit Secrets\n\n"
        "ไปที่ Streamlit Cloud → App → Settings → Secrets แล้วใส่:\n"
        'FMP_API_KEY="YOUR_KEY_HERE"'
    )
    st.stop()

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Symbol", value="AAPL").strip().upper()
    st.caption("ใช้ FMP stable endpoints ผ่าน Secrets เท่านั้น (ไม่แสดง/ไม่รับ key ใน UI)")

run = st.button("▶️ Fetch & Analyze", type="primary")

if not run:
    st.caption("กด Fetch & Analyze เพื่อดึงข้อมูลและเริ่มวิเคราะห์")
    st.stop()

# ---- Fetch Quote / Profile / Historical ----
tab1, tab2, tab3 = st.tabs(["Overview", "Historical Data", "Analytics"])

with tab1:
    st.subheader("Overview")

    q_code, q_ms, q_text = fmp_quote(symbol, api_key)
    p_code, p_ms, p_text = fmp_profile(symbol, api_key)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Quote**")
        st.write(f"HTTP: {q_code} | Latency: {q_ms} ms")
        q_json, q_err = try_json(q_text)
        if q_code != 200:
            st.error("Quote request failed")
            st.code(q_text[:2000], language="json")
        elif q_err:
            st.warning(f"Quote JSON parse error: {q_err}")
            st.code(q_text[:2000], language="text")
        else:
            # Usually list with one object
            if isinstance(q_json, list) and q_json and isinstance(q_json[0], dict):
                st.json(q_json[0])
            else:
                st.json(q_json)

    with colB:
        st.markdown("**Profile**")
        st.write(f"HTTP: {p_code} | Latency: {p_ms} ms")
        p_json, p_err = try_json(p_text)
        if p_code != 200:
            st.error("Profile request failed")
            st.code(p_text[:2000], language="json")
        elif p_err:
            st.warning(f"Profile JSON parse error: {p_err}")
            st.code(p_text[:2000], language="text")
        else:
            if isinstance(p_json, list) and p_json and isinstance(p_json[0], dict):
                st.json(p_json[0])
            else:
                st.json(p_json)

with tab2:
    st.subheader("Historical EOD (Full)")

    h_code, h_ms, h_text = fmp_historical_eod_full(symbol, api_key)
    st.write(f"HTTP: {h_code} | Latency: {h_ms} ms")

    h_json, h_err = try_json(h_text)
    if h_code != 200:
        st.error("Historical request failed")
        st.code(h_text[:2000], language="json")
        st.stop()
    if h_err:
        st.warning(f"Historical JSON parse error: {h_err}")
        st.code(h_text[:2000], language="text")
        st.stop()

    df = parse_historical_eod(h_json)
    if df.empty:
        st.error("Historical payload parsed but no rows found (structure changed or empty data).")
        st.json(h_json if isinstance(h_json, dict) else {"data_type": str(type(h_json))})
        st.stop()

    st.caption(f"Rows: {len(df):,} | Range: {df.index.min().date()} → {df.index.max().date()}")

    # Show table sample
    st.dataframe(df.tail(50))

    # Download CSV
    csv = df.reset_index().to_csv(index=False)
    st.download_button(
        "⬇️ Download historical CSV",
        data=csv,
        file_name=f"{symbol}_historical_eod.csv",
        mime="text/csv",
    )

    # Simple price chart
    if "close" in df.columns:
        fig = plt.figure()
        plt.plot(df.index, df["close"])
        plt.title(f"{symbol} Close (EOD)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(fig)

with tab3:
    st.subheader("Analytics")

    # Reuse historical from tab2 by refetching quickly to keep code simple/robust
    # (Streamlit rerun model: data not persisted across tabs unless cached/session stored)
    h_code, h_ms, h_text = fmp_historical_eod_full(symbol, api_key)
    h_json, h_err = try_json(h_text)
    if h_code != 200 or h_err:
        st.error("Cannot analyze because historical data fetch/parse failed.")
        st.code(h_text[:2000], language="json")
        st.stop()

    df = parse_historical_eod(h_json)
    if df.empty or "close" not in df.columns:
        st.error("No usable close series found for analytics.")
        st.stop()

    close = df["close"].dropna()
    daily_ret = close.pct_change()
    equity = (1 + daily_ret.fillna(0)).cumprod()

    vol_ann = float(daily_ret.dropna().std(ddof=1) * np.sqrt(252)) if daily_ret.dropna().size > 1 else None
    mdd = max_drawdown(equity)
    cagr_val = cagr(equity, equity.index.to_series())
    sharpe_val = sharpe_simple(daily_ret)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", "-" if cagr_val is None else f"{cagr_val*100:.2f}%")
    c2.metric("Vol (ann.)", "-" if vol_ann is None else f"{vol_ann*100:.2f}%")
    c3.metric("Max Drawdown", f"{mdd*100:.2f}%")
    c4.metric("Sharpe (simple)", "-" if sharpe_val is None else f"{sharpe_val:.2f}")

    # Indicators
    ind = pd.DataFrame(index=close.index)
    ind["close"] = close
    ind["SMA20"] = close.rolling(20).mean()
    ind["SMA50"] = close.rolling(50).mean()
    ind["EMA20"] = close.ewm(span=20, adjust=False).mean()
    ind["RSI14"] = compute_rsi(close, 14)

    st.dataframe(ind.tail(30))

    # Price + MAs chart
    fig1 = plt.figure()
    plt.plot(ind.index, ind["close"], label="Close")
    plt.plot(ind.index, ind["SMA20"], label="SMA20")
    plt.plot(ind.index, ind["SMA50"], label="SMA50")
    plt.title(f"{symbol} — Close with SMA20/50")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig1)

    # RSI chart
    fig2 = plt.figure()
    plt.plot(ind.index, ind["RSI14"], label="RSI14")
    plt.axhline(70)
    plt.axhline(30)
    plt.title(f"{symbol} — RSI14")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    st.pyplot(fig2)

    # Export analytics
    out = ind.reset_index().rename(columns={"index": "date"})
    st.download_button(
        "⬇️ Download indicators CSV",
        data=out.to_csv(index=False),
        file_name=f"{symbol}_indicators.csv",
        mime="text/csv",
    )
