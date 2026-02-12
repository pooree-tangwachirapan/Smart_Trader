import json
import time
from typing import Any, Dict, Optional, Tuple, List

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = "https://financialmodelingprep.com/stable"
HTTP_TIMEOUT_SEC = 12  # fixed


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
    try:
        key = st.secrets["FMP_API_KEY"]
        if isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    return None


# ----------------------------
# FMP fetchers (stable)
# ----------------------------
def fmp_get(path: str, params: Dict[str, Any], api_key: str) -> Tuple[Optional[int], int, str]:
    url = f"{BASE}/{path.lstrip('/')}"
    p = dict(params)
    p["apikey"] = api_key
    return safe_get(url, p)


def parse_historical_eod(payload: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for k in ["historical", "data", "prices"]:
            if k in payload and isinstance(payload[k], list):
                rows = payload[k]
                break
        if not rows:
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows = v
                    break
    elif isinstance(payload, list):
        rows = payload

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="SmartTrader Lab — Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 SmartTrader Lab — Stock Analysis (FMP Stable)")

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

run = st.button("▶️ Fetch & Analyze", type="primary")
if not run:
    st.caption("กด Fetch & Analyze เพื่อดึงข้อมูลและเริ่มวิเคราะห์")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Price (Day)", "FMP Data Catalog"])

# ----------------------------
# Overview: Quote + Profile (NO HTTP/LATENCY shown)
# ----------------------------
with tab1:
    st.subheader("Overview")

    q_code, q_ms, q_text = fmp_get("quote", {"symbol": symbol}, api_key)
    p_code, p_ms, p_text = fmp_get("profile", {"symbol": symbol}, api_key)

    # Quote
    st.markdown("### Quote (key fields)")
    q_json, q_err = try_json(q_text)
    if q_code != 200 or q_err:
        st.error("Quote fetch failed")
        st.code(q_text[:2000], language="json")
    else:
        obj = q_json[0] if isinstance(q_json, list) and q_json and isinstance(q_json[0], dict) else q_json
        if isinstance(obj, dict):
            # Show important fields if present (fallback to json)
            fields = ["symbol", "name", "price", "changesPercentage", "change", "dayLow", "dayHigh", "yearLow", "yearHigh", "volume", "avgVolume", "marketCap", "pe", "eps"]
            picked = {k: obj.get(k) for k in fields if k in obj}
            if picked:
                st.json(picked)
            else:
                st.json(obj)
        else:
            st.json(q_json)

    # Profile
    st.markdown("### Profile (key fields)")
    p_json, p_err = try_json(p_text)
    if p_code != 200 or p_err:
        st.error("Profile fetch failed")
        st.code(p_text[:2000], language="json")
    else:
        obj = p_json[0] if isinstance(p_json, list) and p_json and isinstance(p_json[0], dict) else p_json
        if isinstance(obj, dict):
            fields = ["symbol", "companyName", "industry", "sector", "country", "exchangeShortName", "website", "description", "ceo", "fullTimeEmployees", "mktCap", "beta", "ipoDate"]
            picked = {k: obj.get(k) for k in fields if k in obj}
            st.json(picked if picked else obj)
        else:
            st.json(p_json)

# ----------------------------
# Price (Day): Historical EOD + daily chart + indicators
# ----------------------------
with tab2:
    st.subheader("Price (Timeframe: Day / EOD)")

    h_code, h_ms, h_text = fmp_get("historical-price-eod/full", {"symbol": symbol}, api_key)
    h_json, h_err = try_json(h_text)

    if h_code != 200 or h_err:
        st.error("Historical EOD fetch failed")
        st.code(h_text[:2000], language="json")
        st.stop()

    df = parse_historical_eod(h_json)
    if df.empty or "close" not in df.columns:
        st.error("Historical payload parsed but no usable close series found.")
        st.json(h_json if isinstance(h_json, dict) else {"data_type": str(type(h_json))})
        st.stop()

    st.caption(f"Rows: {len(df):,} | Range: {df.index.min().date()} → {df.index.max().date()}")

    # Indicators for chart
    ind = pd.DataFrame(index=df.index)
    ind["Close"] = df["close"]
    ind["SMA20"] = df["close"].rolling(20).mean()
    ind["SMA50"] = df["close"].rolling(50).mean()
    ind["RSI14"] = compute_rsi(df["close"], 14)

    # Daily price chart
    fig1 = plt.figure()
    plt.plot(ind.index, ind["Close"], label="Close")
    plt.plot(ind.index, ind["SMA20"], label="SMA20")
    plt.plot(ind.index, ind["SMA50"], label="SMA50")
    plt.title(f"{symbol} — Daily Close (EOD)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig1)

    # RSI chart
    fig2 = plt.figure()
    plt.plot(ind.index, ind["RSI14"], label="RSI14")
    plt.axhline(70)
    plt.axhline(30)
    plt.title(f"{symbol} — RSI14 (Daily)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    st.pyplot(fig2)

    st.dataframe(df.tail(50))

    st.download_button(
        "⬇️ Download historical CSV",
        data=df.reset_index().to_csv(index=False),
        file_name=f"{symbol}_historical_eod.csv",
        mime="text/csv",
    )

# ----------------------------
# FMP Data Catalog: what we can pull + click to fetch + show keys
# ----------------------------
with tab3:
    st.subheader("FMP Data Catalog (Stocks)")

    st.markdown(
        "เลือก dataset แล้วกด Fetch เพื่อดู **โครงสร้าง field ทั้งหมด (keys)** + JSON ตัวอย่าง 1 ก้อน\n\n"
        "หมายเหตุ: ทั้งหมดนี้อิงจาก FMP **stable endpoints** ในเอกสารทางการ"
    )

    CATALOG = [
        # Market / price
        ("Quote", "quote", {"symbol": symbol}),
        ("Quote Short", "quote-short", {"symbol": symbol}),
        ("Historical EOD (Full)", "historical-price-eod/full", {"symbol": symbol}),
        # Financial statements
        ("Income Statement", "income-statement", {"symbol": symbol}),
        ("Balance Sheet", "balance-sheet-statement", {"symbol": symbol}),
        ("Cash Flow", "cashflow-statement", {"symbol": symbol}),
        # Corporate actions
        ("Dividends (Company)", "dividends-company", {"symbol": symbol}),
        # Growth (fundamentals growth)
        ("Financial Growth", "financial-growth", {"symbol": symbol}),
        # Company profile
        ("Profile", "profile", {"symbol": symbol}),
    ]

    name_to_item = {n: (path, params) for n, path, params in CATALOG}
    choice = st.selectbox("Choose dataset", list(name_to_item.keys()))
    fetch = st.button("📥 Fetch selected dataset")

    if fetch:
        path, params = name_to_item[choice]
        code, ms, text = fmp_get(path, params, api_key)
        data, err = try_json(text)

        if code != 200 or err:
            st.error(f"Fetch failed: {choice}")
            st.code(text[:3000], language="json")
        else:
            st.success(f"Fetched: {choice}")
            # show keys
            if isinstance(data, list) and data and isinstance(data[0], dict):
                st.caption(f"Returned list[{len(data)}] — keys of first object:")
                st.write(sorted(list(data[0].keys())))
                st.json(data[0])
            elif isinstance(data, dict):
                st.caption("Returned dict — keys:")
                st.write(sorted(list(data.keys())))
                st.json(data)
            else:
                st.json(data)

            st.download_button(
                f"⬇️ Download {choice} JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"{symbol}_{choice.replace(' ', '_').lower()}.json",
                mime="application/json",
            )
