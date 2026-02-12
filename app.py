# app.py — SmartTrader Lab (Stocks + FMP Data Catalog + Options Planner)
# ใช้ FMP_API_KEY จาก Streamlit Secrets เท่านั้น (ห้ามใส่ key ในโค้ด/Repo)

import json
import time
import math
from typing import Any, Dict, Optional, Tuple, List

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
BASE = "https://financialmodelingprep.com/stable"
HTTP_TIMEOUT_SEC = 12


# =========================
# Helpers: HTTP / JSON
# =========================
def safe_get(url: str, params: Dict[str, Any]) -> Tuple[Optional[int], int, str]:
    start = time.perf_counter()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
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


def fmp_get(path: str, params: Dict[str, Any], api_key: str) -> Tuple[Optional[int], int, str]:
    url = f"{BASE}/{path.lstrip('/')}"
    p = dict(params)
    p["apikey"] = api_key
    return safe_get(url, p)


# =========================
# Helpers: data parsing / indicators
# =========================
def parse_historical_eod(payload: Any) -> pd.DataFrame:
    """
    พยายาม parse ให้ robust:
    - ถ้า payload เป็น dict: หา list ใต้คีย์ common เช่น historical/data/prices
    - ถ้าเป็น list ก็ใช้ตรงๆ
    """
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


# =========================
# Options: Black–Scholes (European Call) + solver
# =========================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    # S, K: dollars, T: years, r/q/sigma: decimals
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        fwd = S * math.exp((r - q) * T)
        return max(fwd - K, 0.0) * math.exp(-r * T)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def solve_for_underlying_given_call(
    target_call: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    S_low: float,
    S_high: float,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> float:
    """
    Bisection: หา S ที่ทำให้ BS_call(S) = target_call
    ต้อง bracket root ให้ได้ (f(lo) กับ f(hi) คนละเครื่องหมาย)
    """
    def f(S: float) -> float:
        return bs_call_price(S, K, T, r, q, sigma) - target_call

    lo, hi = S_low, S_high
    flo, fhi = f(lo), f(hi)

    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("Root not bracketed: เพิ่มช่วง S_low/S_high ให้กว้างขึ้น")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def infer_linear_slope(S1: float, C1: float, S2: float, C2: float) -> float:
    if S2 == S1:
        return float("nan")
    return (C2 - C1) / (S2 - S1)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SmartTrader Lab — Stocks", page_icon="📈", layout="wide")
st.title("📈 SmartTrader Lab — Stocks (FMP Stable + Data Catalog + Options Planner)")

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
    st.caption("ใช้ Secrets เท่านั้น (ไม่รับ/ไม่แสดง API Key ใน UI)")

# --- FIX: จำสถานะ ready เพื่อไม่เด้งกลับเมื่อกดปุ่มอื่น ---
if "ready" not in st.session_state:
    st.session_state["ready"] = False

if st.button("▶️ Fetch & Analyze", type="primary", key="btn_ready"):
    st.session_state["ready"] = True

if not st.session_state["ready"]:
    st.caption("กด Fetch & Analyze เพื่อเริ่ม (หลังจากนั้นกดปุ่มอื่นได้โดยไม่เด้งกลับ)")
    st.stop()

# --- เก็บผลล่าสุดของ Data Catalog ---
if "catalog_last" not in st.session_state:
    st.session_state["catalog_last"] = None

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Price (Day)", "FMP Data Catalog", "Options Planner"])


# =========================
# Tab 1: Overview (Quote/Profile) — ไม่โชว์ HTTP/Latency
# =========================
with tab1:
    st.subheader("Overview")

    q_code, q_ms, q_text = fmp_get("quote", {"symbol": symbol}, api_key)
    p_code, p_ms, p_text = fmp_get("profile", {"symbol": symbol}, api_key)

    # Quote
    st.markdown("### Quote (key fields)")
    q_json, q_err = try_json(q_text)
    if q_code != 200 or q_err:
        st.error("Quote fetch failed")
        st.code(q_text[:2500], language="json")
    else:
        obj = q_json[0] if isinstance(q_json, list) and q_json and isinstance(q_json[0], dict) else q_json
        if isinstance(obj, dict):
            fields = [
                "symbol", "name", "price", "changesPercentage", "change",
                "dayLow", "dayHigh", "yearLow", "yearHigh",
                "volume", "avgVolume", "marketCap", "pe", "eps"
            ]
            picked = {k: obj.get(k) for k in fields if k in obj}
            st.json(picked if picked else obj)
        else:
            st.json(q_json)

    # Profile
    st.markdown("### Profile (key fields)")
    p_json, p_err = try_json(p_text)
    if p_code != 200 or p_err:
        st.error("Profile fetch failed")
        st.code(p_text[:2500], language="json")
    else:
        obj = p_json[0] if isinstance(p_json, list) and p_json and isinstance(p_json[0], dict) else p_json
        if isinstance(obj, dict):
            fields = [
                "symbol", "companyName", "industry", "sector", "country",
                "exchangeShortName", "website", "ceo", "fullTimeEmployees",
                "mktCap", "beta", "ipoDate", "description"
            ]
            picked = {k: obj.get(k) for k in fields if k in obj}
            st.json(picked if picked else obj)
        else:
            st.json(p_json)


# =========================
# Tab 2: Price (Day) — Daily/EOD + กราฟ
# =========================
with tab2:
    st.subheader("Price (Timeframe: Day / EOD)")

    h_code, h_ms, h_text = fmp_get("historical-price-eod/full", {"symbol": symbol}, api_key)
    h_json, h_err = try_json(h_text)
    if h_code != 200 or h_err:
        st.error("Historical EOD fetch failed")
        st.code(h_text[:2500], language="json")
        st.stop()

    df = parse_historical_eod(h_json)
    if df.empty or "close" not in df.columns:
        st.error("Historical parsed แต่ไม่พบ series 'close' ที่ใช้งานได้")
        st.json(h_json if isinstance(h_json, dict) else {"type": str(type(h_json))})
        st.stop()

    st.caption(f"Rows: {len(df):,} | Range: {df.index.min().date()} → {df.index.max().date()}")

    ind = pd.DataFrame(index=df.index)
    ind["Close"] = df["close"]
    ind["SMA20"] = df["close"].rolling(20).mean()
    ind["SMA50"] = df["close"].rolling(50).mean()
    ind["RSI14"] = compute_rsi(df["close"], 14)

    fig1 = plt.figure()
    plt.plot(ind.index, ind["Close"], label="Close")
    plt.plot(ind.index, ind["SMA20"], label="SMA20")
    plt.plot(ind.index, ind["SMA50"], label="SMA50")
    plt.title(f"{symbol} — Daily Close (EOD)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(ind.index, ind["RSI14"], label="RSI14")
    plt.axhline(70)
    plt.axhline(30)
    plt.title(f"{symbol} — RSI14 (Daily)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    st.pyplot(fig2)

    st.dataframe(df.tail(60))

    st.download_button(
        "⬇️ Download historical CSV",
        data=df.reset_index().to_csv(index=False),
        file_name=f"{symbol}_historical_eod.csv",
        mime="text/csv",
        key="dl_hist_csv",
    )


# =========================
# Tab 3: FMP Data Catalog — กด Fetch แล้วไม่เด้งกลับ, แสดงผลล่าสุด
# =========================
with tab3:
    st.subheader("FMP Data Catalog (Stocks)")

    st.markdown(
        "เลือก dataset แล้วกด **Fetch** เพื่อดู **keys/โครงสร้างข้อมูล** + JSON ตัวอย่าง + ดาวน์โหลด JSON\n\n"
        "หมายเหตุ: ใช้ FMP **stable endpoints**"
    )

    CATALOG = [
        # Market / price
        ("Quote", "quote", {"symbol": symbol}),
        ("Quote Short", "quote-short", {"symbol": symbol}),
        ("Historical EOD (Full)", "historical-price-eod/full", {"symbol": symbol}),
        # Company profile
        ("Profile", "profile", {"symbol": symbol}),
        # Financial statements
        ("Income Statement", "income-statement", {"symbol": symbol}),
        ("Balance Sheet", "balance-sheet-statement", {"symbol": symbol}),
        ("Cash Flow", "cashflow-statement", {"symbol": symbol}),
        # Growth / corporate actions
        ("Financial Growth", "financial-growth", {"symbol": symbol}),
        ("Dividends (Company)", "dividends-company", {"symbol": symbol}),
    ]

    name_to_item = {n: (path, params) for n, path, params in CATALOG}

    choice = st.selectbox("Choose dataset", list(name_to_item.keys()), key="catalog_choice")
    fetch = st.button("📥 Fetch selected dataset", key="fetch_catalog")

    if fetch:
        path, params = name_to_item[choice]
        code, ms, text = fmp_get(path, params, api_key)
        data, err = try_json(text)
        st.session_state["catalog_last"] = {
            "choice": choice,
            "path": path,
            "code": code,
            "ms": ms,
            "text": text,
            "data": data,
            "err": err,
        }

    last = st.session_state.get("catalog_last")
    if last:
        st.divider()
        st.subheader(f"Latest result: {last['choice']}")

        # (คุณไม่ได้ขอให้เอา HTTP/Latency ออกสำหรับ catalog — เลยแสดงไว้ช่วย debug)
        st.caption(f"HTTP: {last['code']} | Latency: {last['ms']} ms | Path: {last['path']}")

        if last["code"] != 200 or last["err"]:
            st.error("Fetch failed")
            st.code((last["text"] or "")[:3000], language="json")
        else:
            data = last["data"]
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
                f"⬇️ Download {last['choice']} JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"{symbol}_{last['choice'].replace(' ', '_').lower()}.json",
                mime="application/json",
                key="dl_catalog_json",
            )
    else:
        st.caption("ยังไม่มีผลลัพธ์ — เลือก dataset แล้วกด Fetch")


# =========================
# Tab 4: Options Planner — หาราคาหุ้นล่วงหน้าให้พรีเมียมถึงเป้า
# =========================
with tab4:
    st.subheader("Options Planner (Call) — คำนวณเพื่อวาง Limit Order ล่วงหน้า")

    st.caption(
        "คำนวณแบบ Black–Scholes (ราคาทฤษฎี) และแก้สมการกลับหา 'ราคาหุ้น' ที่ทำให้ Call premium ถึงเป้าหมาย\n"
        "ข้อจำกัด: ตลาดจริงมี bid/ask, IV เปลี่ยน, liquidity ฯลฯ (ผลไม่ใช่การันตีการ fill)"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        S_now = st.number_input("Underlying now (S)", value=275.0, step=0.5)
        K = st.number_input("Strike (K)", value=285.0, step=0.5)
        days = st.number_input("Days to expiry", value=30, step=1, min_value=1)

    with c2:
        r = st.number_input("Risk-free rate r (annual, %)", value=4.5, step=0.1) / 100.0
        q = st.number_input("Dividend yield q (annual, %)", value=0.5, step=0.1) / 100.0
        sigma = st.number_input("Implied Vol σ (annual, %)", value=35.0, step=0.5) / 100.0

    with c3:
        target_call = st.number_input("Target Call Premium ($/share)", value=2.00, step=0.05)
        st.write(f"≈ {target_call*100:.0f} USD / contract (x100)")

    T = float(days) / 365.0

    st.divider()
    st.markdown("### 1) ราคา Call ทฤษฎี ณ S ปัจจุบัน")
    theo = bs_call_price(S_now, K, T, r, q, sigma)
    st.metric("Theoretical Call ($/share)", f"{theo:.4f}")
    st.write(f"≈ {theo*100:.2f} USD / contract")

    st.divider()
    st.markdown("### 2) หา S* ที่ทำให้ Call = เป้าหมาย (เพื่อวาง Limit รอ)")
    S_low = max(0.01, S_now * 0.3)
    S_high = S_now * 3.0

    if st.button("Solve underlying price for target premium", key="solve_underlying"):
        try:
            S_star = solve_for_underlying_given_call(
                target_call=target_call,
                K=K,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                S_low=S_low,
                S_high=S_high,
            )
            st.success(f"S* ≈ {S_star:.4f} เพื่อให้ Call ≈ {target_call:.4f} ($/share)")
            st.write(f"≈ {target_call*100:.2f} USD / contract")
        except Exception as e:
            st.error(f"Solve failed: {e}")

    st.divider()
    st.markdown("### 3) โหมดประมาณแบบ 2 จุด (ตามตัวอย่างที่คุณให้)")
    st.caption("ตัวอย่าง: S=275 → Call=$2.00 และ S=285 → Call=$3.00 (หน่วย $/share)")
    ex1 = st.number_input("Example S1", value=275.0, step=0.5)
    exC1 = st.number_input("Example Call1 ($/share)", value=2.00, step=0.05)
    ex2 = st.number_input("Example S2", value=285.0, step=0.5)
    exC2 = st.number_input("Example Call2 ($/share)", value=3.00, step=0.05)

    slope = infer_linear_slope(ex1, exC1, ex2, exC2)
    if math.isnan(slope) or math.isinf(slope):
        st.warning("คำนวณ slope ไม่ได้ (S1 กับ S2 เท่ากัน)")
    else:
        st.write(f"Linear slope ≈ {slope:.4f} $ option / $ underlying")
        # ตัวอย่างคำนวณกลับแบบเส้นตรง: ถ้าอยากให้ call ถึง target_call ต้องให้ S ประมาณเท่าไร
        est_S = ex1 + (target_call - exC1) / slope if slope != 0 else float("nan")
        if not (math.isnan(est_S) or math.isinf(est_S)):
            st.write(f"Linear estimate S* ≈ {est_S:.4f} (approx)")
        st.caption("หมายเหตุ: เส้นตรงเป็น approximation (ไม่รวมความโค้ง/IV change)")
