import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st


BASE = "https://financialmodelingprep.com/stable"


def safe_get(url: str, params: Dict[str, Any], timeout: int = 12) -> Tuple[Optional[int], int, str]:
    start = time.perf_counter()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
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


st.set_page_config(page_title="FMP Stock Probe", page_icon="📈", layout="wide")
st.title("📈 FMP Stock Probe (AAPL) — What data do we get?")

with st.sidebar:
    symbol = st.text_input("Symbol", value="AAPL").strip().upper()
    timeout = st.number_input("Timeout (sec)", min_value=3, max_value=60, value=12, step=1)

    # Read key from Streamlit Secrets first, fallback to manual input
    key_from_secrets = st.secrets.get("FMP_API_KEY", "")
    api_key = st.text_input("FMP API Key (prefer Secrets)", value=key_from_secrets, type="password").strip()

run = st.button("▶️ Run FMP Tests", type="primary")

if run:
    if not api_key:
        st.error("ยังไม่มี FMP_API_KEY — ใส่ใน Streamlit Secrets หรือกรอกในช่องด้านซ้ายก่อน")
        st.stop()

    endpoints = [
        ("Quote", f"{BASE}/quote", {"symbol": symbol}),
        ("Quote Short", f"{BASE}/quote-short", {"symbol": symbol}),
        ("Profile", f"{BASE}/profile", {"symbol": symbol}),
        ("Historical EOD (Full)", f"{BASE}/historical-price-eod/full", {"symbol": symbol}),
    ]

    st.info(f"Testing {symbol} on FMP stable endpoints…")

    for name, url, params in endpoints:
        params = dict(params)
        params["apikey"] = api_key

        code, ms, text = safe_get(url, params=params, timeout=int(timeout))
        st.subheader(name)
        cols = st.columns(3)
        cols[0].metric("HTTP", "-" if code is None else str(code))
        cols[1].metric("Latency (ms)", str(ms))
        cols[2].metric("Endpoint", url)

        data, err = try_json(text)

        if code != 200:
            st.error("Request failed")
            st.code(text[:2000], language="json")
            continue

        if err:
            st.warning(f"JSON parse error: {err}")
            st.code(text[:2000], language="text")
            continue

        # Show keys / structure
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            st.caption(f"Returned: list[{len(data)}], showing first object keys")
            st.write(sorted(list(data[0].keys())))
            st.json(data[0])
            st.download_button(
                f"⬇️ Download {name} (JSON)",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"{symbol}_{name.replace(' ', '_').lower()}.json",
                mime="application/json",
            )
        elif isinstance(data, dict):
            st.caption("Returned: dict, showing keys")
            st.write(sorted(list(data.keys())))
            st.json(data)
            st.download_button(
                f"⬇️ Download {name} (JSON)",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"{symbol}_{name.replace(' ', '_').lower()}.json",
                mime="application/json",
            )
        else:
            st.json(data)
