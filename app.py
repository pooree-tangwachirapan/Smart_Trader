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

# แสดงผล “ล่าสุด” ถ้ามี
last = st.session_state["catalog_last"]
if last:
    st.divider()
    st.subheader(f"Latest result: {last['choice']}")
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
            key="dl_catalog",
        )
