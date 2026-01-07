import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Trader", page_icon="📈", layout="wide")

# ---- Helpers ----
def api_get_symbols():
    # Backend'de /symbols varsa oradan çek
    try:
        r = requests.get(f"{API_URL}/symbols", timeout=10)
        if r.status_code == 200:
            data = r.json().get("symbols", [])
            if data:
                return data
    except Exception:
        pass

    # Fallback (backend'e eklemediysen)
    return [
        {"symbol": "ASELS.IS", "name": "Aselsan"},
        {"symbol": "THYAO.IS", "name": "Türk Hava Yolları"},
        {"symbol": "GARAN.IS", "name": "Garanti BBVA"},
        {"symbol": "AKBNK.IS", "name": "Akbank"},
        {"symbol": "KCHOL.IS", "name": "Koç Holding"},
        {"symbol": "BIMAS.IS", "name": "BİM"},
        {"symbol": "EREGL.IS", "name": "Ereğli Demir Çelik"},
        {"symbol": "TUPRS.IS", "name": "Tüpraş"},
        {"symbol": "SISE.IS", "name": "Şişecam"},
        {"symbol": "FROTO.IS", "name": "Ford Otosan"},
    ]

def predict(symbol: str, start: str, window: int):
    payload = {"symbol": symbol, "start": start, "window": int(window)}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()

def fetch_ohlcv_for_chart(symbol: str, start: str):
    # Grafik için basitçe yfinance verisini backend yerine direkt çekmek istersen API genişletilir.
    # Şimdilik hızlı çözüm: rapor csv varsa onu göstereceğiz.
    # İstersen backend'e /ohlcv endpoint ekleyip buradan çekeriz.
    return None

def pill(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(99,102,241,0.25);
            font-size: 12px;">
            {text}
        </span>
        """,
        unsafe_allow_html=True
    )

# ---- Header ----
st.markdown(
    """
    <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:16px;">
      <div>
        <div style="font-size:32px; font-weight:800; line-height:1.1;">📈 AI Trader</div>
        <div style="opacity:0.75; margin-top:4px;">LSTM ile <b>Next Close</b> tahmini • FastAPI + Streamlit</div>
      </div>
      <div style="opacity:0.65; font-size:12px; text-align:right;">
        Local API: <code>127.0.0.1:8000</code><br/>
        Not: Önce modeli eğit: <code>python -m ml.train</code>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

symbols = api_get_symbols()
symbol_map = {f"{s['name']} — {s['symbol']}": s["symbol"] for s in symbols}
symbol_labels = list(symbol_map.keys())

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    selected_label = st.selectbox("Hisse seç", symbol_labels, index=0)
    symbol = symbol_map[selected_label]

    start = st.text_input("Start (YYYY-MM-DD)", value="2018-01-01")
    window = st.slider("Window (lookback)", min_value=20, max_value=200, value=60, step=5)

    st.markdown("---")
    st.markdown("### 🔥 Hızlı İşlemler")
    do_predict = st.button("🚀 Tahmin üret", use_container_width=True)
    do_multi = st.button("📊 10 hisseyi toplu tahmin et", use_container_width=True)

# ---- Main layout ----
tab1, tab2 = st.tabs(["🔮 Tek Hisse Tahmin", "📊 Toplu Tahmin (10 Hisse)"])

def render_prediction_view(result: dict, title: str):
    last_close = result["last_close"]
    pred = result["pred_next_close"]
    diff = result.get("diff", pred - last_close)
    pct = result.get("pct", (diff / last_close) * 100 if last_close else 0.0)

    # Trend pill
    if diff > 0:
        pill(f"⬆️ Pozitif beklenti • {pct:.2f}%")
    elif diff < 0:
        pill(f"⬇️ Negatif beklenti • {pct:.2f}%")
    else:
        pill("➖ Nötr")

    st.markdown(f"#### {title}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{last_close:.2f}")
    c2.metric("Pred Next Close", f"{pred:.2f}")
    c3.metric("Diff", f"{diff:.2f}")
    c4.metric("Pct", f"{pct:.2f}%")

    # Basit “mini-chart”: last vs pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=["Last Close", "Pred Next Close"],
        y=[last_close, pred],
        mode="lines+markers",
        name="Close"
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("İpucu: Daha zengin grafik için backend’e OHLCV endpoint ekleyebiliriz (candlestick + indikatörler).")

# ---- Tab 1: Single ----
with tab1:
    st.markdown("### 🎯 Seçili hisse")
    st.write(f"**{selected_label}**")

    if do_predict:
        with st.spinner("Tahmin üretiliyor..."):
            try:
                res = predict(symbol, start, window)
                # Diff/pct backend'den geliyor ama yine de garanti edelim
                res["diff"] = float(res.get("diff", res["pred_next_close"] - res["last_close"]))
                res["pct"] = float(res.get("pct", (res["diff"]/res["last_close"])*100 if res["last_close"] else 0.0))
                render_prediction_view(res, title="Tahmin Sonucu")
            except Exception as e:
                st.error(f"Hata: {e}")

    else:
        st.info("Soldaki **“🚀 Tahmin üret”** butonuna bas.")

# ---- Tab 2: Multi ----
with tab2:
    st.markdown("### 🧪 10 hisse için toplu tahmin")
    st.caption("Not: Her hisse için model dosyası yoksa 404 döner. (İstersen tek model + multi symbol eğitim akışı da yaparız.)")

    if do_multi:
        rows = []
        with st.spinner("Toplu tahmin üretiliyor..."):
            for s in symbols:
                sym = s["symbol"]
                name = s["name"]
                try:
                    res = predict(sym, start, window)
                    diff = res.get("diff", res["pred_next_close"] - res["last_close"])
                    pct = res.get("pct", (diff / res["last_close"]) * 100 if res["last_close"] else 0.0)
                    rows.append({
                        "Name": name,
                        "Symbol": sym,
                        "Last Close": round(res["last_close"], 2),
                        "Pred Next Close": round(res["pred_next_close"], 2),
                        "Diff": round(diff, 2),
                        "Pct %": round(pct, 2),
                        "Status": "OK"
                    })
                except Exception as e:
                    rows.append({
                        "Name": name,
                        "Symbol": sym,
                        "Last Close": None,
                        "Pred Next Close": None,
                        "Diff": None,
                        "Pct %": None,
                        "Status": f"ERR"
                    })

        df = pd.DataFrame(rows)
        # En yüksek pct üstte
        df_sorted = df.sort_values(by="Pct %", ascending=False, na_position="last")

        st.dataframe(df_sorted, use_container_width=True, hide_index=True)

        # Basit bar chart (Pct %)
        ok_df = df_sorted.dropna(subset=["Pct %"]).head(10)
        if not ok_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ok_df["Name"],
                y=ok_df["Pct %"],
                name="Pct %"
            ))
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_white",
                yaxis_title="Pct %",
                xaxis_title="Hisseler",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Soldaki **“📊 10 hisseyi toplu tahmin et”** butonuna bas.")
