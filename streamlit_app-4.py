"""
SMC Sniper — Liquidity Sweep -> Order Block -> FVG screener
NYSE / NASDAQ · daily & intraday timeframes · Streamlit dashboard
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from smc_engine import detect_setup
from universe import get_all_tickers

st.set_page_config(page_title="SMC Sniper — Sweep · OB · FVG",
                   page_icon="🎯", layout="wide")

# ----------------------------------------------------------------------------- 
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Archivo:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Archivo', sans-serif; }
h1, h2, h3 { font-family: 'Archivo', sans-serif; font-weight: 800; letter-spacing: -0.02em; }
[data-testid="stMetricValue"], code, .mono { font-family: 'IBM Plex Mono', monospace; }

.badge {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
  padding: 2px 10px; border-radius: 3px; letter-spacing: 0.06em;
}
.badge-entry   { background:#0e3a2f; color:#3ddc97; border:1px solid #3ddc97; }
.badge-wait    { background:#2b2b1a; color:#e8c547; border:1px solid #e8c547; }
.badge-trig    { background:#1a2233; color:#7aa2f7; border:1px solid #7aa2f7; }
.badge-dead    { background:#331a1a; color:#f77a7a; border:1px solid #f77a7a; }

.story { display:flex; gap:0; margin: 4px 0 14px 0; }
.story .step {
  flex:1; text-align:center; padding:7px 4px; font-family:'IBM Plex Mono',monospace;
  font-size:0.68rem; letter-spacing:0.04em; border-top:2px solid #333;
  color:#666;
}
.story .step.done { border-top:2px solid #3ddc97; color:#3ddc97; }
.story .step.live { border-top:2px solid #e8c547; color:#e8c547; animation: pulse 1.6s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.45;} }
</style>
""", unsafe_allow_html=True)

st.title("🎯 SMC Sniper")
st.caption("Liquidity sweep → order block → fair value gap · one entry model, "
           "five mechanical steps · NYSE + NASDAQ")

# -----------------------------------------------------------------------------
# Sidebar — scan configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Scan settings")

    src = st.radio("Universe", ["Yahoo screener (NYSE + NASDAQ)", "Paste tickers"])
    if src == "Paste tickers":
        raw = st.text_area("Tickers (comma / space / newline separated)",
                           "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AMD, JPM, XOM")
        max_tickers = None
        min_mcap = 5.0
    else:
        min_mcap = st.number_input("Min market cap ($B)", 1.0, 500.0, 5.0, 1.0)
        max_tickers = st.slider("Max tickers (by mcap, desc)", 25, 1500, 300, 25)
        raw = None

    tf = st.selectbox("Timeframe", ["1d", "1h", "4h (resampled from 1h)"])
    if tf == "1d":
        period, interval = st.selectbox("Lookback", ["6mo", "1y", "2y"], index=1), "1d"
    else:
        period, interval = "60d", "1h"

    st.divider()
    direction = st.selectbox("Direction", ["both", "long", "short"])
    swing_n = st.slider("Swing length (fractal n)", 2, 7, 3)
    bos_window = st.slider("Max bars: sweep → BOS", 5, 60, 25)
    min_disp = st.slider("Min displacement (× ATR)", 0.5, 3.0, 1.0, 0.25)
    max_age = st.slider("Max signal age (bars since BOS)", 5, 120, 40, 5)
    only_active = st.checkbox("Only actionable (in zone / awaiting)", value=True)

    run = st.button("🔍 Run scan", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_universe(min_mcap, max_tickers):
    return get_all_tickers(min_mcap_b=min_mcap, max_tickers=max_tickers,
                           log=lambda *_: None)


@st.cache_data(ttl=900, show_spinner=False)
def download_batch(tickers, period, interval):
    data = yf.download(tickers, period=period, interval=interval,
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    return data


def resample_4h(df):
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample("4h").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return out


def extract(data, t, multi):
    try:
        df = data[t].copy() if multi else data.copy()
    except (KeyError, TypeError):
        return None
    df = df.dropna(subset=["Close"])
    return df if len(df) > 60 else None


# -----------------------------------------------------------------------------
# Run scan
# -----------------------------------------------------------------------------
if run:
    if raw is not None:
        tickers = [t.strip().upper() for t in
                   raw.replace(",", " ").replace("\n", " ").split() if t.strip()]
    else:
        with st.spinner("Building universe from Yahoo screener…"):
            tickers = load_universe(min_mcap, max_tickers)

    if not tickers:
        st.error("No tickers to scan.")
        st.stop()

    st.session_state["tf_label"] = tf
    rows, charts = [], {}
    prog = st.progress(0.0, text=f"Downloading {len(tickers)} tickers…")

    CHUNK = 100
    n_done = 0
    for ci in range(0, len(tickers), CHUNK):
        chunk = tickers[ci:ci + CHUNK]
        data = download_batch(tuple(chunk), period, interval)
        multi = len(chunk) > 1
        for t in chunk:
            n_done += 1
            prog.progress(n_done / len(tickers),
                          text=f"Scanning {t} ({n_done}/{len(tickers)})")
            df = extract(data, t, multi)
            if df is None:
                continue
            if tf.startswith("4h"):
                df = resample_4h(df)
                if len(df) < 60:
                    continue
            try:
                setups = detect_setup(df, swing_n=swing_n, bos_window=bos_window,
                                      min_displacement_atr=min_disp,
                                      max_age_bars=max_age, direction=direction)
            except Exception:
                continue
            for s in setups:
                if only_active and s["status"] not in ("IN ZONE — ENTRY",
                                                       "AWAITING PULLBACK"):
                    continue
                rows.append({
                    "Ticker": t,
                    "Dir": "LONG" if s["direction"] == "long" else "SHORT",
                    "Status": s["status"],
                    "Zone": s["zone_type"],
                    "Zone top": round(s["zone_top"], 2),
                    "Zone bot": round(s["zone_bottom"], 2),
                    "Entry": round(s["entry"], 2),
                    "Stop": round(s["stop"], 2),
                    "Target (2R)": round(s["target"], 2),
                    "Risk %": round(s["risk_pct"], 2),
                    "Last": round(s["last_close"], 2),
                    "Dist to zone %": round(s["dist_to_zone_pct"], 2),
                    "Age (bars)": s["age_bars"],
                    "Sweep": str(s["sweep_date"])[:16],
                    "BOS": str(s["bos_date"])[:16],
                    "Outcome": s["outcome"] or "—",
                })
                charts[(t, s["direction"])] = (df, s)
    prog.empty()
    st.session_state["results"] = pd.DataFrame(rows)
    st.session_state["charts"] = charts

results = st.session_state.get("results")
charts = st.session_state.get("charts", {})

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_scan, tab_chart, tab_rules = st.tabs(["📡 Scanner", "📈 Chart", "📜 Playbook"])

STATUS_ORDER = {"IN ZONE — ENTRY": 0, "AWAITING PULLBACK": 1,
                "TRIGGERED": 2, "INVALIDATED": 3}

with tab_scan:
    if results is None:
        st.info("Configure the scan in the sidebar and hit **Run scan**.")
    elif results.empty:
        st.warning("No setups matched the filters. Loosen displacement / age, "
                   "or widen the universe.")
    else:
        r = results.copy()
        r["_o"] = r["Status"].map(STATUS_ORDER)
        r = r.sort_values(["_o", "Dist to zone %"]).drop(columns="_o")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Setups", len(r))
        c2.metric("In zone now", int((r["Status"] == "IN ZONE — ENTRY").sum()))
        c3.metric("Awaiting pullback", int((r["Status"] == "AWAITING PULLBACK").sum()))
        c4.metric("Longs / Shorts",
                  f'{int((r["Dir"]=="LONG").sum())} / {int((r["Dir"]=="SHORT").sum())}')

        st.dataframe(r, use_container_width=True, hide_index=True, height=560)
        st.download_button("⬇ Export CSV", r.to_csv(index=False),
                           "smc_signals.csv", "text/csv")

with tab_chart:
    if not charts:
        st.info("Run a scan first — every result becomes selectable here.")
    else:
        keys = sorted(charts.keys())
        labels = [f"{t} · {d.upper()}" for t, d in keys]
        sel = st.selectbox("Setup", labels)
        t, d = keys[labels.index(sel)]
        df, s = charts[(t, d)]

        # ---- narrative strip (the 5-step story) --------------------------------
        done = lambda cond: "done" if cond else ""
        live = "live" if s["status"] in ("AWAITING PULLBACK", "IN ZONE — ENTRY") else \
               ("done" if s["status"] == "TRIGGERED" else "")
        st.markdown(f"""
        <div class="story">
          <div class="step done">1 · STRUCTURE</div>
          <div class="step done">2 · SWEEP {str(s['sweep_date'])[:10]}</div>
          <div class="step done">3 · ORDER BLOCK</div>
          <div class="step {done(s['zone_type'] != 'OB')}">4 · FVG ({s['zone_type']})</div>
          <div class="step {live}">5 · RE-ENTRY — {s['status']}</div>
        </div>""", unsafe_allow_html=True)

        badge = {"IN ZONE — ENTRY": "badge-entry", "AWAITING PULLBACK": "badge-wait",
                 "TRIGGERED": "badge-trig", "INVALIDATED": "badge-dead"}[s["status"]]
        st.markdown(
            f'<span class="badge {badge}">{s["status"]}</span> &nbsp; '
            f'<span class="mono">entry {s["entry"]:.2f} · stop {s["stop"]:.2f} · '
            f'target {s["target"]:.2f} · risk {s["risk_pct"]:.2f}% · '
            f'outcome {s["outcome"] or "—"}</span>',
            unsafe_allow_html=True)

        # ---- chart --------------------------------------------------------------
        view = df.iloc[max(0, s["sweep_idx"] - 40):]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], name=t,
            increasing_line_color="#3ddc97", decreasing_line_color="#f77a7a"))

        x0, x1 = view.index[0], view.index[-1]
        sweep_x = df.index[s["sweep_idx"]]
        bos_x = df.index[s["bos_idx"]]

        # order block (gray)
        fig.add_shape(type="rect", x0=sweep_x, x1=x1,
                      y0=s["ob_bottom"], y1=s["ob_top"],
                      fillcolor="rgba(160,160,160,0.16)",
                      line=dict(color="rgba(160,160,160,0.5)", width=1))
        # fair value gaps in the leg (blue)
        for f in s["fvgs"]:
            fx = df.index[f["idx"]]
            fig.add_shape(type="rect", x0=fx, x1=x1,
                          y0=f["bottom"], y1=f["top"],
                          fillcolor="rgba(90,140,255,0.18)",
                          line=dict(color="rgba(90,140,255,0.55)", width=1))
        # swept liquidity level
        fig.add_shape(type="line", x0=x0, x1=sweep_x,
                      y0=s["sweep_level"], y1=s["sweep_level"],
                      line=dict(color="#e8c547", width=1, dash="dot"))
        fig.add_annotation(x=sweep_x, y=s["sweep_level"],
                           text="liquidity swept", showarrow=True, arrowhead=2,
                           font=dict(color="#e8c547", size=11), ay=30 if d == "long" else -30)
        # BOS level
        fig.add_shape(type="line", x0=x0, x1=bos_x,
                      y0=s["bos_level"], y1=s["bos_level"],
                      line=dict(color="#7aa2f7", width=1, dash="dash"))
        fig.add_annotation(x=bos_x, y=s["bos_level"], text="BOS",
                           font=dict(color="#7aa2f7", size=11), showarrow=False,
                           yshift=10)
        # entry / stop / target
        for y, col, lbl in [(s["entry"], "#3ddc97", "entry"),
                            (s["stop"], "#f77a7a", "stop"),
                            (s["target"], "#3ddc97", "2R target")]:
            fig.add_shape(type="line", x0=sweep_x, x1=x1, y0=y, y1=y,
                          line=dict(color=col, width=1))
            fig.add_annotation(x=x1, y=y, text=lbl, showarrow=False,
                               xanchor="left", font=dict(color=col, size=10))

        fig.update_layout(height=620, template="plotly_dark",
                          xaxis_rangeslider_visible=False,
                          margin=dict(l=10, r=70, t=20, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(14,14,16,1)",
                          font=dict(family="IBM Plex Mono"))
        if st.session_state.get("tf_label", "1d") != "1d":
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)

with tab_rules:
    st.markdown("""
### The entry model — one story, five steps

**The sweep is the trap. The order block is the zone. The FVG is the entry.**

**Step 1 — Structure.** Map swing highs and lows (fractal, n bars each side).
A close through the last swing high (long) or swing low (short) is the break
of structure that defines trend direction and the swing range in play.

**Step 2 — Liquidity sweep.** Price must wick through a prior swing low (long)
or swing high (short) — taking out resting stops — and reclaim the level within
a few bars. No sweep, no trade. Sit on your hands.

**Step 3 — Order block.** The origin of the displacement: the sweep candle's
full range (range method), marked high-to-low. This is where the reversal was
fuelled and where price is expected to gravitate back to. One method, applied
mechanically — never switched mid-stream.

**Step 4 — Fair value gap.** Inside the displacement leg, a bullish FVG exists
where `low[i+1] > high[i-1]` (bearish: `high[i+1] < low[i-1]`). The FVG refines
the entry inside the OB. Priority: **FVG inside the OB → nearest FVG → the OB
itself** if the leg left no imbalance.

**Step 5 — Re-entry.** Do nothing until price pulls back and *mitigates* the
zone. Aggressive: enter on first touch of the zone edge. Conservative: wait
for a lower-timeframe shift, then enter.

**Risk.** Stop below the zone (long) / above it (short) with a small ATR
buffer; if the zone is wide, refine to below the entry candle *only when it is
a protected low/high that itself swept liquidity*. Target fixed at **2R**.
Trade invalidated if the sweep extreme is taken out before mitigation.

**Filter, don't collect.** A sweep at a meaningless level, weak displacement,
or a random mid-consolidation candle marked as an OB is not a setup. The goal
is not the most setups — it's the cleanest ones.
    """)
