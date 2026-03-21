# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import altair as alt
from datetime import date, timedelta

# Import helpers from your repo script
# Ensure build_liquid_optionable_universe.py exposes these functions or adjust names accordingly
from build_liquid_optionable_universe import (
    SEED_SYMBOLS,
    extract_symbol_df,
    tsi,
    cci,
    backtest_symbol,
)

st.set_page_config(page_title="TSI + CCI Dashboard", layout="wide")
st.title("TSI + CCI Exhaustion Dashboard")

# Defaults and checkpoint
DEFAULT_TICKERS = SEED_SYMBOLS if "SEED_SYMBOLS" in globals() else [
    "SPY","QQQ","IWM","DIA","XLF","XLK","SMH","XLE","TLT","GLD",
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","INTC","MU","PLTR"
]
CHECKPOINT_FILE = "liquidity_checkpoint.csv"

# -----------------------------
# Helper utilities
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_yf(tickers, start_date, end_date):
    import yfinance as yf
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return raw

@st.cache_data(ttl=3600)
def cached_backtest(symbol, raw, tsi_params, cci_len, tsi_pct_threshold, cci_state):
    df = extract_symbol_df(raw, symbol)
    return backtest_symbol(df, tsi_params, cci_len, tsi_pct_threshold, cci_state)

def safe_float(x):
    return float(x) if pd.notna(x) else np.nan

def make_altair_price_chart(df, symbol):
    dfc = df.reset_index().rename(columns={"index": "Date"})
    base = alt.Chart(dfc).encode(x="Date:T")
    price = base.mark_line(color="#1f77b4").encode(y=alt.Y("Close:Q", title="Price"))
    tsi_line = base.mark_line(color="#ff7f0e").encode(y=alt.Y("TSI:Q", title="TSI"))
    tsi_signal = base.mark_line(color="#2ca02c").encode(y="TSI_SIGNAL:Q")
    cci_line = base.mark_line(color="#9467bd").encode(y="CCI:Q")
    price_chart = price.properties(height=300, width=700, title=f"{symbol} Price")
    indicator_chart = alt.layer(tsi_line, tsi_signal, cci_line).properties(height=200, width=700, title=f"{symbol} Indicators")
    return price_chart & indicator_chart

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area("Tickers (comma-separated)", value=",".join(DEFAULT_TICKERS), height=150)
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

    st.subheader("TSI")
    tsi_choice = st.selectbox("TSI combo", options=["4,2,4", "6,3,6", "7,4,7"], index=0)
    tsi_params = tuple(int(x) for x in tsi_choice.split(","))

    st.subheader("CCI")
    cci_len = st.selectbox("CCI length", options=[5, 7, 10, 14], index=1)
    cci_state = st.selectbox("CCI state", options=["down_1d", "down_2d", "up_1d", "any"], index=0)

    st.subheader("Signal rule")
    tsi_pct_threshold = st.slider("TSI percentile threshold", 85, 99, 97)
    years = st.slider("Years of history", 2, 10, 5)
    min_signals = st.slider("Minimum historical signals", 3, 50, 8)

    st.subheader("Run options")
    batch_size = st.slider("Batch size for downloads", 1, 25, 10)
    delay = st.slider("Delay between symbols (s)", 0, 5, 1)
    run_btn = st.button("Run dashboard", type="primary")
    if os.path.exists(CHECKPOINT_FILE):
        if st.button("Load checkpoint"):
            df_ck = pd.read_csv(CHECKPOINT_FILE)
            st.sidebar.write(f"Loaded checkpoint with {len(df_ck)} rows")
        if st.button("Clear checkpoint"):
            os.remove(CHECKPOINT_FILE)
            st.sidebar.success("Checkpoint removed")
    else:
        df_ck = None

# -----------------------------
# Main run logic
# -----------------------------
if run_btn:
    end_date = date.today() + timedelta(days=1)
    start_date = date.today() - timedelta(days=int(365.25 * years))

    # Download in batches to avoid memory spikes
    results_rows = []
    charts = {}
    total = len(tickers)
    progress = st.progress(0)
    status = st.empty()

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        status.info(f"Downloading batch {i//batch_size + 1} of {((total-1)//batch_size)+1}")
        raw_batch = load_data_yf(batch, start_date, end_date)
        for j, sym in enumerate(batch, start=i+1):
            status.text(f"Processing {sym} ({j}/{total})")
            try:
                df = extract_symbol_df(raw_batch, sym)
                res = backtest_symbol(
                    df=df,
                    tsi_params=tsi_params,
                    cci_len=cci_len,
                    tsi_pct_threshold=tsi_pct_threshold,
                    cci_state=cci_state,
                )
            except Exception as e:
                res = {"symbol": sym, "error": str(e)}
            if res is None:
                continue

            # Ensure consistent types and safe floats
            res_safe = {}
            for k, v in res.items():
                if k in ("latest_tsi", "latest_tsi_pct", "latest_cci", "latest_cci_delta", "avg_1d_return", "avg_2d_return", "next_day_red_pct", "two_day_red_pct"):
                    res_safe[k] = safe_float(v)
                else:
                    res_safe[k] = v

            # Save chart df separately
            chart_df = res_safe.pop("chart_df", None)
            if chart_df is not None and not chart_df.empty:
                charts[sym] = chart_df

            row = {"symbol": sym, **res_safe}
            results_rows.append(row)

            # Append to checkpoint immediately
            df_row = pd.DataFrame([row])
            if os.path.exists(CHECKPOINT_FILE):
                df_row.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)
            else:
                df_row.to_csv(CHECKPOINT_FILE, index=False)

            progress.progress(int(j/total*100))
            time.sleep(delay)

    # Build results DataFrame
    if not results_rows:
        st.warning("No usable data returned. Try fewer tickers or rerun.")
    else:
        results = pd.DataFrame(results_rows)
        results = results[results["signals"].fillna(0) >= min_signals].copy()

        if results.empty:
            st.warning("No symbols met the minimum signal count. Lower the minimum signals filter.")
        else:
            # Compute score and sort
            results["score"] = (
                results["next_day_red_pct"].fillna(0) * 0.55
                + (-results["avg_1d_return"].fillna(0)) * 15 * 0.25
                + results["signals"].clip(upper=50) * 0.20
            )
            results = results.sort_values(
                ["latest_signal", "score", "next_day_red_pct", "signals"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Symbols passing", len(results))
            c2.metric("Live signals now", int(results["latest_signal"].sum()))
            c3.metric("Best next-day red %", f"{results['next_day_red_pct'].max():.1f}%")
            c4.metric("Best avg 1D return", f"{results['avg_1d_return'].max():.2f}%")

            st.subheader("Ranked results")
            display_cols = [
                "symbol", "latest_signal", "latest_tsi", "latest_tsi_pct", "latest_cci",
                "latest_cci_delta", "signals", "next_day_red_pct", "two_day_red_pct",
                "avg_1d_return", "avg_2d_return", "score"
            ]
            st.dataframe(results[display_cols].fillna(""), use_container_width=True, hide_index=True)

            st.subheader("Chart review")
            selected = st.selectbox("Select symbol", options=results["symbol"].tolist())
            chart_df = charts.get(selected)
            if chart_df is not None:
                chart_df = chart_df.copy()
                chart_df["TSI"], chart_df["TSI_SIGNAL"] = tsi(chart_df["Close"], *tsi_params)
                chart_df["CCI"] = cci(chart_df["High"], chart_df["Low"], chart_df["Close"], cci_len)
                st.altair_chart(make_altair_price_chart(chart_df, selected), use_container_width=True)

            # Downloads
            st.download_button("Download CSV", results.to_csv(index=False), "liquidity_snapshot.csv")
            try:
                st.download_button("Download Parquet", results.to_parquet(index=False), "liquidity_snapshot.parquet")
            except Exception:
                pass
else:
    st.info("Choose your tickers and settings in the sidebar, then click 'Run dashboard'.")
