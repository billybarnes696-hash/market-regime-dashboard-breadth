# app.py
"""
Streamlit app: TSI + CCI + ADX scanner, fast sweet-spot grid, persistence, and live monitor.
Optimizations included:
 - batched yfinance downloads
 - rank-based rolling percentile (faster than python lambdas)
 - per-symbol parallel sweet-spot computation (optional)
 - caching of downloads and per-symbol backtests
 - checkpointing and Parquet persistence for sweet spots
 - lightweight live monitor that reads persisted sweet spots
Save this file as app.py next to your environment and run with `streamlit run app.py`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from datetime import date, timedelta, datetime
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Dict, Tuple

# -----------------------------
# Config / defaults
# -----------------------------
st.set_page_config(page_title="TSI + CCI + ADX Dashboard (fast)", layout="wide")
DEFAULT_TICKERS = [
    "SPY","QQQ","IWM","DIA","XLF","XLK","SMH","XLE","TLT","GLD",
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","INTC","MU","PLTR"
]
SWEET_SPOTS_PATH = "sweet_spots.parquet"
RAW_CACHE_DIR = "raw_history"  # optional per-symbol raw history cache
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

# -----------------------------
# Indicator functions
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def tsi(close: pd.Series, long_len: int, short_len: int, signal_len: int):
    mtm = close.diff()
    abs_mtm = mtm.abs()
    double_smoothed_mtm = ema(ema(mtm, long_len), short_len)
    double_smoothed_abs = ema(ema(abs_mtm, long_len), short_len)
    tsi_val = 100 * (double_smoothed_mtm / double_smoothed_abs.replace(0, np.nan))
    tsi_signal = ema(tsi_val, signal_len)
    return tsi_val, tsi_signal

def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * (plus_dm_sm / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_sm / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx_val, plus_di, minus_di

# -----------------------------
# Data loading (batched + cached)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def batch_download_yf(tickers: Tuple[str], start_date: date, end_date: date):
    """Download a batch of tickers via yfinance. Returns raw DataFrame (may be MultiIndex)."""
    try:
        raw = yf.download(
            tickers=list(tickers),
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        return raw
    except Exception as e:
        raise RuntimeError(f"yfinance download failed for batch {tickers[:3]}...: {e}")

def extract_symbol_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize raw yfinance output to a tidy DataFrame for a single symbol."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        df = raw[symbol].copy()
    else:
        df = raw.copy()
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    needed = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[needed].dropna().copy()
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# -----------------------------
# Backtest / signal logic (fast percentile)
# -----------------------------
def rolling_tsi_pct(series: pd.Series, window: int = 252, min_periods: int = 60):
    """Compute rolling percentile using rank-based approach (fast)."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

def apply_states(df: pd.DataFrame, cci_state: str, adx_state: str):
    if cci_state == "down_1d":
        cond_cci = df["CCI_DELTA_1"] < 0
    elif cci_state == "down_2d":
        cond_cci = (df["CCI_DELTA_1"] < 0) & (df["CCI_DELTA_2"] < 0)
    elif cci_state == "up_1d":
        cond_cci = df["CCI_DELTA_1"] > 0
    else:
        cond_cci = pd.Series(True, index=df.index)

    if adx_state == "down_1d":
        cond_adx = df["ADX_DELTA_1"] < 0
    elif adx_state == "flat_or_down":
        cond_adx = df["ADX_DELTA_1"] <= 0
    elif adx_state == "up_1d":
        cond_adx = df["ADX_DELTA_1"] > 0
    else:
        cond_adx = pd.Series(True, index=df.index)

    return cond_cci, cond_adx

def backtest_symbol(df: pd.DataFrame, tsi_params, cci_len, adx_len, tsi_pct_threshold, cci_state, adx_state):
    """Return a dict of metrics for a single symbol and parameter combo. Fast and robust."""
    if df.empty or len(df) < 260:
        return None

    t_long, t_short, t_signal = tsi_params
    out = df.copy()
    out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], t_long, t_short, t_signal)
    out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
    out["ADX"], out["DI_PLUS"], out["DI_MINUS"] = adx(out["High"], out["Low"], out["Close"], adx_len)
    out["RET_1D"] = out["Close"].shift(-1) / out["Close"] - 1
    out["RET_2D"] = out["Close"].shift(-2) / out["Close"] - 1

    # faster percentile
    out["TSI_PCT"] = rolling_tsi_pct(out["TSI"], window=252, min_periods=60)

    out["CCI_DELTA_1"] = out["CCI"].diff()
    out["CCI_DELTA_2"] = out["CCI"].diff(2)
    out["ADX_DELTA_1"] = out["ADX"].diff()

    cond_cci, cond_adx = apply_states(out, cci_state, adx_state)
    signal = (out["TSI_PCT"] >= tsi_pct_threshold) & cond_cci & cond_adx
    signal = signal.fillna(False)
    hits = out[signal].copy()
    hits = hits.dropna(subset=["RET_1D", "RET_2D", "TSI", "CCI", "ADX", "TSI_PCT"])

    latest_signal = bool(signal.iloc[-1]) if len(signal) else False

    base = {
        "latest_signal": latest_signal,
        "latest_tsi": float(out["TSI"].iloc[-1]) if pd.notna(out["TSI"].iloc[-1]) else np.nan,
        "latest_tsi_pct": float(out["TSI_PCT"].iloc[-1]) if pd.notna(out["TSI_PCT"].iloc[-1]) else np.nan,
        "latest_cci": float(out["CCI"].iloc[-1]) if pd.notna(out["CCI"].iloc[-1]) else np.nan,
        "latest_cci_delta": float(out["CCI_DELTA_1"].iloc[-1]) if pd.notna(out["CCI_DELTA_1"].iloc[-1]) else np.nan,
        "latest_adx": float(out["ADX"].iloc[-1]) if pd.notna(out["ADX"].iloc[-1]) else np.nan,
        "latest_adx_delta": float(out["ADX_DELTA_1"].iloc[-1]) if pd.notna(out["ADX_DELTA_1"].iloc[-1]) else np.nan,
        "chart_df": out.tail(180).copy(),
    }

    if hits.empty:
        return {
            "signals": 0,
            "next_day_red_pct": np.nan,
            "two_day_red_pct": np.nan,
            "avg_1d_return": np.nan,
            "avg_2d_return": np.nan,
            **base,
        }

    return {
        "signals": int(len(hits)),
        "next_day_red_pct": float((hits["RET_1D"] < 0).mean() * 100),
        "two_day_red_pct": float((hits["RET_2D"] < 0).mean() * 100),
        "avg_1d_return": float(hits["RET_1D"].mean() * 100),
        "avg_2d_return": float(hits["RET_2D"].mean() * 100),
        **base,
    }

# -----------------------------
# Sweet-spot analysis (parallel per-symbol)
# -----------------------------
def compute_symbol_combos(args):
    """Worker function for ProcessPoolExecutor: compute combos for one symbol and return rows."""
    sym, df, tsi_options, cci_options, adx_options, cci_states, adx_states, tsi_thresholds, min_signals = args
    rows = []
    for tsi_params, cci_len, adx_len, cci_state, adx_state, tsi_pct_threshold in product(
        tsi_options, cci_options, adx_options, cci_states, adx_states, tsi_thresholds
    ):
        res = backtest_symbol(df, tsi_params, cci_len, adx_len, tsi_pct_threshold, cci_state, adx_state)
        if res is None:
            continue
        signals = res["signals"]
        if signals < min_signals:
            continue
        score = (
            res["next_day_red_pct"] * 0.55
            + (-res["avg_1d_return"]) * 15 * 0.25
            + min(signals, 50) * 0.20
        )
        rows.append({
            "symbol": sym,
            "tsi_combo": f"{tsi_params[0]},{tsi_params[1]},{tsi_params[2]}",
            "cci_len": cci_len,
            "adx_len": adx_len,
            "tsi_pct_threshold": tsi_pct_threshold,
            "cci_state": cci_state,
            "adx_state": adx_state,
            "signals": signals,
            "next_day_red_pct": res["next_day_red_pct"],
            "two_day_red_pct": res["two_day_red_pct"],
            "avg_1d_return": res["avg_1d_return"],
            "avg_2d_return": res["avg_2d_return"],
            "latest_signal": res["latest_signal"],
            "latest_tsi": res["latest_tsi"],
            "latest_tsi_pct": res["latest_tsi_pct"],
            "latest_cci": res["latest_cci"],
            "latest_cci_delta": res["latest_cci_delta"],
            "latest_adx": res["latest_adx"],
            "latest_adx_delta": res["latest_adx_delta"],
            "score": score,
        })
    return rows

@st.cache_data(ttl=3600, show_spinner=False)
def sweet_spot_analysis_parallel(raw_map: Dict[str, pd.DataFrame], tickers: List[str], years: int, min_signals: int,
                                 top_n: int = 10, workers: int = 4, use_processes: bool = True):
    """
    Compute sweet-spot combos for a list of tickers.
    raw_map: dict symbol -> DataFrame (pre-downloaded)
    This function parallelizes across symbols using ProcessPoolExecutor (CPU-bound).
    """
    tsi_options = [(4, 2, 4), (6, 3, 6), (7, 4, 7)]
    cci_options = [5, 7, 10, 14]
    adx_options = [7, 10, 14]
    cci_states = ["down_1d", "down_2d"]
    adx_states = ["flat_or_down", "down_1d", "any"]
    tsi_thresholds = [95, 97, 99]

    tasks = []
    for sym in tickers:
        df = raw_map.get(sym)
        if df is None or df.empty or len(df) < 260:
            continue
        tasks.append((sym, df, tsi_options, cci_options, adx_options, cci_states, adx_states, tsi_thresholds, min_signals))

    all_rows = []
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Choose executor type
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with Executor(max_workers=workers) as ex:
        futures = [ex.submit(compute_symbol_combos, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
            except Exception:
                continue

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combos = pd.DataFrame(all_rows)
    best = combos.sort_values(
        ["symbol", "score", "next_day_red_pct", "signals"],
        ascending=[True, False, False, False],
    ).groupby("symbol", as_index=False).first()

    # aggregated top-N metrics
    def top_n_agg(group):
        top = group.nlargest(top_n, "score")
        return pd.Series({
            "top_n_count": len(top),
            "top_n_mean_score": float(top["score"].mean()) if len(top) else np.nan,
            "top_n_median_score": float(top["score"].median()) if len(top) else np.nan,
            "top_n_median_tsi_pct_threshold": float(top["tsi_pct_threshold"].median()) if len(top) else np.nan,
            "top_n_median_cci_len": float(top["cci_len"].median()) if len(top) else np.nan,
            "top_n_median_adx_len": float(top["adx_len"].median()) if len(top) else np.nan,
        })

    agg = combos.groupby("symbol").apply(top_n_agg).reset_index()
    best = best.merge(agg, on="symbol", how="left")
    best = best.sort_values(["latest_signal", "top_n_mean_score"], ascending=[False, False]).reset_index(drop=True)
    return combos, best, agg

# -----------------------------
# Persistence helpers
# -----------------------------
def save_sweet_spots(df_best: pd.DataFrame, path: str = SWEET_SPOTS_PATH):
    if df_best is None or df_best.empty:
        return
    df = df_best.copy()
    df["saved_at"] = datetime.utcnow()
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, df], ignore_index=True)
        merged.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)

def load_sweet_spots(path: str = SWEET_SPOTS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------
# Live match check
# -----------------------------
def check_match(current_df: pd.DataFrame, combo_row: pd.Series, tsi_tolerance: float = 3.0):
    """Return dict with match/approach and indicator snapshots for the combo_row on current_df."""
    if current_df is None or current_df.empty or len(current_df) < 60:
        return {"match": False, "approach": False, "reason": "insufficient history"}

    tsi_params = tuple(int(x) for x in str(combo_row["tsi_combo"]).split(","))
    cci_len = int(combo_row["cci_len"])
    adx_len = int(combo_row["adx_len"])
    threshold = float(combo_row["tsi_pct_threshold"])

    df = current_df.copy()
    df["TSI"], df["TSI_SIGNAL"] = tsi(df["Close"], *tsi_params)
    df["CCI"] = cci(df["High"], df["Low"], df["Close"], cci_len)
    df["ADX"], _, _ = adx(df["High"], df["Low"], df["Close"], adx_len)
    df["TSI_PCT"] = rolling_tsi_pct(df["TSI"], window=252, min_periods=60)
    df["CCI_DELTA_1"] = df["CCI"].diff()
    df["CCI_DELTA_2"] = df["CCI"].diff(2)
    df["ADX_DELTA_1"] = df["ADX"].diff()

    tsi_pct = float(df["TSI_PCT"].iloc[-1]) if pd.notna(df["TSI_PCT"].iloc[-1]) else np.nan
    cci_delta1 = float(df["CCI_DELTA_1"].iloc[-1]) if pd.notna(df["CCI_DELTA_1"].iloc[-1]) else np.nan
    adx_delta1 = float(df["ADX_DELTA_1"].iloc[-1]) if pd.notna(df["ADX_DELTA_1"].iloc[-1]) else np.nan

    # evaluate states
    cond_cci, cond_adx = apply_states(df, combo_row["cci_state"], combo_row["adx_state"])
    state_now = bool(cond_cci.iloc[-1] and cond_adx.iloc[-1])

    exact = (tsi_pct >= threshold) and state_now
    approach = (tsi_pct >= threshold - tsi_tolerance) and (tsi_pct < threshold) and state_now

    return {
        "match": bool(exact),
        "approach": bool(approach),
        "tsi_pct": tsi_pct,
        "cci_delta1": cci_delta1,
        "adx_delta1": adx_delta1,
        "state_now": state_now,
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("TSI + CCI + ADX Exhaustion Dashboard (fast)")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area("Tickers (comma-separated) or upload a file below",
                                value=",".join(DEFAULT_TICKERS), height=120)
    uploaded = st.file_uploader("Upload optionable tickers (one per line)", type=["csv", "txt"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded, header=None, squeeze=True)
            tickers = [t.strip().upper() for t in df_in.tolist() if isinstance(t, str) and t.strip()]
        except Exception:
            tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    else:
        tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

    years = st.slider("Years of history", 2, 10, 5)
    min_signals = st.slider("Minimum historical signals", 3, 50, 8)

    st.subheader("Single combo scanner")
    tsi_choice = st.selectbox("TSI combo", options=["4,2,4", "6,3,6", "7,4,7"], index=0)
    tsi_params = tuple(int(x) for x in tsi_choice.split(","))
    cci_len = st.selectbox("CCI length", options=[5, 7, 10, 14], index=1)
    adx_len = st.selectbox("ADX length", options=[7, 10, 14], index=1)
    cci_state = st.selectbox("CCI state", options=["down_1d", "down_2d", "up_1d", "any"], index=0)
    adx_state = st.selectbox("ADX state", options=["flat_or_down", "down_1d", "up_1d", "any"], index=0)
    tsi_pct_threshold = st.slider("TSI percentile threshold", 85, 99, 97)

    st.subheader("Sweet-spot analysis options")
    top_n = st.slider("Top N combos to aggregate", 3, 50, 10)
    parallel_workers = st.slider("Parallel workers (processes)", 1, 8, 4)
    use_processes = st.checkbox("Use process parallelism for grid (faster, more CPU)", value=True)

    run_scanner = st.button("Run scanner")
    run_sweet_spot = st.button("Run sweet-spot analysis (heavy)")
    run_live_check = st.button("Run live monitor (saved sweet spots)")

# compute date range only when needed
end_date = date.today() + timedelta(days=1)
start_date = date.today() - timedelta(days=int(365.25 * years))

# Tabs: Scanner, Sweet Spots, Live Monitor
tabs = st.tabs(["Scanner", "Sweet Spots", "Live Monitor"])

# -----------------------------
# Tab 1: Scanner (single combo)
# -----------------------------
with tabs[0]:
    st.header("Single combo scanner")
    if run_scanner:
        st.info("Downloading history (batched) and running single combo scan...")
        # batch download all tickers in chunks to reduce overhead
        batch_size = 40
        raw_map = {}
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        progress = st.progress(0)
        for i, batch in enumerate(batches, start=1):
            raw = batch_download_yf(tuple(batch), start_date, end_date)
            for sym in batch:
                df = extract_symbol_df(raw, sym)
                raw_map[sym] = df
            progress.progress(int(i/len(batches)*100))
        rows = []
        charts = {}
        for sym in tickers:
            df = raw_map.get(sym, pd.DataFrame())
            res = backtest_symbol(df, tsi_params, cci_len, adx_len, tsi_pct_threshold, cci_state, adx_state)
            if res is None:
                continue
            chart_df = res.pop("chart_df", None)
            if chart_df is not None:
                charts[sym] = chart_df
            rows.append({"symbol": sym, **res})
        if not rows:
            st.warning("No usable data returned. Try fewer tickers or rerun.")
        else:
            results = pd.DataFrame(rows)
            results = results[results["signals"] >= min_signals].copy()
            if results.empty:
                st.warning("No symbols met the minimum signal count. Lower the minimum signals filter.")
            else:
                results["score"] = (
                    results["next_day_red_pct"].fillna(0) * 0.55
                    + (-results["avg_1d_return"].fillna(0)) * 15 * 0.25
                    + results["signals"].clip(upper=50) * 0.20
                )
                results = results.sort_values(
                    ["latest_signal", "score", "next_day_red_pct", "signals"],
                    ascending=[False, False, False, False],
                ).reset_index(drop=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Symbols passing", len(results))
                c2.metric("Live signals now", int(results["latest_signal"].sum()))
                c3.metric("Best next-day red %", f"{results['next_day_red_pct'].max():.1f}%")
                c4.metric("Best avg 1D return", f"{results['avg_1d_return'].max():.2f}%")
                st.subheader("Scanner results")
                display_cols = [
                    "symbol", "latest_signal", "latest_tsi", "latest_tsi_pct", "latest_cci",
                    "latest_cci_delta", "latest_adx", "latest_adx_delta", "signals",
                    "next_day_red_pct", "two_day_red_pct", "avg_1d_return", "avg_2d_return", "score"
                ]
                st.dataframe(results[display_cols].fillna(""), use_container_width=True, hide_index=True)
                st.subheader("Chart review")
                selected = st.selectbox("Select symbol", options=results["symbol"].tolist())
                if selected:
                    chart_df = charts.get(selected)
                    if chart_df is not None:
                        chart_df = chart_df.copy()
                        chart_df["TSI"], chart_df["TSI_SIGNAL"] = tsi(chart_df["Close"], *tsi_params)
                        chart_df["CCI"] = cci(chart_df["High"], chart_df["Low"], chart_df["Close"], cci_len)
                        chart_df["ADX"], chart_df["DI_PLUS"], chart_df["DI_MINUS"] = adx(chart_df["High"], chart_df["Low"], chart_df["Close"], adx_len)
                        t1, t2 = st.tabs(["Price", "Indicators"])
                        with t1:
                            st.line_chart(chart_df[["Close"]])
                        with t2:
                            st.line_chart(chart_df[["TSI", "TSI_SIGNAL", "CCI", "ADX"]])
    else:
        st.info("Configure scanner options in the sidebar and click Run scanner.")

# -----------------------------
# Tab 2: Sweet Spots (heavy)
# -----------------------------
with tabs[1]:
    st.header("Historical sweet-spot analysis")
    if run_sweet_spot:
        st.info("Downloading history (batched) and running sweet-spot grid (this can take time).")
        # batch download and build raw_map
        batch_size = 40
        raw_map = {}
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        progress = st.progress(0)
        for i, batch in enumerate(batches, start=1):
            raw = batch_download_yf(tuple(batch), start_date, end_date)
            for sym in batch:
                df = extract_symbol_df(raw, sym)
                raw_map[sym] = df
            progress.progress(int(i/len(batches)*100))
        st.info("Starting parallel sweet-spot computation...")
        with st.spinner("Computing combos..."):
            combos, best, agg = sweet_spot_analysis_parallel(raw_map, tickers, years, min_signals,
                                                            top_n=top_n, workers=parallel_workers, use_processes=use_processes)
        if best.empty:
            st.warning("No sweet-spot results found. Try lowering the minimum signal threshold or reducing the ticker list.")
        else:
            # persist best combos
            save_sweet_spots(best, SWEET_SPOTS_PATH)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tickers analyzed", best["symbol"].nunique())
            c2.metric("Best combos active now", int(best["latest_signal"].sum()))
            c3.metric("Top sweet-spot score", f"{best['score'].max():.2f}")
            st.subheader("Best historical combo by symbol (with latest CCI/ADX and top-N aggregates)")
            display_cols = [
                "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold",
                "cci_state", "adx_state", "signals", "next_day_red_pct",
                "two_day_red_pct", "avg_1d_return", "avg_2d_return",
                "latest_signal", "latest_tsi_pct", "latest_cci", "latest_adx",
                "top_n_count", "top_n_mean_score", "top_n_median_score", "top_n_median_tsi_pct_threshold"
            ]
            st.dataframe(best[display_cols].fillna(""), use_container_width=True, hide_index=True)
            st.subheader("Full combo table (top combos first)")
            st.dataframe(combos.sort_values(["score", "next_day_red_pct"], ascending=[False, False]).reset_index(drop=True), use_container_width=True, hide_index=True)
            st.success(f"Saved best combos to {SWEET_SPOTS_PATH}")
    else:
        st.info("Run the sweet-spot analysis from the sidebar to compute and persist best combos.")

# -----------------------------
# Tab 3: Live Monitor
# -----------------------------
with tabs[2]:
    st.header("Live Monitor (checks saved sweet spots)")
    saved = load_sweet_spots(SWEET_SPOTS_PATH)
    if saved.empty:
        st.info("No saved sweet spots found. Run the sweet-spot analysis first.")
    else:
        st.write(f"Loaded {len(saved)} saved sweet-spot rows (most recent runs).")
        symbols_available = sorted(saved["symbol"].unique())
        use_all = st.checkbox("Check all saved sweet spots", value=True)
        selected_symbols = st.multiselect("Symbols to check", options=symbols_available, default=None if use_all else symbols_available[:10])
        tsi_tol = st.slider("TSI approach tolerance (percentile points)", 1, 10, 3)
        run_live = st.button("Run live check now")
        if run_live:
            to_check = saved if use_all else saved[saved["symbol"].isin(selected_symbols)]
            # group by symbol and pick the most recent saved combo rows per symbol to check (or check all)
            results = []
            progress = st.progress(0)
            groups = list(to_check.groupby("symbol"))
            for i, (sym, group) in enumerate(groups, start=1):
                # download only this symbol's history (small)
                raw = batch_download_yf((sym,), start_date, end_date)
                df = extract_symbol_df(raw, sym)
                # check each combo row for this symbol (we'll check top N combos if present)
                for _, combo_row in group.iterrows():
                    chk = check_match(df, combo_row, tsi_tolerance=tsi_tol)
                    results.append({**combo_row.to_dict(), **chk})
                progress.progress(int(i/len(groups)*100))
            df_res = pd.DataFrame(results)
            if df_res.empty:
                st.warning("No matches found.")
            else:
                # prioritize exact matches, then approaches, then score
                df_res["match_rank"] = df_res["match"].astype(int) * 2 + df_res["approach"].astype(int)
                df_res = df_res.sort_values(["match_rank", "score"], ascending=[False, False]).reset_index(drop=True)
                st.dataframe(df_res[[
                    "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold",
                    "cci_state", "adx_state", "score", "match", "approach", "tsi_pct",
                    "latest_cci", "latest_adx", "cci_delta1", "adx_delta1"
                ]].fillna(""), use_container_width=True, hide_index=True)
                st.success("Live check complete.")
        else:
            st.info("Select symbols and click 'Run live check now' to evaluate saved sweet spots against current data.")

