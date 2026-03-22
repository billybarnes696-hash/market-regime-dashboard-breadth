import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import re
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# CONFIG
# -----------------------------
BASE_CACHE_DIR = "base_feature_cache"
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

TSI_OPTIONS = [(4, 2, 4), (6, 3, 6), (7, 4, 7)]
CCI_OPTIONS = [5, 7, 10, 14]
ADX_OPTIONS = [7, 10, 14]
TSI_THRESHOLDS = [95, 97, 99]


# -----------------------------
# INDICATORS
# -----------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def tsi(close, long_len, short_len, signal_len):
    mtm = close.diff()
    abs_mtm = mtm.abs()
    double_smoothed_mtm = ema(ema(mtm, long_len), short_len)
    double_smoothed_abs = ema(ema(abs_mtm, long_len), short_len)
    tsi_val = 100 * (double_smoothed_mtm / double_smoothed_abs.replace(0, np.nan))
    tsi_signal = ema(tsi_val, signal_len)
    return tsi_val, tsi_signal


def cci(high, low, close, length):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def adx(high, low, close, length):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1 / length, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1 / length, adjust=False).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx_val, plus_di, minus_di, atr


def rolling_tsi_pct(series, window=252, min_periods=60):
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )


# -----------------------------
# CANDLE SHAPE FEATURES
# -----------------------------
def candle_features(df):
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["close_loc"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    return df


# -----------------------------
# DATA HELPERS
# -----------------------------
def base_cache_path(symbol):
    return os.path.join(BASE_CACHE_DIR, f"{symbol}.parquet")


def load_or_download(symbol, start_date, end_date):
    path = base_cache_path(symbol)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    df = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=False)
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=str.title)
    df = candle_features(df)
    df.to_parquet(path)
    return df


# -----------------------------
# SWEET-SPOT WORKER
# -----------------------------
def process_symbol(symbol, start_date, end_date, min_signals, direction):
    try:
        df = load_or_download(symbol, start_date, end_date)
        if df.empty or len(df) < 260:
            return {"symbol": symbol, "error": "insufficient data"}

        results = []

        for tsi_params in TSI_OPTIONS:
            for cci_len in CCI_OPTIONS:
                for adx_len in ADX_OPTIONS:
                    for thr in TSI_THRESHOLDS:

                        out = df.copy()

                        out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], *tsi_params)
                        out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
                        out["ADX"], _, _, _ = adx(out["High"], out["Low"], out["Close"], adx_len)
                        out["TSI_PCT"] = rolling_tsi_pct(out["TSI"])

                        out["RET_1D"] = out["Close"].shift(-1) / out["Close"] - 1
                        out["RET_2D"] = out["Close"].shift(-2) / out["Close"] - 1

                        if direction == "bearish":
                            signal = out["TSI_PCT"] >= thr
                        else:
                            signal = out["TSI_PCT"] <= (100 - thr)

                        hits = out[signal].dropna(subset=["RET_1D", "RET_2D"])
                        if len(hits) < min_signals:
                            continue

                        if direction == "bearish":
                            hit_rate_1d = (hits["RET_1D"] < 0).mean() * 100
                        else:
                            hit_rate_1d = (hits["RET_1D"] > 0).mean() * 100

                        results.append({
                            "symbol": symbol,
                            "tsi_combo": f"{tsi_params[0]},{tsi_params[1]},{tsi_params[2]}",
                            "cci_len": cci_len,
                            "adx_len": adx_len,
                            "tsi_pct_threshold": thr,
                            "signals": len(hits),
                            "hit_rate_1d": hit_rate_1d,
                            "avg_1d_return": hits["RET_1D"].mean() * 100,
                            "avg_2d_return": hits["RET_2D"].mean() * 100,
                            "latest_tsi": out["TSI"].iloc[-1],
                            "latest_cci": out["CCI"].iloc[-1],
                            "latest_adx": out["ADX"].iloc[-1],
                            "latest_close_loc": out["close_loc"].iloc[-1],
                            "latest_upper_wick": out["upper_wick"].iloc[-1],
                            "latest_lower_wick": out["lower_wick"].iloc[-1],
                            "score": hit_rate_1d
                        })

        if not results:
            return {"symbol": symbol, "error": "no valid combos"}

        df_res = pd.DataFrame(results)
        best = df_res.sort_values("score", ascending=False).head(1)
        return best

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Sweet-Spot Batch Runner (with Candle Shapes)")

symbols_input = st.text_area("Paste symbols (comma, space, or newline separated)")
uploaded = st.file_uploader("Or upload CSV/TXT", type=["csv", "txt"])

symbols = []

if uploaded is not None:
    df_in = pd.read_csv(uploaded, header=None)
    symbols = [str(x).strip().upper() for x in df_in.iloc[:, 0] if str(x).strip()]
else:
    raw = symbols_input.upper()
    tokens = re.split(r"[,\s;]+", raw)
    symbols = [t for t in tokens if t]

symbols = sorted(set(symbols))

st.write(f"Detected {len(symbols)} symbols.")

years = st.slider("Years of history", 2, 10, 5)
min_signals = st.slider("Minimum signals per combo", 3, 50, 8)
direction = st.radio("Direction", ["bearish", "bullish"])
workers = st.slider("Parallel workers", 1, 16, 8)

run_btn = st.button("Run Batch", type="primary")

if run_btn:
    if not symbols:
        st.warning("No symbols provided.")
    else:
        start_date = date.today() - timedelta(days=int(365.25 * years))
        end_date = date.today() + timedelta(days=1)

        st.write(f"Running batch for {len(symbols)} symbols...")
        progress = st.progress(0.0)

        results = []
        error_list = []

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(process_symbol, sym, start_date, end_date, min_signals, direction): sym
                for sym in symbols
            }

            for i, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                if isinstance(res, dict) and "error" in res:
                    error_list.append(res)
                elif isinstance(res, pd.DataFrame):
                    results.append(res)
                progress.progress(i / len(symbols))

        if results:
            final = pd.concat(results, ignore_index=True)
            st.subheader("Best Combos")
            st.dataframe(final, use_container_width=True)

            st.download_button(
                "Download CSV",
                data=final.to_csv(index=False).encode("utf-8"),
                file_name="sweet_spots_batch.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download Parquet",
                data=final.to_parquet(index=False),
                file_name="sweet_spots_batch.parquet",
                mime="application/octet-stream",
            )

        if error_list:
            st.subheader("Errors")
            st.dataframe(pd.DataFrame(error_list))

