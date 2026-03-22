import os
from datetime import date, timedelta
from io import BytesIO
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="TSI/CCI/ADX Sweet-Spot Batch Runner", layout="wide")

BASE_CACHE_DIR = "base_feature_cache"
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

TSI_OPTIONS = [(4, 2, 4), (6, 3, 6), (7, 4, 7)]
CCI_OPTIONS = [5, 7, 10, 14]
ADX_OPTIONS = [7, 10, 14]
TSI_THRESHOLDS = [95, 97, 99]

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "SMH", "XLE", "TLT", "GLD",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "INTC", "MU", "PLTR",
]


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

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

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
    return adx_val, plus_di, minus_di, atr


def rolling_percentile(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100,
        raw=False,
    )


def base_cache_path(symbol: str) -> str:
    return os.path.join(BASE_CACHE_DIR, f"{symbol}.parquet")


@st.cache_data(ttl=3600, show_spinner=False)
def batch_download_yf(tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    return yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def extract_symbol_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        df = raw[symbol].copy()
    else:
        df = raw.copy()

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna().copy()
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, _, atr = adx(out["High"], out["Low"], out["Close"], 14)
    out["ATR_14"] = atr
    out["EMA_5"] = ema(out["Close"], 5)
    out["EMA_8"] = ema(out["Close"], 8)

    out["RANGE"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["RANGE_POS"] = (out["Close"] - out["Low"]) / out["RANGE"]
    out["UPPER_WICK"] = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["LOWER_WICK"] = out[["Open", "Close"]].min(axis=1) - out["Low"]
    out["BODY"] = (out["Close"] - out["Open"]).abs()
    out["UPPER_WICK_TO_BODY"] = out["UPPER_WICK"] / out["BODY"].replace(0, np.nan)
    out["LOWER_WICK_TO_BODY"] = out["LOWER_WICK"] / out["BODY"].replace(0, np.nan)
    out["BEAR_REJECTION"] = (out["RANGE_POS"] <= 0.35) & (out["UPPER_WICK_TO_BODY"] >= 1.0)
    out["BULL_REJECTION"] = (out["RANGE_POS"] >= 0.65) & (out["LOWER_WICK_TO_BODY"] >= 1.0)
    return out


def save_base_features(symbol: str, df: pd.DataFrame) -> None:
    df.to_parquet(base_cache_path(symbol), index=True)


def load_base_features(symbol: str) -> pd.DataFrame:
    path = base_cache_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def cache_is_stale(symbol: str, today_str: str) -> bool:
    path = base_cache_path(symbol)
    if not os.path.exists(path):
        return True
    try:
        cached = pd.read_parquet(path)
        if cached.empty:
            return True
        last_date = pd.to_datetime(cached.index).max().date().isoformat()
        return last_date < today_str
    except Exception:
        return True


def ensure_base_features(symbols: list[str], start_date: date, end_date: date, batch_size: int = 40, force_refresh: bool = False):
    today_str = date.today().isoformat()
    needed = [s for s in symbols if force_refresh or cache_is_stale(s, today_str)]
    if not needed:
        return

    batches = [needed[i:i + batch_size] for i in range(0, len(needed), batch_size)]
    progress = st.progress(0.0)
    for i, batch in enumerate(batches, start=1):
        raw = batch_download_yf(tuple(batch), start_date, end_date)
        for sym in batch:
            df = extract_symbol_df(raw, sym)
            if not df.empty:
                save_base_features(sym, compute_base_features(df))
        progress.progress(i / len(batches))
    progress.empty()


def apply_states(out: pd.DataFrame, cci_state: str, adx_state: str, direction: str, require_rejection: bool, require_extension: bool):
    if cci_state == "down_1d":
        cond_cci = out["CCI_DELTA_1"] < 0
    elif cci_state == "down_2d":
        cond_cci = (out["CCI_DELTA_1"] < 0) & (out["CCI_DELTA_2"] < 0)
    elif cci_state == "up_1d":
        cond_cci = out["CCI_DELTA_1"] > 0
    else:
        cond_cci = pd.Series(True, index=out.index)

    if adx_state == "down_1d":
        cond_adx = out["ADX_DELTA_1"] < 0
    elif adx_state == "flat_or_down":
        cond_adx = out["ADX_DELTA_1"] <= 0
    elif adx_state == "up_1d":
        cond_adx = out["ADX_DELTA_1"] > 0
    else:
        cond_adx = pd.Series(True, index=out.index)

    if direction == "bearish":
        cond_rejection = out["BEAR_REJECTION"] if require_rejection else pd.Series(True, index=out.index)
        cond_extension = (out["DIST_EMA5_ATR"] >= 1.0) if require_extension else pd.Series(True, index=out.index)
    else:
        cond_rejection = out["BULL_REJECTION"] if require_rejection else pd.Series(True, index=out.index)
        cond_extension = (out["DIST_EMA5_ATR"] <= -1.0) if require_extension else pd.Series(True, index=out.index)

    return cond_cci, cond_adx, cond_rejection, cond_extension


def summarize_hits(hits: pd.DataFrame, direction: str) -> dict:
    if hits.empty:
        return {
            "signals": 0,
            "hit_rate_1d": np.nan,
            "hit_rate_2d": np.nan,
            "avg_1d_return": np.nan,
            "avg_2d_return": np.nan,
            "avg_3d_return": np.nan,
            "median_1d_return": np.nan,
            "best_excursion_2d": np.nan,
            "median_tsi": np.nan,
            "median_cci": np.nan,
            "median_adx": np.nan,
            "avg_tsi": np.nan,
            "avg_cci": np.nan,
            "avg_adx": np.nan,
            "median_tsi_pct": np.nan,
            "avg_tsi_pct": np.nan,
            "median_cci_delta": np.nan,
            "avg_cci_delta": np.nan,
            "median_adx_delta": np.nan,
            "avg_adx_delta": np.nan,
        }

    if direction == "bearish":
        hit_rate_1d = float((hits["RET_1D"] < 0).mean() * 100)
        hit_rate_2d = float((hits["RET_2D"] < 0).mean() * 100)
        best_excursion_2d = float((-hits["FWD_LOW_2D"]).mean() * 100)
    else:
        hit_rate_1d = float((hits["RET_1D"] > 0).mean() * 100)
        hit_rate_2d = float((hits["RET_2D"] > 0).mean() * 100)
        best_excursion_2d = float((hits["FWD_HIGH_2D"]).mean() * 100)

    return {
        "signals": int(len(hits)),
        "hit_rate_1d": hit_rate_1d,
        "hit_rate_2d": hit_rate_2d,
        "avg_1d_return": float(hits["RET_1D"].mean() * 100),
        "avg_2d_return": float(hits["RET_2D"].mean() * 100),
        "avg_3d_return": float(hits["RET_3D"].mean() * 100),
        "median_1d_return": float(hits["RET_1D"].median() * 100),
        "best_excursion_2d": best_excursion_2d,
        "median_tsi": float(hits["TSI"].median()),
        "median_cci": float(hits["CCI"].median()),
        "median_adx": float(hits["ADX"].median()),
        "avg_tsi": float(hits["TSI"].mean()),
        "avg_cci": float(hits["CCI"].mean()),
        "avg_adx": float(hits["ADX"].mean()),
        "median_tsi_pct": float(hits["TSI_PCT"].median()),
        "avg_tsi_pct": float(hits["TSI_PCT"].mean()),
        "median_cci_delta": float(hits["CCI_DELTA_1"].median()),
        "avg_cci_delta": float(hits["CCI_DELTA_1"].mean()),
        "median_adx_delta": float(hits["ADX_DELTA_1"].median()),
        "avg_adx_delta": float(hits["ADX_DELTA_1"].mean()),
    }


def score_combo(summary: dict, direction: str) -> float:
    if summary["signals"] == 0 or pd.isna(summary["hit_rate_1d"]):
        return np.nan
    if direction == "bearish":
        return_component = -summary["avg_1d_return"]
    else:
        return_component = summary["avg_1d_return"]

    return (
        summary["hit_rate_1d"] * 0.45
        + max(return_component, 0) * 18 * 0.25
        + summary["hit_rate_2d"] * 0.10
        + max(summary["best_excursion_2d"], 0) * 8 * 0.10
        + min(summary["signals"], 50) * 0.10
    )


def backtest_combo(base_df: pd.DataFrame, tsi_params, cci_len: int, adx_len: int, tsi_pct_threshold: int,
                  cci_state: str, adx_state: str, direction: str, require_rejection: bool, require_extension: bool):
    if base_df is None or base_df.empty or len(base_df) < 260:
        return None

    out = base_df.copy()
    out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], *tsi_params)
    out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
    out["ADX"], out["DI_PLUS"], out["DI_MINUS"], _ = adx(out["High"], out["Low"], out["Close"], adx_len)

    out["RET_1D"] = out["Close"].shift(-1) / out["Close"] - 1
    out["RET_2D"] = out["Close"].shift(-2) / out["Close"] - 1
    out["RET_3D"] = out["Close"].shift(-3) / out["Close"] - 1
    out["FWD_LOW_2D"] = out["Low"].shift(-1).rolling(2).min() / out["Close"] - 1
    out["FWD_HIGH_2D"] = out["High"].shift(-1).rolling(2).max() / out["Close"] - 1

    out["TSI_PCT"] = rolling_percentile(out["TSI"], window=252, min_periods=60)
    out["CCI_DELTA_1"] = out["CCI"].diff()
    out["CCI_DELTA_2"] = out["CCI"].diff(2)
    out["ADX_DELTA_1"] = out["ADX"].diff()
    out["DIST_EMA5_ATR"] = (out["Close"] - out["EMA_5"]) / out["ATR_14"].replace(0, np.nan)

    cond_cci, cond_adx, cond_rejection, cond_extension = apply_states(
        out, cci_state, adx_state, direction, require_rejection, require_extension
    )

    if direction == "bearish":
        base_signal = out["TSI_PCT"] >= tsi_pct_threshold
    else:
        base_signal = out["TSI_PCT"] <= (100 - tsi_pct_threshold)

    signal = (base_signal & cond_cci & cond_adx & cond_rejection & cond_extension).fillna(False)
    hits = out[signal].copy()
    hits = hits.dropna(subset=["RET_1D", "RET_2D", "RET_3D", "TSI", "CCI", "ADX", "TSI_PCT"])

    summary = summarize_hits(hits, direction)
    summary["score"] = score_combo(summary, direction)
    summary.update({
        "latest_signal": bool(signal.iloc[-1]) if len(signal) else False,
        "latest_tsi": float(out["TSI"].iloc[-1]) if pd.notna(out["TSI"].iloc[-1]) else np.nan,
        "latest_tsi_pct": float(out["TSI_PCT"].iloc[-1]) if pd.notna(out["TSI_PCT"].iloc[-1]) else np.nan,
        "latest_cci": float(out["CCI"].iloc[-1]) if pd.notna(out["CCI"].iloc[-1]) else np.nan,
        "latest_cci_delta": float(out["CCI_DELTA_1"].iloc[-1]) if pd.notna(out["CCI_DELTA_1"].iloc[-1]) else np.nan,
        "latest_adx": float(out["ADX"].iloc[-1]) if pd.notna(out["ADX"].iloc[-1]) else np.nan,
        "latest_adx_delta": float(out["ADX_DELTA_1"].iloc[-1]) if pd.notna(out["ADX_DELTA_1"].iloc[-1]) else np.nan,
    })
    return summary


def process_symbol(symbol: str, min_signals: int, direction: str, require_rejection: bool, require_extension: bool, top_n: int):
    base = load_base_features(symbol)
    if base.empty or len(base) < 260:
        return None, None

    combo_rows = []
    cci_states = ["down_1d", "down_2d"] if direction == "bearish" else ["up_1d", "any"]
    adx_states = ["flat_or_down", "down_1d", "any"] if direction == "bearish" else ["up_1d", "any"]

    for tsi_params, cci_len, adx_len, tsi_threshold, cci_state, adx_state in product(
        TSI_OPTIONS, CCI_OPTIONS, ADX_OPTIONS, TSI_THRESHOLDS, cci_states, adx_states
    ):
        res = backtest_combo(
            base_df=base,
            tsi_params=tsi_params,
            cci_len=cci_len,
            adx_len=adx_len,
            tsi_pct_threshold=tsi_threshold,
            cci_state=cci_state,
            adx_state=adx_state,
            direction=direction,
            require_rejection=require_rejection,
            require_extension=require_extension,
        )
        if res is None or res["signals"] < min_signals or pd.isna(res["score"]):
            continue

        combo_rows.append({
            "symbol": symbol,
            "direction": direction,
            "tsi_combo": f"{tsi_params[0]},{tsi_params[1]},{tsi_params[2]}",
            "tsi_long": tsi_params[0],
            "tsi_short": tsi_params[1],
            "tsi_signal": tsi_params[2],
            "cci_len": cci_len,
            "adx_len": adx_len,
            "tsi_pct_threshold": tsi_threshold,
            "cci_state": cci_state,
            "adx_state": adx_state,
            "require_rejection": require_rejection,
            "require_extension": require_extension,
            **res,
        })

    if not combo_rows:
        return None, None

    combos = pd.DataFrame(combo_rows).sort_values(["score", "hit_rate_1d", "signals"], ascending=[False, False, False]).reset_index(drop=True)
    best = combos.iloc[[0]].copy()

    top = combos.head(top_n).copy()
    aggregate = {
        "symbol": symbol,
        "top_n_used": len(top),
        "agg_mean_score_top_n": float(top["score"].mean()),
        "agg_median_score_top_n": float(top["score"].median()),
        "agg_mean_hit_rate_1d_top_n": float(top["hit_rate_1d"].mean()),
        "agg_median_hit_rate_1d_top_n": float(top["hit_rate_1d"].median()),
        "agg_mean_avg_1d_return_top_n": float(top["avg_1d_return"].mean()),
        "agg_median_avg_1d_return_top_n": float(top["avg_1d_return"].median()),
        "agg_mean_tsi_top_n": float(top["median_tsi"].mean()),
        "agg_median_tsi_top_n": float(top["median_tsi"].median()),
        "agg_mean_cci_top_n": float(top["median_cci"].mean()),
        "agg_median_cci_top_n": float(top["median_cci"].median()),
        "agg_mean_adx_top_n": float(top["median_adx"].mean()),
        "agg_median_adx_top_n": float(top["median_adx"].median()),
        "agg_mean_tsi_pct_top_n": float(top["median_tsi_pct"].mean()),
        "agg_median_tsi_pct_top_n": float(top["median_tsi_pct"].median()),
        "agg_mean_cci_delta_top_n": float(top["median_cci_delta"].mean()),
        "agg_median_cci_delta_top_n": float(top["median_cci_delta"].median()),
        "agg_mean_adx_delta_top_n": float(top["median_adx_delta"].mean()),
        "agg_median_adx_delta_top_n": float(top["median_adx_delta"].median()),
        "agg_mean_tsi_long_top_n": float(top["tsi_long"].mean()),
        "agg_median_tsi_long_top_n": float(top["tsi_long"].median()),
        "agg_mean_tsi_short_top_n": float(top["tsi_short"].mean()),
        "agg_median_tsi_short_top_n": float(top["tsi_short"].median()),
        "agg_mean_tsi_signal_top_n": float(top["tsi_signal"].mean()),
        "agg_median_tsi_signal_top_n": float(top["tsi_signal"].median()),
        "agg_mean_cci_len_top_n": float(top["cci_len"].mean()),
        "agg_median_cci_len_top_n": float(top["cci_len"].median()),
        "agg_mean_adx_len_top_n": float(top["adx_len"].mean()),
        "agg_median_adx_len_top_n": float(top["adx_len"].median()),
        "agg_mean_threshold_top_n": float(top["tsi_pct_threshold"].mean()),
        "agg_median_threshold_top_n": float(top["tsi_pct_threshold"].median()),
    }

    for k, v in aggregate.items():
        if k != "symbol":
            best[k] = v

    return combos, best


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


st.title("Full TSI / CCI / ADX Sweet-Spot Finder")
st.write("Find the best historical TSI/CCI/ADX combination per symbol, plus top-N aggregate medians/averages for export into your real-time scanner.")

symbols_input = st.text_area("Paste symbols (comma or newline separated)", value=",".join(DEFAULT_SYMBOLS), height=120)
uploaded = st.file_uploader("Or upload CSV/TXT", type=["csv", "txt"])

if uploaded is not None:
    df_in = pd.read_csv(uploaded, header=None)
    symbols = [str(x).strip().upper() for x in df_in.iloc[:, 0].tolist() if str(x).strip()]
else:
    symbols = [x.strip().upper() for x in symbols_input.replace("\n", ",").split(",") if x.strip()]

years = st.slider("Years of history", 2, 10, 5)
min_signals = st.slider("Minimum signals", 3, 50, 8)
direction = st.radio("Direction", ["bearish", "bullish"], horizontal=True)
require_rejection = st.checkbox("Require candle rejection", value=False)
require_extension = st.checkbox("Require EMA/ATR extension", value=False)
force_refresh = st.checkbox("Force refresh cached price data", value=False)
workers = st.slider("Thread workers", 1, 16, 8)
top_n = st.slider("Top N combos for aggregates", 3, 25, 10)
batch_size = st.slider("Download batch size", 10, 80, 40)

run_btn = st.button("Run Full Sweet-Spot Batch", type="primary")

if run_btn and symbols:
    start_date = date.today() - timedelta(days=int(365.25 * years))
    end_date = date.today() + timedelta(days=1)

    ensure_base_features(symbols, start_date, end_date, batch_size=batch_size, force_refresh=force_refresh)

    progress = st.progress(0.0)
    all_combo_frames = []
    best_frames = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_symbol, sym, min_signals, direction, require_rejection, require_extension, top_n): sym
            for sym in symbols
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            combos_df, best_df = fut.result()
            if combos_df is not None and not combos_df.empty:
                all_combo_frames.append(combos_df)
            if best_df is not None and not best_df.empty:
                best_frames.append(best_df)
            progress.progress(i / len(symbols))

    progress.empty()

    if not best_frames:
        st.warning("No results found.")
    else:
        best_final = pd.concat(best_frames, ignore_index=True)
        all_combos_final = pd.concat(all_combo_frames, ignore_index=True) if all_combo_frames else pd.DataFrame()

        best_final = best_final.sort_values(["score", "hit_rate_1d", "signals"], ascending=[False, False, False]).reset_index(drop=True)
        all_combos_final = all_combos_final.sort_values(["score", "hit_rate_1d", "signals"], ascending=[False, False, False]).reset_index(drop=True)

        st.subheader("Best combo per symbol + top-N aggregate summary")
        st.dataframe(best_final, use_container_width=True, hide_index=True)

        st.subheader("All tested combos")
        st.dataframe(all_combos_final, use_container_width=True, hide_index=True)

        st.download_button(
            "Download best combos CSV",
            best_final.to_csv(index=False).encode("utf-8"),
            "sweet_spots_best.csv",
            "text/csv",
        )
        st.download_button(
            "Download all combos CSV",
            all_combos_final.to_csv(index=False).encode("utf-8"),
            "sweet_spots_all_combos.csv",
            "text/csv",
        )
        st.download_button(
            "Download best combos Parquet",
            to_parquet_bytes(best_final),
            "sweet_spots_best.parquet",
            "application/octet-stream",
        )
        st.download_button(
            "Download all combos Parquet",
            to_parquet_bytes(all_combos_final),
            "sweet_spots_all_combos.parquet",
            "application/octet-stream",
        )
else:
    st.info("Paste or upload symbols, choose settings, then run the batch.")

