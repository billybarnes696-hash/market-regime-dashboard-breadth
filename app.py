import os
from datetime import date, timedelta
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# Config / defaults
# -----------------------------
st.set_page_config(page_title="TSI + CCI + ADX Sweet-Spot Dashboard", layout="wide")

DEFAULT_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "SMH", "XLE", "TLT", "GLD"]
DEFAULT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "INTC", "MU", "PLTR"]
DEFAULT_TICKERS = DEFAULT_ETFS + DEFAULT_STOCKS

BASE_CACHE_DIR = "base_feature_cache"
SWEET_SPOTS_PATH = "sweet_spots_latest.parquet"
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

TSI_OPTIONS = [(4, 2, 4), (6, 3, 6), (7, 4, 7)]
CCI_OPTIONS = [5, 7, 10, 14]
ADX_OPTIONS = [7, 10, 14]
TSI_THRESHOLDS = [95, 97, 99]
DIRECTIONS = ["bearish", "bullish"]


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


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def batch_download_yf(tickers: Tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
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
    needed = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[needed].dropna().copy()

    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def base_cache_path(symbol: str) -> str:
    return os.path.join(BASE_CACHE_DIR, f"{symbol}.parquet")


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


# -----------------------------
# Feature engineering
# -----------------------------
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


def ensure_base_features(
    tickers: List[str],
    start_date: date,
    end_date: date,
    batch_size: int = 40,
    force_refresh: bool = False,
) -> None:
    today_str = date.today().isoformat()

    needed = []
    for t in tickers:
        if force_refresh or cache_is_stale(t, today_str):
            needed.append(t)

    if not needed:
        return

    batches = [needed[i:i + batch_size] for i in range(0, len(needed), batch_size)]
    progress = st.progress(0.0)

    for i, batch in enumerate(batches, start=1):
        raw = batch_download_yf(tuple(batch), start_date, end_date)
        for sym in batch:
            df = extract_symbol_df(raw, sym)
            if df.empty:
                continue
            base = compute_base_features(df)
            save_base_features(sym, base)
        progress.progress(i / len(batches))

    progress.empty()


# -----------------------------
# Signal logic
# -----------------------------
def apply_states(
    out: pd.DataFrame,
    cci_state: str,
    adx_state: str,
    direction: str,
    require_rejection: bool,
    require_extension: bool,
):
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
    }


def score_combo(summary: dict, direction: str) -> float:
    signals = summary["signals"]
    if signals == 0 or pd.isna(summary["hit_rate_1d"]):
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
        + min(signals, 50) * 0.10
    )


def backtest_combo(
    base_df: pd.DataFrame,
    tsi_params: Tuple[int, int, int],
    cci_len: int,
    adx_len: int,
    tsi_pct_threshold: int,
    cci_state: str,
    adx_state: str,
    direction: str,
    require_rejection: bool,
    require_extension: bool,
):
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
    summary.update(
        {
            "latest_signal": bool(signal.iloc[-1]) if len(signal) else False,
            "latest_tsi": float(out["TSI"].iloc[-1]) if pd.notna(out["TSI"].iloc[-1]) else np.nan,
            "latest_tsi_pct": float(out["TSI_PCT"].iloc[-1]) if pd.notna(out["TSI_PCT"].iloc[-1]) else np.nan,
            "latest_cci": float(out["CCI"].iloc[-1]) if pd.notna(out["CCI"].iloc[-1]) else np.nan,
            "latest_cci_delta": float(out["CCI_DELTA_1"].iloc[-1]) if pd.notna(out["CCI_DELTA_1"].iloc[-1]) else np.nan,
            "latest_adx": float(out["ADX"].iloc[-1]) if pd.notna(out["ADX"].iloc[-1]) else np.nan,
            "latest_adx_delta": float(out["ADX_DELTA_1"].iloc[-1]) if pd.notna(out["ADX_DELTA_1"].iloc[-1]) else np.nan,
            "latest_dist_ema5_atr": float(out["DIST_EMA5_ATR"].iloc[-1]) if pd.notna(out["DIST_EMA5_ATR"].iloc[-1]) else np.nan,
            "chart_df": out.tail(220).copy(),
        }
    )
    return summary


# -----------------------------
# Sweet-spot analysis
# -----------------------------
def run_sweet_spot_serial(
    tickers: List[str],
    min_signals: int,
    direction: str,
    require_rejection: bool,
    require_extension: bool,
):
    combo_rows = []
    progress = st.progress(0.0)

    for i, sym in enumerate(tickers, start=1):
        base = load_base_features(sym)
        if base.empty or len(base) < 260:
            progress.progress(i / len(tickers))
            continue

        cci_states = ["down_1d", "down_2d"] if direction == "bearish" else ["up_1d", "any"]
        adx_states = ["flat_or_down", "down_1d", "any"] if direction == "bearish" else ["up_1d", "any"]

        for tsi_params, cci_len, adx_len, tsi_pct_threshold, cci_state, adx_state in product(
            TSI_OPTIONS,
            CCI_OPTIONS,
            ADX_OPTIONS,
            TSI_THRESHOLDS,
            cci_states,
            adx_states,
        ):
            res = backtest_combo(
                base_df=base,
                tsi_params=tsi_params,
                cci_len=cci_len,
                adx_len=adx_len,
                tsi_pct_threshold=tsi_pct_threshold,
                cci_state=cci_state,
                adx_state=adx_state,
                direction=direction,
                require_rejection=require_rejection,
                require_extension=require_extension,
            )
            if res is None or res["signals"] < min_signals or pd.isna(res["score"]):
                continue

            combo_rows.append(
                {
                    "symbol": sym,
                    "direction": direction,
                    "tsi_combo": f"{tsi_params[0]},{tsi_params[1]},{tsi_params[2]}",
                    "cci_len": cci_len,
                    "adx_len": adx_len,
                    "tsi_pct_threshold": tsi_pct_threshold,
                    "cci_state": cci_state,
                    "adx_state": adx_state,
                    "require_rejection": require_rejection,
                    "require_extension": require_extension,
                    **{k: v for k, v in res.items() if k != "chart_df"},
                }
            )

        progress.progress(i / len(tickers))

    progress.empty()

    if not combo_rows:
        return pd.DataFrame(), pd.DataFrame()

    combos = pd.DataFrame(combo_rows)
    best = (
        combos.sort_values(
            ["symbol", "score", "hit_rate_1d", "signals"],
            ascending=[True, False, False, False],
        )
        .groupby("symbol", as_index=False)
        .first()
    )
    best = best.sort_values(["latest_signal", "score"], ascending=[False, False]).reset_index(drop=True)
    return combos, best


# -----------------------------
# Persistence / live monitor
# -----------------------------
def save_sweet_spots(df_best: pd.DataFrame, meta: dict, path: str = SWEET_SPOTS_PATH) -> None:
    if df_best is None or df_best.empty:
        return
    df = df_best.copy()
    df["saved_at"] = pd.Timestamp.utcnow()
    for k, v in meta.items():
        df[k] = v
    df.to_parquet(path, index=False)


def load_sweet_spots(path: str = SWEET_SPOTS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def check_combo_live_from_base(base_df: pd.DataFrame, combo_row: pd.Series, tsi_tolerance: float = 3.0):
    if base_df is None or base_df.empty or len(base_df) < 60:
        return {"match": False, "approach": False}

    tsi_params = tuple(int(x) for x in str(combo_row["tsi_combo"]).split(","))
    cci_len = int(combo_row["cci_len"])
    adx_len = int(combo_row["adx_len"])
    threshold = float(combo_row["tsi_pct_threshold"])
    require_rejection = bool(combo_row.get("require_rejection", False))
    require_extension = bool(combo_row.get("require_extension", False))
    direction = combo_row.get("direction", "bearish")

    out = base_df.copy()
    out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], *tsi_params)
    out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
    out["ADX"], _, _, _ = adx(out["High"], out["Low"], out["Close"], adx_len)
    out["TSI_PCT"] = rolling_percentile(out["TSI"], window=252, min_periods=60)
    out["CCI_DELTA_1"] = out["CCI"].diff()
    out["CCI_DELTA_2"] = out["CCI"].diff(2)
    out["ADX_DELTA_1"] = out["ADX"].diff()
    out["DIST_EMA5_ATR"] = (out["Close"] - out["EMA_5"]) / out["ATR_14"].replace(0, np.nan)

    cond_cci, cond_adx, cond_rejection, cond_extension = apply_states(
        out,
        combo_row["cci_state"],
        combo_row["adx_state"],
        direction,
        require_rejection,
        require_extension,
    )

    state_now = bool(cond_cci.iloc[-1] and cond_adx.iloc[-1] and cond_rejection.iloc[-1] and cond_extension.iloc[-1])
    tsi_pct_now = float(out["TSI_PCT"].iloc[-1]) if pd.notna(out["TSI_PCT"].iloc[-1]) else np.nan

    if direction == "bearish":
        exact = (tsi_pct_now >= threshold) and state_now
        approach = (tsi_pct_now >= threshold - tsi_tolerance) and (tsi_pct_now < threshold) and state_now
    else:
        exact = (tsi_pct_now <= (100 - threshold)) and state_now
        approach = (tsi_pct_now <= (100 - threshold + tsi_tolerance)) and (tsi_pct_now > (100 - threshold)) and state_now

    return {
        "match": bool(exact),
        "approach": bool(approach),
        "tsi_pct_now": tsi_pct_now,
        "latest_cci_now": float(out["CCI"].iloc[-1]) if pd.notna(out["CCI"].iloc[-1]) else np.nan,
        "latest_adx_now": float(out["ADX"].iloc[-1]) if pd.notna(out["ADX"].iloc[-1]) else np.nan,
        "cci_delta1_now": float(out["CCI_DELTA_1"].iloc[-1]) if pd.notna(out["CCI_DELTA_1"].iloc[-1]) else np.nan,
        "adx_delta1_now": float(out["ADX_DELTA_1"].iloc[-1]) if pd.notna(out["ADX_DELTA_1"].iloc[-1]) else np.nan,
    }


def csv_download(df: pd.DataFrame, filename: str) -> None:
    if df is not None and not df.empty:
        st.download_button(
            label=f"Download {filename}",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )


# -----------------------------
# Sidebar
# -----------------------------
st.title("TSI + CCI + ADX Sweet-Spot Dashboard")

with st.sidebar:
    st.header("Universe")
    mode = st.radio("Ticker input", ["Default ETFs + stocks", "Custom list"], index=0)

    if mode == "Default ETFs + stocks":
        use_etfs = st.checkbox("Include ETFs", value=True)
        use_stocks = st.checkbox("Include stocks", value=True)
        tickers = []
        if use_etfs:
            tickers.extend(DEFAULT_ETFS)
        if use_stocks:
            tickers.extend(DEFAULT_STOCKS)
    else:
        tickers_text = st.text_area("Tickers (comma-separated)", value=",".join(DEFAULT_TICKERS), height=160)
        uploaded = st.file_uploader("Upload tickers (CSV/TXT, one per line or first column)", type=["csv", "txt"])
        if uploaded is not None:
            df_in = pd.read_csv(uploaded, header=None)
            tickers = [str(x).strip().upper() for x in df_in.iloc[:, 0].tolist() if str(x).strip()]
        else:
            tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

    years = st.slider("Years of history", 2, 10, 5)
    min_signals = st.slider("Minimum historical signals", 3, 50, 8)
    direction = st.radio("Signal direction", DIRECTIONS, index=0)
    require_rejection = st.checkbox("Require candle rejection", value=False)
    require_extension = st.checkbox("Require EMA/ATR extension", value=False)
    force_refresh = st.checkbox("Force refresh cached data", value=False)

    st.header("Single combo scanner")
    tsi_choice = st.selectbox("TSI combo", options=["4,2,4", "6,3,6", "7,4,7"], index=0)
    tsi_params = tuple(int(x) for x in tsi_choice.split(","))
    cci_len = st.selectbox("CCI length", options=CCI_OPTIONS, index=1)
    adx_len = st.selectbox("ADX length", options=ADX_OPTIONS, index=1)

    if direction == "bearish":
        cci_default = 0
        adx_default = 0
    else:
        cci_default = 2
        adx_default = 2

    cci_state = st.selectbox("CCI state", options=["down_1d", "down_2d", "up_1d", "any"], index=cci_default)
    adx_state = st.selectbox("ADX state", options=["flat_or_down", "down_1d", "up_1d", "any"], index=adx_default)
    tsi_pct_threshold = st.slider("TSI percentile threshold", 85, 99, 97)
    batch_size = st.slider("Download batch size", 10, 80, 40)

    run_scanner = st.button("Run scanner", type="primary")
    run_sweet_spot = st.button("Run sweet-spot analysis")
    run_live_monitor = st.button("Run live monitor")


# -----------------------------
# Data prep
# -----------------------------
end_date = date.today() + timedelta(days=1)
start_date = date.today() - timedelta(days=int(365.25 * years))

ensure_base_features(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    batch_size=batch_size,
    force_refresh=force_refresh,
)


# -----------------------------
# Tabs
# -----------------------------
overview_tab, scanner_tab, sweet_tab, live_tab = st.tabs(
    ["Overview", "Scanner", "Sweet Spots", "Live Monitor"]
)

with overview_tab:
    st.write(f"Universe size: **{len(tickers)}**")
    st.write(f"Direction: **{direction.title()}**")
    st.write(f"Years loaded: **{years}**")
    st.write("Workflow: scan one combo, or run sweet-spot analysis to find the best historical combo by symbol.")

with scanner_tab:
    if run_scanner:
        rows = []
        charts = {}

        for sym in tickers:
            base = load_base_features(sym)
            if base.empty:
                continue

            res = backtest_combo(
                base_df=base,
                tsi_params=tsi_params,
                cci_len=cci_len,
                adx_len=adx_len,
                tsi_pct_threshold=tsi_pct_threshold,
                cci_state=cci_state,
                adx_state=adx_state,
                direction=direction,
                require_rejection=require_rejection,
                require_extension=require_extension,
            )
            if res is None:
                continue

            chart_df = res.pop("chart_df", None)
            if chart_df is not None:
                charts[sym] = chart_df
            rows.append({"symbol": sym, **res})

        if not rows:
            st.warning("No usable results returned.")
        else:
            results = pd.DataFrame(rows)
            results = results[results["signals"] >= min_signals].copy()

            if results.empty:
                st.warning("No symbols met the minimum signal count.")
            else:
                results = results.sort_values(
                    ["latest_signal", "score", "hit_rate_1d", "signals"],
                    ascending=[False, False, False, False],
                ).reset_index(drop=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Symbols passing", len(results))
                c2.metric("Live signals now", int(results["latest_signal"].sum()))
                c3.metric("Best 1D hit rate", f"{results['hit_rate_1d'].max():.1f}%")
                best_avg = results["avg_1d_return"].max() if direction == "bullish" else results["avg_1d_return"].min()
                c4.metric("Best avg 1D return", f"{best_avg:.2f}%")

                display_cols = [
                    "symbol", "latest_signal", "latest_tsi", "latest_tsi_pct", "latest_cci", "latest_cci_delta",
                    "latest_adx", "latest_adx_delta", "latest_dist_ema5_atr", "signals", "hit_rate_1d", "hit_rate_2d",
                    "avg_1d_return", "avg_2d_return", "best_excursion_2d", "score"
                ]
                st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
                csv_download(results[display_cols], "scanner_results.csv")

                selected = st.selectbox("Select symbol for chart review", options=results["symbol"].tolist())
                chart_df = charts[selected].copy()

                t1, t2 = st.tabs(["Price", "Indicators"])
                with t1:
                    st.line_chart(chart_df[["Close", "EMA_5", "EMA_8"]])
                with t2:
                    st.line_chart(chart_df[["TSI", "TSI_SIGNAL", "CCI", "ADX"]])
    else:
        st.info("Click 'Run scanner' to test one chosen setup across the current universe.")

with sweet_tab:
    if run_sweet_spot:
        combos, best = run_sweet_spot_serial(
            tickers=tickers,
            min_signals=min_signals,
            direction=direction,
            require_rejection=require_rejection,
            require_extension=require_extension,
        )

        if best.empty:
            st.warning("No sweet-spot results found.")
        else:
            save_sweet_spots(
                best,
                {
                    "years": years,
                    "direction": direction,
                    "min_signals": min_signals,
                },
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Tickers analyzed", best["symbol"].nunique())
            c2.metric("Best combos active now", int(best["latest_signal"].sum()))
            c3.metric("Top sweet-spot score", f"{best['score'].max():.2f}")

            best_cols = [
                "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold", "cci_state", "adx_state",
                "require_rejection", "require_extension", "signals", "hit_rate_1d", "hit_rate_2d",
                "avg_1d_return", "avg_2d_return", "best_excursion_2d", "latest_signal", "latest_tsi_pct", "score"
            ]
            st.dataframe(best[best_cols], use_container_width=True, hide_index=True)
            csv_download(best[best_cols], "best_combos_by_symbol.csv")

            combo_cols = [
                "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold", "cci_state", "adx_state",
                "signals", "hit_rate_1d", "hit_rate_2d", "avg_1d_return", "avg_2d_return",
                "best_excursion_2d", "latest_signal", "score"
            ]
            st.dataframe(
                combos.sort_values(["score", "hit_rate_1d", "signals"], ascending=[False, False, False])[combo_cols],
                use_container_width=True,
                hide_index=True,
            )
            csv_download(combos[combo_cols], "all_combo_results.csv")
    else:
        st.info("Click 'Run sweet-spot analysis' to search across TSI / CCI / ADX combinations.")

with live_tab:
    saved = load_sweet_spots()

    if saved.empty:
        st.info("No saved sweet spots found. Run sweet-spot analysis first.")
    else:
        st.write(f"Loaded {len(saved)} saved sweet spots from the latest run.")
        tsi_tol = st.slider("TSI approach tolerance", 1, 10, 3)

        if run_live_monitor:
            rows = []
            progress = st.progress(0.0)

            for i, (_, combo_row) in enumerate(saved.iterrows(), start=1):
                sym = combo_row["symbol"]
                base = load_base_features(sym)
                if base.empty:
                    progress.progress(i / len(saved))
                    continue

                chk = check_combo_live_from_base(base, combo_row, tsi_tolerance=tsi_tol)
                rows.append({**combo_row.to_dict(), **chk})
                progress.progress(i / len(saved))

            progress.empty()

            if rows:
                df_res = pd.DataFrame(rows)
                df_res["match_rank"] = df_res["match"].astype(int) * 2 + df_res["approach"].astype(int)
                df_res = df_res.sort_values(["match_rank", "score"], ascending=[False, False]).reset_index(drop=True)

                display_cols = [
                    "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold", "cci_state", "adx_state",
                    "score", "match", "approach", "tsi_pct_now", "latest_cci_now", "latest_adx_now",
                    "cci_delta1_now", "adx_delta1_now"
                ]
                st.dataframe(df_res[display_cols], use_container_width=True, hide_index=True)
                csv_download(df_res[display_cols], "live_monitor_results.csv")
            else:
                st.warning("No live monitor rows to show.")
    else:
        st.info("Click 'Run live monitor' to compare the latest saved sweet spots against current conditions.")
