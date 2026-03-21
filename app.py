import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from itertools import product

st.set_page_config(page_title="TSI + CCI + ADX Dashboard", layout="wide")

DEFAULT_TICKERS = [
    "SPY","QQQ","IWM","DIA","XLF","XLK","SMH","XLE","TLT","GLD",
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","INTC","MU","PLTR"
]

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
    return adx_val, plus_di, minus_di


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start_date, end_date):
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return data


def extract_symbol_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
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
    df.index = pd.to_datetime(df.index)
    return df


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
    if df.empty or len(df) < 260:
        return None

    t_long, t_short, t_signal = tsi_params
    out = df.copy()
    out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], t_long, t_short, t_signal)
    out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
    out["ADX"], out["DI_PLUS"], out["DI_MINUS"] = adx(out["High"], out["Low"], out["Close"], adx_len)
    out["RET_1D"] = out["Close"].shift(-1) / out["Close"] - 1
    out["RET_2D"] = out["Close"].shift(-2) / out["Close"] - 1

    out["TSI_PCT"] = out["TSI"].rolling(252, min_periods=60).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False
    )
    out["CCI_DELTA_1"] = out["CCI"].diff()
    out["CCI_DELTA_2"] = out["CCI"].diff(2)
    out["ADX_DELTA_1"] = out["ADX"].diff()

    cond_cci, cond_adx = apply_states(out, cci_state, adx_state)
    signal = (out["TSI_PCT"] >= tsi_pct_threshold) & cond_cci & cond_adx
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


@st.cache_data(ttl=3600, show_spinner=False)
def sweet_spot_analysis(raw, tickers, years, min_signals):
    tsi_options = [(4, 2, 4), (6, 3, 6), (7, 4, 7)]
    cci_options = [5, 7, 10, 14]
    adx_options = [7, 10, 14]
    cci_states = ["down_1d", "down_2d"]
    adx_states = ["flat_or_down", "down_1d", "any"]
    tsi_thresholds = [95, 97, 99]

    combo_rows = []

    for sym in tickers:
        df = extract_symbol_df(raw, sym)
        if df.empty or len(df) < 260:
            continue

        for tsi_params, cci_len, adx_len, cci_state, adx_state, tsi_pct_threshold in product(
            tsi_options, cci_options, adx_options, cci_states, adx_states, tsi_thresholds
        ):
            result = backtest_symbol(
                df=df,
                tsi_params=tsi_params,
                cci_len=cci_len,
                adx_len=adx_len,
                tsi_pct_threshold=tsi_pct_threshold,
                cci_state=cci_state,
                adx_state=adx_state,
            )
            if result is None:
                continue

            signals = result["signals"]
            if signals < min_signals:
                continue

            score = (
                result["next_day_red_pct"] * 0.55
                + (-result["avg_1d_return"]) * 15 * 0.25
                + min(signals, 50) * 0.20
            )

            combo_rows.append({
                "symbol": sym,
                "tsi_combo": f"{tsi_params[0]},{tsi_params[1]},{tsi_params[2]}",
                "cci_len": cci_len,
                "adx_len": adx_len,
                "tsi_pct_threshold": tsi_pct_threshold,
                "cci_state": cci_state,
                "adx_state": adx_state,
                "signals": signals,
                "next_day_red_pct": result["next_day_red_pct"],
                "two_day_red_pct": result["two_day_red_pct"],
                "avg_1d_return": result["avg_1d_return"],
                "avg_2d_return": result["avg_2d_return"],
                "latest_signal": result["latest_signal"],
                "latest_tsi": result["latest_tsi"],
                "latest_tsi_pct": result["latest_tsi_pct"],
                "latest_cci": result["latest_cci"],
                "latest_cci_delta": result["latest_cci_delta"],
                "latest_adx": result["latest_adx"],
                "latest_adx_delta": result["latest_adx_delta"],
                "score": score,
            })

    if not combo_rows:
        return pd.DataFrame(), pd.DataFrame()

    combos = pd.DataFrame(combo_rows)
    best = combos.sort_values(
        ["symbol", "score", "next_day_red_pct", "signals"],
        ascending=[True, False, False, False],
    ).groupby("symbol", as_index=False).first()

    best = best.sort_values(["latest_signal", "score"], ascending=[False, False]).reset_index(drop=True)
    return combos, best


st.title("TSI + CCI + ADX Exhaustion Dashboard")
st.caption("Focused Streamlit scanner plus historical sweet-spot analyzer")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
        height=150,
    )
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

    run_scanner = st.button("Run scanner", type="primary")
    run_sweet_spot = st.button("Run sweet-spot analysis")

end_date = date.today() + timedelta(days=1)
start_date = date.today() - timedelta(days=int(365.25 * years))
raw = load_data(tickers, start_date, end_date)

if run_scanner:
    rows = []
    charts = {}

    for sym in tickers:
        df = extract_symbol_df(raw, sym)
        result = backtest_symbol(
            df=df,
            tsi_params=tsi_params,
            cci_len=cci_len,
            adx_len=adx_len,
            tsi_pct_threshold=tsi_pct_threshold,
            cci_state=cci_state,
            adx_state=adx_state,
        )
        if result is None:
            continue

        chart_df = result.pop("chart_df", None)
        if chart_df is not None:
            charts[sym] = chart_df
        rows.append({"symbol": sym, **result})

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
            c4.metric("Best avg 1D return", f"{results['avg_1d_return'].min():.2f}%")

            st.subheader("Scanner results")
            display = results[[
                "symbol", "latest_signal", "latest_tsi", "latest_tsi_pct", "latest_cci",
                "latest_cci_delta", "latest_adx", "latest_adx_delta", "signals",
                "next_day_red_pct", "two_day_red_pct", "avg_1d_return", "avg_2d_return", "score"
            ]].copy()
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.subheader("Chart review")
            selected = st.selectbox("Select symbol", options=results["symbol"].tolist())
            chart_df = charts[selected].copy()
            chart_df["TSI"], chart_df["TSI_SIGNAL"] = tsi(chart_df["Close"], *tsi_params)
            chart_df["CCI"] = cci(chart_df["High"], chart_df["Low"], chart_df["Close"], cci_len)
            chart_df["ADX"], chart_df["DI_PLUS"], chart_df["DI_MINUS"] = adx(chart_df["High"], chart_df["Low"], chart_df["Close"], adx_len)

            t1, t2 = st.tabs(["Price", "Indicators"])
            with t1:
                st.line_chart(chart_df[["Close"]])
            with t2:
                st.line_chart(chart_df[["TSI", "TSI_SIGNAL", "CCI", "ADX"]])

if run_sweet_spot:
    with st.spinner("Running historical sweet-spot analysis across TSI / CCI / ADX combos..."):
        combos, best = sweet_spot_analysis(raw, tickers, years, min_signals)

    if best.empty:
        st.warning("No sweet-spot results found. Try lowering the minimum signal threshold or reducing the ticker list.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers analyzed", best["symbol"].nunique())
        c2.metric("Best combos active now", int(best["latest_signal"].sum()))
        c3.metric("Top sweet-spot score", f"{best['score'].max():.2f}")

        st.subheader("Best historical combo by symbol")
        st.dataframe(best[[
            "symbol", "tsi_combo", "cci_len", "adx_len", "tsi_pct_threshold",
            "cci_state", "adx_state", "signals", "next_day_red_pct",
            "two_day_red_pct", "avg_1d_return", "avg_2d_return",
            "latest_signal", "latest_tsi_pct", "score"
        ]], use_container_width=True, hide_index=True)

        st.subheader("Full combo table")
        st.dataframe(combos.sort_values(["score", "next_day_red_pct"], ascending=[False, False]), use_container_width=True, hide_index=True)
else:
    if not run_scanner:
        st.info("Use 'Run scanner' for one chosen setup, or 'Run sweet-spot analysis' to find each symbol's best historical TSI/CCI/ADX combo.")

