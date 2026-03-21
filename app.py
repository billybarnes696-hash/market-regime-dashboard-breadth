import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="TSI + CCI Dashboard", layout="wide")

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


def percentile_rank_last(series: pd.Series, window: int = 252):
    s = series.dropna()
    if len(s) < max(window, 30):
        return np.nan
    recent = s.iloc[-window:]
    last_val = recent.iloc[-1]
    return float((recent <= last_val).mean() * 100)


def backtest_symbol(df: pd.DataFrame, tsi_params, cci_len, tsi_pct_threshold, cci_state):
    if df.empty or len(df) < 260:
        return None

    t_long, t_short, t_signal = tsi_params
    out = df.copy()
    out["TSI"], out["TSI_SIGNAL"] = tsi(out["Close"], t_long, t_short, t_signal)
    out["CCI"] = cci(out["High"], out["Low"], out["Close"], cci_len)
    out["RET_1D"] = out["Close"].shift(-1) / out["Close"] - 1
    out["RET_2D"] = out["Close"].shift(-2) / out["Close"] - 1

    # Rolling percentile of TSI using only prior data up to each row
    out["TSI_PCT"] = out["TSI"].rolling(252, min_periods=60).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False
    )

    out["CCI_DELTA_1"] = out["CCI"].diff()
    out["CCI_DELTA_2"] = out["CCI"].diff(2)

    if cci_state == "down_1d":
        cond_cci = out["CCI_DELTA_1"] < 0
    elif cci_state == "down_2d":
        cond_cci = (out["CCI_DELTA_1"] < 0) & (out["CCI_DELTA_2"] < 0)
    elif cci_state == "up_1d":
        cond_cci = out["CCI_DELTA_1"] > 0
    else:
        cond_cci = pd.Series(True, index=out.index)

    signal = (out["TSI_PCT"] >= tsi_pct_threshold) & cond_cci
    hits = out[signal].copy()
    hits = hits.dropna(subset=["RET_1D", "RET_2D", "TSI", "CCI", "TSI_PCT"])

    if hits.empty:
        return {
            "signals": 0,
            "next_day_red_pct": np.nan,
            "two_day_red_pct": np.nan,
            "avg_1d_return": np.nan,
            "avg_2d_return": np.nan,
            "latest_signal": False,
            "latest_tsi": out["TSI"].iloc[-1],
            "latest_tsi_pct": out["TSI_PCT"].iloc[-1],
            "latest_cci": out["CCI"].iloc[-1],
            "latest_cci_delta": out["CCI_DELTA_1"].iloc[-1],
        }

    latest_signal = bool(signal.iloc[-1]) if len(signal) else False

    return {
        "signals": int(len(hits)),
        "next_day_red_pct": float((hits["RET_1D"] < 0).mean() * 100),
        "two_day_red_pct": float((hits["RET_2D"] < 0).mean() * 100),
        "avg_1d_return": float(hits["RET_1D"].mean() * 100),
        "avg_2d_return": float(hits["RET_2D"].mean() * 100),
        "latest_signal": latest_signal,
        "latest_tsi": float(out["TSI"].iloc[-1]),
        "latest_tsi_pct": float(out["TSI_PCT"].iloc[-1]) if pd.notna(out["TSI_PCT"].iloc[-1]) else np.nan,
        "latest_cci": float(out["CCI"].iloc[-1]),
        "latest_cci_delta": float(out["CCI_DELTA_1"].iloc[-1]) if pd.notna(out["CCI_DELTA_1"].iloc[-1]) else np.nan,
        "chart_df": out.tail(180).copy(),
    }


st.title("TSI + CCI Exhaustion Dashboard")
st.caption("Simple Streamlit scanner for a focused ETF/stock watchlist")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
        height=150,
    )
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

    st.subheader("TSI")
    tsi_choice = st.selectbox(
        "TSI combo",
        options=["4,2,4", "6,3,6", "7,4,7"],
        index=0,
    )
    tsi_params = tuple(int(x) for x in tsi_choice.split(","))

    st.subheader("CCI")
    cci_len = st.selectbox("CCI length", options=[5, 7, 10, 14], index=1)
    cci_state = st.selectbox(
        "CCI state",
        options=["down_1d", "down_2d", "up_1d", "any"],
        index=0,
    )

    st.subheader("Signal rule")
    tsi_pct_threshold = st.slider("TSI percentile threshold", 85, 99, 97)
    years = st.slider("Years of history", 2, 10, 5)
    min_signals = st.slider("Minimum historical signals", 3, 50, 8)

    run_btn = st.button("Run dashboard", type="primary")

if run_btn:
    end_date = date.today() + timedelta(days=1)
    start_date = date.today() - timedelta(days=int(365.25 * years))

    raw = load_data(tickers, start_date, end_date)

    rows = []
    charts = {}

    for sym in tickers:
        df = extract_symbol_df(raw, sym)
        result = backtest_symbol(
            df=df,
            tsi_params=tsi_params,
            cci_len=cci_len,
            tsi_pct_threshold=tsi_pct_threshold,
            cci_state=cci_state,
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

            st.subheader("Ranked results")
            display = results[[
                "symbol", "latest_signal", "latest_tsi", "latest_tsi_pct", "latest_cci",
                "latest_cci_delta", "signals", "next_day_red_pct", "two_day_red_pct",
                "avg_1d_return", "avg_2d_return", "score"
            ]].copy()
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.subheader("Chart review")
            selected = st.selectbox("Select symbol", options=results["symbol"].tolist())
            chart_df = charts[selected].copy()
            chart_df["TSI"], chart_df["TSI_SIGNAL"] = tsi(chart_df["Close"], *tsi_params)
            chart_df["CCI"] = cci(chart_df["High"], chart_df["Low"], chart_df["Close"], cci_len)

            t1, t2 = st.tabs(["Price", "Indicators"])
            with t1:
                st.line_chart(chart_df[["Close"]])
            with t2:
                st.line_chart(chart_df[["TSI", "TSI_SIGNAL", "CCI"]])
else:
    st.info("Choose your tickers and settings in the sidebar, then click 'Run dashboard'.")
