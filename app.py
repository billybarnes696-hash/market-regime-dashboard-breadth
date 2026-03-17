
from __future__ import annotations
import io, re
from dataclasses import dataclass, asdict
from datetime import datetime, time
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import fitz
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Unified Breadth System", layout="wide")
st.title("📊 Unified Breadth System")
st.caption("PDF = truth, API = intraday speed, PNG = visual review only")

@dataclass
class FileAudit:
    file_name: str
    file_type: str
    inferred_timestamp: Optional[str]
    phase: str
    eod_mode: bool
    timestamp_source: str
    confidence: str

@dataclass
class SnapshotRow:
    file_name: str
    timestamp: Optional[str]
    symbol: str
    last: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    change: Optional[float] = None
    pct_b: Optional[float] = None
    rsi: Optional[float] = None
    roc: Optional[float] = None
    cci: Optional[float] = None
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None
    source_method: str = "pdf_extract"
    confidence: str = "Medium"

if "audit_rows" not in st.session_state:
    st.session_state.audit_rows = []
if "snapshot_rows" not in st.session_state:
    st.session_state.snapshot_rows = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "live_history_df" not in st.session_state:
    st.session_state.live_history_df = pd.DataFrame()
if "official_state" not in st.session_state:
    st.session_state.official_state = {"ema19_prev": 0.0, "ema39_prev": 0.0, "nysi_prev": 0.0, "nysi_proxy_prev": 0.0}

SYMBOLS = ["RSP","URSP","SPX","BPSPX","BPNYA","NYMO","NYSI","NYAD","NYHL","CPCE","SPXA50R","OEXA50R","OEXA150R","OEXA200R","RSP:SPY","IWM:SPY","SMH:SPY","XLF:SPY","HYG:TLT","HYG:IEF","VXX","SPXS:SVOL"]
DEFAULT_TICKERS = {"SPY":"SPY","RSP":"RSP","IWM":"IWM","SMH":"SMH","XLF":"XLF","HYG":"HYG","TLT":"TLT","IEF":"IEF","VXX":"VXX","SVOL":"SVOL","SPXS":"SPXS"}
DEFAULT_PROXY_WEIGHTS = {"rsp_vs_spy":0.28,"iwm_vs_spy":0.18,"smh_vs_spy":0.14,"xlf_vs_spy":0.10,"hyg_vs_tlt":0.14,"hyg_vs_ief":0.06,"spxs_vs_svol_inv":0.05,"vxx_inv":0.05}

def safe_float(x):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def infer_timestamp_from_filename(name: str):
    lower = name.lower()
    year = month = day = None
    ymd = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", lower)
    if ymd:
        year, month, day = int(ymd.group(1)), int(ymd.group(2)), int(ymd.group(3))
    else:
        md = re.search(r"(\d{1,2})[-_](\d{1,2})(?:[-_](\d{2,4}))?", lower)
        if md:
            month, day = int(md.group(1)), int(md.group(2))
            yr = md.group(3)
            year = datetime.now().year if not yr else int(yr)
            if year < 100:
                year += 2000
    hour = None
    minute = 0
    source = "filename"
    tmatch = re.search(r"\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)\b", lower)
    if tmatch:
        hour = int(tmatch.group(1))
        minute = int(tmatch.group(2) or 0)
        ap = tmatch.group(3)
        if ap == "pm" and hour != 12:
            hour += 12
        if ap == "am" and hour == 12:
            hour = 0
    elif "morning" in lower:
        hour, minute, source = 10, 0, "filename_morning"
    elif "noon" in lower:
        hour, minute, source = 12, 0, "filename_noon"
    if year and month and day:
        if hour is None:
            hour, source = 10, "filename_date_only"
        try:
            return datetime(year, month, day, hour, minute), source, "High"
        except Exception:
            return None, "unresolved", "Low"
    if hour is not None:
        now = datetime.now()
        return datetime(now.year, now.month, now.day, hour, minute), "filename_time_only", "Medium"
    return None, "unresolved", "Low"

def infer_phase(ts):
    if ts is None:
        return "10:00 AM Opening Read", False
    t = ts.time()
    if t < time(11, 0):
        return "10:00 AM Opening Read", False
    if t < time(14, 30):
        return "1:00 PM Midday Confirmation", False
    if t < time(18, 0):
        return "3:30 PM Late-Day Intent", False
    return "Post-6:00 PM EOD Regime Update", True

def as_df(items):
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([asdict(x) for x in items])

def normalize_text(s: str):
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s)

def pdf_extract_text(file_bytes: bytes):
    if not PDF_OK:
        return ""
    texts = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                texts.append(page.get_text("text"))
    except Exception:
        return ""
    return "\n".join(texts)

def png_ocr_text(file_bytes: bytes):
    if not (OCR_OK and PIL_OK):
        return ""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    except Exception:
        return ""

def extract_symbol_section(text: str, symbol: str, window: int = 1200):
    t = text.upper()
    s = symbol.upper()
    idx = t.find(s)
    if idx == -1:
        return ""
    return text[max(0, idx - 100): min(len(text), idx + window)]

def extract_metric(section: str, keys):
    for key in keys:
        pat = rf"{re.escape(key)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pat, section, flags=re.IGNORECASE)
        if m:
            return safe_float(m.group(1))
    return None

def extract_price_bundle(section):
    out = {"last": None, "open": None, "high": None, "low": None, "change": None}
    out["last"] = extract_metric(section, ["last", "close", "value"])
    out["open"] = extract_metric(section, ["open"])
    out["high"] = extract_metric(section, ["high"])
    out["low"] = extract_metric(section, ["low"])
    out["change"] = extract_metric(section, ["change"])
    if out["last"] is None:
        m = re.search(r"\$\s*(-?\d+(?:\.\d+)?)", section)
        if m:
            out["last"] = safe_float(m.group(1))
    return out

def parse_symbol_from_text(file_name: str, ts, text: str, source_method: str):
    rows = []
    text = normalize_text(text)
    for symbol in SYMBOLS:
        section = extract_symbol_section(text, symbol)
        if not section:
            continue
        price = extract_price_bundle(section)
        row = SnapshotRow(
            file_name=file_name,
            timestamp=ts.isoformat() if ts else None,
            symbol=symbol,
            last=price["last"],
            open=price["open"],
            high=price["high"],
            low=price["low"],
            change=price["change"],
            pct_b=extract_metric(section, ["%B", "BB%", "Percent B"]),
            rsi=extract_metric(section, ["RSI", "RSI(14)"]),
            roc=extract_metric(section, ["ROC", "ROC(12)", "ROC(14)"]),
            cci=extract_metric(section, ["CCI", "CCI(20)", "CCI(14)"]),
            adx=extract_metric(section, ["ADX", "ADX(14)"]),
            di_plus=extract_metric(section, ["DI+", "+DI"]),
            di_minus=extract_metric(section, ["DI-", "-DI"]),
            source_method=source_method,
            confidence="Medium",
        )
        if any(v is not None for v in [row.last,row.open,row.high,row.low,row.change,row.pct_b,row.rsi,row.roc,row.cci,row.adx,row.di_plus,row.di_minus]):
            rows.append(row)
    return rows

def get_latest_row(df, file_name: str, symbol: str):
    sub = df[(df["file_name"] == file_name) & (df["symbol"] == symbol)]
    if sub.empty:
        return None
    return sub.iloc[-1]

def ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

def mcclellan_oscillator_from_feed(adv_issues, dec_issues, ema19_prev, ema39_prev):
    total = adv_issues + dec_issues
    net = 0.0 if total <= 0 else 1000.0 * (adv_issues - dec_issues) / total
    alpha19 = 2.0 / 20.0
    alpha39 = 2.0 / 40.0
    ema19_now = ema(ema19_prev, net, alpha19)
    ema39_now = ema(ema39_prev, net, alpha39)
    return float(ema19_now - ema39_now), float(ema19_now), float(ema39_now)

def nysi_from_nymo(nysi_prev, nymo_now):
    return float(nysi_prev + nymo_now)

def cpce_from_feed(put_volume, call_volume):
    if call_volume <= 0:
        return None
    return float(put_volume / call_volume)

def fetch_latest_quotes_yf(tickers):
    rows = []
    for label, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="5m", auto_adjust=False)
            if hist.empty:
                rows.append({"label":label,"ticker":ticker,"last":np.nan,"prev_close":np.nan})
                continue
            rows.append({"label":label,"ticker":ticker,"last":float(hist["Close"].iloc[-1]),"prev_close":float(hist["Close"].iloc[0])})
        except Exception:
            rows.append({"label":label,"ticker":ticker,"last":np.nan,"prev_close":np.nan})
    return pd.DataFrame(rows)

def compute_proxy_feature_table(quotes):
    q = quotes.copy()
    q["ret"] = (q["last"] / q["prev_close"]) - 1.0
    q = q.set_index("label")
    def get_ret(label):
        if label not in q.index or pd.isna(q.loc[label, "ret"]):
            return 0.0
        return float(q.loc[label, "ret"])
    rows = [("rsp_vs_spy", get_ret("RSP") - get_ret("SPY")),("iwm_vs_spy", get_ret("IWM") - get_ret("SPY")),("smh_vs_spy", get_ret("SMH") - get_ret("SPY")),("xlf_vs_spy", get_ret("XLF") - get_ret("SPY")),("hyg_vs_tlt", get_ret("HYG") - get_ret("TLT")),("hyg_vs_ief", get_ret("HYG") - get_ret("IEF")),("spxs_vs_svol_inv", -(get_ret("SPXS") - get_ret("SVOL"))),("vxx_inv", -get_ret("VXX"))]
    return pd.DataFrame(rows, columns=["feature","value"])

def compute_nymo_proxy_eq(feature_df, weights):
    merged = feature_df.copy()
    merged["weight"] = merged["feature"].map(weights).fillna(0.0)
    merged["contribution"] = merged["value"] * 1000.0 * merged["weight"]
    return float(np.clip(merged["contribution"].sum(), -120.0, 120.0)), merged

def compute_nysi_proxy_eq(nysi_prev, nymo_proxy_eq):
    return float(nysi_prev + (nymo_proxy_eq * 0.35))

def compute_cpce_proxy_eq(quotes):
    q = quotes.set_index("label")
    def get_ret(label):
        if label not in q.index or pd.isna(q.loc[label, "last"]) or pd.isna(q.loc[label, "prev_close"]):
            return 0.0
        return float((q.loc[label, "last"] / q.loc[label, "prev_close"]) - 1.0)
    stress = (1.6 * get_ret("VXX")) + (0.8 * get_ret("SPXS")) - (0.8 * get_ret("SVOL"))
    return float(np.clip(0.60 + (stress * 10.0), 0.35, 1.20))

def participation_factor(bpspx, bpnya):
    def one(v):
        if v is None or pd.isna(v): return 0.0
        if v > 70: return 2
        if v > 55: return 1
        if v >= 40: return 0
        if v >= 25: return -1
        return -2
    return round((one(bpspx) + one(bpnya)) / 2.0, 2)

def momentum_factor(nymo_like):
    if nymo_like is None or pd.isna(nymo_like): return 0.0
    if nymo_like > 50: return 2
    if nymo_like > 0: return 1
    if nymo_like >= -50: return 0
    if nymo_like >= -80: return -1
    return -2

def breadth_depth_factor(spxa50r, oexa50r, oexa150r):
    def one(v):
        if v is None or pd.isna(v): return 0.0
        if v > 60: return 2
        if v >= 50: return 1
        if v >= 40: return 0
        if v >= 30: return -1
        return -2
    vals = [one(spxa50r), one(oexa50r), one(oexa150r)]
    return round(sum(vals) / len(vals), 2)

def sentiment_factor(cpce_like, vxx_last, vxx_prev):
    score = 0.0
    if cpce_like is not None and not pd.isna(cpce_like):
        if cpce_like > 0.80: score += 2
        elif cpce_like >= 0.60: score += 1
        elif cpce_like >= 0.45: score += 0
        elif cpce_like >= 0.35: score += -1
        else: score += -2
    if vxx_last is not None and vxx_prev is not None and not pd.isna(vxx_last) and not pd.isna(vxx_prev):
        score += 0.5 if vxx_last <= vxx_prev else -0.5
    return max(-2, min(2, score))

def confirmation_factor(rsp_last, rsp_prev, rsp_spy, hyg_tlt):
    score = 0.0
    if rsp_last is not None and rsp_prev is not None and not pd.isna(rsp_last) and not pd.isna(rsp_prev):
        score += 1 if rsp_last > rsp_prev else -1
    if rsp_spy is not None and not pd.isna(rsp_spy):
        score += 0.5 if rsp_spy > 0 else -0.5
    if hyg_tlt is not None and not pd.isna(hyg_tlt):
        score += 0.5 if hyg_tlt > 0.90 else -0.5
    return max(-2, min(2, score))

def classify_score(score):
    if score > 6: return "Strong Breadth Expansion"
    if score >= 3: return "Constructive"
    if score >= -2: return "Neutral / Transition"
    if score >= -6: return "Weakening / Correction"
    return "Capitulation / Breakdown"

def summarize_pdf_scores(snapshot_df, audit_df):
    if snapshot_df.empty or audit_df.empty:
        return pd.DataFrame()
    summaries = []
    for _, audit in audit_df.iterrows():
        file_name = audit["file_name"]
        eod_mode = bool(audit["eod_mode"])
        def val(sym, col="last"):
            row = get_latest_row(snapshot_df, file_name, sym)
            if row is None: return None
            return row.get(col)
        bpspx = val("BPSPX","last")
        bpnya = val("BPNYA","last")
        nymo = val("NYMO","last") if eod_mode else None
        cpce = val("CPCE","last") if eod_mode else None
        vxx = val("VXX","last")
        spxa50r = val("SPXA50R","last")
        oexa50r = val("OEXA50R","last")
        oexa150r = val("OEXA150R","last")
        rsp_pct_b = val("RSP","pct_b")
        rsp = val("RSP","last")
        rsp_spy = val("RSP:SPY","last")
        hyg_tlt = val("HYG:TLT","last")
        parts = {"participation":participation_factor(bpspx,bpnya),"momentum":momentum_factor(nymo if eod_mode else spxa50r),"breadth_depth":breadth_depth_factor(spxa50r,oexa50r,oexa150r),"sentiment":sentiment_factor(cpce,vxx,vxx),"confirmation":confirmation_factor(rsp,rsp,rsp_spy,hyg_tlt)}
        score = float(np.clip(sum(parts.values()), -10.0, 10.0))
        summaries.append({"file_name":file_name,"timestamp":audit["inferred_timestamp"],"phase":audit["phase"],"eod_mode":audit["eod_mode"],"score":round(score,2),"regime":classify_score(score),**parts})
    out = pd.DataFrame(summaries)
    if out.empty: return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["timestamp","file_name"], na_position="last")
    out["change_from_prior_upload"] = out["score"].diff()
    out["trade_date"] = out["timestamp"].dt.date
    prior_day_last = out.groupby("trade_date")["score"].last().shift(1)
    out["prior_day_baseline"] = out["trade_date"].map(prior_day_last)
    out["change_from_prior_day"] = out["score"] - out["prior_day_baseline"]
    return out

with st.sidebar:
    show_ocr = st.toggle("Show OCR preview for PNGs", value=False)
    uploaded_pdf = st.file_uploader("Upload PDF chart packs (primary truth source)", type=["pdf"], accept_multiple_files=True)
    uploaded_png = st.file_uploader("Upload PNGs for visual review only (optional)", type=["png"], accept_multiple_files=True)
    st.subheader("Live intraday engine")
    mode = st.radio("Intraday engine mode", ["Proxy Mode", "Official Feed Mode"], index=0)
    tickers = DEFAULT_TICKERS.copy()
    for k, v in list(tickers.items()):
        tickers[k] = st.text_input(f"{k} ticker", value=v)
    proxy_weights = DEFAULT_PROXY_WEIGHTS.copy()
    for k, v in list(proxy_weights.items()):
        proxy_weights[k] = st.slider(k, 0.0, 0.5, float(v), 0.01)
    adv_issues = st.number_input("Advancing issues", value=0.0, step=1.0)
    dec_issues = st.number_input("Declining issues", value=0.0, step=1.0)
    put_volume = st.number_input("Put volume", value=0.0, step=1.0)
    call_volume = st.number_input("Call volume", value=0.0, step=1.0)
    bpspx_override = st.text_input("BPSPX", value="")
    bpnya_override = st.text_input("BPNYA", value="")
    spxa50r_override = st.text_input("SPXA50R", value="")
    oexa50r_override = st.text_input("OEXA50R", value="")
    oexa150r_override = st.text_input("OEXA150R", value="")
    rsp_spy_override = st.text_input("RSP:SPY", value="")
    hyg_tlt_override = st.text_input("HYG:TLT", value="")
    run_live = st.button("Refresh live dashboard", type="primary")
    if st.button("Clear session data"):
        st.session_state.audit_rows = []
        st.session_state.snapshot_rows = []
        st.session_state.uploaded_files = {}
        st.session_state.live_history_df = pd.DataFrame()
        st.rerun()

if uploaded_pdf:
    for f in uploaded_pdf:
        if f.name not in st.session_state.uploaded_files:
            file_bytes = f.read()
            st.session_state.uploaded_files[f.name] = {"type":"pdf","bytes":file_bytes}
            ts, ts_source, conf = infer_timestamp_from_filename(f.name)
            phase, eod_mode = infer_phase(ts)
            st.session_state.audit_rows.append(FileAudit(file_name=f.name,file_type="pdf",inferred_timestamp=ts.isoformat() if ts else None,phase=phase,eod_mode=eod_mode,timestamp_source=ts_source,confidence=conf))
            text = pdf_extract_text(file_bytes)
            rows = parse_symbol_from_text(f.name, ts, text, "pdf_text")
            st.session_state.snapshot_rows.extend(rows)

if uploaded_png:
    for f in uploaded_png:
        if f.name not in st.session_state.uploaded_files:
            file_bytes = f.read()
            st.session_state.uploaded_files[f.name] = {"type":"png","bytes":file_bytes}
            ts, ts_source, conf = infer_timestamp_from_filename(f.name)
            phase, eod_mode = infer_phase(ts)
            st.session_state.audit_rows.append(FileAudit(file_name=f.name,file_type="png",inferred_timestamp=ts.isoformat() if ts else None,phase=phase,eod_mode=eod_mode,timestamp_source=ts_source,confidence=conf))

audit_df = as_df(st.session_state.audit_rows)
if not audit_df.empty:
    audit_df["inferred_timestamp"] = pd.to_datetime(audit_df["inferred_timestamp"], errors="coerce")
    audit_df = audit_df.sort_values(["inferred_timestamp","file_name"], na_position="last")

snapshot_df = as_df(st.session_state.snapshot_rows)
if not snapshot_df.empty:
    snapshot_df["timestamp"] = pd.to_datetime(snapshot_df["timestamp"], errors="coerce")

pdf_audit_df = audit_df[audit_df["file_type"] == "pdf"].copy() if not audit_df.empty else pd.DataFrame()
png_audit_df = audit_df[audit_df["file_type"] == "png"].copy() if not audit_df.empty else pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["PDF Truth Engine","PNG Review","Live Intraday Engine","Unified Dashboard"])

with tab1:
    st.subheader("📄 PDF Truth Engine")
    if pdf_audit_df.empty:
        st.info("Upload PDF chart packs in the sidebar.")
    else:
        st.dataframe(pdf_audit_df, use_container_width=True)
        if snapshot_df.empty:
            st.warning("No rows parsed from PDF text.")
        else:
            st.dataframe(snapshot_df.sort_values(["timestamp","file_name","symbol"], na_position="last"), use_container_width=True)
        pdf_score_df = summarize_pdf_scores(snapshot_df, pdf_audit_df)
        if not pdf_score_df.empty:
            st.subheader("PDF-derived breadth scores")
            st.dataframe(pdf_score_df, use_container_width=True)
            if PLOTLY_OK:
                st.plotly_chart(px.line(pdf_score_df, x="timestamp", y="score", markers=True, title="PDF Truth Score History"), use_container_width=True)

with tab2:
    st.subheader("🖼️ PNG Review Only")
    if png_audit_df.empty:
        st.info("Optional PNG uploads will appear here for visual review.")
    else:
        st.dataframe(png_audit_df, use_container_width=True)
        chosen = st.selectbox("Choose PNG", png_audit_df["file_name"].tolist(), key="png_choose")
        data = st.session_state.uploaded_files[chosen]["bytes"]
        st.image(data, caption=chosen, use_container_width=True)
        if show_ocr:
            st.text_area("OCR preview", png_ocr_text(data), height=220)

with tab3:
    st.subheader("⚡ Live Intraday Engine")
    if run_live:
        now = datetime.now()
        phase = infer_phase(now)[0]
        quotes = fetch_latest_quotes_yf(tickers)
        q = quotes.set_index("label")
        def qlast(label):
            if label not in q.index: return None
            v = q.loc[label, "last"]
            return None if pd.isna(v) else float(v)
        def qprev(label):
            if label not in q.index: return None
            v = q.loc[label, "prev_close"]
            return None if pd.isna(v) else float(v)
        feature_df = compute_proxy_feature_table(quotes)
        nymo_proxy_eq, feature_detail = compute_nymo_proxy_eq(feature_df, proxy_weights)
        cpce_proxy_eq = compute_cpce_proxy_eq(quotes)
        nysi_proxy_eq = compute_nysi_proxy_eq(st.session_state.official_state["nysi_proxy_prev"], nymo_proxy_eq)
        st.session_state.official_state["nysi_proxy_prev"] = nysi_proxy_eq
        official_nymo_eq = official_nysi_eq = official_cpce = None
        if mode == "Official Feed Mode" and adv_issues > 0 and dec_issues > 0:
            official_nymo_eq, ema19_now, ema39_now = mcclellan_oscillator_from_feed(adv_issues, dec_issues, st.session_state.official_state["ema19_prev"], st.session_state.official_state["ema39_prev"])
            st.session_state.official_state["ema19_prev"] = ema19_now
            st.session_state.official_state["ema39_prev"] = ema39_now
            official_nysi_eq = nysi_from_nymo(st.session_state.official_state["nysi_prev"], official_nymo_eq)
            st.session_state.official_state["nysi_prev"] = official_nysi_eq
            if put_volume > 0 and call_volume > 0:
                official_cpce = cpce_from_feed(put_volume, call_volume)
        active_nymo_eq = official_nymo_eq if official_nymo_eq is not None else nymo_proxy_eq
        active_nysi_eq = official_nysi_eq if official_nysi_eq is not None else nysi_proxy_eq
        active_cpce_eq = official_cpce if official_cpce is not None else cpce_proxy_eq
        bpspx = safe_float(bpspx_override)
        bpnya = safe_float(bpnya_override)
        spxa50r = safe_float(spxa50r_override)
        oexa50r = safe_float(oexa50r_override)
        oexa150r = safe_float(oexa150r_override)
        rsp_spy_val = safe_float(rsp_spy_override)
        hyg_tlt_val = safe_float(hyg_tlt_override)
        if rsp_spy_val is None:
            rsp, spy = qlast("RSP"), qlast("SPY")
            rsp_spy_val = ((rsp / spy) - 1.0) if (rsp is not None and spy is not None and spy != 0) else 0.0
        if hyg_tlt_val is None:
            hyg, tlt = qlast("HYG"), qlast("TLT")
            hyg_tlt_val = (hyg / tlt) if (hyg is not None and tlt is not None and tlt != 0) else None
        parts = {"participation":participation_factor(bpspx,bpnya),"momentum":momentum_factor(active_nymo_eq),"breadth_depth":breadth_depth_factor(spxa50r,oexa50r,oexa150r),"sentiment":sentiment_factor(active_cpce_eq, qlast("VXX"), qprev("VXX")),"confirmation":confirmation_factor(qlast("RSP"), qprev("RSP"), rsp_spy_val, hyg_tlt_val)}
        score = float(np.clip(sum(parts.values()), -10.0, 10.0))
        row = {"timestamp":pd.Timestamp(now),"phase":phase,"mode":mode,"score":round(score,2),"regime":classify_score(score),"nymo_eq":active_nymo_eq,"nysi_eq":active_nysi_eq,"cpce_eq":active_cpce_eq,"rsp":qlast("RSP"),"spy":qlast("SPY"),"vxx":qlast("VXX"),"bpspx_manual":bpspx,"bpnya_manual":bpnya,"spxa50r_manual":spxa50r,"oexa50r_manual":oexa50r,"oexa150r_manual":oexa150r,**parts}
        st.session_state.live_history_df = pd.concat([st.session_state.live_history_df, pd.DataFrame([row])], ignore_index=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Phase", phase)
        c2.metric("Breadth score", round(score,2))
        c3.metric("NYMO eq", None if active_nymo_eq is None else round(active_nymo_eq,2))
        c4.metric("NYSI eq", None if active_nysi_eq is None else round(active_nysi_eq,2))
        c5.metric("CPCE eq", None if active_cpce_eq is None else round(active_cpce_eq,3))
        st.dataframe(quotes, use_container_width=True)
        st.dataframe(feature_detail, use_container_width=True)
        hist = st.session_state.live_history_df.copy()
        if not hist.empty:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
            if PLOTLY_OK:
                st.plotly_chart(px.line(hist, x="timestamp", y="score", markers=True, title="Live Breadth Score History"), use_container_width=True)
                st.plotly_chart(px.line(hist, x="timestamp", y=["nymo_eq","nysi_eq","cpce_eq"], title="Live Equivalent History"), use_container_width=True)
            st.dataframe(hist.tail(50), use_container_width=True)
    else:
        st.info("Set live inputs in the sidebar and click Refresh live dashboard.")

with tab4:
    st.subheader("🧠 Unified Dashboard")
    left, right = st.columns(2)
    with left:
        st.markdown("### PDF Truth Snapshot")
        if pdf_audit_df.empty:
            st.info("No PDF truth data yet.")
        else:
            pdf_score_df = summarize_pdf_scores(snapshot_df, pdf_audit_df)
            if pdf_score_df.empty:
                st.info("No score from PDFs yet.")
            else:
                latest_pdf = pdf_score_df.sort_values("timestamp", na_position="last").iloc[-1]
                st.metric("Latest PDF score", latest_pdf["score"])
                st.write(f"Regime: **{latest_pdf['regime']}**")
                st.write(f"Phase: **{latest_pdf['phase']}**")
    with right:
        st.markdown("### Live Intraday Snapshot")
        if st.session_state.live_history_df.empty:
            st.info("No live data yet.")
        else:
            latest_live = st.session_state.live_history_df.sort_values("timestamp").iloc[-1]
            st.metric("Latest live score", latest_live["score"])
            st.write(f"Regime: **{latest_live['regime']}**")
            st.write(f"Phase: **{latest_live['phase']}**")
    st.markdown("### Recommended workflow")
    st.markdown("1. Upload **PDFs** for exact historical/current chart values.  
2. Use the **live intraday engine** for NYMO/NYSI/CPCE equivalents during the day.  
3. Use **PNGs** only for visual review, not as the primary data source.  
4. Manually enter BPSPX/BPNYA/SPXA50R/OEX values in the sidebar when you have them intraday.")

st.markdown("---")
st.markdown("### Honest limitation")
st.markdown("There is no universal free public API that gives official real-time NYMO, NYSI, and CPCE exactly like StockCharts. This app solves that in two ways: direct formulas when you provide advancing/declining and put/call inputs, and transparent proxy equivalents when you do not.")
