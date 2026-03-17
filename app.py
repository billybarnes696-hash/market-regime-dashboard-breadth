import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

# Optional imports. App still runs if some are missing.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = None
    ImageOps = None
    ImageFilter = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Breadth PDF Dashboard", layout="wide")
st.title("📊 Breadth PDF Dashboard")
st.caption("Upload historical and current StockCharts PDF packs. The app sorts by file timestamp, extracts chart data, and builds a live dashboard.")


# =========================================================
# MODELS
# =========================================================
@dataclass
class FileAudit:
    file_name: str
    inferred_timestamp: Optional[str]
    phase: str
    eod_mode: bool
    source: str
    confidence: str


@dataclass
class IndicatorRow:
    file_name: str
    timestamp: Optional[str]
    indicator: str
    current_value: Optional[float] = None
    pct_b: Optional[float] = None
    rsi: Optional[float] = None
    roc: Optional[float] = None
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None
    extraction_source: str = ""
    extraction_confidence: str = "Low"


# =========================================================
# HELPERS: TIME / PHASE
# =========================================================
def infer_timestamp_from_filename(name: str) -> Tuple[Optional[datetime], str, str]:
    lower = name.lower()

    # Explicit common patterns
    # Examples: 3-16 morning, noon, 240, 1036am, 1:40pm, 330pm
    date_match = re.search(r'(\d{1,2})[-_/](\d{1,2})(?:[-_/](\d{2,4}))?', lower)
    hour = None
    minute = 0
    suffix = None

    # 1036am / 330pm / 1:40pm / 12pm
    m = re.search(r'\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)\b', lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        suffix = m.group(3)
    elif 'noon' in lower:
        hour, minute, suffix = 12, 0, 'pm'
    elif 'morning' in lower:
        hour, minute, suffix = 10, 0, 'am'

    if hour is not None and suffix:
        if suffix == 'pm' and hour != 12:
            hour += 12
        if suffix == 'am' and hour == 12:
            hour = 0

    if date_match:
        month = int(date_match.group(1))
        day = int(date_match.group(2))
        year_raw = date_match.group(3)
        year = datetime.now().year if not year_raw else int(year_raw)
        if year < 100:
            year += 2000
        if hour is None:
            hour = 10
        try:
            return datetime(year, month, day, hour, minute), "filename", "High"
        except Exception:
            pass

    # time only, assume current year and today's date if missing
    if hour is not None:
        now = datetime.now()
        try:
            return datetime(now.year, now.month, now.day, hour, minute), "filename_time_only", "Medium"
        except Exception:
            pass

    return None, "unresolved", "Low"


def infer_phase(ts: Optional[datetime]) -> Tuple[str, bool]:
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


# =========================================================
# OCR / PDF EXTRACTION
# =========================================================
def preprocess_image(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def render_pdf_page(doc, page_num: int, zoom: float = 3.0):
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    if Image is None:
        return None
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_pdf_text_primary(file_bytes: bytes) -> str:
    if fitz is None:
        return ""
    text_chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            try:
                text_chunks.append(page.get_text("text"))
            except Exception:
                pass
    return "\n".join(text_chunks)


def extract_pdf_text_ocr(file_bytes: bytes, max_pages: int = 25) -> str:
    if fitz is None or pytesseract is None or Image is None:
        return ""
    text_chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            try:
                img = render_pdf_page(doc, i, zoom=3.5)
                if img is None:
                    continue
                img = preprocess_image(img)
                txt = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
                text_chunks.append(txt)
            except Exception:
                pass
    return "\n".join(text_chunks)


def extract_all_text(file_bytes: bytes, use_ocr_fallback: bool = True) -> Tuple[str, str]:
    primary = extract_pdf_text_primary(file_bytes)
    primary_len = len(primary.strip())

    if primary_len > 1000:
        return primary, "pdf_text"

    if use_ocr_fallback:
        ocr_text = extract_pdf_text_ocr(file_bytes)
        combined = (primary + "\n" + ocr_text).strip()
        return combined, "pdf_text+ocr"

    return primary, "pdf_text"


# =========================================================
# CHART / VALUE PARSING
# =========================================================
INDICATORS = [
    "BPSPX", "BPNYA", "NYMO", "NYSI", "NYAD", "NYHL", "CPCE",
    "SPXA50R", "OEXA50R", "OEXA150R", "OEXA200R",
    "RSP", "URSP", "RSP:SPY", "IWM:SPY", "SMH:SPY", "XLF:SPY",
    "HYG:IEF", "HYG:TLT", "VXX", "SPXS:SVOL"
]


def clean_text(text: str) -> str:
    text = text.replace("$", " $")
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_section_near_indicator(text: str, indicator: str, window: int = 900) -> str:
    idx = text.upper().find(indicator.upper())
    if idx == -1:
        return ""
    start = max(0, idx - 120)
    end = min(len(text), idx + window)
    return text[start:end]


def find_numeric_after_keywords(section: str, keywords: List[str]) -> Optional[float]:
    for kw in keywords:
        pattern = rf'{re.escape(kw)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)'
        m = re.search(pattern, section, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def find_first_price_like(section: str) -> Optional[float]:
    # Avoid obvious indicators first; look for dollars or price-like values.
    patterns = [
        r'\$\s*(-?\d+(?:\.\d+)?)',
        r'close\s*[:=]?\s*(-?\d+(?:\.\d+)?)',
        r'last\s*[:=]?\s*(-?\d+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.search(p, section, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def parse_indicator_values(text: str, file_name: str, ts: Optional[datetime], source: str) -> List[IndicatorRow]:
    rows: List[IndicatorRow] = []
    normalized = clean_text(text)

    for ind in INDICATORS:
        section = extract_section_near_indicator(normalized, ind)
        if not section:
            continue

        current = None
        if ind in {"RSP", "URSP", "VXX", "RSP:SPY", "IWM:SPY", "SMH:SPY", "XLF:SPY", "HYG:IEF", "HYG:TLT", "SPXS:SVOL"}:
            current = find_first_price_like(section)
        if current is None:
            current = find_numeric_after_keywords(section, [ind, "value", "current", "close", "last"])

        pct_b = find_numeric_after_keywords(section, ["%B", "Bollinger %B", "PctB", "Percent B"])
        rsi = find_numeric_after_keywords(section, ["RSI", "RSI(14)"])
        roc = find_numeric_after_keywords(section, ["ROC", "ROC(12)", "ROC(14)"])
        adx = find_numeric_after_keywords(section, ["ADX", "ADX(14)"])
        di_plus = find_numeric_after_keywords(section, ["DI+", "+DI"])
        di_minus = find_numeric_after_keywords(section, ["DI-", "-DI"])

        confidence = "Low"
        hits = sum(v is not None for v in [current, pct_b, rsi, roc, adx, di_plus, di_minus])
        if hits >= 4:
            confidence = "High"
        elif hits >= 2:
            confidence = "Medium"

        rows.append(
            IndicatorRow(
                file_name=file_name,
                timestamp=ts.isoformat() if ts else None,
                indicator=ind,
                current_value=current,
                pct_b=pct_b,
                rsi=rsi,
                roc=roc,
                adx=adx,
                di_plus=di_plus,
                di_minus=di_minus,
                extraction_source=source,
                extraction_confidence=confidence,
            )
        )

    return rows


# =========================================================
# SCORING / TRIGGERS
# =========================================================
def participation_factor(bpspx: Optional[float], bpnya: Optional[float]) -> float:
    def one(v):
        if v is None:
            return 0.0
        if v > 70:
            return 2
        if v > 55:
            return 1
        if v >= 40:
            return 0
        if v >= 25:
            return -1
        return -2
    return round((one(bpspx) + one(bpnya)) / 2.0, 2)


def momentum_factor(nymo: Optional[float], nysi: Optional[float], eod_mode: bool, intraday_proxy: Optional[float]) -> float:
    if eod_mode:
        base = 0.0
        if nymo is not None:
            if nymo > 50:
                base = 2
            elif nymo > 0:
                base = 1
            elif nymo >= -50:
                base = 0
            elif nymo >= -80:
                base = -1
            else:
                base = -2
        slope_mod = 0.0
        # nysi level proxy, not perfect; you can improve later.
        if nysi is not None:
            slope_mod = 0.0
        return max(-2, min(2, base + slope_mod))
    if intraday_proxy is None:
        return 0.0
    if intraday_proxy > 2:
        return 0.5
    if intraday_proxy < -2:
        return -0.5
    return 0.0


def breadth_depth_factor(spxa50r: Optional[float], oexa50r: Optional[float], oexa150r: Optional[float]) -> float:
    def one(v):
        if v is None:
            return 0.0
        if v > 60:
            return 2
        if v >= 50:
            return 1
        if v >= 40:
            return 0
        if v >= 30:
            return -1
        return -2
    vals = [one(spxa50r), one(oexa50r), one(oexa150r)]
    return round(sum(vals) / len(vals), 2)


def sentiment_factor(cpce: Optional[float], vxx: Optional[float], eod_mode: bool) -> float:
    score = 0.0
    if eod_mode and cpce is not None:
        if cpce > 0.80:
            score += 2
        elif cpce >= 0.60:
            score += 1
        elif cpce >= 0.45:
            score += 0
        elif cpce >= 0.35:
            score += -1
        else:
            score += -2
    # If VXX exists, lower VXX is constructive.
    if vxx is not None:
        score += 0.5 if vxx < 35 else -0.5
    return max(-2, min(2, score))


def confirmation_factor(rsp_pct_b: Optional[float], rsp_spy: Optional[float], hyg_tlt: Optional[float]) -> float:
    score = 0.0
    if rsp_pct_b is not None:
        score += 1 if rsp_pct_b > 0.20 else -1
    if rsp_spy is not None:
        score += 0.5 if rsp_spy > 0 else -0.5
    if hyg_tlt is not None:
        score += 0.5 if hyg_tlt > 0.90 else -0.5
    return max(-2, min(2, score))


def breadth_score_from_snapshot(snapshot: Dict[str, Dict[str, float]], eod_mode: bool) -> Tuple[float, Dict[str, float]]:
    bpspx = snapshot.get("BPSPX", {}).get("current_value")
    bpnya = snapshot.get("BPNYA", {}).get("current_value")
    nymo = snapshot.get("NYMO", {}).get("current_value")
    nysi = snapshot.get("NYSI", {}).get("current_value")
    spxa50r = snapshot.get("SPXA50R", {}).get("current_value")
    oexa50r = snapshot.get("OEXA50R", {}).get("current_value")
    oexa150r = snapshot.get("OEXA150R", {}).get("current_value")
    cpce = snapshot.get("CPCE", {}).get("current_value")
    vxx = snapshot.get("VXX", {}).get("current_value")
    rsp_pct_b = snapshot.get("RSP", {}).get("pct_b")
    rsp_spy = snapshot.get("RSP:SPY", {}).get("current_value")
    hyg_tlt = snapshot.get("HYG:TLT", {}).get("current_value")

    comps = {
        "participation": participation_factor(bpspx, bpnya),
        "momentum": momentum_factor(nymo, nysi, eod_mode=eod_mode, intraday_proxy=spxa50r),
        "breadth_depth": breadth_depth_factor(spxa50r, oexa50r, oexa150r),
        "sentiment": sentiment_factor(cpce, vxx, eod_mode=eod_mode),
        "confirmation": confirmation_factor(rsp_pct_b, rsp_spy, hyg_tlt),
        "capitulation_adjustment": -3 if eod_mode and nymo is not None and bpspx is not None and nymo < -80 and bpspx < 45 else 0,
    }
    total = round(sum(comps.values()), 2)
    total = max(-10, min(10, total))
    return total, comps


def classify_score(score: float) -> str:
    if score > 6:
        return "Strong Breadth Expansion"
    if score >= 3:
        return "Constructive"
    if score >= -2:
        return "Neutral / Transition"
    if score >= -6:
        return "Weakening / Correction"
    return "Capitulation / Breakdown"


# =========================================================
# DASHBOARD LOGIC
# =========================================================
def rows_to_dataframe(rows: List[IndicatorRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in rows])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def build_snapshot(df: pd.DataFrame, file_name: str) -> Dict[str, Dict[str, float]]:
    snap = {}
    sub = df[df["file_name"] == file_name].copy()
    for _, row in sub.iterrows():
        snap[row["indicator"]] = {
            "current_value": row.get("current_value"),
            "pct_b": row.get("pct_b"),
            "rsi": row.get("rsi"),
            "roc": row.get("roc"),
            "adx": row.get("adx"),
            "di_plus": row.get("di_plus"),
            "di_minus": row.get("di_minus"),
        }
    return snap


def compute_file_summary(df: pd.DataFrame, audits: List[FileAudit]) -> pd.DataFrame:
    out = []
    for audit in audits:
        snap = build_snapshot(df, audit.file_name)
        score, comps = breadth_score_from_snapshot(snap, eod_mode=audit.eod_mode)
        out.append({
            "file_name": audit.file_name,
            "timestamp": audit.inferred_timestamp,
            "phase": audit.phase,
            "eod_mode": audit.eod_mode,
            "score": score,
            "regime": classify_score(score),
            **comps,
        })
    out_df = pd.DataFrame(out)
    if not out_df.empty:
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce")
        out_df = out_df.sort_values(["timestamp", "file_name"], na_position="last")
        out_df["change_from_prior_upload"] = out_df["score"].diff()
        if "timestamp" in out_df.columns:
            out_df["trade_date"] = out_df["timestamp"].dt.date
            day_last = out_df.groupby("trade_date")["score"].last().shift(1)
            out_df["prior_day_baseline"] = out_df["trade_date"].map(day_last)
            out_df["change_from_prior_day"] = out_df["score"] - out_df["prior_day_baseline"]
    return out_df


# =========================================================
# UI
# =========================================================
with st.sidebar:
    st.header("Settings")
    use_ocr = st.toggle("Use OCR fallback", value=True)
    st.caption("Primary extraction uses PDF text first. OCR fallback uses Tesseract if installed.")
    show_raw = st.toggle("Show raw extracted text preview", value=False)

uploaded_files = st.file_uploader(
    "Upload PDF chart packs",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more historical/current PDF chart packs to build the dashboard.")
    st.stop()

all_rows: List[IndicatorRow] = []
audits: List[FileAudit] = []
raw_text_cache: Dict[str, str] = {}

progress = st.progress(0)
for i, file in enumerate(uploaded_files, start=1):
    file_bytes = file.read()
    ts, ts_source, conf = infer_timestamp_from_filename(file.name)
    phase, eod_mode = infer_phase(ts)
    text, extraction_source = extract_all_text(file_bytes, use_ocr_fallback=use_ocr)
    raw_text_cache[file.name] = text[:12000]

    audits.append(
        FileAudit(
            file_name=file.name,
            inferred_timestamp=ts.isoformat() if ts else None,
            phase=phase,
            eod_mode=eod_mode,
            source=ts_source,
            confidence=conf,
        )
    )

    rows = parse_indicator_values(text, file.name, ts, extraction_source)
    all_rows.extend(rows)
    progress.progress(i / len(uploaded_files))

progress.empty()

df = rows_to_dataframe(all_rows)
audit_df = pd.DataFrame([asdict(a) for a in audits])
summary_df = compute_file_summary(df, audits)

st.subheader("🕒 File Timestamp Audit")
st.dataframe(audit_df, use_container_width=True)

if df.empty:
    st.error("No indicator values were parsed from the uploads. This usually means the PDFs are image-heavy and OCR support is missing or the chart layout changed.")
    st.stop()

st.subheader("🧮 Extracted Indicator Rows")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Real-Time / Historical Score Dashboard")
st.dataframe(summary_df, use_container_width=True)

if px is not None and not summary_df.empty:
    plot_df = summary_df.dropna(subset=["timestamp"]).copy()
    if not plot_df.empty:
        fig = px.line(plot_df, x="timestamp", y="score", markers=True, title="Breadth Confluence Score by Upload")
        st.plotly_chart(fig, use_container_width=True)

        comp_cols = [c for c in ["participation", "momentum", "breadth_depth", "sentiment", "confirmation", "capitulation_adjustment"] if c in plot_df.columns]
        if comp_cols:
            fig2 = go.Figure()
            for c in comp_cols:
                fig2.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df[c], mode="lines+markers", name=c))
            fig2.update_layout(title="Score Components by Upload")
            st.plotly_chart(fig2, use_container_width=True)

# Latest snapshot detail
latest_file = summary_df.iloc[-1]["file_name"] if not summary_df.empty else None
if latest_file:
    st.subheader(f"🎯 Latest Snapshot: {latest_file}")
    latest = df[df["file_name"] == latest_file].copy()
    st.dataframe(latest, use_container_width=True)

    rsp_row = latest[latest["indicator"] == "RSP"]
    if not rsp_row.empty:
        r = rsp_row.iloc[0]
        st.markdown("### RSP System Check")
        c1, c2, c3 = st.columns(3)
        c1.metric("RSP Current", None if pd.isna(r.get("current_value")) else round(float(r.get("current_value")), 2))
        c2.metric("RSP %B", None if pd.isna(r.get("pct_b")) else round(float(r.get("pct_b")), 2))
        c3.metric("RSP RSI", None if pd.isna(r.get("rsi")) else round(float(r.get("rsi")), 2))

# Raw text debug
if show_raw:
    st.subheader("📝 Raw Extracted Text Preview")
    selected = st.selectbox("Choose file", list(raw_text_cache.keys()))
    st.text_area("Extracted Text", raw_text_cache[selected], height=350)

# Export
st.subheader("💾 Export")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download extracted indicators CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="breadth_indicator_extracts.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Download score summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="breadth_score_summary.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    """
- Timestamp inference is based on filename first. Add consistent names like `3-17 1030am.pdf`, `3-17 1pm.pdf`, `3-17 330pm.pdf`, `3-17 6:15pm.pdf` for best results.
- OCR quality is strongest when Tesseract is installed and the PDFs are high-resolution. This app uses a multi-pass approach: PDF text first, then OCR fallback.
- For truly best-in-class OCR on chart-heavy PDFs, a commercial OCR/vision service usually beats local OCR. This script is the strongest local-first Streamlit version I can give you in a single file.
"""
)

