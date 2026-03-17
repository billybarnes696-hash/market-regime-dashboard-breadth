
"""
streamlit_png_breadth_dashboard.py

PNG-first Streamlit dashboard for market breadth / RSP workflow.

Run:
    streamlit run streamlit_png_breadth_dashboard.py
"""

from __future__ import annotations

import re
import io
from dataclasses import dataclass, asdict
from datetime import datetime, time
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False


st.set_page_config(page_title="PNG Breadth Dashboard", layout="wide")
st.title("📊 PNG Breadth Dashboard")
st.caption("Upload PNG chart images, capture exact printed values, and build a rolling breadth / RSP dashboard.")


@dataclass
class ImageAudit:
    file_name: str
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
    source_method: str = "manual"
    confidence: str = "High"


if "audit_rows" not in st.session_state:
    st.session_state.audit_rows = []
if "snapshot_rows" not in st.session_state:
    st.session_state.snapshot_rows = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}


def infer_timestamp_from_filename(name: str) -> Tuple[Optional[datetime], str, str]:
    lower = name.lower()
    year = None
    month = None
    day = None

    ymd = re.search(r'(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})', lower)
    if ymd:
        year = int(ymd.group(1))
        month = int(ymd.group(2))
        day = int(ymd.group(3))
    else:
        md = re.search(r'(\d{1,2})[-_/](\d{1,2})(?:[-_/](\d{2,4}))?', lower)
        if md:
            month = int(md.group(1))
            day = int(md.group(2))
            year_raw = md.group(3)
            if year_raw:
                year = int(year_raw)
                if year < 100:
                    year += 2000
            else:
                year = datetime.now().year

    hour = None
    minute = 0
    time_source = "filename"

    tmatch = re.search(r'\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)\b', lower)
    if tmatch:
        hour = int(tmatch.group(1))
        minute = int(tmatch.group(2) or 0)
        ap = tmatch.group(3)
        if ap == "pm" and hour != 12:
            hour += 12
        if ap == "am" and hour == 12:
            hour = 0
    elif "noon" in lower:
        hour, minute = 12, 0
        time_source = "filename_noon"
    elif "morning" in lower:
        hour, minute = 10, 0
        time_source = "filename_morning"

    if year and month and day:
        if hour is None:
            hour = 10
            time_source = "filename_date_only"
        try:
            return datetime(year, month, day, hour, minute), time_source, "High"
        except Exception:
            return None, "unresolved", "Low"

    if hour is not None:
        now = datetime.now()
        try:
            return datetime(now.year, now.month, now.day, hour, minute), "filename_time_only", "Medium"
        except Exception:
            return None, "unresolved", "Low"

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


def as_df(items: List[object]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([asdict(x) for x in items])


def ocr_preview(image_bytes: bytes) -> str:
    if not OCR_OK or Image is None:
        return "OCR unavailable. Install pytesseract and the Tesseract binary."
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    except Exception as e:
        return f"OCR failed: {e}"


def get_latest_row(df: pd.DataFrame, file_name: str, symbol: str):
    sub = df[(df["file_name"] == file_name) & (df["symbol"] == symbol)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def participation_factor(bpspx, bpnya) -> float:
    def one(v):
        if v is None or pd.isna(v):
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


def momentum_factor(nymo, spxa50r, eod_mode: bool) -> float:
    if eod_mode and nymo is not None and not pd.isna(nymo):
        if nymo > 50:
            return 2
        if nymo > 0:
            return 1
        if nymo >= -50:
            return 0
        if nymo >= -80:
            return -1
        return -2
    if spxa50r is None or pd.isna(spxa50r):
        return 0.0
    if spxa50r > 60:
        return 1.0
    if spxa50r < 30:
        return -1.0
    return 0.0


def breadth_depth_factor(spxa50r, oexa50r, oexa150r) -> float:
    def one(v):
        if v is None or pd.isna(v):
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


def sentiment_factor(cpce, vxx, eod_mode: bool) -> float:
    score = 0.0
    if eod_mode and cpce is not None and not pd.isna(cpce):
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
    if vxx is not None and not pd.isna(vxx):
        score += 0.5 if vxx < 35 else -0.5
    return max(-2, min(2, score))


def confirmation_factor(rsp_pct_b, rsp_spy, hyg_tlt) -> float:
    score = 0.0
    if rsp_pct_b is not None and not pd.isna(rsp_pct_b):
        score += 1 if rsp_pct_b > 0.20 else -1
    if rsp_spy is not None and not pd.isna(rsp_spy):
        score += 0.5 if rsp_spy > 0 else -0.5
    if hyg_tlt is not None and not pd.isna(hyg_tlt):
        score += 0.5 if hyg_tlt > 0.90 else -0.5
    return max(-2, min(2, score))


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


def summarize_scores(snapshot_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty or audit_df.empty:
        return pd.DataFrame()

    summaries = []
    for _, audit in audit_df.iterrows():
        file_name = audit["file_name"]
        eod_mode = bool(audit["eod_mode"])

        def val(sym: str, col: str = "last"):
            row = get_latest_row(snapshot_df, file_name, sym)
            if row is None:
                return None
            return row.get(col)

        bpspx = val("BPSPX", "last")
        bpnya = val("BPNYA", "last")
        nymo = val("NYMO", "last")
        cpce = val("CPCE", "last")
        vxx = val("VXX", "last")
        spxa50r = val("SPXA50R", "last")
        oexa50r = val("OEXA50R", "last")
        oexa150r = val("OEXA150R", "last")
        rsp_pct_b = val("RSP", "pct_b")
        rsp_spy = val("RSP:SPY", "last")
        hyg_tlt = val("HYG:TLT", "last")

        participation = participation_factor(bpspx, bpnya)
        momentum = momentum_factor(nymo, spxa50r, eod_mode=eod_mode)
        breadth_depth = breadth_depth_factor(spxa50r, oexa50r, oexa150r)
        sentiment = sentiment_factor(cpce, vxx, eod_mode=eod_mode)
        confirmation = confirmation_factor(rsp_pct_b, rsp_spy, hyg_tlt)
        capitulation_adjustment = -3 if eod_mode and nymo is not None and bpspx is not None and nymo < -80 and bpspx < 45 else 0

        score = round(participation + momentum + breadth_depth + sentiment + confirmation + capitulation_adjustment, 2)
        score = max(-10, min(10, score))

        summaries.append({
            "file_name": file_name,
            "timestamp": audit["inferred_timestamp"],
            "phase": audit["phase"],
            "eod_mode": audit["eod_mode"],
            "score": score,
            "regime": classify_score(score),
            "participation": participation,
            "momentum": momentum,
            "breadth_depth": breadth_depth,
            "sentiment": sentiment,
            "confirmation": confirmation,
            "capitulation_adjustment": capitulation_adjustment,
        })

    out = pd.DataFrame(summaries)
    if out.empty:
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["timestamp", "file_name"], na_position="last")
    out["change_from_prior_upload"] = out["score"].diff()
    out["trade_date"] = out["timestamp"].dt.date
    prior_day_last = out.groupby("trade_date")["score"].last().shift(1)
    out["prior_day_baseline"] = out["trade_date"].map(prior_day_last)
    out["change_from_prior_day"] = out["score"] - out["prior_day_baseline"]
    return out


def price_validation_message(row) -> str:
    needed = ["open", "high", "low", "last"]
    if any(pd.isna(row.get(x)) for x in needed):
        return "Insufficient data"
    lo = row["low"]
    hi = row["high"]
    op = row["open"]
    last = row["last"]
    valid = (lo <= op <= hi) and (lo <= last <= hi) and (hi >= lo)
    return "Valid" if valid else "Invalid"


with st.sidebar:
    st.header("Settings")
    show_ocr = st.toggle("Show OCR preview", value=False)
    st.caption("OCR is helper-only. Exact printed values should be entered manually for highest accuracy.")
    if st.button("Clear session data"):
        st.session_state.audit_rows = []
        st.session_state.snapshot_rows = []
        st.session_state.uploaded_images = {}
        st.rerun()


uploaded = st.file_uploader(
    "Upload PNG chart images",
    type=["png"],
    accept_multiple_files=True,
)

if uploaded:
    for f in uploaded:
        if f.name not in st.session_state.uploaded_images:
            image_bytes = f.read()
            ts, ts_source, conf = infer_timestamp_from_filename(f.name)
            phase, eod_mode = infer_phase(ts)
            st.session_state.uploaded_images[f.name] = image_bytes
            st.session_state.audit_rows.append(
                ImageAudit(
                    file_name=f.name,
                    inferred_timestamp=ts.isoformat() if ts else None,
                    phase=phase,
                    eod_mode=eod_mode,
                    timestamp_source=ts_source,
                    confidence=conf,
                )
            )

audit_df = as_df(st.session_state.audit_rows)
if not audit_df.empty:
    audit_df["inferred_timestamp"] = pd.to_datetime(audit_df["inferred_timestamp"], errors="coerce")
    audit_df = audit_df.sort_values(["inferred_timestamp", "file_name"], na_position="last")

snapshot_df = as_df(st.session_state.snapshot_rows)
if not snapshot_df.empty:
    snapshot_df["timestamp"] = pd.to_datetime(snapshot_df["timestamp"], errors="coerce")

st.subheader("🕒 Image Timestamp Audit")
if audit_df.empty:
    st.info("Upload one or more PNG chart images to begin.")
else:
    st.dataframe(audit_df, use_container_width=True)

if audit_df.empty:
    st.stop()

st.subheader("🖼️ Image Preview")
selected_file = st.selectbox("Choose PNG", audit_df["file_name"].tolist())
selected_bytes = st.session_state.uploaded_images[selected_file]
st.image(selected_bytes, caption=selected_file, use_container_width=True)

if show_ocr:
    st.subheader("📝 OCR Preview")
    st.text_area("OCR text", ocr_preview(selected_bytes), height=220)

st.subheader("✍️ Exact Header / Panel Values")
st.caption("Enter the values visibly printed on the PNG. These are treated as gospel.")

audit_match = audit_df[audit_df["file_name"] == selected_file].iloc[0]
ts_value = audit_match["inferred_timestamp"]

col1, col2, col3, col4 = st.columns(4)
symbol = col1.selectbox(
    "Symbol / Indicator",
    [
        "RSP", "URSP", "BPSPX", "BPNYA", "NYMO", "NYSI", "NYAD", "NYHL", "CPCE",
        "SPXA50R", "OEXA50R", "OEXA150R", "OEXA200R",
        "RSP:SPY", "IWM:SPY", "SMH:SPY", "XLF:SPY", "HYG:TLT", "HYG:IEF", "VXX", "SPXS:SVOL"
    ],
)
source_method = col2.selectbox("Source", ["manual", "ocr_assisted", "mixed"])
confidence = col3.selectbox("Confidence", ["High", "Medium", "Low"], index=0)
custom_ts = col4.text_input("Override timestamp (optional)", value="")

if custom_ts.strip():
    try:
        ts_parsed = pd.to_datetime(custom_ts).to_pydatetime()
        ts_iso = ts_parsed.isoformat()
    except Exception:
        ts_iso = ts_value.isoformat() if pd.notna(ts_value) else None
else:
    ts_iso = ts_value.isoformat() if pd.notna(ts_value) else None

st.markdown("**Price block**")
p1, p2, p3, p4, p5 = st.columns(5)
last = p1.text_input("Last", value="")
open_ = p2.text_input("Open", value="")
high = p3.text_input("High", value="")
low = p4.text_input("Low", value="")
change = p5.text_input("Change", value="")

st.markdown("**Printed indicator values**")
i1, i2, i3, i4 = st.columns(4)
pct_b = i1.text_input("%B / BB%", value="")
rsi = i2.text_input("RSI", value="")
roc = i3.text_input("ROC", value="")
cci = i4.text_input("CCI", value="")

i5, i6, i7 = st.columns(3)
adx = i5.text_input("ADX", value="")
di_plus = i6.text_input("DI+", value="")
di_minus = i7.text_input("DI-", value="")


def to_float(x: str):
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x.replace(",", ""))
    except Exception:
        return None


if st.button("Save snapshot row", type="primary"):
    st.session_state.snapshot_rows = [
        row for row in st.session_state.snapshot_rows
        if not (row.file_name == selected_file and row.symbol == symbol)
    ]
    st.session_state.snapshot_rows.append(
        SnapshotRow(
            file_name=selected_file,
            timestamp=ts_iso,
            symbol=symbol,
            last=to_float(last),
            open=to_float(open_),
            high=to_float(high),
            low=to_float(low),
            change=to_float(change),
            pct_b=to_float(pct_b),
            rsi=to_float(rsi),
            roc=to_float(roc),
            cci=to_float(cci),
            adx=to_float(adx),
            di_plus=to_float(di_plus),
            di_minus=to_float(di_minus),
            source_method=source_method,
            confidence=confidence,
        )
    )
    st.success(f"Saved {symbol} for {selected_file}")
    st.rerun()

snapshot_df = as_df(st.session_state.snapshot_rows)
if not snapshot_df.empty:
    snapshot_df["timestamp"] = pd.to_datetime(snapshot_df["timestamp"], errors="coerce")

st.subheader("🧮 Snapshot Rows")
if snapshot_df.empty:
    st.warning("No exact values saved yet.")
else:
    snapshot_df["price_validation"] = snapshot_df.apply(price_validation_message, axis=1)
    st.dataframe(snapshot_df.sort_values(["timestamp", "file_name", "symbol"], na_position="last"), use_container_width=True)

summary_df = summarize_scores(snapshot_df, audit_df)

st.subheader("📈 Rolling Breadth Dashboard")
if summary_df.empty:
    st.info("Scores will populate after you enter exact values for key symbols like BPSPX, BPNYA, SPXA50R, RSP, VXX, etc.")
else:
    st.dataframe(summary_df, use_container_width=True)
    if px is not None:
        plot_df = summary_df.dropna(subset=["timestamp"]).copy()
        if not plot_df.empty:
            fig = px.line(plot_df, x="timestamp", y="score", markers=True, title="Breadth Confluence Score")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            for col in ["participation", "momentum", "breadth_depth", "sentiment", "confirmation", "capitulation_adjustment"]:
                fig2.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df[col], mode="lines+markers", name=col))
            fig2.update_layout(title="Score Components")
            st.plotly_chart(fig2, use_container_width=True)

st.subheader("🎯 Latest RSP System Check")
if snapshot_df.empty:
    st.info("Enter an RSP row to populate this section.")
else:
    latest_file = audit_df.sort_values(["inferred_timestamp", "file_name"], na_position="last").iloc[-1]["file_name"]
    rsp_row = get_latest_row(snapshot_df, latest_file, "RSP")
    if rsp_row is None:
        st.info("No RSP row saved for latest file.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last", rsp_row.get("last"))
        c2.metric("Open", rsp_row.get("open"))
        c3.metric("High", rsp_row.get("high"))
        c4.metric("Low", rsp_row.get("low"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("%B", rsp_row.get("pct_b"))
        c6.metric("RSI", rsp_row.get("rsi"))
        c7.metric("ROC", rsp_row.get("roc"))
        c8.metric("CCI", rsp_row.get("cci"))

        c9, c10, c11 = st.columns(3)
        c9.metric("ADX", rsp_row.get("adx"))
        c10.metric("DI+", rsp_row.get("di_plus"))
        c11.metric("DI-", rsp_row.get("di_minus"))

        st.write(f"Price validation: **{price_validation_message(rsp_row)}**")

st.subheader("💾 Export")
e1, e2, e3 = st.columns(3)
with e1:
    st.download_button(
        "Download audit CSV",
        data=audit_df.to_csv(index=False).encode("utf-8"),
        file_name="png_audit.csv",
        mime="text/csv",
    )
with e2:
    st.download_button(
        "Download snapshot CSV",
        data=snapshot_df.to_csv(index=False).encode("utf-8") if not snapshot_df.empty else b"",
        file_name="png_snapshots.csv",
        mime="text/csv",
    )
with e3:
    st.download_button(
        "Download score CSV",
        data=summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b"",
        file_name="png_score_summary.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    """
- This version is **PNG-only**.
- Exact printed values are meant to be entered from the image and treated as gospel.
- OCR preview is helper-only and should not override visible printed values.
- Best filename examples:
  - `3-17 1030am RSP.png`
  - `3-17 1pm BPSPX.png`
  - `3-17 330pm breadth.png`
  - `3-17 615pm eod.png`
- For fully automatic chart reading from PNGs, the next upgrade is a **vision model**, not OCR alone.
"""
)
