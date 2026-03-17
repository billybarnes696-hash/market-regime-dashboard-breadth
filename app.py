import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Breadth Traffic Light", layout="wide")


# -----------------------------
# Helper models
# -----------------------------
@dataclass
class MetricInput:
    name: str
    value: float
    confidence: str


CONFIDENCE_MAP = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
    "Unresolved": 3,
}


def is_low_quality(confidence: str) -> bool:
    return CONFIDENCE_MAP.get(confidence, 3) >= 2


# -----------------------------
# Scoring logic
# -----------------------------
def classify_bpspx_zone(bpspx_pb: float) -> str:
    if bpspx_pb < 0.10:
        return "Capitulation"
    if bpspx_pb < 0.20:
        return "Repair"
    if bpspx_pb < 0.30:
        return "Confirmation"
    return "Strong Confirmation"


def score_traffic_light(inputs: Dict[str, float], confidence_flags: Dict[str, str]) -> Tuple[str, int, List[str], List[str], List[str]]:
    """
    Returns:
        light, points, reasons, upgrades, downgrades
    """
    reasons: List[str] = []
    upgrades: List[str] = []
    downgrades: List[str] = []

    core_conf_keys = [
        "bpspx_pb_conf",
        "nymo_conf",
        "rsp_pb_conf",
        "spxa50r_pb_conf",
        "oexa50r_pb_conf",
    ]
    low_quality_count = sum(is_low_quality(confidence_flags[k]) for k in core_conf_keys)

    # Fail-fast quality override
    if low_quality_count > 1:
        reasons.append("More than 20% of core inputs are low-confidence or unresolved.")
        downgrades.append("Improve data quality before taking risk.")
        return "🔴 RED", 0, reasons, upgrades, downgrades

    points = 0

    # 1) BPSPX %B
    if inputs["bpspx_pb"] > 0.20:
        points += 1
        reasons.append(f"BPSPX %B is confirming at {inputs['bpspx_pb']:.2f}.")
    else:
        reasons.append(f"BPSPX %B remains in the {classify_bpspx_zone(inputs['bpspx_pb']).lower()} zone at {inputs['bpspx_pb']:.2f}.")
        upgrades.append("BPSPX %B > 0.20")
        if inputs["bpspx_pb"] < 0.10:
            downgrades.append("BPSPX %B < 0.10 deepens capitulation risk.")

    # 2) NYMO
    if inputs["nymo"] > -20:
        points += 1
        reasons.append(f"NYMO has improved to {inputs['nymo']:.2f}, above the -20 repair threshold.")
    else:
        reasons.append(f"NYMO is still weak at {inputs['nymo']:.2f}.")
        upgrades.append("NYMO > -20")
        if inputs["nymo"] < -40:
            downgrades.append("NYMO < -40 would reinforce a defensive posture.")

    # 3) RSP %B
    if inputs["rsp_pb"] > 0.20:
        points += 1
        reasons.append(f"RSP %B is constructive at {inputs['rsp_pb']:.2f}.")
    else:
        reasons.append(f"RSP %B is not yet confirming at {inputs['rsp_pb']:.2f}.")
        upgrades.append("RSP %B > 0.20")
        downgrades.append("RSP price/quality deterioration can invalidate probe setups.")

    # 4) SPXA50R %B
    if inputs["spxa50r_pb"] > 0.20:
        points += 1
        reasons.append(f"SPXA50R %B confirms improving breadth depth at {inputs['spxa50r_pb']:.2f}.")
    else:
        reasons.append(f"SPXA50R %B remains soft at {inputs['spxa50r_pb']:.2f}.")
        upgrades.append("SPXA50R %B > 0.20")

    # 5) OEXA50R %B
    if inputs["oexa50r_pb"] > 0.20:
        points += 1
        reasons.append(f"OEXA50R %B confirms improving large-cap breadth depth at {inputs['oexa50r_pb']:.2f}.")
    else:
        reasons.append(f"OEXA50R %B remains soft at {inputs['oexa50r_pb']:.2f}.")
        upgrades.append("OEXA50R %B > 0.20")

    # 6) VXX behavior
    if not inputs["vxx_expanding"]:
        points += 1
        reasons.append("VXX is stable/down, so volatility is not actively fighting the setup.")
    else:
        reasons.append("VXX is expanding, which is a headwind for long entries.")
        downgrades.append("VXX rolling higher keeps risk elevated.")

    # 7) Relative strength
    if inputs["rsp_spy_flat_to_up"]:
        points += 1
        reasons.append("RSP:SPY is flat to rising, which avoids a leadership penalty.")
    else:
        reasons.append("RSP:SPY is falling, so equal-weight leadership is not confirmed.")
        downgrades.append("Falling RSP:SPY weakens long conviction.")

    # Light mapping
    if points <= 2:
        light = "🔴 RED"
    elif points <= 4:
        light = "🟡 YELLOW"
    else:
        light = "🟢 GREEN"

    # Override rules
    if light == "🟢 GREEN" and inputs["bpspx_pb"] <= 0.20:
        light = "🟡 YELLOW"
        reasons.append("Green downgraded to Yellow because participation has not fully confirmed.")

    if inputs["bpspx_pb"] < 0.10:
        light = "🔴 RED"
        reasons.append("Forced Red override due to BPSPX %B < 0.10.")

    return light, points, reasons, upgrades, downgrades


# -----------------------------
# UI
# -----------------------------
st.title("🚦 Breadth Traffic Light Dashboard")
st.caption("A fail-fast breadth model for RSP / URSP decision support")

with st.sidebar:
    st.header("Input Values")

    st.subheader("Core Breadth")
    bpspx_pb = st.number_input("BPSPX %B", min_value=-1.0, max_value=2.0, value=0.15, step=0.01, format="%.2f")
    nymo = st.number_input("NYMO", min_value=-200.0, max_value=200.0, value=-31.50, step=0.50, format="%.2f")
    rsp_pb = st.number_input("RSP %B", min_value=-1.0, max_value=2.0, value=0.22, step=0.01, format="%.2f")
    spxa50r_pb = st.number_input("SPXA50R %B", min_value=-1.0, max_value=2.0, value=0.16, step=0.01, format="%.2f")
    oexa50r_pb = st.number_input("OEXA50R %B", min_value=-1.0, max_value=2.0, value=0.25, step=0.01, format="%.2f")

    st.subheader("Filters")
    vxx_expanding = st.checkbox("VXX expanding / rising sharply", value=False)
    rsp_spy_flat_to_up = st.checkbox("RSP:SPY flat to rising", value=True)

    st.subheader("Confidence")
    conf_options = ["High", "Medium", "Low", "Unresolved"]
    bpspx_pb_conf = st.selectbox("BPSPX confidence", conf_options, index=0)
    nymo_conf = st.selectbox("NYMO confidence", conf_options, index=0)
    rsp_pb_conf = st.selectbox("RSP %B confidence", conf_options, index=0)
    spxa50r_pb_conf = st.selectbox("SPXA50R %B confidence", conf_options, index=0)
    oexa50r_pb_conf = st.selectbox("OEXA50R %B confidence", conf_options, index=0)

    st.subheader("Optional Prices")
    rsp_price = st.number_input("RSP price", min_value=0.0, value=196.05, step=0.05, format="%.2f")
    rsp_trigger = st.number_input("RSP trigger level", min_value=0.0, value=195.50, step=0.05, format="%.2f")

inputs = {
    "bpspx_pb": bpspx_pb,
    "nymo": nymo,
    "rsp_pb": rsp_pb,
    "spxa50r_pb": spxa50r_pb,
    "oexa50r_pb": oexa50r_pb,
    "vxx_expanding": vxx_expanding,
    "rsp_spy_flat_to_up": rsp_spy_flat_to_up,
    "rsp_price": rsp_price,
    "rsp_trigger": rsp_trigger,
}

confidence_flags = {
    "bpspx_pb_conf": bpspx_pb_conf,
    "nymo_conf": nymo_conf,
    "rsp_pb_conf": rsp_pb_conf,
    "spxa50r_pb_conf": spxa50r_pb_conf,
    "oexa50r_pb_conf": oexa50r_pb_conf,
}

light, points, reasons, upgrades, downgrades = score_traffic_light(inputs, confidence_flags)

# -----------------------------
# Summary cards
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Traffic Light", light)
with col2:
    st.metric("Score", f"{points}/7")
with col3:
    if rsp_price >= rsp_trigger:
        st.metric("Price vs Trigger", f"{rsp_price:.2f} ≥ {rsp_trigger:.2f}")
    else:
        st.metric("Price vs Trigger", f"{rsp_price:.2f} < {rsp_trigger:.2f}")

# -----------------------------
# Narrative section
# -----------------------------
st.subheader("Decision Summary")
if light == "🟢 GREEN":
    st.success("Go regime: breadth and price are broadly aligned.")
    action_text = "RSP 40–60%; URSP 0–25% only if confirmation is broad, not just price-led."
elif light == "🟡 YELLOW":
    st.warning("Probe regime: some improvement, but breadth is not fully confirmed.")
    action_text = "RSP 10–15% max; no URSP; require quick follow-through or exit."
else:
    st.error("No-trade / defensive regime: internals are weak or data quality is insufficient.")
    action_text = "No new longs; hold cash or hedges; wait for repair."

st.write(f"**Suggested action:** {action_text}")

# -----------------------------
# Diagnostics
# -----------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Why")
    for reason in reasons:
        st.write(f"- {reason}")

with col_b:
    st.subheader("Upgrade / Downgrade Triggers")
    if upgrades:
        st.write("**Upgrade to stronger stance if:**")
        for item in upgrades[:5]:
            st.write(f"- {item}")
    if downgrades:
        st.write("**Downgrade risk if:**")
        for item in downgrades[:5]:
            st.write(f"- {item}")

# -----------------------------
# Rule grid
# -----------------------------
st.subheader("Rule Grid")
rule_df = pd.DataFrame(
    [
        ["BPSPX %B", bpspx_pb, "> 0.20", "Pass" if bpspx_pb > 0.20 else "Fail", bpspx_pb_conf],
        ["NYMO", nymo, "> -20", "Pass" if nymo > -20 else "Fail", nymo_conf],
        ["RSP %B", rsp_pb, "> 0.20", "Pass" if rsp_pb > 0.20 else "Fail", rsp_pb_conf],
        ["SPXA50R %B", spxa50r_pb, "> 0.20", "Pass" if spxa50r_pb > 0.20 else "Fail", spxa50r_pb_conf],
        ["OEXA50R %B", oexa50r_pb, "> 0.20", "Pass" if oexa50r_pb > 0.20 else "Fail", oexa50r_pb_conf],
        ["VXX", "Stable/Down" if not vxx_expanding else "Expanding", "Stable/Down", "Pass" if not vxx_expanding else "Fail", "n/a"],
        ["RSP:SPY", "Flat/Up" if rsp_spy_flat_to_up else "Down", "Flat/Up", "Pass" if rsp_spy_flat_to_up else "Fail", "n/a"],
    ],
    columns=["Factor", "Current", "Threshold", "Status", "Confidence"],
)
st.dataframe(rule_df, use_container_width=True, hide_index=True)

# -----------------------------
# Self-audit
# -----------------------------
st.subheader("Self-Audit")
core_drivers = [
    ("BPSPX %B", bpspx_pb, bpspx_pb_conf),
    ("NYMO", nymo, nymo_conf),
    ("RSP %B", rsp_pb, rsp_pb_conf),
]

st.write("**Top 3 drivers**")
for name, value, conf in core_drivers:
    st.write(f"- {name}: {value} ({conf})")

st.write("**Invalidation test**")
if light == "🟡 YELLOW":
    st.write("- This thesis is invalidated if BPSPX %B falls below 0.10 or price loses the session support/trigger structure.")
    st.write("- This thesis upgrades if BPSPX %B > 0.20 and NYMO > -20.")
elif light == "🟢 GREEN":
    st.write("- This thesis is invalidated if participation falls back under confirmation and volatility expands.")
else:
    st.write("- This thesis is invalidated if participation and momentum suddenly confirm together, which would shift the model out of Red.")

st.write("**Probability framing**")
if light == "🟢 GREEN":
    st.write("- Estimated probability of constructive long conditions: High (>70%).")
elif light == "🟡 YELLOW":
    st.write("- Estimated probability of durable follow-through: Moderate (50–70%), pending confirmation.")
else:
    st.write("- Estimated probability of attractive long conditions: Low (<50%).")

st.caption("Note: This dashboard is a rules-based support tool, not investment advice.")
