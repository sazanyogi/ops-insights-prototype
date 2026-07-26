"""Ops Insights Prototype -- Streamlit dashboard.

Upload or generate shift/production data, see the operational trends, then
ask Gemini to summarize the period and flag anomalies worth a supervisor's
attention.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analyzer import analyze
from src.data_generator import add_derived_metrics, generate_shift_data

load_dotenv()

st.set_page_config(page_title="Ops Insights Prototype", page_icon="🏭", layout="wide")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

st.title("🏭 Ops Insights Prototype")
st.caption(
    "LLM-powered summary and anomaly detection for shift/production data -- "
    "built on patterns from real production-floor reporting (OEE, downtime reason codes, shift handoffs)."
)

with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["Generate sample data", "Upload CSV"])

    if source == "Generate sample data":
        days = st.slider("Days of shift history", 7, 30, 14)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("Regenerate", use_container_width=True) or "df" not in st.session_state:
            st.session_state.df = generate_shift_data(days=days, seed=int(seed))
    else:
        uploaded = st.file_uploader(
            "CSV with columns: date, shift, line, units_target, units_produced, "
            "downtime_minutes, downtime_reason, defect_count, availability, performance"
        )
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            st.session_state.df = add_derived_metrics(raw)

    if "OPS_INSIGHTS_MODEL" in os.environ:
        st.caption(f"Model: `{os.environ['OPS_INSIGHTS_MODEL']}`")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY not set -- copy `.env.example` to `.env` to enable analysis.", icon="⚠️")

if "df" not in st.session_state:
    st.info("Generate or upload data to get started.")
    st.stop()

df = st.session_state.df

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg throughput", f"{df['throughput_pct'].mean():.1f}%")
col2.metric("Avg defect rate", f"{df['defect_rate_pct'].mean():.2f}%")
col3.metric("Avg OEE", f"{df['oee_pct'].mean():.1f}%")
col4.metric("Total downtime", f"{df['downtime_minutes'].sum() / 60:.1f} hrs")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("OEE trend by line")
    oee_by_date_line = df.pivot_table(index="date", columns="line", values="oee_pct", aggfunc="mean")
    st.line_chart(oee_by_date_line)

with chart_col2:
    st.subheader("Downtime by reason")
    downtime_by_reason = df.groupby("downtime_reason")["downtime_minutes"].sum().sort_values(ascending=False)
    st.bar_chart(downtime_by_reason)

st.subheader("Defect rate by shift")
st.bar_chart(df.groupby("shift")["defect_rate_pct"].mean().reindex(["Day", "Evening", "Night"]))

with st.expander("Raw shift log"):
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Gemini analysis")

if st.button("Analyze with Gemini", type="primary", disabled=not os.getenv("GEMINI_API_KEY")):
    with st.spinner("Reviewing the shift log..."):
        try:
            st.session_state.result = analyze(df)
        except Exception as exc:  # surfaced directly to the user, not swallowed
            st.error(f"Analysis failed: {exc}")
            st.session_state.pop("result", None)

if "result" in st.session_state:
    result = st.session_state.result
    st.markdown(f"**Summary:** {result['period_summary']}")

    anomalies = sorted(result["anomalies"], key=lambda a: SEVERITY_ORDER.get(a["severity"], 3))
    if anomalies:
        st.markdown("#### Flagged anomalies")
        for a in anomalies:
            icon = SEVERITY_ICON.get(a["severity"], "⚪")
            with st.expander(f"{icon} **{a['title']}** — {a['scope']}"):
                st.markdown(f"**Evidence:** {a['evidence']}")
                st.markdown(f"**Likely cause:** {a['likely_cause']}")
                st.markdown(f"**Recommended action:** {a['recommended_action']}")
    else:
        st.success("No anomalies flagged for this period.")

    if result.get("highlights"):
        st.markdown("#### Highlights")
        for h in result["highlights"]:
            st.markdown(f"- {h}")
