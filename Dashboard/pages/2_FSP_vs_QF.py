# fsp_qf_dashboard.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="FSP vs QF – GAP Dashboard", layout="wide")

# --- Config ---
# DEFAULT_CSV = r"C:\Users\61432\Downloads\FSP_QF.csv"

# --- Config ---
# Resolve project root and csv_files folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FOLDER = PROJECT_ROOT / "csv_files"
DEFAULT_CSV = CSV_FOLDER / "FSP_QF.csv"

# --- Helpers ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Detect date column
    date_col = None
    for cand in ["date", "Date", "DATE"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("No date column found.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    rename_map = {date_col: "date"}

    # Match GAP columns
    if "GAP_delay (QF-FSP)" in df.columns:
        rename_map["GAP_delay (QF-FSP)"] = "gap_fsp_qf"
    else:
        raise ValueError("Column 'GAP_delay (QF-FSP)' not found.")

    if "GAP_delay (QF-ACTUAL)" in df.columns:
        rename_map["GAP_delay (QF-ACTUAL)"] = "gap_actual_qf"
    else:
        raise ValueError("Column 'GAP_delay (QF-ACTUAL)' not found.")

    df = df.rename(columns=rename_map)
    return df


def compute_percentiles(series: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame({"percentile": probs, "value": [np.nan] * len(probs)})
    values = np.quantile(series, probs, method="linear")
    return pd.DataFrame({"percentile": probs, "value": values})


# --- Load ---
try:
    df = load_data(DEFAULT_CSV)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

st.title("FSP vs QF – GAP Dashboard")

# --- Filters ---
min_date = df["date"].min().date()
max_date = df["date"].max().date()
start_date, end_date = st.date_input(
    "Date interval",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
)

mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
filtered = df.loc[mask]

if filtered.empty:
    st.warning("No data after applying date filter.")
    st.stop()

# --- Chart 0: Combined view of FSP, Actual, and QF delays ---
st.subheader("Maintenance Delays: FSP vs Actual vs QF")

fig0 = px.line(
    filtered.sort_values("date"),
    x="date",
    y=["FSP_mtx_delay", "Actual_mtx_delay", "QF_mtx_delay"],
    markers=True,
    labels={
        "value": "Delay",
        "variable": "Series",
        "date": "Date",
        "FSP_mtx_delay": "FSP_mtx_delay",
        "Actual_mtx_delay": "Actual_mtx_delay",
        "QF_mtx_delay": "QF_mtx_delay",
    },
)
fig0.update_layout(hovermode="x unified", legend_title="Series")
st.plotly_chart(fig0, use_container_width=True)

# --- Percentiles config ---
probs = np.arange(0.05, 1.01, 0.05)

# --- Chart Row 1: GAP_delay (QF-FSP) ---
st.subheader("GAP_delay (QF-FSP)")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(
        filtered.sort_values("date"),
        x="date",
        y="gap_fsp_qf",
        markers=True,
        labels={"date": "Date", "gap_fsp_qf": "GAP_delay (QF-FSP)"},
    )
    fig1.update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    p1 = compute_percentiles(filtered["gap_fsp_qf"], probs)
    fig2 = px.scatter(
        p1, x="percentile", y="value",
        labels={"percentile": "Percentile", "value": "Value"},
    )
    fig2.update_traces(mode="lines+markers")
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# --- Chart Row 2: GAP_delay (ACTUAL-QF) ---
st.subheader("GAP_delay (QF-ACTUAL)")
col3, col4 = st.columns(2)

with col3:
    fig3 = px.line(
        filtered.sort_values("date"),
        x="date",
        y="gap_actual_qf",
        markers=True,
        labels={"date": "Date", "gap_actual_qf": "GAP_delay (QF-ACTUAL)"},
    )
    fig3.update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    p2 = compute_percentiles(filtered["gap_actual_qf"], probs)
    fig4 = px.scatter(
        p2, x="percentile", y="value",
        labels={"percentile": "Percentile", "value": "Value"},
    )
    fig4.update_traces(mode="lines+markers")
    fig4.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)

# --- Footer ---
st.caption("Percentiles are computed on filtered data. Charts show all mines aggregated.")
