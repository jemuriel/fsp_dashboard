# fsp_actual_dashboard.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="FSP vs ACTUAL – GAP Dashboard", layout="wide")

# --- Config ---
# DEFAULT_CSV = r"C:\Users\61432\Downloads\FSP_ACTUAL.csv"

# --- Config ---
# Resolve project root and csv_files folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FOLDER = PROJECT_ROOT / "csv_files"
DEFAULT_CSV = CSV_FOLDER / "FSP_ACTUAL.csv"

# --- Helpers ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Read and normalise columns (strip spaces, unify case)
    df = pd.read_csv(path)
    original_cols = df.columns.tolist()
    normalised = [c.strip() for c in original_cols]
    df.columns = normalised

    # Try a few likely date column casings
    date_col = None
    for cand in ["date", "Date", "DATE"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("Could not find a 'date' column in the CSV.")

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Standardise key column names for internal use
    rename_map = {}
    # Mine column
    for cand in ["mine", "Mine", "MINE"]:
        if cand in df.columns:
            rename_map[cand] = "mine"
            break
    if "mine" not in rename_map and "mine" not in df.columns:
        raise ValueError("Could not find a 'mine' column in the CSV.")

    # GAP columns (with parentheses)
    gap_delay_candidates = [c for c in df.columns if c.lower().replace(" ", "") in ["gap_delay(fsp-actual)", "gap_delay(fsp–actual)","gap_delay(fsp—actual)"]]
    gap_consist_candidates = [c for c in df.columns if c.lower().replace(" ", "") in ["gap_consist(fsp-actual)", "gap_consist(fsp–actual)","gap_consist(fsp—actual)"]]

    if not gap_delay_candidates:
        # Fallback: exact match from the prompt
        if "GAP_delay (FSP-ACTUAL)" in df.columns:
            gap_delay_col = "GAP_delay (FSP-ACTUAL)"
        else:
            raise ValueError("Could not find the column 'GAP_delay (FSP-ACTUAL)'.")
    else:
        gap_delay_col = gap_delay_candidates[0]

    if not gap_consist_candidates:
        if "GAP_CONSIST (FSP-ACTUAL)" in df.columns:
            gap_consist_col = "GAP_CONSIST (FSP-ACTUAL)"
        else:
            raise ValueError("Could not find the column 'GAP_CONSIST (FSP-ACTUAL)'.")
    else:
        gap_consist_col = gap_consist_candidates[0]

    # Apply renames
    rename_map[date_col] = "date"
    rename_map[gap_delay_col] = "gap_delay"
    rename_map[gap_consist_col] = "gap_consist"
    df = df.rename(columns=rename_map)

    # Keep only relevant columns + any others for potential debugging
    # Make sure the key columns exist
    for needed in ["date", "mine", "gap_delay", "gap_consist"]:
        if needed not in df.columns:
            raise ValueError(f"Missing required column after renaming: {needed}")

    return df


def compute_percentiles(series: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame({"percentile": probs, "value": [np.nan] * len(probs)})
    values = np.quantile(series, probs, method="linear")
    return pd.DataFrame({"percentile": probs, "value": values})


# --- Load data ---
try:
    df = load_data(DEFAULT_CSV)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.title("FSP vs ACTUAL – GAP Dashboard")

# --- Filters ---
with st.expander("Filters", expanded=True):
    left, right = st.columns([1, 2], gap="large")

    with left:
        mines = sorted(df["mine"].dropna().astype(str).unique().tolist())
        selected_mines = st.multiselect("Mine", options=mines, default=mines)

    with right:
        min_date = pd.to_datetime(df["date"].min()).date()
        max_date = pd.to_datetime(df["date"].max()).date()
        start_date, end_date = st.date_input(
            "Date interval",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )

# Apply filters
mask = (
    df["mine"].astype(str).isin(selected_mines)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data after applying the filters.")
    st.stop()

# --- Chart 0: Maintenance Delay Comparison (FSP vs Actual) ---
st.subheader("Maintenance Delay Comparison – FSP vs Actual")

fig0 = px.area(
    filtered.sort_values("date"),
    x="date",
    y=["FSP_mtx_delay_sum", "Actual_mtx_delay"],
    labels={
        "value": "Delay",
        "variable": "Series",
        "date": "Date",
        "FSP_mtx_delay_sum": "FSP_mtx_delay_sum",
        "Actual_mtx_delay": "Actual_mtx_delay",
    },
)
fig0.update_traces(mode="lines")  # keep line outlines visible
fig0.update_layout(hovermode="x unified", legend_title="Series")
# fig0.update_layout(stackgroup=None)
st.plotly_chart(fig0, use_container_width=True)

# --- Chart: GAP_delay aggregated (all mines combined) ---
st.subheader("GAP_delay (FSP-ACTUAL) – All Mines Combined")

# Aggregate by date across all selected mines
agg_delay = (
    filtered.groupby("date", as_index=False)["gap_delay"]
    .mean()   # could use .median() if that fits your analysis better
)

fig0 = px.line(
    agg_delay.sort_values("date"),
    x="date",
    y="gap_delay",
    markers=True,
    labels={"date": "Date", "gap_delay": "GAP_delay (FSP-ACTUAL)"},
)
fig0.update_layout(hovermode="x unified", showlegend=False)
st.plotly_chart(fig0, use_container_width=True)


# --- Percentiles config (5% to 100% in 5% steps) ---
probs = np.arange(0.05, 1.01, 0.05)

# --- Chart Row 1: GAP_delay ---
st.subheader("GAP_delay (FSP-ACTUAL)")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(
        filtered.sort_values("date"),
        x="date",
        y="gap_delay",
        color="mine",
        markers=True,
        labels={"date": "Date", "gap_delay": "GAP_delay (FSP-ACTUAL)", "mine": "Mine"},
    )
    fig1.update_layout(legend_title="Mine", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    p_delay = compute_percentiles(filtered["gap_delay"], probs)
    fig3 = px.scatter(
        p_delay, x="percentile", y="value",
        labels={"percentile": "Percentile", "value": "Value"},
    )
    fig3.update_traces(mode="lines+markers")
    fig3.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

# --- Chart Row 2: GAP_CONSIST ---
st.subheader("GAP_CONSIST (FSP-ACTUAL)")
col3, col4 = st.columns(2)

with col3:
    fig2 = px.line(
        filtered.sort_values("date"),
        x="date",
        y="gap_consist",
        color="mine",
        markers=True,
        labels={"date": "Date", "gap_consist": "GAP_CONSIST (FSP-ACTUAL)", "mine": "Mine"},
    )
    fig2.update_layout(legend_title="Mine", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    p_consist = compute_percentiles(filtered["gap_consist"], probs)
    fig4 = px.scatter(
        p_consist, x="percentile", y="value",
        labels={"percentile": "Percentile", "value": "Value"},
    )
    fig4.update_traces(mode="lines+markers")
    fig4.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)
