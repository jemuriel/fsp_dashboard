# Home.py
import streamlit as st

st.set_page_config(page_title="FSP Dashboards", layout="wide")

st.title("📊 FSP Dashboards")
st.markdown(
    """
    Welcome to the **FSP Dashboards** suite.  
    Use the sidebar to navigate:

    - **FSP vs ACTUAL – GAP Dashboard**  
    - **FSP vs QF – GAP Dashboard**

    ---
    """
)

st.info("👈 Select a dashboard from the sidebar to get started.")
