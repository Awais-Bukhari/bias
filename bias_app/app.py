import streamlit as st
import pandas as pd
from src.pipeline import run_pipeline, build_pooled_dataset
from src.metrics import stats_scatter_fig

st.set_page_config(page_title="Precipitation Bias Correction", layout="wide")
st.title("🌧️ ERA5 Precipitation Bias Correction")
st.caption("Sirf apna observed station data upload karo — ERA5 satellite data khud fetch hoga.")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader(
        "Upload observed data CSV (columns: STATION, LAT, LON, DATE, OBS_mm)",
        type=["csv"]
    )
    train_start = st.number_input("Train start year", 2000, 2030, 2015)
    train_end = st.number_input("Train end year", 2000, 2030, 2024)
    outlier_factor = st.slider("Outlier IQR factor", 1.0, 5.0, 3.0)
    run_btn = st.button("Fetch ERA5 & Run", type="primary", use_container_width=True)

if uploaded is None:
    st.info("👈 CSV upload karo. Columns hone chahiye: STATION, LAT, LON, DATE, OBS_mm")
    st.stop()

obs_df = pd.read_csv(uploaded, parse_dates=["DATE"])
required_cols = {"STATION", "LAT", "LON", "DATE", "OBS_mm"}
if not required_cols.issubset(obs_df.columns):
    st.error(f"CSV mein ye columns hone chahiye: {required_cols}")
    st.stop()

if run_btn:
    with st.spinner("ERA5 data fetch ho raha hai satellite se... (stations ke hisaab se time lagega)"):
        pooled = build_pooled_dataset(obs_df)

    if pooled.empty:
        st.error("Koi overlapping data nahi mila. Dates/coordinates check karo.")
        st.stop()

    st.success(f"✅ {pooled['STATION'].nunique()} stations, {len(pooled)} merged rows fetched.")

    with st.spinner("Bias correction chal raha hai..."):
        results = run_pipeline(pooled, train_start, train_end, outlier_factor)

    st.success(f"🏆 Winner method: **{results['winner']}**")

    tab1, tab2, tab3 = st.tabs(["CV Summary", "Before vs After Scatter", "Corrected Data"])

    with tab1:
        st.dataframe(results["cv_summary"], use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = stats_scatter_fig(
                results["pooled_train_corrected"]["OBS_mm"],
                results["pooled_train_corrected"]["ERA5_mm"],
                "Raw ERA5 vs Observed"
            )
            st.pyplot(fig1)
        with c2:
            fig2 = stats_scatter_fig(
                results["pooled_train_corrected"]["OBS_mm"],
                results["pooled_train_corrected"]["CORR_mm"],
                f"Corrected ({results['winner']}) vs Observed"
            )
            st.pyplot(fig2)

    with tab3:
        st.dataframe(results["pooled_train_corrected"], use_container_width=True)
        csv = results["pooled_train_corrected"].to_csv(index=False).encode("utf-8")
        st.download_button("Download corrected CSV", csv, "corrected_data.csv", "text/csv")