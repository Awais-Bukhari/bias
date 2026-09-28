import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline import run_pipeline, build_pooled_dataset
from src.metrics import stats_scatter_fig, compute_metrics
from src.config import TRAIN_START_DEFAULT, TRAIN_END_DEFAULT

st.set_page_config(
    page_title="Precipitation Bias Correction",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional visual system
# -----------------------------
st.markdown("""
<style>
:root {
    --bg:#f6f8fb; --surface:#ffffff; --border:#e5e7eb;
    --text:#17202a; --muted:#667085; --primary:#0f766e;
    --primary-dark:#115e59; --soft:#ecfdf5;
}
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1400px; padding-top:2rem; padding-bottom:4rem; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--border); }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3 { color:var(--text)!important; letter-spacing:-.02em; }
.eyebrow { color:var(--primary); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.page-title { font-size:2.35rem; line-height:1.1; font-weight:800; margin:0; }
.page-subtitle { color:var(--muted); font-size:1rem; max-width:760px; line-height:1.6; margin-top:.6rem; }
.card,.metric-card,.method-card { background:#fff; border:1px solid var(--border); border-radius:14px; box-shadow:0 1px 2px rgba(16,24,40,.03); }
.card { padding:1.2rem 1.3rem; }
.metric-card { padding:1rem 1.1rem; min-height:105px; }
.metric-label { color:var(--muted); font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { color:var(--text); font-size:1.45rem; font-weight:800; margin-top:.25rem; }
.metric-help,.helper { color:var(--muted); font-size:.82rem; line-height:1.5; }
.method-card { padding:1rem; min-height:125px; }
.method-name { font-weight:800; margin-bottom:.35rem; }
.method-description { color:var(--muted); font-size:.84rem; line-height:1.45; }
.selected-method { border:1px solid #99f6e4; background:var(--soft); border-radius:14px; padding:1.2rem; }
.selected-method-name { color:var(--primary-dark); font-size:1.35rem; font-weight:800; }
.workflow { display:flex; align-items:center; margin:1.7rem 0 2rem; overflow-x:auto; }
.workflow-step { display:flex; align-items:center; white-space:nowrap; color:#98a2b3; font-size:.82rem; font-weight:700; }
.workflow-step.active { color:var(--primary-dark); }
.workflow-number { width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; margin-right:.45rem; background:#eef2f6; border:1px solid #d0d5dd; }
.workflow-step.active .workflow-number { background:var(--primary); color:#fff; border-color:var(--primary); }
.workflow-line { flex:1; min-width:35px; height:1px; background:#d0d5dd; margin:0 .7rem; }
.stButton>button,.stDownloadButton>button { border-radius:9px; font-weight:700; min-height:42px; }
[data-testid="stFileUploader"] section { border:1.5px dashed #94a3b8; border-radius:14px; background:#fbfdff; }
.stTabs [data-baseweb="tab"] { font-weight:700; }
</style>
""", unsafe_allow_html=True)


def workflow(active=1, completed=0):
    steps = ["Data", "Configure", "Process", "Results"]
    html = '<div class="workflow">'
    for i, name in enumerate(steps, 1):
        cls = "active" if i == active or i <= completed else ""
        html += f'<div class="workflow-step {cls}"><span class="workflow-number">{i}</span>{name}</div>'
        if i < len(steps):
            html += '<div class="workflow-line"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label, value, help_text=""):
    return f'''<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-help">{help_text}</div>
    </div>'''


def method_card(name, description):
    return f'''<div class="method-card">
        <div class="method-name">{name}</div>
        <div class="method-description">{description}</div>
    </div>'''


def validate_dataset(df):
    required = {"STATION", "LAT", "LON", "DATE", "OBS_mm"}
    errors, warnings = [], []
    missing = required - set(df.columns)
    if missing:
        return ["Missing required columns: " + ", ".join(sorted(missing))], warnings
    if df.empty:
        return ["The uploaded CSV is empty."], warnings

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    if df["DATE"].isna().any():
        errors.append(f"{df['DATE'].isna().sum():,} rows contain invalid dates.")
    for col in ["LAT", "LON", "OBS_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int(df[col].isna().sum())
        if bad:
            errors.append(f"{bad:,} rows contain invalid {col} values.")
    if ((df["LAT"] < -90) | (df["LAT"] > 90)).any():
        errors.append("Some latitude values are outside -90 to 90.")
    if ((df["LON"] < -180) | (df["LON"] > 180)).any():
        errors.append("Some longitude values are outside -180 to 180.")
    if (df["OBS_mm"] < 0).any():
        errors.append("Observed precipitation cannot be negative.")
    duplicates = int(df.duplicated(subset=["STATION", "DATE"]).sum())
    if duplicates:
        warnings.append(f"{duplicates:,} duplicate station/date records found.")
    missing_obs = int(df["OBS_mm"].isna().sum())
    if missing_obs:
        warnings.append(f"{missing_obs:,} missing observed precipitation values.")
    return errors, warnings


def fmt(v):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="eyebrow">CLIMATE DATA ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Precipitation Bias Correction</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Evaluate and correct ERA5 reanalysis precipitation using observed weather-station data.</div>', unsafe_allow_html=True)

if "obs_df" not in st.session_state: st.session_state.obs_df = None
if "pooled" not in st.session_state: st.session_state.pooled = None
if "results" not in st.session_state: st.session_state.results = None

with st.sidebar:
    st.markdown("### Project")
    st.markdown("**Precipitation Bias Correction**")
    st.caption("ERA5 reanalysis + station observations")
    st.divider()
    st.markdown("### About")
    st.caption("Retrieve ERA5-based daily precipitation, evaluate correction methods using temporal cross-validation, and export corrected precipitation.")
    st.divider()
    st.caption("Streamlit scientific dashboard")

# -----------------------------
# Workflow tabs
# -----------------------------
tab_data, tab_config, tab_results = st.tabs(["01  Data", "02  Configure", "03  Results"])

with tab_data:
    workflow(1)
    st.markdown("## Upload observed precipitation data")
    st.markdown('<div class="helper">Upload a CSV containing station observations. ERA5 reanalysis data will be retrieved automatically from station coordinates and dates.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed", help="Required: STATION, LAT, LON, DATE, OBS_mm")
    if uploaded is None:
        st.markdown('<div class="card"><strong>Required columns</strong><br><span class="helper">STATION • LAT • LON • DATE • OBS_mm</span></div>', unsafe_allow_html=True)
        st.info("Upload your station CSV to continue.")
    else:
        try:
            obs_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        errors, warnings = validate_dataset(obs_df)
        st.session_state.obs_df = obs_df

        if errors:
            st.error("Dataset validation failed.")
            for e in errors: st.markdown(f"- {e}")
        else:
            st.success(f"Dataset ready: {uploaded.name}")
            if warnings:
                with st.expander("Review data-quality warnings"):
                    for w in warnings: st.warning(w)

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(metric_card("Stations", f"{obs_df.STATION.nunique():,}", "Unique station IDs"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card("Observations", f"{len(obs_df):,}", "Rows in uploaded CSV"), unsafe_allow_html=True)
            with c3: st.markdown(metric_card("Date range", f"{obs_df.DATE.min():%d %b %Y} – {obs_df.DATE.max():%d %b %Y}", "Observed data period"), unsafe_allow_html=True)
            with c4: st.markdown(metric_card("Missing OBS", f"{obs_df.OBS_mm.isna().sum():,}", "Missing precipitation values"), unsafe_allow_html=True)

            st.markdown("### Dataset preview")
            st.dataframe(obs_df.head(10), use_container_width=True, hide_index=True)

            map_df = obs_df[["STATION","LAT","LON"]].dropna().drop_duplicates("STATION")
            if not map_df.empty:
                st.markdown("### Station locations")
                st.map(map_df.rename(columns={"LAT":"lat","LON":"lon"})[["lat","lon"]])

with tab_config:
    workflow(2, 1 if st.session_state.obs_df is not None else 0)
    if st.session_state.obs_df is None:
        st.info("Upload a dataset in the Data tab first.")
    else:
        obs_df = st.session_state.obs_df
        min_year, max_year = int(obs_df.DATE.dt.year.min()), int(obs_df.DATE.dt.year.max())
        st.markdown("## Analysis configuration")
        st.markdown('<div class="helper">Configure the training period and outlier handling before running ERA5 retrieval and bias correction.</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            train_start = st.number_input("Training start year", min_year, max_year, max(min_year,min(TRAIN_START_DEFAULT,max_year)))
        with c2:
            train_end = st.number_input("Training end year", min_year, max_year, min(TRAIN_END_DEFAULT,max_year))
        if train_start >= train_end: st.warning("Training end year should be later than the start year.")

        st.markdown("### Outlier handling")
        outlier_factor = st.slider("IQR multiplier", 1.0, 5.0, 3.0, 0.5, help="Controls how aggressively extreme ERA5-observation residuals are removed.")

        st.markdown("### Correction methods")
        methods = [
            ("Linear Scaling", "Adjusts ERA5 rainfall using the observed-to-ERA5 mean ratio."),
            ("Variance Scaling", "Adjusts both mean and variability of precipitation."),
            ("Power Transformation", "Uses a nonlinear power relationship between datasets."),
            ("LOCI", "Corrects wet-day occurrence and rainfall intensity."),
            ("Parametric Quantile Mapping", "Maps ERA5 rainfall probabilities to the observed distribution."),
        ]
        cols = st.columns(3)
        for i,(name,desc) in enumerate(methods):
            with cols[i%3]: st.markdown(method_card(name,desc), unsafe_allow_html=True)

        st.divider()
        run_btn = st.button("Run ERA5 retrieval & bias correction", type="primary", use_container_width=True, disabled=train_start>=train_end)
        if run_btn:
            progress = st.progress(0, text="Preparing analysis...")
            status = st.empty()
            try:
                status.info("Fetching ERA5 reanalysis for station locations...")
                progress.progress(25, text="Fetching ERA5 reanalysis...")
                pooled = build_pooled_dataset(obs_df.copy())
                if pooled.empty:
                    st.error("No overlapping station/ERA5 records were found. Check dates and coordinates.")
                    st.stop()
                st.session_state.pooled = pooled
                status.info("Running temporal cross-validation and bias correction...")
                progress.progress(65, text="Running bias-correction methods...")
                st.session_state.results = run_pipeline(pooled, int(train_start), int(train_end), float(outlier_factor))
                progress.progress(100, text="Analysis complete")
                status.success(f"Completed: {pooled.STATION.nunique():,} stations and {len(pooled):,} merged records.")
                st.success("Analysis completed. Open the Results tab.")
            except Exception as exc:
                progress.empty(); status.empty(); st.exception(exc)

with tab_results:
    has_results = st.session_state.results is not None
    workflow(4 if has_results else 3, 3 if has_results else 2)
    if not has_results:
        st.info("Run the analysis from the Configure tab to see results here.")
    else:
        results = st.session_state.results
        pooled = st.session_state.pooled
        corrected = results["pooled_train_corrected"]
        summary = results["cv_summary"].copy()
        winner = results["winner"]
        st.markdown("## Analysis results")
        st.markdown('<div class="helper">Compare cross-validation performance, inspect raw versus corrected precipitation, and export the results.</div>', unsafe_allow_html=True)

        winner_metrics = summary.loc[winner] if winner in summary.index else None
        raw = compute_metrics(corrected.OBS_mm, corrected.ERA5_mm)
        corr = compute_metrics(corrected.OBS_mm, corrected.CORR_mm)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(metric_card("Stations", f"{pooled.STATION.nunique():,}", "Merged station records"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Merged records", f"{len(pooled):,}", "Observed + ERA5"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Selected method", winner.replace("_"," "), "Based on temporal CV"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Mean NSE", fmt(winner_metrics.NSE if winner_metrics is not None else corr["NSE"]), "Cross-validation performance"), unsafe_allow_html=True)

        st.markdown("### Selected correction method")
        st.markdown(f'''<div class="selected-method"><div class="selected-method-name">{winner.replace("_"," ")}</div><div class="helper">Selected using mean temporal cross-validation performance.</div></div>''', unsafe_allow_html=True)

        a,b,c,d = st.columns(4)
        with a: st.metric("RMSE", f"{fmt(corr['RMSE_mm'])} mm")
        with b: st.metric("R²", fmt(corr["R2"]))
        with c: st.metric("NSE", fmt(corr["NSE"]))
        with d: st.metric("Bias", f"{fmt(np.mean(corrected.CORR_mm-corrected.OBS_mm))} mm")

        r1,r2,r3,r4 = st.tabs(["Performance","Before vs After","Station Data","Export"])
        with r1:
            st.markdown("### Cross-validation summary")
            display = summary.reset_index().rename(columns={"RMSE_mm":"RMSE (mm)","R2":"R²"})
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.markdown("### NSE by method")
            st.bar_chart(summary["NSE"])
        with r2:
            st.markdown("### Agreement with observed precipitation")
            c1,c2 = st.columns(2)
            with c1:
                st.pyplot(stats_scatter_fig(corrected.OBS_mm, corrected.ERA5_mm, "Raw ERA5 vs Observed"), use_container_width=True)
            with c2:
                st.pyplot(stats_scatter_fig(corrected.OBS_mm, corrected.CORR_mm, f"Corrected ({winner.replace('_',' ')}) vs Observed"), use_container_width=True)
            p1,p2,p3 = st.columns(3)
            with p1: st.metric("RMSE", f"{raw['RMSE_mm']:.2f} → {corr['RMSE_mm']:.2f} mm")
            with p2: st.metric("R²", f"{raw['R2']:.2f} → {corr['R2']:.2f}")
            with p3: st.metric("NSE", f"{raw['NSE']:.2f} → {corr['NSE']:.2f}")
        with r3:
            stations = sorted(corrected.STATION.dropna().astype(str).unique())
            selected = st.selectbox("Station", ["All stations"] + stations)
            view = corrected if selected == "All stations" else corrected[corrected.STATION.astype(str)==selected]
            st.caption(f"Showing {len(view):,} records")
            st.dataframe(view, use_container_width=True, hide_index=True)
        with r4:
            st.markdown("### Export results")
            c1,c2 = st.columns(2)
            with c1:
                st.download_button("Download corrected CSV", corrected.to_csv(index=False).encode("utf-8"), "corrected_precipitation.csv", "text/csv", use_container_width=True)
            with c2:
                st.download_button("Download CV summary", summary.reset_index().to_csv(index=False).encode("utf-8"), "cross_validation_summary.csv", "text/csv", use_container_width=True)
