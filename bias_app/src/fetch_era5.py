import requests
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def fetch_era5_daily_precip(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily precipitation (ERA5-based) from Open-Meteo Archive API. No API key needed."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "DATE": pd.to_datetime(data["daily"]["time"]),
        "ERA5_mm": data["daily"]["precipitation_sum"],
    })
    return df