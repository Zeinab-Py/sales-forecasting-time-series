import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# --- Title and Image ---
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("\U0001F4C8 NextQ Forecast")
st.subheader("Your Sales Forecasting Dashboard")

# Load header image if available
image_path = r"C:\Users\zeina\OneDrive\Desktop\Bootcamp\final project\Sales Forecasting Project\app\Designer.jpeg"
if os.path.exists(image_path):
    st.image(image_path)

# --- Description ---
st.markdown("""
Upload last year's sales data, optionally include external factors, and forecast Q1 sales for the upcoming year using Facebook Prophet.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Upload Your Data")
with st.sidebar.expander("Expected Format"):
    st.markdown("""
    **Your CSV should contain at least these columns:**
    - `date` (YYYY-MM-DD)
    - `store`
    - `item`
    - `sales`

    Optional:
    - `year`, `month`, `day`, `weekday` (for manual inspection)
    """)

sales_file = st.sidebar.file_uploader("Sales Data (CSV)", type="csv")
external_file = st.sidebar.file_uploader("Optional: External Variables (CSV)", type="csv")

# --- Load Holidays ---
@st.cache_data

def load_holidays():
    holidays_path = r"C:\Users\zeina\OneDrive\Desktop\Bootcamp\final project\Sales Forecasting Project\data\US Bank holidays (up to 2020)"
    df = pd.read_csv(holidays_path, header=None, names=["date", "holiday"])
    df["date"] = pd.to_datetime(df["date"])
    return df

holidays = load_holidays()

# --- Main Logic ---
if sales_file:
    df = pd.read_csv(sales_file, parse_dates=["date"])
    latest_year = df["date"].dt.year.max()
    next_year = latest_year + 1
    df = df[df["date"].dt.year == latest_year]

    # Merge with holidays
    df = df.merge(holidays, on="date", how="left")
    df["holiday_bool"] = df["holiday"].notnull().astype(int)
    df = pd.get_dummies(df, columns=["month", "holiday", "weekday"], prefix=["month", "holiday", "weekday"], drop_first=True)

    # Merge external variables
    if external_file:
        ext_df = pd.read_csv(external_file, parse_dates=["date"])
        df = df.merge(ext_df, on="date", how="left")
        extra_vars = [col for col in ext_df.columns if col != "date"]
        selected_vars = st.multiselect("Choose external variables", extra_vars, default=extra_vars)
    else:
        selected_vars = []

    # --- Store/Item Selection ---
    
    st.sidebar.header("Filter Data")
    all_stores = df["store"].unique().tolist()
    all_items = df["item"].unique().tolist()
    select_all_stores = st.sidebar.checkbox("Select All Stores", value=True)
    select_all_items = st.sidebar.checkbox("Select All Items", value=True)
    #stores = st.sidebar.multiselect("Select Store(s)", all_stores, default=all_stores)
    #items = st.sidebar.multiselect("Select Item(s)", all_items, default=all_items)

    # Multiselects (enabled only if "Select All" is unchecked)
    if select_all_stores:
        stores = all_stores
    else:
        stores = st.sidebar.multiselect("Select Store(s)", all_stores)

    if select_all_items:
        items = all_items
    else:
        items = st.sidebar.multiselect("Select Item(s)", all_items)

    # Filter the data
    filtered = df[df["store"].isin(stores) & df["item"].isin(items)]

    if filtered.empty:  
        st.error("⚠️ No data available after filtering. Please adjust your store/item selection.")
        st.stop() 

    #filtered = df[df["store"].isin(stores) & df["item"].isin(items)]
    #if filtered.empty:
        #st.error("No data available after filtering. Please adjust store/item selection.")
        #st.stop()

    agg = filtered.groupby("date")["sales"].sum().reset_index()
    prophet_df = agg.rename(columns={"date": "ds", "sales": "y"})
    

    # --- Forecasting Parameters ---
    st.sidebar.header("Model Settings")
    changepoint_scale = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
    fourier_order = st.sidebar.slider("Fourier Order (Monthly)", 2, 10, 5)

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_scale
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=fourier_order)
    m.add_country_holidays(country_name="US")
    for var in selected_vars:
        m.add_regressor(var)

    m.fit(prophet_df)

    future = pd.DataFrame(pd.date_range(f"{next_year}-01-01", f"{next_year}-03-31", freq="D"), columns=["ds"])
    forecast = m.predict(future)

    # --- Plot Forecast ---
    st.subheader(f"Forecast for Q1 {next_year}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prophet_df["ds"], prophet_df["y"], label=f"Actual ({latest_year})")
    ax.plot(forecast["ds"], forecast["yhat"], label=f"Forecast ({next_year})")
    ax.set_title("Sales Forecast")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # --- Table ---
    table = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecasted Sales"})
    if stores != all_stores or items != all_items:
        table["Store"] = ", ".join(map(str, stores))
        table["Item"] = ", ".join(map(str, items))
        table = table[["Store", "Item", "Date", "Forecasted Sales"]]
    st.dataframe(table, use_container_width=True)

else:
    st.info("Please upload your sales data CSV to begin.")
