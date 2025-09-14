# refinery_dashboard.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Refinery Profitability Dashboard",
    layout="wide"
)

# -------------------------
# Load Data & Model
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("integrated_data.csv", parse_dates=["date"])
    data.set_index("date", inplace=True)
    return data

@st.cache_resource
def load_model():
    with open("arima_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_transaction_data():
    tx_data = pd.read_csv("cleaned_dataset_1.csv", parse_dates=["date"])
    return tx_data

data = load_data()
model = load_model()
transaction_data = load_transaction_data()

# -------------------------
# Forecast
# -------------------------
forecast_steps = 30
forecast = model.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

forecast_df["date"] = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=forecast_steps,
    freq="D"
)
forecast_df.set_index("date", inplace=True)
forecast_csv = forecast_df.to_csv().encode("utf-8")

# -------------------------
# Sidebar Controls
# -------------------------
metric_options = {
    "Profit": "profit_usd",
    "Diesel Sales": "diesel_sales_usd",
    "Gasoline Sales": "gasoline_sales_usd",
    "Crude Cost": "crude_cost_usd",
    "Operational Cost": "operational_cost_usd"
}

selected_label = st.sidebar.selectbox("Choose a Metric to Explore", list(metric_options.keys()))
selected_metric = metric_options[selected_label]

# -------------------------
# Dashboard Title
# -------------------------
st.title("⛽ Refinery Profitability Dashboard")
st.markdown("Track refinery performance, costs, profitability, and potential retail leakage with a **30-day forecast**.")

# -------------------------
# Metric Over Time
# -------------------------
st.subheader(f"{selected_label} Over Time")
fig_metric = px.line(
    data,
    x=data.index,
    y=selected_metric,
    title=f"{selected_label} Trend",
    labels={"x": "Date", selected_metric: selected_label}
)
st.plotly_chart(fig_metric, use_container_width=True)

# -------------------------
# Key Trends
# -------------------------
st.subheader("Key Trends")
col1, col2 = st.columns(2)

with col1:
    fig_profit = px.line(data, x=data.index, y="profit_usd", title="Profit Over Time")
    st.plotly_chart(fig_profit, use_container_width=True)

with col2:
    fig_sales = px.line(
        data,
        x=data.index,
        y=["diesel_sales_usd", "gasoline_sales_usd"],
        title="Diesel vs Gasoline Sales"
    )
    st.plotly_chart(fig_sales, use_container_width=True)

# -------------------------
# Costs & Correlation
# -------------------------
col3, col4 = st.columns(2)

with col3:
    fig_costs = px.line(
        data,
        x=data.index,
        y=["crude_cost_usd", "operational_cost_usd"],
        title="Cost Breakdown"
    )
    st.plotly_chart(fig_costs, use_container_width=True)

with col4:
    corr = data.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Forecast & Residuals
# -------------------------
st.subheader("Profitability Forecast (30 Days)")
col5, col6 = st.columns(2)

with col5:
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=data.index, y=data["profit_usd"], mode="lines", name="History", line=dict(color="blue")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["mean"], mode="lines", name="Forecast", line=dict(color="orange")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["mean_ci_upper"], mode="lines",
        name="Upper CI", line=dict(color="orange", dash="dot")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["mean_ci_lower"], mode="lines",
        name="Lower CI", line=dict(color="orange", dash="dot"),
        fill="tonexty", fillcolor="rgba(255,165,0,0.2)"
    ))
    fig_forecast.update_layout(title="30-Day Profit Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

with col6:
    residuals = model.resid
    fig_resid = px.line(
        x=residuals.index,
        y=residuals.values,
        title="Model Residuals",
        labels={"x": "Date", "y": "Residual"}
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_resid, use_container_width=True)

# -------------------------
# -------------------------
# Station-level Retail Leakage (Day-by-Day Corrected)

st.subheader("ℹ️ How Leakage / Suspicious Transactions Are Derived")
st.markdown(
    """
    **How Leakage / Suspicious Transactions are Calculated:**  
    1. For each station, we track daily inventory (`inventory_level_liters`) and sales (`volume_liters`).  
    2. The expected inventory for a day is calculated as:  
       *Previous Day's Inventory - Previous Day's Volume Sold*  
    3. If the actual inventory for the day is **less than the expected inventory**, the difference is considered **leakage / suspicious transaction**.  
    4. Leakage % is calculated relative to the expected inventory.  
    5. This method allows us to detect unaccounted-for stock movement across stations.
    """
)
# -------------------------
st.subheader("🚨 Station-level Retail Leakage (Day-by-Day)")

# Sort transaction data
transaction_data_sorted = transaction_data.sort_values(by=['station_id', 'date']).copy()
transaction_data_sorted['leakage_liters'] = 0
transaction_data_sorted['expected_inventory'] = 0

for station in transaction_data_sorted['station_id'].unique():
    station_data = transaction_data_sorted[transaction_data_sorted['station_id'] == station]
    prev_inventory = None

    for idx, row in station_data.iterrows():
        if prev_inventory is None:
            # First day, we don't have previous day, so we just set expected = actual
            expected_inventory = row['inventory_level_liters']
            leakage = 0
        else:
            # Expected inventory = previous day's inventory - previous day's volume sold
            expected_inventory = prev_inventory - prev_volume_sold
            # Leakage = positive difference between expected and actual
            leakage = max(0, expected_inventory - row['inventory_level_liters'])

        # Save values
        transaction_data_sorted.at[idx, 'expected_inventory'] = expected_inventory
        transaction_data_sorted.at[idx, 'leakage_liters'] = leakage

        # Update previous day info for next iteration
        prev_inventory = row['inventory_level_liters']
        prev_volume_sold = row['volume_liters']

# Calculate leakage %
transaction_data_sorted['leakage_pct'] = (transaction_data_sorted['leakage_liters'] / transaction_data_sorted['expected_inventory']) * 100

# Total leakage across all stations
total_station_leakage = transaction_data_sorted['leakage_liters'].sum()
st.metric("Total Station-level Leakage (Liters)", f"{total_station_leakage:,.0f}")

# Suspicious leakage detection
threshold = 0.01  # 1% of expected inventory
suspicious = transaction_data_sorted[transaction_data_sorted['leakage_pct'] > threshold]

if not suspicious.empty:
    st.warning("⚠️ Suspicious Leakage Detected!")
    st.dataframe(suspicious[['station_id', 'date', 'volume_liters', 'inventory_level_liters', 'expected_inventory', 'leakage_liters', 'leakage_pct']])
else:
    st.success("✅ No suspicious leakage detected")

# Top stations by cumulative leakage
top_leakage_stations = transaction_data_sorted.groupby('station_id')['leakage_liters'].sum().sort_values(ascending=False).head(5)
st.write("Top 5 stations with highest cumulative leakage (Liters):")
st.dataframe(top_leakage_stations)

# Plotly bar chart
fig_station = px.bar(
    top_leakage_stations.reset_index(),
    x='station_id',
    y='leakage_liters',
    title="Top 5 Stations by Cumulative Leakage (Liters)",
    text='leakage_liters'
)
fig_station.update_traces(texttemplate='%{text:.0f}', textposition='outside')
fig_station.update_layout(yaxis_title="Leakage (Liters)", xaxis_title="Station ID")
st.plotly_chart(fig_station)

transaction_data_sorted['date'] = pd.to_datetime(transaction_data_sorted['date'])

# Daily leakage trend per station
st.subheader("📊 Station Daily Leakage Trend")
fig_station_trend = px.line(
    transaction_data_sorted,
    x='date',
    y='leakage_liters',
    color='station_id',
    title="Daily Leakage Trend per Station",
    labels={'leakage_liters': 'Leakage (Liters)', 'date': 'Date', 'station_id': 'Station'}
)
st.plotly_chart(fig_station_trend, use_container_width=True)

# -------------------------
# Download Button
# -------------------------
st.download_button(
    "📥 Download Forecast Data as CSV",
    forecast_csv,
    "profit_forecast.csv",
    "text/csv"
)
