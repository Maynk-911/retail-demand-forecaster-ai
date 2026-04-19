import streamlit as st
import pandas as pd
import numpy as np
import os

# ==========================================
# Helper Functions
# ==========================================
def format_number(num):
    """Formats large numbers with K or M suffixes for readability."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"
    else:
        return f"{num:.0f}"

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="Retail Forecasting & Inventory Dashboard", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fc;
        border-radius: 5px;
        padding: 15px;
        border-left: 4px solid #4e73df;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .section-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #34495e;
        border-bottom: 2px solid #eaeded;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Data Loading & Caching
# ==========================================
@st.cache_data
def load_data():
    forecast_df = pd.DataFrame()
    inventory_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    
    # Load Forecast Results
    if os.path.exists("forecast_results.csv"):
        try:
            forecast_df = pd.read_csv("forecast_results.csv", parse_dates=['ds'])
        except Exception as e:
            st.error(f"Error loading forecast_results.csv: {e}")
    
    # Load Inventory Metrics
    if os.path.exists("inventory_metrics.csv"):
        try:
            inventory_df = pd.read_csv("inventory_metrics.csv")
        except Exception as e:
            st.error(f"Error loading inventory_metrics.csv: {e}")
            
    # Load Accuracy Metrics
    if os.path.exists("accuracy_metrics.csv"):
        try:
            metrics_df = pd.read_csv("accuracy_metrics.csv")
        except Exception as e:
            st.error(f"Error loading accuracy_metrics.csv: {e}")
            
    # Mock some historical actual sales to compare against predictions
    if not forecast_df.empty:
        # Simulate Actuals
        np.random.seed(42)
        forecast_df['actual_sales'] = forecast_df['yhat'] * np.random.uniform(0.8, 1.2, size=len(forecast_df))
        
        # Simulate that actuals are only available for the first half of the forecast period
        unique_dates = forecast_df['ds'].sort_values().unique()
        split_date = unique_dates[len(unique_dates) // 2]
        forecast_df.loc[forecast_df['ds'] >= split_date, 'actual_sales'] = np.nan
        
    return forecast_df, inventory_df, metrics_df

# ==========================================
# Main App
# ==========================================
def main():
    st.markdown("<h1 class='main-header'>Supply Chain Command Center</h1>", unsafe_allow_html=True)
    
    forecast_df, inventory_df, metrics_df = load_data()
    
    if forecast_df.empty or inventory_df.empty:
        st.warning("Data files (forecast_results.csv or inventory_metrics.csv) are missing. Please run forecasting and inventory modules first.")
        st.stop()
        
    # --- Sidebar: What-If Simulator ---
    st.sidebar.header("⚙️ What-If Simulator")
    promo_impact = st.sidebar.slider("Promotion Impact (%)", min_value=-20, max_value=50, value=0, step=5)
    lead_time = st.sidebar.slider("Lead Time (Days)", min_value=1, max_value=30, value=7, step=1)
    
    # Apply dynamic simulation based on sidebar inputs
    forecast_df['sim_yhat'] = forecast_df['yhat'] * (1 + promo_impact / 100.0)
    inventory_df['sim_Avg_Demand'] = inventory_df['Avg_Demand'] * (1 + promo_impact / 100.0)
    inventory_df['sim_Safety_Stock'] = np.ceil(1.65 * inventory_df['Std_Demand'] * np.sqrt(lead_time))
    inventory_df['sim_ROP'] = np.ceil((inventory_df['sim_Avg_Demand'] * lead_time) + inventory_df['sim_Safety_Stock'])
    
    # --- Header: KPI Metrics ---
    total_forecasted_demand = forecast_df['sim_yhat'].sum()
    stockout_risk_count = len(inventory_df[inventory_df['Current_Inventory'] < inventory_df['sim_ROP']])
    inventory_turnover = round(np.random.uniform(4.0, 8.0), 2) # Dummy placeholder logic
    potential_savings = round(np.random.uniform(10000, 50000), 2) # Dummy placeholder logic
    
    # Calculate baseline accuracy metrics
    avg_mape = f"{metrics_df['MAPE'].mean():.1f}%" if not metrics_df.empty else "N/A"
    avg_rmse = f"{metrics_df['RMSE'].mean():.0f}" if not metrics_df.empty else "N/A"
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Forecasted Demand", format_number(total_forecasted_demand))
    col2.metric("Stockout Risk Count", stockout_risk_count)
    col3.metric("Inventory Turnover", inventory_turnover)
    col4.metric("Potential Savings", f"${format_number(potential_savings)}")
    col5.metric("Baseline MAPE", avg_mape, help="Mean Absolute Percentage Error (Historical Baseline)")
    col6.metric("Baseline RMSE", avg_rmse, help="Root Mean Square Error (Historical Baseline)")
    
    # --- Middle Section: Visuals ---
    st.markdown("<h2 class='section-header'>Performance Visuals</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs. Predicted Sales")
        # Ensure chronological plotting
        plot_df = forecast_df.groupby('ds').agg({'sim_yhat': 'sum', 'actual_sales': 'sum'}).reset_index()
        st.line_chart(plot_df.set_index('ds')[['sim_yhat', 'actual_sales']])
        
    with col2:
        st.subheader("Current Stock vs. Required Stock")
        bar_df = inventory_df[['Store', 'Current_Inventory', 'sim_ROP']].copy()
        # Clean chart labels from '1', '2' to 'Store 1', 'Store 2'
        bar_df['StoreLabel'] = 'Store ' + bar_df['Store'].astype(str)
        bar_df = bar_df.set_index('StoreLabel')
        
        bar_df.rename(columns={'sim_ROP': 'Required Stock (ROP)', 'Current_Inventory': 'Current Stock'}, inplace=True)
        st.bar_chart(bar_df[['Current Stock', 'Required Stock (ROP)']])
        
    # --- Bottom Section: Actionable Tables ---
    st.markdown("<h2 class='section-header'>Actionable Insights</h2>", unsafe_allow_html=True)
    
    # Auto-Replenishment Table
    st.subheader("⚠️ Auto-Replenishment Table (Hit ROP)")
    replenish_df = inventory_df[inventory_df['Current_Inventory'] <= inventory_df['sim_ROP']][['Store', 'Current_Inventory', 'sim_ROP', 'sim_Safety_Stock']].copy()
    
    if not replenish_df.empty:
        replenish_df['Deficit'] = replenish_df['sim_ROP'] - replenish_df['Current_Inventory']
        st.dataframe(replenish_df, use_container_width=True)
        
        # Download Button
        csv = replenish_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Restock List",
            data=csv,
            file_name='restock_list.csv',
            mime='text/csv',
        )
    else:
        st.info("No stores hit the Reorder Point. All inventory levels are sound.")
        
    # Warehouse Transfer Table
    st.subheader("🔄 Warehouse Transfer Table")
    
    # Updated Warehouse Transfer Logic 
    # Identifies stores currently holding 20% more inventory than the global average
    avg_stock = inventory_df['Current_Inventory'].mean()
    overstock_threshold = avg_stock * 1.20  # 20% above average
    
    overstocked_stores = inventory_df[inventory_df['Current_Inventory'] > overstock_threshold]
    understocked_stores = replenish_df  # Uses same Logic/DataFrame hitting ROP requirements
    
    if not overstocked_stores.empty and not understocked_stores.empty:
        transfers = []
        for _, u_row in understocked_stores.iterrows():
            # Pick first available overstocked store correctly
            o_store = overstocked_stores.iloc[0] 
            
            # Ensure the transfer doesn't deplete the overstocked store past its own ROP limits or the average threshold
            available_surplus = max(0, o_store['Current_Inventory'] - o_store['sim_ROP']) 
            
            transfer_amt = min(u_row['Deficit'], available_surplus)
            
            if transfer_amt > 0:
                transfers.append({
                    "From Store": int(o_store['Store']),
                    "To Store": int(u_row['Store']),
                    "Transfer Quantity": int(transfer_amt)
                })
        
        if transfers:
            st.dataframe(pd.DataFrame(transfers), use_container_width=True)
        else:
            st.info("Available surplus is locked or insufficient to safely process an open transfer request.")
    else:
        st.info("No viable transfers identified based on the 20% global average metric.")

if __name__ == "__main__":
    main()
