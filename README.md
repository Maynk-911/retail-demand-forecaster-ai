📦 Supply Chain Analytics & Inventory Optimization Engine
🚀 Overview
Ye ek end-to-end forecasting solution hai jo retail sales data ko use karke inventory levels ko optimize karta hai. Is project ka main goal stockouts ko rokna aur excess inventory cost ko kam karna hai.

✨ Key Features
Time-Series Forecasting: Meta ke Prophet model ka use karke daily sales predict ki gayi hain.

Multiplicative Seasonality: Retail sales ke complex patterns ko handle karne ke liye bias-corrected seasonality use ki gayi hai.

What-If Simulator: Streamlit dashboard par interactive sliders hain jisse "Promotion Impact" aur "Lead Time" ka live asar dekha ja sakta hai.

Inventory Automation: Har store ke liye Reorder Point (ROP) aur Safety Stock auto-calculate hota hai.

Actionable Alerts: Dashboard turant flag karta hai ki kaunsa store "Stockout Risk" par hai.

📊 Model Performance
MAPE (Mean Absolute Percentage Error): 11.8% (~88% Accuracy)

RMSE (Root Mean Square Error): 852 (On 10k-15k avg. volume)

Potential Savings: Estimated $39K through optimized stock levels.

🛠️ Tech Stack
Language: Python 3.9+

Libraries: Pandas, NumPy, Prophet, Scikit-learn

Visualization: Streamlit, Plotly

Environment: Virtualenv

📁 Project Structure
ETL.py: Data cleaning aur feature engineering.

forecast.py: Prophet model training aur metrics generation.

app.py: Streamlit dashboard code.

accuracy_metrics.csv: Per-store accuracy tracking.
# retail-demand-forecaster-ai
An end to end Supply Chain Forecasting engine using Meta's Prophet and Streamlit. Features a What If Simulator for promotion impact, automated Reorder Point (ROP) logic and interactive inventory KPIs. Achieved 11.8% MAPE with bias corrected multiplicative seasonality. 🚀📈
