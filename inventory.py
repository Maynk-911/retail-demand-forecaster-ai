import pandas as pd
import numpy as np
import logging
import sys
import os

# ==========================================
# Configuration & Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inventory_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def evaluate_inventory(forecast_file, output_file, lead_time=7, service_level_z=1.65):
    """
    Calculates Safety Stock and Reorder Point (ROP) based on forecasted demand.
    Identifies stores requiring immediate inventory transfers and saves a recommendation table.
    """
    if not os.path.exists(forecast_file):
        logger.error(f"Input file '{forecast_file}' not found. Please run the forecasting engine first.")
        sys.exit(1)

    try:
        logger.info(f"Loading forecast results from {forecast_file}...")
        df = pd.read_csv(forecast_file)

        # Basic data validation
        required_cols = ['Store', 'yhat']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in forecast data: {missing_cols}")

        if df['yhat'].isnull().any():
            logger.warning("Missing values found in 'yhat' column. Filling NaNs with 0 to prevent math errors.")
            df['yhat'] = df['yhat'].fillna(0)

        # 1. Grouped Metrics Calculation
        logger.info("Calculating Demand Metrics grouped by Store...")
        
        # Group computations per store to avoid data leakage
        inventory_metrics = df.groupby('Store').agg(
            Avg_Demand=('yhat', 'mean'),
            Std_Demand=('yhat', 'std')
        ).reset_index()

        # Handle stores with 0 standard deviation (e.g. only 1 data point or flat line forecast)
        inventory_metrics['Std_Demand'] = inventory_metrics['Std_Demand'].fillna(0)

        # 2. Safety Stock & ROP Formulas
        logger.info("Applying Supply Chain Math for Safety Stock and ROP...")
        
        # Safety Stock = Z * Std_Demand * sqrt(Lead_Time)
        inventory_metrics['Safety_Stock'] = service_level_z * inventory_metrics['Std_Demand'] * np.sqrt(lead_time)
        
        # ROP = (Avg_Demand * Lead_Time) + Safety_Stock
        inventory_metrics['ROP'] = (inventory_metrics['Avg_Demand'] * lead_time) + inventory_metrics['Safety_Stock']
        
        # Round up stock values as we can't order fractional items
        inventory_metrics['Safety_Stock'] = np.ceil(inventory_metrics['Safety_Stock'])
        inventory_metrics['ROP'] = np.ceil(inventory_metrics['ROP'])

        # 3. Simulate Current Inventory Levels
        # (Assuming current inventory oscillates around ROP to visualize the 'needs transfer' logic)
        np.random.seed(42) # For reproducibility
        inventory_metrics['Current_Inventory'] = inventory_metrics['ROP'] * np.random.uniform(0.5, 1.5, size=len(inventory_metrics))
        inventory_metrics['Current_Inventory'] = np.floor(inventory_metrics['Current_Inventory'])

        # 4. Generate Restock Recommendation Flag
        inventory_metrics['Needs_Restock'] = inventory_metrics['Current_Inventory'] < inventory_metrics['ROP']

        # Save complete dataset
        logger.info(f"Saving comprehensive inventory metrics to {output_file}...")
        inventory_metrics.to_csv(output_file, index=False)

        # 5. Extract Restock Recommendations 
        recommendations = inventory_metrics[inventory_metrics['Needs_Restock'] == True]
        
        print("\n" + "="*50)
        print("RESTOCK RECOMMENDATION TABLE (Immediate Transfers)")
        print("="*50)
        if recommendations.empty:
            print("All stores have sufficient inventory. No immediate transfers needed.")
        else:
            # Drop the boolean flag for display, it is implied
            display_df = recommendations.drop(columns=['Needs_Restock'])
            print(display_df.to_string(index=False))
            print("="*50 + "\n")
            
        logger.info("Inventory evaluation complete!")

    except Exception as e:
        logger.error(f"An error occurred during inventory planning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    evaluate_inventory("forecast_results.csv", "inventory_metrics.csv")
