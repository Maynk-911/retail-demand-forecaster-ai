import pandas as pd
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
        logging.FileHandler("etl_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# SQL KPI Queries
SQL_KPI_QUERIES = """
-- 1. Stockout % 
-- (Proxy: Days with 0 Sales when Store was Open)
SELECT 
    Store,
    (SUM(CASE WHEN Sales = 0 AND Open = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS Stockout_Percentage
FROM processed_data
GROUP BY Store;

-- 2. Inventory Turnover Ratio
-- (Template: Cost of Goods Sold / Average Inventory)
-- Note: Requires external inventory data. Using Sales/Customers as a performance proxy here.
SELECT 
    Store,
    SUM(Sales) / NULLIF(AVG(Sales), 0) as Sales_Volume_Turnover_Proxy
FROM processed_data
GROUP BY Store;

-- 3. Weekly Store Sales Trends
SELECT 
    Store,
    strftime('%Y-%W', Date) as Week_Year,
    SUM(Sales) as Weekly_Sales
FROM processed_data
GROUP BY Store, Week_Year
ORDER BY Store, Week_Year;
"""

def extract_data(train_path, store_path):
    """Loads CSV files into DataFrames."""
    try:
        logger.info(f"Loading data from {train_path} and {store_path}...")
        train = pd.read_csv(train_path, low_memory=False)
        store = pd.read_csv(store_path)
        logger.info("Extraction successful.")
        return train, store
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        sys.exit(1)

def transform_data(train, store):
    """Merges, cleans, and filters the data."""
    try:
        logger.info("Starting data transformation...")

        # 1. Handle missing values in store.csv
        # CompetitionDistance: fill with median or high value
        store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
        # Fill other competition/promo fields with 0
        store.fillna(0, inplace=True)

        # 2. Merge data on Store
        merged_df = pd.merge(train, store, on='Store', how='left')
        logger.info(f"Merged Data Shape: {merged_df.shape}")

        # 3. Convert Date to datetime
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])

        # 4. Filter for Open stores
        # We only care about trends when the store is actually open
        merged_df = merged_df[merged_df['Open'] == 1]
        logger.info(f"Filtered for Open stores. New Shape: {merged_df.shape}")

        # 5. Handle missing values in train if any (Open stores usually have Sales/Customers)
        merged_df['Sales'] = merged_df['Sales'].fillna(0)
        merged_df['Customers'] = merged_df['Customers'].fillna(0)

        logger.info("Transformation complete.")
        return merged_df
    except Exception as e:
        logger.error(f"Error during transformation: {e}")
        sys.exit(1)

def load_processed_data(df, output_path):
    """Saves the cleaned data to CSV."""
    try:
        logger.info(f"Saving processed data to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        sys.exit(1)

def main():
    train_file = "train.csv"
    store_file = "store.csv"
    output_file = "processed_data.csv"

    # Extraction
    train_df, store_df = extract_data(train_file, store_file)

    # Transformation
    processed_df = transform_data(train_df, store_df)

    # Loading
    load_processed_data(processed_df, output_file)

    # Output SQL Queries
    print("-" * 50)
    print("SQL KPI QUERIES")
    print("-" * 50)
    print(SQL_KPI_QUERIES)
    print("-" * 50)
    logger.info("ETL Pipeline completed successfully.")

if __name__ == "__main__":
    main()
