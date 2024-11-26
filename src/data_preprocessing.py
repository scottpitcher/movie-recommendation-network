# data_preprocessing.py

# Step 1: Importing packages
print("Importing packages...")
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging

# Step 2: Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Packages imported successfully.")

# Step 3: Define file paths
RAW_DATA_PATH = "data/full_data_with_credits.csv"
PROCESSED_DATA_PATH = "data/processed_data/full_data_binned_log.csv"

# Step 4: Load the raw data
logging.info(f"Loading raw data from {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)
logging.info("Raw data loaded successfully.")

# Step 5: Log-transform the popularity column
logging.info("Applying log transformation to the 'popularity' column...")
df['log_popularity'] = np.log1p(df['popularity']) 

# Step 6: Bin the log-transformed values
logging.info("Binning the log-transformed popularity values into categories...")
df['popularity_bin'] = pd.qcut(
    df['log_popularity'], 
    q=6, 
    labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Top']
)

# Step 7: Save the processed data
logging.info(f"Saving processed data to {PROCESSED_DATA_PATH}...")
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)  # Create the directory if it doesn't exist
df.to_csv(PROCESSED_DATA_PATH, index=False)
logging.info("Processed data saved successfully.")
