import pandas as pd
import os
from pycaret.time_series import TSForecastingExperiment

# Proactively import to prevent potential discovery issues with LightGBM
try:
    from lightgbm import LGBMRegressor
except ImportError:
    print("Warning: lightgbm not found. This might cause issues if not installed.")
    pass

def train_and_save_model(csv_path, model_alias, pkl_output_path, forecast_horizon=24):
    """
    Reads a CSV, trains a PyCaret time series model, and saves it.
    Assumes the CSV has two columns: the first is the datetime, the second is the value.

    Args:
        csv_path (str): Path to the input CSV file.
        model_alias (str): The alias of the model to train (e.g., 'lightgbm_cds_dt').
        pkl_output_path (str): Path to save the output .pkl file (without extension).
        forecast_horizon (int): The number of periods to forecast.
    """
    print(f"--- Starting Process for: {csv_path} ---")

    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found at '{csv_path}'. Skipping.")
        return

    # 1. Read Data with robust encoding
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    print(f"✅ Data loaded. Found columns: {df.columns.tolist()}")

    # 2. Prepare Data
    # Dynamically rename the first two columns for consistency
    original_cols = df.columns
    df = df.rename(columns={original_cols[0]: 'datetime', original_cols[1]: 'value'})
    
    # Convert to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # --- Timestamp Manipulation to make forecast start on 2026-01-01 ---
    # We shift the entire timeline so the training data ends on 2025-12-31 23:00:00.
    last_timestamp = df['datetime'].max()
    target_end_timestamp = pd.Timestamp('2025-12-31 23:00:00')
    time_offset = target_end_timestamp - last_timestamp
    df['datetime'] = df['datetime'] + time_offset
    # --- End of Timestamp Manipulation ---

    # Set datetime as index
    df = df.set_index('datetime')
    
    # Ensure and set frequency (crucial for pycaret)
    # Using asfreq and ffill is more robust if there are missing hourly data points
    df = df.asfreq('H', method='ffill')
    print("✅ Data prepared: Timestamps shifted, index set, and frequency ensured.")
    
    # 3. PyCaret Time Series Experiment
    print(f"Setting up PyCaret experiment (Target='value', Forecast Horizon={forecast_horizon})...")
    exp = TSForecastingExperiment()
    # Using verbose=False to keep the output clean
    exp.setup(data=df[['value']], target='value', fh=forecast_horizon, session_id=123, verbose=False)
    
    # 4. Create and train the specified model
    print(f"Creating model: '{model_alias}'...")
    model = exp.create_model(model_alias)
    
    # 5. Finalize the model for prediction
    print("Finalizing model...")
    final_model = exp.finalize_model(model)
    
    # 6. Save the model
    # PyCaret automatically adds the .pkl extension
    print(f"Saving model to '{pkl_output_path}.pkl'...")
    exp.save_model(final_model, pkl_output_path)
    
    print(f"--- ✅ Successfully finished processing for {csv_path} ---")


if __name__ == "__main__":
    # --- Task 1: Train Load Forecasting Model ---
    train_and_save_model(
        csv_path='load.csv',
        model_alias='lightgbm_cds_dt',
        pkl_output_path='load'
    )

    print("\n" + "="*60 + "\n")

    # --- Task 2: Train Power Generation Forecasting Model ---
    train_and_save_model(
        csv_path='power.csv',
        model_alias='lightgbm_cds_dt',
        pkl_output_path='power'
    )

    print("\nAll tasks completed.")
