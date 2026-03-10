import pandas as pd
from pycaret.time_series import load_model, predict_model
import os

def load_and_predict(model_path, model_description):
    """
    Loads a saved PyCaret model and prints its future predictions.

    Args:
        model_path (str): The file path of the .pkl model (without extension).
        model_description (str): A description of the model for printing.
    """
    print(f"--- Loading and Predicting for: {model_description} ---")

    full_model_path = f"{model_path}.pkl"

    if not os.path.exists(full_model_path):
        print(f"❌ ERROR: Model file not found at '{full_model_path}'. Skipping.")
        print("Hint: Please make sure you have run 'pycart_model.py' first to generate the model files.")
        return

    # Load the saved model (PyCaret's load_model doesn't need the extension)
    print(f"Loading model from '{full_model_path}'...")
    saved_model = load_model(model_path, verbose=False)

    # Generate future predictions (default horizon is what it was trained with, i.e., 24)
    print("Generating future predictions...")
    future_predictions = predict_model(saved_model)

    print(f"\n--- ✅ Future 24-Hour Forecast for {model_description} ---")
    print(future_predictions)
    print("-" * 60)

if __name__ == "__main__":
    # --- Task 1: View Load Forecast ---
    load_and_predict(
        model_path='load',
        model_description="Data Center Load"
    )

    print("\n" + "="*60 + "\n")

    # --- Task 2: View Power Generation Forecast ---
    load_and_predict(
        model_path='power',
        model_description="Wind/Solar Power Generation"
    )

    print("\nAll prediction views completed.")
