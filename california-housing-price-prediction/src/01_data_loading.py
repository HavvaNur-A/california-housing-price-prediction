# src/01_data_loading.py
# California Housing Price Prediction
# Step 1: Load dataset from sklearn and save to CSV

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def create_directory(path: str) -> None:
    """Creates directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def main() -> pd.DataFrame:
    print("Loading California Housing dataset...")

    # Set up data directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    create_directory(data_dir)

    # Load dataset from sklearn
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target

    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    # Save to CSV
    csv_path = os.path.join(data_dir, "california_housing.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to: {csv_path}")

    return df


if __name__ == "__main__":
    main()
