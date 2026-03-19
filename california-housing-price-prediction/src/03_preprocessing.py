# src/03_preprocessing.py
# California Housing Price Prediction
# Step 3: Data Preprocessing — outlier handling, feature engineering, scaling, train/test split

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Caps outliers using IQR method (1.5 * IQR)."""
    print("\n=== Outlier Detection & Capping ===")
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"  {col}: {n_outliers} outliers ({n_outliers / len(df) * 100:.2f}%)")
        df[col] = np.clip(df[col], lower, upper)
    print("Outliers capped.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones."""
    print("\n=== Feature Engineering ===")
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"]      = df["AveBedrms"] / df["AveRooms"]
    df["PopulationDensity"] = df["Population"] / df["AveOccup"]
    print("New features added: RoomsPerHousehold, BedroomRatio, PopulationDensity")
    return df


def main() -> dict:
    print("Starting data preprocessing...")

    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir   = os.path.join(root_dir, "data")
    models_dir = os.path.join(root_dir, "models")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, "california_housing.csv"))

    # 1. Outliers
    df = cap_outliers(df)

    # 2. Missing values
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() > 0:
        imputer = SimpleImputer(strategy="median")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        joblib.dump(imputer, os.path.join(models_dir, "imputer.pkl"))
        print("Missing values filled with median. Imputer saved.")
    else:
        print("No missing values found.")

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Split features / target
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # 5. Scaling
    print("\n=== Scaling ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("Features scaled. Scaler saved.")

    # 6. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Save processed data
    processed_data = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": X.columns.tolist()
    }
    joblib.dump(processed_data, os.path.join(data_dir, "processed_data.pkl"))
    print("Processed data saved: data/processed_data.pkl")

    # Save processed CSV
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df["Price"] = y.values
    scaled_df.to_csv(os.path.join(data_dir, "processed_california_housing.csv"), index=False)
    print("Processed CSV saved: data/processed_california_housing.csv")

    # Correlation heatmap (post-processing)
    plt.figure(figsize=(14, 12))
    sns.heatmap(scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix (After Preprocessing)")
    plt.savefig(os.path.join(figures_dir, "correlation_matrix_after_preprocessing.png"))
    plt.close()
    print("Saved: figures/correlation_matrix_after_preprocessing.png")

    print("\nPreprocessing complete.")
    return processed_data


if __name__ == "__main__":
    main()
