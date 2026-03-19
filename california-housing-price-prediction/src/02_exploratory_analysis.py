# src/02_exploratory_analysis.py
# California Housing Price Prediction
# Step 2: Exploratory Data Analysis (EDA) and visualizations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> pd.DataFrame:
    print("Starting Exploratory Data Analysis...")

    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(data_dir, "california_housing.csv"))

    # --- Basic Info ---
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")

    # --- Price Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Price"], kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Price (100k USD)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(figures_dir, "price_distribution.png"))
    plt.close()
    print("Saved: figures/price_distribution.png")

    # --- Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(figures_dir, "correlation_matrix.png"))
    plt.close()
    print("Saved: figures/correlation_matrix.png")

    # --- Top 3 Features vs Price ---
    top_corr = df.corr()["Price"].abs().sort_values(ascending=False)[1:4]
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(top_corr.index):
        plt.subplot(1, 3, i + 1)
        plt.scatter(df[col], df["Price"], alpha=0.4)
        plt.title(f"Price vs {col}")
        plt.xlabel(col)
        plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "price_correlations.png"))
    plt.close()
    print("Saved: figures/price_correlations.png")

    # --- Geographic Distribution ---
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        df["Longitude"], df["Latitude"],
        alpha=0.4,
        s=df["Population"] / 50,
        c=df["Price"],
        cmap="viridis"
    )
    plt.colorbar(scatter, label="Price (100k USD)")
    plt.title("California House Prices - Geographic Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(os.path.join(figures_dir, "geographic_distribution.png"))
    plt.close()
    print("Saved: figures/geographic_distribution.png")

    # --- Boxplots ---
    plt.figure(figsize=(15, 10))
    df.boxplot()
    plt.title("Feature Boxplots")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "boxplots.png"))
    plt.close()
    print("Saved: figures/boxplots.png")

    print("\nExploratory Data Analysis complete.")
    return df


if __name__ == "__main__":
    main()
