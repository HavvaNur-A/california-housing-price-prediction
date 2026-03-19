# src/04_model_training.py
# California Housing Price Prediction
# Step 4: Train and compare multiple regression models

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Trains a model and returns performance metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        "model_name":    model_name,
        "model":         model,
        "training_time": training_time,
        "rmse_train":    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test":     np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae_test":      mean_absolute_error(y_test, y_pred_test),
        "r2_train":      r2_score(y_train, y_pred_train),
        "r2_test":       r2_score(y_test, y_pred_test),
        "y_pred_test":   y_pred_test,
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    metrics["cv_rmse"] = np.sqrt(-cv_scores.mean())

    print(f"\n=== {model_name} ===")
    print(f"  Training time : {training_time:.4f}s")
    print(f"  Train RMSE    : {metrics['rmse_train']:.4f}")
    print(f"  Test  RMSE    : {metrics['rmse_test']:.4f}")
    print(f"  Test  MAE     : {metrics['mae_test']:.4f}")
    print(f"  Test  R²      : {metrics['r2_test']:.4f}")
    print(f"  CV    RMSE    : {metrics['cv_rmse']:.4f}")

    return metrics


def plot_predictions(y_test, y_pred, model_name: str, figures_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} — Actual vs Predicted")
    plt.annotate(f"R² = {r2_score(y_test, y_pred):.4f}", xy=(0.05, 0.95), xycoords="axes fraction")
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(figures_dir, f"{safe_name}_predictions.png"))
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str, figures_dir: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    indices = np.argsort(model.feature_importances_)[::-1]
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_names)), model.feature_importances_[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"{model_name} — Feature Importances")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(figures_dir, f"{safe_name}_feature_importance.png"))
    plt.close()


def main():
    print("Starting model training...")

    root_dir    = os.path.dirname(os.path.dirname(__file__))
    data_dir    = os.path.join(root_dir, "data")
    models_dir  = os.path.join(root_dir, "models")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load preprocessed data
    processed = joblib.load(os.path.join(data_dir, "processed_data.pkl"))
    X_train, X_test = processed["X_train"], processed["X_test"]
    y_train, y_test = processed["y_train"], processed["y_test"]
    feature_names   = processed["feature_names"]
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Define models
    models = {
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(alpha=1.0),
        "Lasso Regression":     Lasso(alpha=0.1),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    all_results = []

    for name, model in models.items():
        results = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        all_results.append(results)

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(models_dir, f"{safe_name}_model.pkl"))

        # Plots
        plot_predictions(y_test, results["y_pred_test"], name, figures_dir)
        plot_feature_importance(model, feature_names, name, figures_dir)

    # Comparison table
    results_df = pd.DataFrame({
        "Model":           [r["model_name"]    for r in all_results],
        "Training Time(s)":[r["training_time"] for r in all_results],
        "Train RMSE":      [r["rmse_train"]    for r in all_results],
        "Test RMSE":       [r["rmse_test"]     for r in all_results],
        "Test MAE":        [r["mae_test"]      for r in all_results],
        "Test R²":         [r["r2_test"]       for r in all_results],
        "CV RMSE":         [r["cv_rmse"]       for r in all_results],
    }).sort_values("Test RMSE")

    print(f"\n=== Model Comparison ===\n{results_df.to_string(index=False)}")
    best = results_df.iloc[0]["Model"]
    print(f"\nBest model: {best}")

    # Comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["Test RMSE", "Test R²", "Test MAE", "Training Time(s)"]
    for ax, metric in zip(axes.flat, metrics):
        ax.bar(results_df["Model"], results_df[metric])
        ax.set_title(metric)
        ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"))
    plt.close()
    print("Saved: figures/model_comparison.png")

    # Save results CSV
    csv_path = os.path.join(data_dir, "model_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    print("\nModel training complete.")
    return all_results, results_df


if __name__ == "__main__":
    main()
