# src/05_model_optimization.py
# California Housing Price Prediction
# Step 5: Hyperparameter tuning with RandomizedSearchCV

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def optimize_model(estimator, param_grid: dict, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Runs RandomizedSearchCV and evaluates the best estimator."""
    print(f"\n=== Optimizing: {model_name} ===")
    start = time.time()

    search = RandomizedSearchCV(
        estimator, param_grid,
        n_iter=20, cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best = search.best_estimator_
    print(f"Best params : {search.best_params_}")
    print(f"Search time : {elapsed:.2f}s")

    y_pred_test  = best.predict(X_test)
    y_pred_train = best.predict(X_train)

    cv_idx   = search.best_index_
    cv_rmse  = np.sqrt(-search.cv_results_["mean_test_score"][cv_idx])

    results = {
        "model_name":    f"Optimized {model_name}",
        "model":         best,
        "best_params":   search.best_params_,
        "training_time": elapsed,
        "rmse_train":    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test":     np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        "mae_test":      mean_absolute_error(y_test, y_pred_test),
        "r2_test":       r2_score(y_test, y_pred_test),
        "cv_rmse":       cv_rmse,
        "y_pred_test":   y_pred_test,
    }

    print(f"  Train RMSE : {results['rmse_train']:.4f}")
    print(f"  Test  RMSE : {results['rmse_test']:.4f}")
    print(f"  Test  MAE  : {results['mae_test']:.4f}")
    print(f"  Test  R²   : {results['r2_test']:.4f}")
    print(f"  CV    RMSE : {results['cv_rmse']:.4f}")

    return results


def plot_predictions(y_test, y_pred, model_name: str, figures_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} — Actual vs Predicted")
    plt.annotate(f"R² = {r2_score(y_test, y_pred):.4f}", xy=(0.05, 0.95), xycoords="axes fraction")
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(figures_dir, f"{safe}_predictions.png"))
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
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(figures_dir, f"{safe}_feature_importance.png"))
    plt.close()


def main():
    print("Starting model optimization...")

    root_dir    = os.path.dirname(os.path.dirname(__file__))
    data_dir    = os.path.join(root_dir, "data")
    models_dir  = os.path.join(root_dir, "models")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    processed = joblib.load(os.path.join(data_dir, "processed_data.pkl"))
    X_train, X_test = processed["X_train"], processed["X_test"]
    y_train, y_test = processed["y_train"], processed["y_test"]
    feature_names   = processed["feature_names"]

    # Load previous results if available
    results_path = os.path.join(data_dir, "model_comparison_results.csv")
    if os.path.exists(results_path):
        prev_df = pd.read_csv(results_path)
        print("Previous results loaded.")
        print(prev_df.to_string(index=False))
    else:
        prev_df = pd.DataFrame()

    # Hyperparameter grids
    rf_params = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":    ["sqrt", "log2"],
    }
    gb_params = {
        "n_estimators":    [100, 200, 300],
        "learning_rate":   [0.01, 0.05, 0.1],
        "max_depth":       [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "subsample":       [0.8, 0.9, 1.0],
    }

    to_optimize = [
        ("Random Forest",     RandomForestRegressor(random_state=42),     rf_params),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42), gb_params),
    ]

    optimized = []
    for name, estimator, params in to_optimize:
        res = optimize_model(estimator, params, X_train, X_test, y_train, y_test, name)
        optimized.append(res)

        safe = f"optimized_{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(res["model"], os.path.join(models_dir, safe))
        print(f"Saved: models/{safe}")

        plot_predictions(y_test, res["y_pred_test"], res["model_name"], figures_dir)
        plot_feature_importance(res["model"], feature_names, res["model_name"], figures_dir)

    # Merge with previous results
    new_df = pd.DataFrame({
        "Model":            [r["model_name"]    for r in optimized],
        "Training Time(s)": [r["training_time"] for r in optimized],
        "Train RMSE":       [r["rmse_train"]    for r in optimized],
        "Test RMSE":        [r["rmse_test"]     for r in optimized],
        "Test MAE":         [r["mae_test"]      for r in optimized],
        "Test R²":          [r["r2_test"]       for r in optimized],
        "CV RMSE":          [r["cv_rmse"]       for r in optimized],
    })

    combined = pd.concat([prev_df, new_df]).sort_values("Test RMSE") if not prev_df.empty else new_df
    print(f"\n=== All Models (Final Comparison) ===\n{combined.to_string(index=False)}")
    print(f"\nBest model overall: {combined.iloc[0]['Model']}")

    # Final comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, ["Test RMSE", "Test R²", "Test MAE", "Training Time(s)"]):
        ax.bar(combined["Model"], combined[metric])
        ax.set_title(metric)
        ax.set_xticklabels(combined["Model"], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "all_models_comparison.png"))
    plt.close()
    print("Saved: figures/all_models_comparison.png")

    combined.to_csv(results_path, index=False)
    print(f"Updated results saved: {results_path}")

    print("\nModel optimization complete.")


if __name__ == "__main__":
    main()
