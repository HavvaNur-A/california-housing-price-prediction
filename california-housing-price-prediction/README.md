# 🏠 California Housing Price Prediction

A complete machine learning pipeline for predicting California housing prices using the classic sklearn dataset. The project covers the full workflow: data loading, exploratory analysis, preprocessing, model training, and hyperparameter optimization.

---

## 📊 Results

| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| **Random Forest** | **0.4962** | **0.3284** | **0.8053** |
| Optimized Random Forest | 0.5033 | 0.3402 | 0.7996 |
| Gradient Boosting | 0.5247 | 0.3615 | 0.7823 |
| Linear Regression | 0.6419 | 0.4709 | 0.6741 |
| Ridge Regression | 0.6419 | 0.4709 | 0.6741 |
| Lasso Regression | 0.7490 | 0.5678 | 0.5563 |

> **Best model: Random Forest** with R² = 0.805 (explains ~80% of price variance)

---

## 📁 Project Structure

```
california-housing-price-prediction/
│
├── src/
│   ├── 01_data_loading.py          # Load dataset & save to CSV
│   ├── 02_exploratory_analysis.py  # EDA & visualizations
│   ├── 03_preprocessing.py         # Outlier handling, feature engineering, scaling
│   ├── 04_model_training.py        # Train & compare 5 models
│   └── 05_model_optimization.py    # Hyperparameter tuning (RandomizedSearchCV)
│
├── data/
│   ├── california_housing.csv
│   ├── processed_california_housing.csv
│   └── model_comparison_results.csv
│
├── models/                         # Saved .pkl model files
├── figures/                        # Generated plots & charts
├── notebooks/                      # Kaggle-style notebook version
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Features

- **Outlier Handling** — IQR-based capping for all numeric features
- **Feature Engineering** — 3 new features derived from existing ones:
  - `RoomsPerHousehold` = AveRooms / AveOccup
  - `BedroomRatio` = AveBedrms / AveRooms
  - `PopulationDensity` = Population / AveOccup
- **Scaling** — StandardScaler applied before model training
- **5 Models Compared** — Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- **Cross-Validation** — 5-fold CV for reliable performance estimates
- **Hyperparameter Tuning** — RandomizedSearchCV on top-2 models

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/california-housing-price-prediction.git
cd california-housing-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the pipeline step by step
python src/01_data_loading.py
python src/02_exploratory_analysis.py
python src/03_preprocessing.py
python src/04_model_training.py
python src/05_model_optimization.py
```

---

## 📈 Key Visualizations

| Price Distribution | Geographic Map | Model Comparison |
|---|---|---|
| ![](figures/price_distribution.png) | ![](figures/geographic_distribution.png) | ![](figures/model_comparison.png) |

---

## 📦 Dataset

**California Housing Dataset** from `sklearn.datasets`  
- 20,640 samples, 8 features  
- Target: median house value (in units of $100,000)  
- Source: 1990 U.S. Census  

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-red)

---

## 📬 Contact

Feel free to open an issue or connect on [LinkedIn](https://www.linkedin.com/in/havvanur-ai/).
