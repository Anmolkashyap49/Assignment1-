# Car Price Prediction (UCI Automobile Dataset)

This project uses the UCI Automobile dataset to predict car prices using a Random Forest Regressor.

## Steps
1. Load dataset dynamically from UCI repository.
2. Clean missing values and convert numeric columns.
3. Preprocess features with OneHotEncoding for categoricals and StandardScaler for numerics.
4. Train a Random Forest model.
5. Evaluate using R² and MAE.
6. Plot Actual vs Predicted prices.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Run
```bash
python main.py
