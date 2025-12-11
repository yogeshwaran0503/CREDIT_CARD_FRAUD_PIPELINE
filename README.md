# Credit Card Fraud Detection — End-to-End ML Pipeline
- `data/creditcard_sample.csv` — small sample dataset included for quick testing
- `src/` — Python modules: preprocessing, training, model, prediction
- `notebooks/EDA.md` — EDA steps (markdown version)
- `config.yaml` — configuration for paths & hyperparameters
- `requirements.txt` — Python packages used
- `artifacts/` — where trained model & scalers will be saved by training script
- `README.md` — this file (how to run)

## About
This project is an end-to-end pipeline for **credit card fraud detection** (binary classification, highly imbalanced).
It demonstrates: EDA, preprocessing, handling class imbalance, training, evaluation, and a simple prediction API script.

## Important — dataset
The popular dataset used for this problem is the "Credit Card Fraud Detection" dataset available on Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud).
**I cannot download the full Kaggle dataset for you here.** To use the full dataset:
1. Download `creditcard.csv` from Kaggle above.
2. Put it into this project's `data/` folder and rename it to `creditcard.csv`.

A tiny sample `data/creditcard_sample.csv` is included so you can run and test the code quickly.

## Quickstart (local)
1. Create a Python venv and activate it.
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training (this will expect `data/creditcard.csv`; if you only have the sample, it will run on sample data):
   ```bash
   python src/train.py
   ```
4. Run prediction on a new CSV:
   ```bash
   python src/predict.py --input data/creditcard_sample.csv --output predictions.csv
   ```

## Project structure
- `src/` contains modular scripts:
  - `preprocessing.py` — feature scaling, optional dimensionality reduction
  - `train.py` — training + evaluation + model saving
  - `predict.py` — load saved model + predict on new data
  - `model.py` — helper functions for model creation
- `artifacts/` will contain saved `model.pkl` and `scaler.pkl`

## For your resume
Write a short bullet like:
> Built an end-to-end ML pipeline for credit card fraud detection: data cleaning, feature engineering, handling imbalanced classes, model evaluation, and production-ready prediction script. Implemented RandomForest with balanced weights and saved model for inference.

---
If you want, I can now **customize the pipeline** further (add hyperparameter tuning, MLflow, CI, Dockerfile, or create a GitHub-ready repo with actions).