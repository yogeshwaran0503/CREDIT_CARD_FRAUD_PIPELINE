# EDA Notes — Credit Card Fraud Detection (short)

1. Inspect dataset shape, types, missing values.
2. Check class imbalance: `Class` column is 0 (legit) or 1 (fraud).
3. Visualize distribution of `Amount`, `Time` and PCA-like features (V1..V28).
4. Consider log-transforming `Amount` and scaling features.
5. Use stratified split for train/test because of imbalance.
6. Evaluate with precision, recall, F1, ROC-AUC and PR-AUC.
7. Consider resampling strategies: SMOTE, RandomUnderSampler, or class_weight in model.
