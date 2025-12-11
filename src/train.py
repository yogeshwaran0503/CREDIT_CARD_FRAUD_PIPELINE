import argparse
import yaml
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from preprocessing import load_data, basic_preprocess
from model import build_model

def main(config_path='config.yaml'):
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)
    data_path = cfg['data'].get('raw_path')
    sample_path = cfg['data'].get('sample_path')
    path_to_use = data_path if os.path.exists(data_path) else sample_path
    print(f"Using dataset: {path_to_use}")
    df = load_data(path_to_use)
    X, y = basic_preprocess(df, fit_scaler=True)
    if y is None:
        raise ValueError('Target column `Class` not found in dataset.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['training']['test_size'], random_state=cfg['training']['random_state'], stratify=y)
    model = build_model(n_estimators=cfg['training']['n_estimators'], max_depth=cfg['training']['max_depth'], random_state=cfg['training']['random_state'])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print('Classification report:\n', classification_report(y_test, preds))
    print('ROC AUC:', roc_auc_score(y_test, probs))
    # Save model
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, cfg['artifacts']['model_path'])
    print('Model saved to', cfg['artifacts']['model_path'])

if __name__ == '__main__':
    main()
