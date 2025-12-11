import argparse, yaml, os, joblib, pandas as pd
from preprocessing import basic_preprocess

def predict(input_path, output_path='predictions.csv', config_path='config.yaml'):
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)
    model = joblib.load(cfg['artifacts']['model_path'])
    X, _ = basic_preprocess(pd.read_csv(input_path), fit_scaler=False, scaler_path=cfg['artifacts']['scaler_path'])
    probs = model.predict_proba(X)[:,1]
    df_out = pd.read_csv(input_path).copy()
    df_out['fraud_prob'] = probs
    df_out.to_csv(output_path, index=False)
    print('Saved predictions to', output_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python predict.py --input <input_csv> [--output out.csv]')
    else:
        # basic CLI
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True)
        parser.add_argument('--output', default='predictions.csv')
        args = parser.parse_args()
        predict(args.input, args.output)
