import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'outputs')

preds_fp = os.path.join(OUT, 'predictions.csv')
metrics_fp = os.path.join(OUT, 'model_metrics.txt')
out_fp = os.path.join(OUT, 'metrics_comparison.csv')

def read_lgb_metrics(path):
    if not os.path.exists(path):
        return None
    txt = open(path, 'r', encoding='utf8').read()
    # look for RMSE and R2
    rmse = None
    r2 = None
    for line in txt.splitlines():
        if 'RMSE' in line:
            try:
                rmse = float(line.split(':')[1].strip())
            except:
                pass
        if line.strip().startswith('R2'):
            try:
                r2 = float(line.split(':')[1].strip())
            except:
                pass
    return {'rmse': rmse, 'r2': r2}

def main():
    rows = []
    if os.path.exists(preds_fp):
        df = pd.read_csv(preds_fp)
        # assume columns: y_true, pred_lin, pred_rf
        if 'y_true' in df.columns and 'pred_lin' in df.columns:
            y = df['y_true']
            from sklearn.metrics import mean_squared_error, r2_score
            rmse_lin = mean_squared_error(y, df['pred_lin']) ** 0.5
            r2_lin = r2_score(y, df['pred_lin'])
            rows.append({'model': 'LinearRegression', 'rmse': rmse_lin, 'r2': r2_lin})
        if 'y_true' in df.columns and 'pred_rf' in df.columns:
            y = df['y_true']
            from sklearn.metrics import mean_squared_error, r2_score
            rmse_rf = mean_squared_error(y, df['pred_rf']) ** 0.5
            r2_rf = r2_score(y, df['pred_rf'])
            rows.append({'model': 'RandomForest', 'rmse': rmse_rf, 'r2': r2_rf})
    # LightGBM metrics
    lgbm = read_lgb_metrics(metrics_fp)
    if lgbm is not None:
        rows.append({'model': 'LightGBM', 'rmse': lgbm.get('rmse'), 'r2': lgbm.get('r2')})

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(out_fp, index=False)
        print('Saved comparison to', out_fp)
        print(out.to_string(index=False))
    else:
        print('No metrics found to compare. Ensure outputs/predictions.csv and outputs/model_metrics.txt exist.')

if __name__ == '__main__':
    main()
