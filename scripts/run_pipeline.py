import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'Top_10000_Movies_IMDb.csv'
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA)
print('Shape', df.shape)

# Basic cleaning and feature engineering
print('Parsing runtime...')
# runtime like '142 min'
def parse_runtime(x):
    try:
        return int(str(x).split()[0])
    except:
        return np.nan

df['runtime_min'] = df['Runtime'].apply(parse_runtime)

# Genre: take top-10 genres and multi-hot encode
print('Encoding genres...')

def split_genres(s):
    return [g.strip() for g in str(s).split(',')]

all_genres = df['Genre'].dropna().apply(split_genres)
from collections import Counter
cnt = Counter([g for sub in all_genres for g in sub])
common = [g for g,_ in cnt.most_common(12)]
for g in common:
    df[f'genre_{g}'] = df['Genre'].apply(lambda s: 1 if pd.notna(s) and g in s else 0)

# Metascore: numeric and flag
print('Processing metascore...')
df['Metascore_num'] = pd.to_numeric(df['Metascore'], errors='coerce')
df['metascore_missing'] = df['Metascore_num'].isna().astype(int)
# impute missing with median
med = df['Metascore_num'].median()
df['Metascore_num'].fillna(med, inplace=True)

# Votes, Gross -> numeric
print('Processing votes/gross...')
df['Votes_num'] = pd.to_numeric(df['Votes'], errors='coerce')
# Gross sometimes equals votes in the sample (strings). try to coerce
# Remove non-digit characters
import re

def parse_gross(x):
    try:
        s = str(x).replace(',', '')
        return float(re.sub(r"[^0-9.]", "", s)) if s!='' else np.nan
    except:
        return np.nan

df['Gross_num'] = df['Gross'].apply(parse_gross)
# If Gross is NaN but equals Votes (some noisy rows), leave as NaN

# Plot length of plot text
print('Text features...')
df['plot_len'] = df['Plot'].fillna('').apply(lambda s: len(s))

# Count stars / directors
print('Count stars/directors...')

def count_list_field(x):
    s = str(x)
    # assume lists like "['A','B']" or comma-separated
    if s.strip().startswith('['):
        return s.count("',")+1 if "'," in s else (0 if s.strip()=="[]" else 1)
    return len([p for p in s.split(',') if p.strip()!=''])

from math import isnan

df['n_stars'] = df['Stars'].apply(count_list_field)
df['n_directors'] = df['Directors'].apply(count_list_field)

# Target
Y = df['Rating'].astype(float)

# Features list
features = ['runtime_min','Metascore_num','metascore_missing','Votes_num','Gross_num','plot_len','n_stars','n_directors']
features += [f'genre_{g}' for g in common]

# Fill numeric NaNs
for f in features:
    if df[f].dtype.kind in 'biufc':
        df[f].fillna(df[f].median(), inplace=True)

X = df[features]

print('Final feature shape', X.shape)

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Baseline: Linear Regression
print('\nTraining LinearRegression...')
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_lin = lin.predict(X_test)
rmse_lin = (mean_squared_error(y_test, pred_lin)) ** 0.5
r2_lin = r2_score(y_test, pred_lin)
print('Linear RMSE:', round(rmse_lin,4), 'R2:', round(r2_lin,4))

# RandomForest
print('\nTraining RandomForestRegressor...')
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rmse_rf = (mean_squared_error(y_test, pred_rf)) ** 0.5
r2_rf = r2_score(y_test, pred_rf)
print('RF RMSE:', round(rmse_rf,4), 'R2:', round(r2_rf,4))

# Save predictions
preds = X_test.copy()
preds['y_true'] = y_test
preds['pred_lin'] = pred_lin
preds['pred_rf'] = pred_rf
preds.to_csv(OUT / 'predictions.csv', index=False)
print('Saved predictions to', OUT / 'predictions.csv')

# Feature importances
fi = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
fi.to_csv(OUT / 'feature_importances.csv', index=False)
print('Saved feature importances to', OUT / 'feature_importances.csv')

# Plot importances
plt.figure(figsize=(8,5))
sns.barplot(data=fi.head(15), x='importance', y='feature')
plt.title('Feature importances (RF)')
plt.tight_layout()
plt.savefig(OUT / 'feature_importances.png')
print('Saved feature importances plot to', OUT / 'feature_importances.png')

# Save models
joblib.dump(lin, OUT / 'lin_model.joblib')
joblib.dump(rf, OUT / 'rf_model.joblib')
print('Saved models')

print('\nDone')
