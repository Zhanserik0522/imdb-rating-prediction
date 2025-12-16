import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'Top_10000_Movies_IMDb.csv'
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)

print('Load data')
df = pd.read_csv(DATA)

# basic cleaning reused
print('Parse runtime')
df['runtime_min'] = df['Runtime'].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) else np.nan)

print('Metascore')
df['Metascore_num'] = pd.to_numeric(df['Metascore'], errors='coerce')
df['metascore_missing'] = df['Metascore_num'].isna().astype(int)
df['Metascore_num'].fillna(df['Metascore_num'].median(), inplace=True)

print('Votes/Gross')
df['Votes_num'] = pd.to_numeric(df['Votes'], errors='coerce')
import re

def parse_gross(x):
    try:
        s = str(x).replace(',', '')
        s = re.sub(r'[^0-9.]', '', s)
        return float(s) if s!='' else np.nan
    except:
        return np.nan

df['Gross_num'] = df['Gross'].apply(parse_gross)

print('Plot length')
df['plot_len'] = df['Plot'].fillna('').apply(len)

# top actors/directors
print('Top actors/directors')
from collections import Counter

all_stars = df['Stars'].fillna('').apply(lambda s: [x.strip() for x in s.strip("[]").split(',') if x.strip()!=''])
flat_stars = [x for sub in all_stars for x in sub]
star_cnt = Counter(flat_stars)
top_stars = [s for s,_ in star_cnt.most_common(50)]
for s in top_stars:
    df[f'star_{s}'] = df['Stars'].apply(lambda x: 1 if pd.notna(x) and s in x else 0)

all_dirs = df['Directors'].fillna('').apply(lambda s: [x.strip() for x in s.strip("[]").split(',') if x.strip()!=''])
flat_dirs = [x for sub in all_dirs for x in sub]
dir_cnt = Counter(flat_dirs)
top_dirs = [d for d,_ in dir_cnt.most_common(20)]
for d in top_dirs:
    df[f'dir_{d}'] = df['Directors'].apply(lambda x: 1 if pd.notna(x) and d in x else 0)

# TF-IDF on Plot -> SVD
print('TF-IDF')
corpus = df['Plot'].fillna('').tolist()
vec = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = vec.fit_transform(corpus)
print('SVD to 50 comps')
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
svd_cols = [f'tfidf_svd_{i}' for i in range(X_svd.shape[1])]
X_svd_df = pd.DataFrame(X_svd, columns=svd_cols)

# numeric features
num_feats = ['runtime_min','Metascore_num','metascore_missing','Votes_num','Gross_num','plot_len']
# fillna numeric
for f in num_feats:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(df[f].median())

# assemble final X
feat_df = pd.concat([df[num_feats].reset_index(drop=True), X_svd_df.reset_index(drop=True)], axis=1)
# add top actors/dirs
for s in top_stars:
    feat_df[f'star_{s}'] = df[f'star_{s}'].values
for d in top_dirs:
    feat_df[f'dir_{d}'] = df[f'dir_{d}'].values

print('Final shape', feat_df.shape)
Y = df['Rating'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(feat_df, Y, test_size=0.2, random_state=42)

# LightGBM with small randomized search
print('Train LightGBM with RandSearch')
train_data = lgb.Dataset(X_train, label=y_train)

param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'max_depth': [-1, 6, 10]
}

lgb_est = lgb.LGBMRegressor(random_state=42)
rs = RandomizedSearchCV(lgb_est, param_grid, n_iter=8, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print('Best params', rs.best_params_)
best = rs.best_estimator_

pred = best.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
print('LightGBM RMSE:', rmse, 'R2:', r2)

# Save model
joblib.dump(best, OUT / 'lgb_model.joblib')

# Save metrics to file
with open(OUT / 'model_metrics.txt','w',encoding='utf8') as f:
    f.write(f'LightGBM RMSE: {rmse}\nR2: {r2}\nBest params: {rs.best_params_}\n')

# SHAP analysis (explain on a sample for speed)
print('Compute SHAP')
explainer = shap.TreeExplainer(best)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(OUT / 'shap_summary.png')
print('Saved SHAP plot')

# Save feature importances from lgb
fi = pd.DataFrame({'feature': X_test.columns, 'importance': best.feature_importances_}).sort_values('importance', ascending=False)
fi.to_csv(OUT / 'lgb_feature_importances.csv', index=False)
print('Saved lgb feature importances')

print('Done')
