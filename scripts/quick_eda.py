import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'Top_10000_Movies_IMDb.csv'
OUT = Path(__file__).resolve().parent.parent / 'outputs'
OUT.mkdir(exist_ok=True)

print('Loading', DATA)
df = pd.read_csv(DATA)
print('Shape:', df.shape)
print('\nColumns:')
print(df.columns.tolist())
print('\nMissing per column:')
print(df.isna().sum())

# Convert Rating to numeric (already likely numeric)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

plt.figure(figsize=(8,5))
sns.histplot(df['Rating'].dropna(), bins=30, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.xlabel('Rating')
plt.savefig(OUT / 'rating_distribution.png')
print('Saved rating histogram to', OUT / 'rating_distribution.png')
print(df['Rating'].describe())
