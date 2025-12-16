import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'Top_10000_Movies_IMDb.csv')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

def parse_numeric_columns(df):
    df['Votes_num'] = pd.to_numeric(df['Votes'].astype(str).str.replace(r"[^0-9]","",regex=True), errors='coerce')
    df['Gross_num'] = pd.to_numeric(df['Gross'].astype(str).str.replace(r"[^0-9.]","",regex=True), errors='coerce')
    df['Metascore_num'] = pd.to_numeric(df['Metascore'], errors='coerce')
    df['runtime_min'] = df['Runtime'].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) else np.nan)
    df['plot_len'] = df['Plot'].fillna('').apply(len)
    df['Votes_log'] = np.log1p(df['Votes_num'])
    df['Gross_log'] = np.log1p(df['Gross_num'])
    return df

def votes_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Votes_log'].dropna(), bins=60)
    plt.title('Distribution of log1p(Votes)')
    plt.xlabel('log1p(Votes)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'votes_distribution.png'), dpi=150)
    plt.close()

def rating_vs_votes(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Votes_num', y='Rating', data=df.sample(frac=0.2, random_state=1))
    sns.regplot(x='Votes_num', y='Rating', data=df.sample(frac=0.2, random_state=2), scatter=False, color='red')
    plt.xscale('log')
    plt.xlabel('Votes (log scale)')
    plt.title('Rating vs Votes')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'rating_vs_votes.png'), dpi=150)
    plt.close()

def rating_vs_metascore(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Metascore_num', y='Rating', data=df)
    sns.regplot(x='Metascore_num', y='Rating', data=df, scatter=False, color='red')
    plt.xlabel('Metascore')
    plt.title('Rating vs Metascore')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'rating_vs_metascore.png'), dpi=150)
    plt.close()

def pairplot_subset(df):
    cols = ['Rating','Metascore_num','runtime_min','Votes_log']
    small = df[cols].dropna().sample(n=min(1000, len(df)), random_state=1)
    g = sns.pairplot(small)
    g.fig.suptitle('Pairplot subset', y=1.02)
    g.savefig(os.path.join(OUT, 'pairplot_subset.png'), dpi=150)
    plt.close()

def correlation_heatmap(df):
    cols = ['Rating','Votes_num','Gross_num','Metascore_num','runtime_min','plot_len']
    corr = df[cols].corr()
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlation heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'correlation_heatmap.png'), dpi=150)
    plt.close()

def main():
    df = pd.read_csv(DATA)
    df = parse_numeric_columns(df)
    votes_distribution(df)
    rating_vs_votes(df)
    rating_vs_metascore(df)
    pairplot_subset(df)
    correlation_heatmap(df)
    print('Saved EDA plots to', OUT)

if __name__ == '__main__':
    main()
