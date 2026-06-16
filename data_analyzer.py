#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df, remove_duplicates=True, fill_missing=True):
    df_clean = df.copy()
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
    return df_clean

def analyze_data(df):
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe(include=[np.number]).to_dict(),
        'categorical_summary': {}
    }
    for col in df.select_dtypes(include=['object']).columns:
        stats['categorical_summary'][col] = df[col].value_counts().head(5).to_dict()
    return stats

def generate_insights(df):
    print("\n" + "="*60)
    print("AUTOMATIC INSIGHTS")
    print("="*60)
    missing = df.isnull().sum()
    high_missing = missing[missing > 0.2*len(df)]
    if not high_missing.empty:
        print(f"⚠️ High missing values in: {', '.join(high_missing.index)}")
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"⚠️ Found {dup} duplicate rows ({dup/len(df)*100:.1f}%)")
    for col in df.select_dtypes(include=[np.number]).columns:
        skew = df[col].skew()
        if abs(skew) > 1:
            print(f"📊 '{col}' is {'right' if skew>0 else 'left'}-skewed (skew={skew:.2f})")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        high_corr = (corr.abs() > 0.7) & (corr.abs() < 1.0)
        if high_corr.any().any():
            print("🔗 Strong correlations found:")
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i,j]) > 0.7:
                        print(f"   {corr.columns[i]} ↔ {corr.columns[j]}: {corr.iloc[i,j]:.2f}")

def plot_histograms(df, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{output_dir}/hist_{col}.png', dpi=100)
        plt.close()

def plot_correlation_heatmap(df, output_dir='plots'):
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        plt.figure(figsize=(10,8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=100)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze CSV data')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('--clean', action='store_true', help='Perform cleaning')
    parser.add_argument('--output', help='Output JSON file for statistics')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    df = load_data(args.input)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    if args.clean:
        df_clean = clean_data(df)
        print(f"After cleaning: {df_clean.shape[0]} rows")
    else:
        df_clean = df
    stats = analyze_data(df_clean)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {args.output}")
    generate_insights(df_clean)
    if args.plots:
        plot_histograms(df_clean)
        plot_correlation_heatmap(df_clean)
        print("Plots saved to ./plots/")
    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    main()