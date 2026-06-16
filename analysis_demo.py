import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("STEP 1: Load Dataset")
print("="*60)
df = pd.read_csv('data/sample_sales.csv')
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "="*60)
print("STEP 2: Data Cleaning")
print("="*60)
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
cat_cols = df_clean.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_val = df_clean[col].mode()
    if not mode_val.empty:
        df_clean[col] = df_clean[col].fillna(mode_val[0])
print(f"After cleaning: {df_clean.shape[0]} rows, missing values: {df_clean.isnull().sum().sum()}")

print("\n" + "="*60)
print("STEP 3: Summary Statistics")
print("="*60)
print(df_clean.describe(include='all'))

print("\n" + "="*60)
print("STEP 4: Generating Charts (saved as PNG files)")
print("="*60)
plt.figure(figsize=(8,4))
sns.histplot(df_clean['Sales'], kde=True, bins=5)
plt.title('Distribution of Sales')
plt.savefig('sales_distribution.png')
plt.close()
print("Saved: sales_distribution.png")

prod_sales = df_clean.groupby('Product')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
prod_sales.plot(kind='bar', color='teal')
plt.title('Total Sales by Product')
plt.ylabel('Sales')
plt.savefig('sales_by_product.png')
plt.close()
print("Saved: sales_by_product.png")

numeric_df = df_clean.select_dtypes(include=[np.number])
if len(numeric_df.columns) >= 2:
    plt.figure(figsize=(6,4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Saved: correlation_heatmap.png")

print("\n" + "="*60)
print("STEP 5: Automatic Insights")
print("="*60)
print(f"- Total missing values originally: {df.isnull().sum().sum()}")
print(f"- Duplicate rows: {df.duplicated().sum()}")
print(f"- Highest selling product: {prod_sales.idxmax()} with ${prod_sales.max():.0f}")
if len(numeric_df.columns) >= 2:
    corr_value = df_clean['Sales'].corr(df_clean['Quantity'])
    print(f"- Correlation between Sales and Quantity: {corr_value:.2f}")

print("\n✅ Analysis complete. Check the generated PNG images.")