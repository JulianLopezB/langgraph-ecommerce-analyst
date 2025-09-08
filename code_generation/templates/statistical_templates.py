"""Statistical analysis code templates."""

DESCRIPTIVE_STATS_TEMPLATE = """
# Descriptive Statistics Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Basic descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(f"Dataset shape: {df.shape}")
print(f"Dataset info:")
print(df.info())
print(f"\\nDescriptive statistics:")
print(df.describe())

# Missing data analysis
print(f"\\n=== MISSING DATA ANALYSIS ===")
missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_pct
})
missing_df = missing_df[missing_df.Missing_Count > 0].sort_values('Missing_Count', ascending=False)
print(missing_df)

# Data types analysis
print(f"\\n=== DATA TYPES ===")
print(df.dtypes.value_counts())

# Numerical columns analysis
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Create visualizations
if numerical_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols[:6], 1):  # Limit to first 6 columns
        plt.subplot(2, 3, i)
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

analysis_results = {
    'shape': df.shape,
    'missing_data': missing_df.to_dict() if not missing_df.empty else {},
    'numerical_columns': numerical_cols,
    'categorical_columns': categorical_cols,
    'summary_stats': df.describe().to_dict()
}
"""

CORRELATION_ANALYSIS_TEMPLATE = """
# Correlation Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numerical_cols) < 2:
    print("Not enough numerical columns for correlation analysis")
    analysis_results = {'error': 'Insufficient numerical columns'}
else:
    print("=== CORRELATION ANALYSIS ===")
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find highly correlated pairs (> 0.7)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print(f"Highly correlated pairs (|r| > 0.7):")
    for pair in high_corr_pairs:
        print(f"{pair['var1']} <-> {pair['var2']}: {pair['correlation']:.3f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    
    analysis_results = {
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlations': high_corr_pairs,
        'numerical_columns': numerical_cols
    }
"""

DISTRIBUTION_ANALYSIS_TEMPLATE = """
# Distribution Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest

print("=== DISTRIBUTION ANALYSIS ===")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
distribution_results = {}

for col in numerical_cols[:5]:  # Analyze first 5 numerical columns
    print(f"\\n--- Analysis for {col} ---")
    data = df[col].dropna()
    
    if len(data) == 0:
        print(f"No valid data for {col}")
        continue
    
    # Basic statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    print(f"Mean: {mean_val:.3f}")
    print(f"Median: {median_val:.3f}")
    print(f"Std Dev: {std_val:.3f}")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    
    # Normality tests
    if len(data) >= 3:
        try:
            shapiro_stat, shapiro_p = shapiro(data.sample(min(5000, len(data))))
            print(f"Shapiro-Wilk p-value: {shapiro_p:.6f}")
            
            normal_test_stat, normal_test_p = normaltest(data)
            print(f"D'Agostino normality p-value: {normal_test_p:.6f}")
            
            is_normal = shapiro_p > 0.05 and normal_test_p > 0.05
            print(f"Appears normally distributed: {is_normal}")
            
        except Exception as e:
            print(f"Normality tests failed: {e}")
            is_normal = False
    else:
        is_normal = False
    
    # Store results
    distribution_results[col] = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_normal': is_normal,
        'count': len(data)
    }

# Create distribution plots
if numerical_cols:
    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols[:6]) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numerical_cols[:6], 1):
        plt.subplot(n_rows, n_cols, i)
        data = df[col].dropna()
        
        # Histogram with KDE
        plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
        
        # Add normal distribution overlay
        if len(data) > 0:
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, normal_curve, 'r-', linewidth=2, label='Normal')
        
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

analysis_results = {
    'distribution_analysis': distribution_results,
    'columns_analyzed': numerical_cols[:5]
}
"""
