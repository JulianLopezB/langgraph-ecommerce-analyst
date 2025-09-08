"""Machine Learning analysis code templates."""

CUSTOMER_SEGMENTATION_TEMPLATE = """
# Customer Segmentation using RFM Analysis and K-Means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta

print("=== CUSTOMER SEGMENTATION ANALYSIS ===")

# Prepare data for RFM analysis
# Assuming df has columns: user_id, order_date, sale_price
if 'user_id' not in df.columns:
    print("Error: user_id column not found")
    analysis_results = {'error': 'Missing user_id column'}
else:
    # Convert order_date to datetime if not already
    if 'created_at' in df.columns:
        df['order_date'] = pd.to_datetime(df['created_at'])
    elif 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    else:
        print("Error: No date column found")
        analysis_results = {'error': 'Missing date column'}
    
    # Calculate RFM metrics
    reference_date = df['order_date'].max() + timedelta(days=1)
    
    rfm_data = df.groupby('user_id').agg({
        'order_date': lambda x: (reference_date - x.max()).days,  # Recency
        'user_id': 'count',  # Frequency (number of orders)
        'sale_price': 'sum'  # Monetary value
    }).reset_index()
    
    rfm_data.columns = ['user_id', 'recency', 'frequency', 'monetary']
    
    print(f"RFM Data Summary:")
    print(rfm_data.describe())
    
    # Handle missing values and outliers
    rfm_data = rfm_data.dropna()
    
    # Remove extreme outliers (top 1% for monetary and frequency)
    monetary_q99 = rfm_data['monetary'].quantile(0.99)
    frequency_q99 = rfm_data['frequency'].quantile(0.99)
    
    rfm_clean = rfm_data[
        (rfm_data['monetary'] <= monetary_q99) & 
        (rfm_data['frequency'] <= frequency_q99)
    ].copy()
    
    print(f"Data after outlier removal: {len(rfm_clean)} customers")
    
    # Standardize the features for clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clean[['recency', 'frequency', 'monetary']])
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
    
    # Find optimal K (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_clean['cluster'] = kmeans_final.fit_predict(rfm_scaled)
    
    # Calculate cluster statistics
    cluster_summary = rfm_clean.groupby('cluster').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'monetary': ['mean', 'median', 'sum'],
        'user_id': 'count'
    }).round(2)
    
    print(f"\\nCluster Summary:")
    print(cluster_summary)
    
    # Calculate Customer Lifetime Value (CLV) by cluster
    rfm_clean['clv'] = rfm_clean['monetary'] * (rfm_clean['frequency'] / rfm_clean['recency']) * 365
    rfm_clean['clv'] = rfm_clean['clv'].replace([np.inf, -np.inf], 0)
    
    clv_by_cluster = rfm_clean.groupby('cluster')['clv'].agg(['mean', 'sum', 'count']).round(2)
    print(f"\\nCLV by Cluster:")
    print(clv_by_cluster)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RFM scatter plot
    scatter = axes[0, 0].scatter(rfm_clean['frequency'], rfm_clean['monetary'], 
                               c=rfm_clean['cluster'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].set_ylabel('Monetary')
    axes[0, 0].set_title('Customer Segments (Frequency vs Monetary)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Cluster size pie chart
    cluster_counts = rfm_clean['cluster'].value_counts().sort_index()
    axes[0, 1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                   autopct='%1.1f%%')
    axes[0, 1].set_title('Customer Distribution by Cluster')
    
    # Recency vs Frequency
    scatter2 = axes[1, 0].scatter(rfm_clean['recency'], rfm_clean['frequency'], 
                                c=rfm_clean['cluster'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Recency (days)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Recency vs Frequency by Cluster')
    plt.colorbar(scatter2, ax=axes[1, 0])
    
    # CLV distribution by cluster
    rfm_clean.boxplot(column='clv', by='cluster', ax=axes[1, 1])
    axes[1, 1].set_title('CLV Distribution by Cluster')
    axes[1, 1].set_xlabel('Cluster')
    
    plt.tight_layout()
    plt.show()
    
    # Assign cluster labels based on characteristics
    cluster_labels = {}
    for cluster_id in rfm_clean['cluster'].unique():
        cluster_data = rfm_clean[rfm_clean['cluster'] == cluster_id]
        avg_recency = cluster_data['recency'].mean()
        avg_frequency = cluster_data['frequency'].mean()
        avg_monetary = cluster_data['monetary'].mean()
        
        if avg_frequency > rfm_clean['frequency'].mean() and avg_monetary > rfm_clean['monetary'].mean():
            cluster_labels[cluster_id] = "Champions"
        elif avg_frequency > rfm_clean['frequency'].mean():
            cluster_labels[cluster_id] = "Loyal Customers"
        elif avg_monetary > rfm_clean['monetary'].mean():
            cluster_labels[cluster_id] = "Big Spenders"
        elif avg_recency < rfm_clean['recency'].mean():
            cluster_labels[cluster_id] = "New Customers"
        else:
            cluster_labels[cluster_id] = "At Risk"
    
    print(f"\\nCluster Labels:")
    for cluster_id, label in cluster_labels.items():
        count = len(rfm_clean[rfm_clean['cluster'] == cluster_id])
        print(f"Cluster {cluster_id}: {label} ({count} customers)")
    
    analysis_results = {
        'optimal_clusters': int(optimal_k),
        'cluster_summary': cluster_summary.to_dict(),
        'cluster_labels': cluster_labels,
        'clv_by_cluster': clv_by_cluster.to_dict(),
        'total_customers': len(rfm_clean),
        'silhouette_score': silhouette_scores[optimal_k - 2]
    }
"""

FORECASTING_TEMPLATE = """
# Sales Forecasting using Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import warnings
warnings.filterwarnings('ignore')

print("=== SALES FORECASTING ANALYSIS ===")

# Prepare time series data
# Assuming df has date and sales/revenue columns
date_col = None
value_col = None

# Try to identify date column
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower():
        date_col = col
        break

# Try to identify value column
for col in df.columns:
    if any(keyword in col.lower() for keyword in ['price', 'revenue', 'sales', 'amount']):
        value_col = col
        break

if not date_col or not value_col:
    print("Error: Could not identify date and value columns")
    analysis_results = {'error': 'Missing required columns for forecasting'}
else:
    # Prepare data for Prophet
    df_prophet = df[[date_col, value_col]].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    # Aggregate by date (sum daily values)
    df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()
    df_prophet = df_prophet.sort_values('ds')
    
    print(f"Time series data: {len(df_prophet)} data points")
    print(f"Date range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()}")
    print(f"Average daily value: {df_prophet['y'].mean():.2f}")
    
    # Remove outliers (values beyond 3 standard deviations)
    mean_val = df_prophet['y'].mean()
    std_val = df_prophet['y'].std()
    df_prophet = df_prophet[
        (df_prophet['y'] >= mean_val - 3*std_val) & 
        (df_prophet['y'] <= mean_val + 3*std_val)
    ]
    
    # Initialize and fit Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    # Create future dataframe for forecasting
    future_periods = 90  # Forecast 90 days ahead
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    
    # Extract forecast results
    forecast_period = forecast.tail(future_periods)
    historical_period = forecast.head(len(df_prophet))
    
    print(f"\\nForecast Summary (next {future_periods} days):")
    print(f"Predicted total: {forecast_period['yhat'].sum():.2f}")
    print(f"Average daily: {forecast_period['yhat'].mean():.2f}")
    print(f"Growth trend: {(forecast_period['yhat'].iloc[-1] / forecast_period['yhat'].iloc[0] - 1) * 100:.1f}%")
    
    # Calculate accuracy metrics on historical data
    if len(df_prophet) > 30:  # Only if we have enough data
        try:
            # Cross validation
            df_cv = cross_validation(model, initial='60 days', period='30 days', horizon='30 days')
            df_performance = performance_metrics(df_cv)
            
            print(f"\\nModel Performance Metrics:")
            print(f"MAPE: {df_performance['mape'].mean():.3f}")
            print(f"MAE: {df_performance['mae'].mean():.2f}")
            print(f"RMSE: {df_performance['rmse'].mean():.2f}")
            
        except Exception as e:
            print(f"Could not calculate cross-validation metrics: {e}")
            df_performance = pd.DataFrame()
    else:
        df_performance = pd.DataFrame()
    
    # Detect anomalies in historical data
    historical_residuals = df_prophet['y'] - historical_period['yhat'][:len(df_prophet)]
    residual_std = historical_residuals.std()
    anomaly_threshold = 2 * residual_std
    
    anomalies = df_prophet[abs(historical_residuals) > anomaly_threshold].copy()
    anomalies['residual'] = historical_residuals[abs(historical_residuals) > anomaly_threshold]
    
    print(f"\\nAnomaly Detection:")
    print(f"Found {len(anomalies)} anomalous data points")
    if len(anomalies) > 0:
        print("Top 5 anomalies:")
        top_anomalies = anomalies.nlargest(5, 'residual')[['ds', 'y', 'residual']]
        print(top_anomalies.to_string(index=False))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main forecast plot
    axes[0, 0].plot(df_prophet['ds'], df_prophet['y'], 'ko', markersize=2, label='Actual')
    axes[0, 0].plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
    axes[0, 0].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                           alpha=0.3, color='blue')
    axes[0, 0].axvline(x=df_prophet['ds'].max(), color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Sales Forecast')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].legend()
    
    # Trend and seasonality
    axes[0, 1].plot(forecast['ds'], forecast['trend'], label='Trend')
    if 'weekly' in forecast.columns:
        axes[0, 1].plot(forecast['ds'], forecast['weekly'], label='Weekly Seasonality')
    axes[0, 1].set_title('Trend and Seasonality')
    axes[0, 1].legend()
    
    # Residuals plot
    if len(historical_residuals) > 0:
        axes[1, 0].scatter(range(len(historical_residuals)), historical_residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].axhline(y=anomaly_threshold, color='orange', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=-anomaly_threshold, color='orange', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Residuals and Anomaly Detection')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Residual')
    
    # Forecast distribution
    axes[1, 1].hist(forecast_period['yhat'], bins=20, alpha=0.7, color='skyblue')
    axes[1, 1].axvline(forecast_period['yhat'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {forecast_period["yhat"].mean():.2f}')
    axes[1, 1].set_title('Forecast Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    analysis_results = {
        'forecast_total': float(forecast_period['yhat'].sum()),
        'forecast_daily_avg': float(forecast_period['yhat'].mean()),
        'forecast_period_days': future_periods,
        'growth_rate_pct': float((forecast_period['yhat'].iloc[-1] / forecast_period['yhat'].iloc[0] - 1) * 100),
        'anomalies_detected': len(anomalies),
        'model_performance': df_performance.to_dict() if not df_performance.empty else {},
        'trend_direction': 'increasing' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else 'decreasing'
    }
"""
