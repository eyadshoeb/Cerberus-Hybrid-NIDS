import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def fit_kmeans(X_benign, k=30):
    """Fits KMeans on LCCDE-predicted Benign traffic."""
    print(f"Fitting KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_benign)
    return kmeans

def apply_heuristics(X_df, cluster_labels, config):
    """
    Applies Cluster-Aware Heuristics.
    Returns a binary mask: 1 = Anomaly, 0 = Normal.
    """
    anomalies = np.zeros(len(X_df))
    
    # Extract config
    t_urg = config['heuristic_thresholds']['urg_flag']
    t_fin = config['heuristic_thresholds']['fin_flag']
    c_urg = config['target_clusters']['urg_cluster']
    c_fin = config['target_clusters']['fin_cluster']
    
    # Assuming column names map to scaled features. 
    # Note: In production, ensure column index mapping is robust.
    # Here we assume X_df is a DataFrame with named columns.
    
    # Rule 1: URG Flag in Cluster 8
    # We need to find the column index/name for 'URG Flag Count'
    # For robust implementation, pass the feature names or indices.
    
    # Simplified Logic (assuming DataFrame input):
    if 'URG Flag Count' in X_df.columns:
        mask_c8 = (cluster_labels == c_urg)
        mask_urg = (X_df.loc[mask_c8, 'URG Flag Count'] > t_urg)
        # Map back to global indices
        anomalies[X_df.index.isin(X_df.loc[mask_c8][mask_urg].index)] = 1
        
    if 'FIN Flag Count' in X_df.columns:
        mask_c24 = (cluster_labels == c_fin)
        mask_fin = (X_df.loc[mask_c24, 'FIN Flag Count'] > t_fin)
        anomalies[X_df.index.isin(X_df.loc[mask_c24][mask_fin].index)] = 1
        
    return anomalies