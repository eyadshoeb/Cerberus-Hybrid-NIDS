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
    anomalies = np.zeros(len(X_df), dtype=int)
    
    # Extract config
    t_urg = config['heuristic_thresholds']['urg_flag']
    t_fin = config['heuristic_thresholds']['fin_flag']
    c_urg = config['target_clusters']['urg_cluster']
    c_fin = config['target_clusters']['fin_cluster']
    
    # Rule 1: URG Flag in Cluster 8
    if 'URG Flag Count' in X_df.columns:
        mask_urg = (cluster_labers == c_urg) & (X_df['URG Flag Count'] > t_urg)
        anomalies[mask_urg] = 1

    # Rule 2: FIN Flag in Cluster 24
    if 'FIN Flag Count' in X_df.columns:
        mask_urg = (cluster_labers == c_fin) & (X_df['URG Flag Count'] > t_fin)
        anomalies[mask_fin] = 1
        
    return anomalies
