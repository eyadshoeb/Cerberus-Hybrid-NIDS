import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class SmartDataSampler:
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.label_encoder = LabelEncoder()

    def load_and_normalize(self):
        """Loads CSV, strips whitespace, and applies Z-score normalization."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        self.df.columns = self.df.columns.str.strip()
        features = self.df.dtypes[self.df.dtypes != 'object'].index
        print("Applying Z-Score Normalization...")
        # (x - mean) / std
        self.df[features] = self.df[features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )
        self.df = self.df.fillna(0)
        if 'Label' in self.df.columns:
            self.df.iloc[:, -1] = self.label_encoder.fit_transform(self.df.iloc[:, -1])
            print("Labels Encoded.")
        return self.df

    def kmeans_undersampling(self, n_clusters=1000, sample_frac=0.008):
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_normalize() first.")

        print(f"Starting K-Means Sampling (Clusters: {n_clusters})...")
        # Assuming Benign is usually class 0 or the most frequent
        counts = self.df['Label'].value_counts()
        majority_class_id = counts.idxmax()
        
        df_minor = self.df[self.df['Label'] != majority_class_id]
        df_major = self.df[self.df['Label'] == majority_class_id].copy()
        
        print(f"Majority Class Size: {len(df_major)}")
        print(f"Minority Class Size: {len(df_minor)}")

        # Drop Label for clustering
        X_major = df_major.drop(['Label'], axis=1)
        
        # Use MiniBatchKMeans for speed on large datasets
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X_major)
        df_major['klabel'] = kmeans.labels_
        print(f"Sampling {sample_frac*100}% from each cluster...")
        
        def typical_sampling(group):
            return group.sample(frac=sample_frac)

        df_sampled_major = df_major.groupby('klabel', group_keys=False).apply(typical_sampling)
        df_sampled_major = df_sampled_major.drop(['klabel'], axis=1, errors='ignore')
        print(f"Reduced Majority Class to: {len(df_sampled_major)} samples")
        result_df = pd.concat([df_sampled_major, df_minor], axis=0)
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Final Dataset Shape: {result_df.shape}")
        return result_df

if __name__ == "__main__":
    sampler = SmartDataSampler('data/Wednesday-workingHours.pcap_ISCX.csv')
    sampler.load_and_normalize()
    final_df = sampler.kmeans_undersampling(n_clusters=1000, sample_frac=0.01)
    
    # Save for training
    final_df.to_csv('data/smart_sampled_dataset.csv', index=False)
    print("Data Pipeline Complete.")
