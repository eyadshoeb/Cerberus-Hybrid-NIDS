import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def fit_preprocessing(df, label_col='Label'):
    """Fits Scaler and Encoder on Training Data."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, scaler, le, X.columns.tolist()

def transform_data(df, scaler, le=None, label_col='Label'):
    """Applies existing Scaler/Encoder to New Data (Inference)."""
    if label_col in df.columns:
        X = df.drop(columns=[label_col])
        y = df[label_col]
        # Handle unseen labels if necessary, or assume test set matches train
        y_encoded = le.transform(y) if le else None
    else:
        X = df
        y_encoded = None
        
    X_scaled = scaler.transform(X)
    return X_scaled, y_encoded
