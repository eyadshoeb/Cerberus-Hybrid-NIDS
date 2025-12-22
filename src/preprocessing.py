import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def fit_preprocessing(df, label_col='Label'):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MUST return scaler and le to save them
    return X_scaled, y_encoded, scaler, le, X.columns.tolist()

def transform_data(df, scaler, le=None, label_col='Label'):
    if label_col in df.columns:
        X = df.drop(columns=[label_col])
    else:
        X = df
        
    X_scaled = scaler.transform(X)
    y_encoded = None
    return X_scaled, y_encoded
