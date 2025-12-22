import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath, sample_size=None):
    """
    Loads network traffic data, cleans NaNs/Infs, and encodes labels.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Clean Column Names
    df.columns = df.columns.str.strip()
    
    # Handle Infinite/Null values common in CICIDS2017
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def preprocess_features(df, label_col='Label'):
    """
    Splits features/target, applies Label Encoding and Standard Scaling.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Encode Targets
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le