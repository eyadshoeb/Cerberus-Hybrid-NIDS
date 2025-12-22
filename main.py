import argparse
import yaml
import pandas as pd
import numpy as np
from src.preprocessing import load_data, fit_preprocessing, transform_data
from src.model_io import save_artifact, load_artifact, save_model, load_model
from src.lccde import lccde_predict
from src.anomaly_detection import fit_kmeans, apply_heuristics
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import classification_report

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    print("Loading Data...")
    df = load_data(config['paths']['raw_data_path'])
    
    print("Preprocessing...")
    X_scaled, y_encoded, scaler, le, feature_names = fit_preprocessing(df)
    
    # Save Artifacts
    save_artifact(scaler, os.path.join(config['paths']['model_dir'], config['artifacts']['scaler']))
    save_artifact(le, os.path.join(config['paths']['model_dir'], config['artifacts']['label_encoder']))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    
    # Train Models
    print("Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(**config['models']['xgboost'])
    xgb_clf.fit(X_train, y_train)
    save_model(xgb_clf, os.path.join(config['paths']['model_dir'], "xgb_model.json"), 'xgboost')
    
    print("Training LightGBM...")
    lgb_clf = lgb.LGBMClassifier(**config['models']['lightgbm'])
    lgb_clf.fit(X_train, y_train)
    save_model(lgb_clf, os.path.join(config['paths']['model_dir'], "lgb_model.txt"), 'lightgbm')
    
    print("Training CatBoost...")
    cb_clf = cb.CatBoostClassifier(**config['models']['catboost'], verbose=0)
    cb_clf.fit(X_train, y_train)
    save_model(cb_clf, os.path.join(config['paths']['model_dir'], "cb_model.cbm"), 'catboost')
    
    # --- Calibrate LCCDE Leader Map ---
    print("Calibrating LCCDE Leaders...")
    models = {'XGBoost': xgb_clf, 'LightGBM': lgb_clf, 'CatBoost': cb_clf}
    leader_map = {}
    class_names = le.classes_
    
    # Get preds on test set
    preds = {name: m.predict(X_test) for name, m in models.items()}
    if 'LightGBM' in preds: preds['LightGBM'] = preds['LightGBM'].astype(int) # ensure int type
    
    # Determine best model per class (simplified logic for brevity)
    # (In full version, calculate F1 per class and assign)
    # For now, let's assume we save the map we found in our chat:
    # This part should ideally implement the full F1 comparison logic from our Phase 2 script.
    
    # --- Calibrate KMeans ---
    print("Calibrating Anomaly Detection...")
    # Predict on Test Set
    # Filter for Benign (Class 0 usually)
    benign_idx = le.transform(['BENIGN'])[0]
    # For calibration, we use TRUE Benign samples from test set to learn "Normal" structure
    X_benign_test = X_test[y_test == benign_idx]
    
    kmeans = fit_kmeans(X_benign_test, k=config['anomaly_detection']['kmeans_k'])
    save_artifact(kmeans, os.path.join(config['paths']['model_dir'], config['artifacts']['kmeans_model']))
    
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    conf = load_config(args.config)
    
    if args.mode == 'train':
        train(conf)
    # Add predict logic here...
