import argparse
import yaml
import os
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
from sklearn.metrics import classification_report, f1_score

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
    class_names = le.classes_
    num_classes = len(class_names)
    
# Get preds on test set to calculate F1 per class
    model_f1_scores = {}
    for name, m in models.items():
        y_pred = m.predict(X_test)
        # Ensure predictions are integers (LGBM sometimes returns floats/different types)
        y_pred = np.array(y_pred).astype(int).flatten()
        
        # Calculate F1 for each class
        f1s = f1_score(y_test, y_pred, average=None, labels=range(num_classes))
        model_f1_scores[name] = f1s

    # Build leader map: for each class, which model has highest F1?
    leader_map = {}
    for i in range(num_classes):
        best_model = max(model_f1_scores.keys(), key=lambda k: model_f1_scores[k][i])
        leader_map[int(i)] = best_model
        print(f"Class {i} ({class_names[i]}): Leader = {best_model} (F1: {model_f1_scores[best_model][i]:.4f})")
    
    save_artifact(leader_map, os.path.join(config['paths']['model_dir'], config['artifacts']['leader_map']))
    
    # --- Calibrate KMeans ---
    print("Calibrating Anomaly Detection...")
    # Filter for Benign (Class 0 usually)
    try:
        benign_label = 'BENIGN' if 'BENIGN' in class_names else class_names[0]
        benign_idx = le.transform([benign_label])[0]
        X_benign_test = X_test[y_test == benign_idx]
        
        if len(X_benign_test) > 0:
            kmeans = fit_kmeans(X_benign_test, k=config['anomaly_detection']['kmeans_k'])
            save_artifact(kmeans, os.path.join(config['paths']['model_dir'], config['artifacts']['kmeans_model']))
        else:
            print("Warning: No benign samples found for KMeans calibration.")
    except Exception as e:
        print(f"Error during KMeans calibration: {e}")
    print("Training Complete.")
    
def predict(config, input_path, output_path=None):
    print(f"Loading Models and Artifacts from {config['paths']['model_dir']}...")
    scaler = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['scaler']))
    le = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['label_encoder']))
    leader_map = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['leader_map']))
    kmeans = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['kmeans_model']))
    
    xgb_clf = load_model(os.path.join(config['paths']['model_dir'], "xgb_model.json"), 'xgboost', config['models']['xgboost'])
    lgb_clf = load_model(os.path.join(config['paths']['model_dir'], "lgb_model.txt"), 'lightgbm', config['models']['lightgbm'])
    cb_clf = load_model(os.path.join(config['paths']['model_dir'], "cb_model.cbm"), 'catboost', config['models']['catboost'])
    
    models = {'XGBoost': xgb_clf, 'LightGBM': lgb_clf, 'CatBoost': cb_clf}
    
    print(f"Loading inference data from {input_path}...")
    df_raw = load_data(input_path)
    X_scaled, _ = transform_data(df_raw, scaler)
    
    # Get individual predictions and probabilities
    print("Generating base predictions...")
    predictions = {}
    probabilities = {}
    
    for name, m in models.items():
        predictions[name] = m.predict(X_scaled).astype(int).flatten()
        probabilities[name] = m.predict_proba(X_scaled)
    
    print("Applying LCCDE Ensemble...")
    final_preds = lccde_predict(X_scaled, models, leader_map, predictions, probabilities)
    
    # Decode predictions
    final_labels = le.inverse_transform(final_preds).astype(object)
    
    # --- Tier 3 & 4: Anomaly Detection ---
    print("Applying Behavioral Profiling (KMeans)...")
    benign_idx = le.transform(['BENIGN'])[0] if 'BENIGN' in le.classes_ else 0
    
    # Identify samples predicted as Benign
    benign_mask = (final_preds == benign_idx)
    if benign_mask.any():
        X_benign = X_scaled[benign_mask]
        cluster_labels = kmeans.predict(X_benign)
        X_benign_df = pd.DataFrame(X_benign, columns=scaler.feature_names_in_)
        
        anomalies = apply_heuristics(X_benign_df, cluster_labels, config['anomaly_detection'])
        
        # Update labels for anomalies
        # We find the indices in the original final_labels array that correspond to the benign samples
        benign_indices = np.where(benign_mask)[0]
        anomaly_indices = benign_indices[anomalies == 1]
        
        final_labels[anomaly_indices] = 'ANOMALY'
        
        num_anomalies = np.sum(anomalies)
        print(f"Detected {num_anomalies} anomalies in traffic predicted as Benign and flagged them.")
    
    df_raw['Cerberus_Prediction'] = final_labels
    
    if output_path:
        df_raw.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print("\nPrediction Summary:")
        print(df_raw['Cerberus_Prediction'].value_counts())

    return df_raw
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--input', help='Path to input CSV for prediction')
    parser.add_argument('--output', help='Path to save prediction results')
    args = parser.parse_args()
    
    conf = load_config(args.config)
    
    if args.mode == 'train':
        train(conf)
    elif args.mode == 'predict':
        if not args.input:
            print("Error: --input is required for predict mode.")
        else:
            predict(conf, args.input, args.output)
