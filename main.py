import argparse
import yaml
import pandas as pd
import numpy as np
import os
from src.preprocessing import load_data, fit_preprocessing, transform_data
from src.model_io import save_artifact, load_artifact, save_model, load_model
from src.lccde import lccde_predict
from src.anomaly_detection import fit_kmeans, apply_heuristics
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(config, dataset_path):
    print(f"Loading Data from {dataset_path}...")
    if os.path.isdir(dataset_path):
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSVs found in {dataset_path}")
        filepath = os.path.join(dataset_path, csv_files[0])
    else:
        filepath = dataset_path

    df = load_data(filepath)
    
    print("Preprocessing...")
    X_scaled, y_encoded, scaler, le, feature_names = fit_preprocessing(df)
    
    # Save Artifacts
    save_artifact(scaler, os.path.join(config['paths']['model_dir'], config['artifacts']['scaler']))
    save_artifact(le, os.path.join(config['paths']['model_dir'], config['artifacts']['label_encoder']))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    
    # Save Test Data for Calibration
    np.savez(os.path.join(config['paths']['model_dir'], "test_data.npz"), X_test=X_test, y_test=y_test)

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
    
    print("Training Complete.")

def evaluate_and_calibrate(config):
    print("Loading Test Data & Artifacts...")
    data = np.load(os.path.join(config['paths']['model_dir'], "test_data.npz"))
    X_test, y_test = data['X_test'], data['y_test']
    
    le = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['label_encoder']))
    
    # Load Models
    xgb_clf = load_model(os.path.join(config['paths']['model_dir'], "xgb_model.json"), 'xgboost')
    lgb_clf = load_model(os.path.join(config['paths']['model_dir'], "lgb_model.txt"), 'lightgbm', config['models']['lightgbm'])
    cb_clf = load_model(os.path.join(config['paths']['model_dir'], "cb_model.cbm"), 'catboost')
    
    models = {'XGBoost': xgb_clf, 'LightGBM': lgb_clf, 'CatBoost': cb_clf}
    
    # Calibrate Leaders (Simplified)
    print("Calibrating Leaders...")
    # In a full implementation, calculate F1 per class here. 
    # For now, we save a default map to ensure the pipeline flows.
    leader_map = {c: 'LightGBM' for c in range(len(le.classes_))} 
    save_artifact(leader_map, os.path.join(config['paths']['model_dir'], config['artifacts']['leader_map']))
    
    # Calibrate KMeans
    print("Calibrating Anomaly Detection...")
    try:
        benign_idx = le.transform(['BENIGN'])[0]
        X_benign = X_test[y_test == benign_idx]
        
        if len(X_benign) > config['anomaly_detection']['kmeans_k']:
            kmeans = fit_kmeans(X_benign, k=config['anomaly_detection']['kmeans_k'])
            save_artifact(kmeans, os.path.join(config['paths']['model_dir'], config['artifacts']['kmeans_model']))
            print("KMeans Calibrated.")
        else:
            print("Not enough BENIGN data for KMeans calibration.")
    except ValueError:
        print("Label 'BENIGN' not found in encoder. Skipping anomaly calibration.")

def predict(config, input_file, output_file):
    print(f"Predicting on {input_file}...")
    df = pd.read_csv(input_file)
    
    # Load Artifacts
    scaler = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['scaler']))
    le = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['label_encoder']))
    leader_map = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['leader_map']))
    kmeans = load_artifact(os.path.join(config['paths']['model_dir'], config['artifacts']['kmeans_model']))
    
    # Load Models
    xgb_clf = load_model(os.path.join(config['paths']['model_dir'], "xgb_model.json"), 'xgboost')
    lgb_clf = load_model(os.path.join(config['paths']['model_dir'], "lgb_model.txt"), 'lightgbm', config['models']['lightgbm'])
    cb_clf = load_model(os.path.join(config['paths']['model_dir'], "cb_model.cbm"), 'catboost')
    models = {'XGBoost': xgb_clf, 'LightGBM': lgb_clf, 'CatBoost': cb_clf}
    
    # Preprocess
    X_scaled, _ = transform_data(df, scaler, le)
    
    # 1. LCCDE Prediction
    preds = {}; probs = {}
    for name, m in models.items():
        probs[name] = m.predict_proba(X_scaled)
        if name == 'LightGBM': preds[name] = m.predict(X_scaled)
        else: preds[name] = np.argmax(probs[name], axis=1)
        
    y_pred = lccde_predict(X_scaled, models, leader_map, preds, probs)
    y_pred_str = le.inverse_transform(y_pred.astype(int))
    
    # 2. Anomaly Detection logic would go here
    # (Simplified for main.py readability)
        
    # Save
    pd.DataFrame({'Prediction': y_pred_str}).to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate_leaders_and_anomaly', 'predict'], required=True)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    conf = load_config(args.config)
    
    if args.mode == 'train':
        train(conf, args.dataset_path)
    elif args.mode == 'evaluate_leaders_and_anomaly':
        evaluate_and_calibrate(conf)
    elif args.mode == 'predict':
        predict(conf, args.input_file, args.output_file)
