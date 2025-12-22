from src.preprocessing import load_and_clean_data, preprocess_features
from src.models import train_all_models
from src.lccde import lccde_predict
from sklearn.metrics import classification_report

# 1. Load Data
# Note: For the repo, put a small sample CSV in a 'data' folder
df = load_and_clean_data('data/network_traffic_sample.csv', sample_size=50000)

# 2. Preprocess
X_train, X_test, y_train, y_test, le = preprocess_features(df)
num_classes = len(le.classes_)

# 3. Train Base Models
models = train_all_models(X_train, y_train, num_classes)

# 4. Define Leader Map (Mock logic for demo - in real life this comes from validation F1 scores)
# This maps Class_ID -> "Best Model Name"
leader_map = {i: "CatBoost" for i in range(num_classes)} 

# 5. Run LCCDE Ensemble
print("Running LCCDE Ensemble Inference...")
y_pred = lccde_predict(X_test, models, leader_map)

# 6. Evaluate
print("\n--- Sentinel-NIDS Performance Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))