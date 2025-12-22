import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_tuned_xgboost(num_classes):
    return xgb.XGBClassifier(
        learning_rate=0.057, n_estimators=900, max_depth=11,
        subsample=0.7, colsample_bytree=0.9,
        objective='multi:softprob', eval_metric='mlogloss',
        random_state=42, n_jobs=-1
    )

def get_tuned_lightgbm(num_classes):
    return lgb.LGBMClassifier(
        learning_rate=0.01, n_estimators=600, num_leaves=294,
        max_depth=12, objective='multiclass',
        random_state=42, n_jobs=-1, verbose=-1
    )

def get_tuned_catboost(num_classes):
    return cb.CatBoostClassifier(
        iterations=600, learning_rate=0.044, depth=9,
        loss_function='MultiClass', random_seed=42, verbose=0
    )

def train_all_models(X_train, y_train, num_classes):
    print("Training XGBoost...")
    xgb_model = get_tuned_xgboost(num_classes)
    xgb_model.fit(X_train, y_train)
    
    print("Training LightGBM...")
    lgb_model = get_tuned_lightgbm(num_classes)
    lgb_model.fit(X_train, y_train)
    
    print("Training CatBoost...")
    cb_model = get_tuned_catboost(num_classes)
    cb_model.fit(X_train, y_train)
    
    return {"XGBoost": xgb_model, "LightGBM": lgb_model, "CatBoost": cb_model}