import joblib
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def save_artifact(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Saved artifact to {filepath}")

def load_artifact(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    raise FileNotFoundError(f"Artifact not found: {filepath}")

def save_model(model, filepath, model_type):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if model_type == 'xgboost':
        model.save_model(filepath)
    elif model_type == 'lightgbm':
        model.booster_.save_model(filepath)
    elif model_type == 'catboost':
        model.save_model(filepath)
    else:
        joblib.dump(model, filepath)

def load_model(filepath, model_type, params=None):
    if model_type == 'xgboost':
        model = xgb.XGBClassifier()
        if params: model.set_params(**params)
        model.load_model(filepath)
        return model
    elif model_type == 'catboost':
        model = cb.CatBoostClassifier()
        model.load_model(filepath)
        return model
    elif model_type == 'lightgbm':
        # Loading LGBM into sklearn wrapper 
        model = lgb.LGBMClassifier(**params)
        booster = lgb.Booster(model_file=filepath)
        model._Booster = booster
        model._n_classes = booster.num_class()
        model._fitted = True
        return model
    return joblib.load(filepath)
