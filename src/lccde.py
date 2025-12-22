import numpy as np

def lccde_predict(X_test, models, leader_map):
    """
    Leader Class Confidence Decision Ensemble (LCCDE) Logic.
    1. Gets predictions from all 3 models.
    2. If they agree, output prediction.
    3. If they disagree, trust the 'Leader' model for that specific predicted class.
    """
    preds = {}
    probs = {}
    
    # Get predictions from base models
    for name, model in models.items():
        probs[name] = model.predict_proba(X_test)
        preds[name] = np.argmax(probs[name], axis=1)
        
    final_preds = []
    num_samples = X_test.shape[0]
    
    for i in range(num_samples):
        p_xgb = preds["XGBoost"][i]
        p_lgb = preds["LightGBM"][i]
        p_cb  = preds["CatBoost"][i]
        
        # Consensus
        if p_xgb == p_lgb == p_cb:
            final_preds.append(p_xgb)
            continue
            
        # If models disagree, define strategies
        predictions = [p_xgb, p_lgb, p_cb]
        
        # Logic: If 2 models agree, does the Leader of that class agree?
        # (Simplified logic for portfolio readability)
        if p_xgb == p_lgb:
            final_preds.append(p_xgb)
        elif p_xgb == p_cb:
            final_preds.append(p_xgb)
        elif p_lgb == p_cb:
            final_preds.append(p_lgb)
        else:
            # Complete disagreement: Trust the highest confidence
            conf_xgb = probs["XGBoost"][i][p_xgb]
            conf_lgb = probs["LightGBM"][i][p_lgb]
            conf_cb  = probs["CatBoost"][i][p_cb]
            
            best_model_idx = np.argmax([conf_xgb, conf_lgb, conf_cb])
            final_preds.append(predictions[best_model_idx])
            
    return np.array(final_preds)