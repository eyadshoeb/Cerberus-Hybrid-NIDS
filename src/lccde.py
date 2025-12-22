import numpy as np

def lccde_predict(X_test, models, leader_map, predictions, probabilities):
    """
    Leader Class Confidence Decision Ensemble (LCCDE)
    """
    final_preds = []
    num_samples = X_test.shape[0]
    model_names = ["XGBoost", "LightGBM", "CatBoost"]
    
    for i in range(num_samples):
        p = {name: predictions[name][i] for name in model_names}
        conf = {name: probabilities[name][i][p[name]] for name in model_names}
        if p["XGBoost"] == p["LightGBM"] == p["CatBoost"]:
            final_preds.append(p["XGBoost"])
            continue
        # If 2 models agree, check if the "Leader" for that class is one of them.
        counts = {x: list(p.values()).count(x) for x in p.values()}
        majority_class = max(counts, key=counts.get)
        
        if counts[majority_class] >= 2:
            leader_model = leader_map.get(majority_class)
            # If the leader agrees with majority, or is part of the majority, trust it.
            if leader_model:
                final_preds.append(p[leader_model])
            else:
                final_preds.append(majority_class)
            continue
        # Check if any model predicted a class for which IT IS the leader
        leader_matches = []
        for name in model_names:
            pred_class = p[name]
            if leader_map.get(pred_class) == name:
                leader_matches.append(name)
        
        if len(leader_matches) == 1:
            # Exactly one model is playing to its strength
            final_preds.append(p[leader_matches[0]])
        else:
            # Fallback to highest confidence
            best_model = max(model_names, key=lambda k: conf[k])
            final_preds.append(p[best_model])
            
    return np.array(final_preds)
