# Cerberus-Hybrid-NIDS

A hybrid Multi-tiered Network Intrusion Detection System (NIDS) trained on the **CICIDS2017** dataset.

This project implements a **Leader Class Confidence Decision Ensemble (LCCDE)**. Instead of using a simple majority vote (which averages out errors but also strengths), this architecture combines **XGBoost, LightGBM, and CatBoost** by assigning a specific "Leader" model to each attack class based on validation F1-scores.

It also includes a secondary **Unsupervised K-Means** layer to detect zero-day anomalies that the supervised models might classify as "Benign."

## The Architecture
Most ensembles just average predictions. Cerberus is stricter:
1.  **Base Layer:** Three gradient boosting models independently predict the traffic class.
2.  **LCCDE Logic:**
    *   If all models agree → Output prediction.
    *   If models disagree → The system defers to the "Leader Model" for the disputed class (e.g., if XGBoost historically detects 'PortScan' best, its vote outweighs the others for that specific label).
3.  **Anomaly Detection:** Traffic classified as 'Benign' is passed through K-Means. Data points that form low-density clusters or are statistical outliers are flagged as potential zero-day threats.

## Technical Highlights
*   **Models:** XGBoost, LightGBM, CatBoost (Optimized via **Optuna**).
*   **Performance:** ~99.8% Accuracy on the test set.
*   **Zero-Day Handling:** unsupervised clustering on negative predictions to catch edge cases.
*   **Data Pipeline:** Full cleaning, scaling, and encoding pipeline for the CICIDS2017 CSVs.
