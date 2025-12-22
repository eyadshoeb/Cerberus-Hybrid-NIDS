# Cerberus-Hybrid-NIDS

A hybrid Multi-tiered Network Intrusion Detection System (NIDS) trained on the **CICIDS2017** dataset.

This project implements a **Leader Class Confidence Decision Ensemble (LCCDE)**. Instead of using a simple majority vote (which averages out errors but also strengths), this architecture combines **XGBoost, LightGBM, and CatBoost** by assigning a specific "Leader" model to each attack class based on validation F1-scores.

It also includes a secondary **Unsupervised K-Means** layer to detect zero-day anomalies that the supervised models might classify as "Benign."

## Smart Sampling:
To handle the massive volume of CICIDS2017 (2.8M rows) without losing rare attack patterns, we implemented K-Means Undersampling.
We clustered the majority class ("Benign") into 1,000 micro-clusters.
We sampled representatively from each cluster, ensuring we kept diverse types of "Normal" traffic while reducing dataset size by 90%.
(Note: Code for this is provided in preprocessing_experiments.py)

## The Architecture: A 4-Tier Defense System
**Cerberus** goes beyond simple classification. It utilizes a hierarchical "Tiered" approach to ensure zero-day threats don't slip through as "Benign."

### Tier 1: The Base Learners
Three independent gradient boosting models (XGBoost, LightGBM, CatBoost) analyze the raw network flow.

### Tier 2: LCCDE Strategy
The **Leader Class Confidence Decision Ensemble** aggregates predictions. If the models disagree, decision power is dynamically handed to the model with the highest historical F1-score for the predicted class.

### Tier 3: Behavioral Profiling (Unsupervised)
Traffic classified as "Benign" by Tier 2 is not trusted blindly. It is passed to a **K-Means Clustering algorithm ($k=30$)**.
*   **Logic:** Attacks that bypass supervised filters often cluster together due to similar statistical anomalies (e.g., high packet variance).
*   **Discovery:** We identified specific clusters (e.g., Cluster 8 and 24) that contained a high density of False Negatives.

### Tier 4: Cluster-Aware Heuristics & Biased Classifiers
Once traffic is mapped to a cluster, we apply localized detection logic:
1.  **Heuristics:** If a flow falls into *Cluster 8*, we check specific flags (e.g., `URG Flag > 0.5`). This simple rule recovered **41% of the attacks** that LCCDE originally missed.
2.  **Biased Classifiers:** For high-risk clusters, we deploy lightweight Decision Trees trained specifically to distinguish "Normal" vs "Anomaly" *within that specific cluster's distribution*.

## Technical Highlights
*   **Models:** XGBoost, LightGBM, CatBoost (Optimized via **Optuna**).
*   **Performance:** ~99.8% Accuracy on the test set.
*   **Zero-Day Handling:** unsupervised clustering on negative predictions to catch edge cases.
*   **Data Pipeline:** Full cleaning, scaling, and encoding pipeline for the CICIDS2017 CSVs.

## Performance & Results
Unlike standard benchmarks that show uniform perfection, our experiments revealed significant variance between base learners. This validates the need for the **LCCDE (Leader Class Confidence)** architecture.

| Model | Accuracy | Macro F1 | Key Weakness |
| :--- | :--- | :--- | :--- |
| **LightGBM** | 90.26% | 0.28 | Failed to detect **Botnet** and **PortScan** (0% Recall). |
| **CatBoost** | 99.63% | 0.70 | Struggled with **Web Attack XSS** (2% Recall). |
| **XGBoost** | 99.77% | 0.88 | Strongest base learner, but weak on **Sql Injection**. |
| **Cerberus (Ensemble)** | **99.81%** | **0.94** | Leverage XGBoost for PortScan but CatBoost for DDoS. |

**Key Insight:**
*   **LightGBM** completely missed the `PortScan` class (0.03 recall), whereas **XGBoost** identified it perfectly (1.00 recall).
*   **CatBoost** detected `FTP-Patator` with 100% precision, while **LightGBM** only managed 18%.
*   **Conclusion:** The LCCDE architecture successfully ignored LightGBM's false negatives by deferring to XGBoost's higher confidence for PortScans.

### Visual Analysis
The confusion matrices below demonstrate how LightGBM (second panel) failed to diagonalize (classify correctly) for minority classes, appearing as a "faded" heatmap. The **Cerberus Ensemble (far right)** restores the diagonal structure, effectively "healing" the blind spots of the weaker models.

![Confusion Matrix Comparison](demo/cerberus_comparison_matrix.png)

## Anomaly Detection Results (The "Safety Net")
The Supervised LCCDE layer is excellent, but no model is perfect. The Unsupervised Layer acts as a safety net for False Negatives (FN).

| Method | Target | Performance | Impact |
| :--- | :--- | :--- | :--- |
| **K-Means Profiling** | LCCDE "Benign" Predictions | 7 Clusters flagged | Isolated high-risk traffic subgroups. |
| **Cluster-Aware Heuristics** | Hidden Zero-Days | **41.3% Recall** | Recovered 12/29 attacks the main models missed. |
| **Biased XGBoost (Cluster 30)** | Edge-case Anomalies | **100% Accuracy** | Successfully distinguished anomalies inside a mixed cluster. |

**Key Finding:**
The primary models struggled with specific `Web Attack` vectors. By analyzing **Cluster 8**, we found that `URG Flag` counts were a dead giveaway for these specific attacks, allowing us to write a heuristic rule that caught them without retraining the massive base models.

### Visualizing the Zero-Day Defense
The matrix below shows the effectiveness of **Tier 4 (Cluster-Aware Heuristics)**. Even after the advanced LCCDE models classified these samples as "Benign," the unsupervised logic successfully flagged them as anomalies based on cluster profiling.

![Anomaly Detection Matrix](demo/heuristic_matrix.png)
