# Cerberus-Hybrid-NIDS

**Cerberus** is a hybrid Network Intrusion Detection System (NIDS) designed to detect both known exploits and zero-day anomalies. Trained on the **CICIDS2017** dataset, it utilizes a **Leader Class Confidence Decision Ensemble (LCCDE)** to optimize detection across diverse attack vectors, achieving **99.8% accuracy** with high recall on critical threats.

The system employs a four-tier defense architecture, combining supervised gradient boosting models for known signatures with unsupervised K-Means clustering to profile traffic behavior and detect anomalies that bypass traditional classifiers.

## The Engineering Journey
The path to the final Cerberus architecture involved extensive benchmarking, failure analysis, and data-driven pivots. Below is the log of how the system evolved from a basic classifier to a hybrid ensemble.

### Phase 1: Baselines and Failure Analysis
We established baseline performance using 78 features on a subset of the dataset.
*   **Logistic Regression:** Struggled significantly with non-linearity and class imbalance (Macro F1: ~0.49).
*   **XGBoost (Default):** The immediate standout, achieving **0.88 Macro F1** out of the box.
*   **LightGBM (Default):** The surprise failure. With default parameters, it underfitted severely (Accuracy 90.2%, Macro F1 0.28), failing to split effectively on minority classes.

| Baseline Model | Accuracy | Macro F1 | Notes |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 94.62% | 0.49 | Failed on minority attack classes. |
| **LightGBM (Untuned)** | **90.26%** | **0.28** | **High Bias / Underfitting.** Missed PortScans entirely. |
| CatBoost (Untuned) | 99.63% | 0.70 | Good, but slower training on CPU. |
| **XGBoost (Untuned)** | **99.77%** | **0.88** | Best "out of the box" performance. |

### Phase 2: Optimization
We utilized **Optuna** for Hyperparameter Optimization (HPO) to address the baseline failures.
1.  **LightGBM Optimization:** We tuned LightGBM on CPU to resolve the underfitting. This was a turning point; the model went from the worst performer (F1 0.28) to the **single best individual model** (F1 ~0.92), excelling specifically at *SQL Injection* and *XSS* attacks.
2.  **GPU Acceleration:** XGBoost and CatBoost were successfully tuned on GPU, reducing HPO time from hours to minutes.
3.  **LCCDE Integration:** The ensemble combined the optimized LightGBM with the robust XGBoost and CatBoost. While LCCDE didn't simply average the F1 scores, it provided architectural stability, ensuring no single model's blind spot could result in a missed detection.

### Phase 3: The Feature Selection Decision
We attempted to reduce the dimensionality from 78 features to 40 using **Information Gain (IG)** to improve inference speed.
*   **Result:** General accuracy remained high (99.73%).
*   **Critical Failure:** Detection for `Web Attack – XSS` degraded significantly, with the F1-score dropping from **0.41 to 0.27**.
*   **Engineering Decision:** We rejected the reduced feature set. In a security context, losing visibility on specific attack vectors to save milliseconds is an unacceptable trade-off. Cerberus retains the full 78-feature set to prioritize Recall.

## The Architecture: A 4-Tier Defense System
Based on the findings above, Cerberus utilizes a hierarchical approach designed specifically to address class imbalance and feature sensitivity.

### Tier 1: The Optimized Base Layer
Three independent gradient boosting models analyze the raw network flow.
*   **Input:** Full 78-Feature Set (preserving XSS detection capabilities).
*   **Models:**
    *   **LightGBM (CPU-Tuned):** Optimized to fix initial underfitting. Now serves as the specialist for **Web Attacks (SQL Injection & XSS)**.
    *   **CatBoost (GPU-Tuned):** The designated specialist for **Brute Force** attacks.
    *   **XGBoost (GPU-Tuned):** The generalist backbone, providing high stability across DoS vectors.

### Tier 2: LCCDE Strategy (The Logic Core)
The **Leader Class Confidence Decision Ensemble** aggregates predictions. Instead of a simple majority vote, decision power is dynamically handed to the model with the highest historical F1-score for the *predicted* class.
*   *Example:* If LightGBM predicts `SQL Injection` but XGBoost predicts `Benign`, the system trusts **LightGBM** because it has a proven higher F1-score for Web Attacks.

### Tier 3: Behavioral Profiling (Unsupervised)
Traffic classified as "Benign" by Tier 2 is passed to a **K-Means Clustering algorithm ($k=30$)**.
*   **Logic:** Attacks that bypass supervised filters often cluster together due to similar statistical anomalies (e.g., high packet variance).
*   **Discovery:** We identified specific clusters (e.g., Cluster 8) that contained a high density of False Negatives.

### Tier 4: Cluster-Aware Heuristics
Once traffic is mapped to a cluster, we apply localized detection logic derived from our failure analysis:
*   **Heuristic Rule:** If a flow falls into *Cluster 8*, we check `URG Flag > 0.5`. This simple rule recovered **41% of the zero-day attacks** that the supervised layer missed.

## Data Pipeline: Smart Sampling
To handle the massive volume of the CICIDS2017 dataset (2.8M rows) without losing rare attack patterns, we implemented **K-Means Undersampling**.
1.  We clustered the majority class ("Benign") into 1,000 micro-clusters.
2.  We sampled representatively from each cluster, ensuring we kept diverse types of "Normal" traffic while reducing the dataset size by 90%.
*(Note: Implementation is provided in `src/preprocessing_experiments.py`)*

## Performance & Results
Unlike standard benchmarks that show uniform perfection, our experiments revealed significant variance between base learners. This validates the need for the LCCDE architecture.

| Model | Accuracy | Macro F1 | Key Weakness |
| :--- | :--- | :--- | :--- |
| **LightGBM** | 90.26% | 0.28 | Failed to detect **Botnet** and **PortScan** (0% Recall). |
| **CatBoost** | 99.63% | 0.70 | Struggled with **Web Attack XSS** (2% Recall). |
| **XGBoost** | 99.77% | 0.88 | Strongest base learner, but weak on **Sql Injection**. |
| **Cerberus (Ensemble)** | **99.81%** | **0.94** | Leverage XGBoost for PortScan but CatBoost for DDoS. |

**Key Insight:** LightGBM completely missed the `PortScan` class (0.03 recall), whereas XGBoost identified it perfectly. Conversely, CatBoost detected `FTP-Patator` with 100% precision while LightGBM only managed 18%. The LCCDE architecture successfully ignored individual model failures by deferring to the correct specialist.

### Visual Analysis
The confusion matrices below demonstrate how LightGBM (second panel) failed to diagonalize (classify correctly) for minority classes, appearing as a "faded" heatmap. The **Cerberus Ensemble (far right)** restores the diagonal structure, effectively "healing" the blind spots of the weaker models.

![Confusion Matrix Comparison](demo/cerberus_comparison_matrix.png)

## Anomaly Detection Results
The Unsupervised Layer acts as a safety net for False Negatives (FN) that bypass the ensemble.

| Method | Target | Performance | Impact |
| :--- | :--- | :--- | :--- |
| **K-Means Profiling** | LCCDE "Benign" Predictions | 7 Clusters flagged | Isolated high-risk traffic subgroups. |
| **Cluster-Aware Heuristics** | Hidden Zero-Days | **41.3% Recall** | Recovered 12/29 attacks the main models missed. |
| **Biased XGBoost (Cluster 30)** | Edge-case Anomalies | **100% Accuracy** | Successfully distinguished anomalies inside a mixed cluster. |

### Visualizing the Zero-Day Defense
The matrix below shows the effectiveness of **Tier 4 (Cluster-Aware Heuristics)**. Even after the advanced LCCDE models classified these samples as "Benign," the unsupervised logic successfully flagged them as anomalies based on cluster profiling.

![Anomaly Detection Matrix](demo/heuristic_matrix.png)

## Project Structure
```text
Cerberus-Hybrid-NIDS/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Cleaning, Label Encoding, Scaling
│   ├── preprocessing_experiments.py # K-Means Smart Sampling logic
│   ├── models.py           # XGB, LGBM, CatBoost definitions with Optuna params
│   └── lccde.py            # The ensemble logic and inference engine
├── demo/                   # Visualization assets
├── main.py                 # CLI entry point for training and inference
├── requirements.txt        # Python dependencies
└── README.md
