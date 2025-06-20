# Credit Card Fraud Detection 🕵️‍♂️💳

This project focuses on detecting fraudulent credit card transactions using **machine learning** — specifically the **XGBoost** model — and enhancing interpretability with **SHAP (SHapley Additive Explanations)**.

---

## 🔍 Objective

To accurately identify **fraudulent transactions** (extremely rare events) using **anomaly detection and supervised classification techniques**, and explain the model's decisions using feature impact analysis.

---

## 📁 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions (only 492 are frauds — highly imbalanced)
- All features are anonymized via PCA except `Time` and `Amount`
- `Class = 1` → Fraud; `Class = 0` → Legitimate

---

## ⚙️ Tools & Libraries

- Python 3.x
- pandas, numpy
- scikit-learn
- XGBoost
- SHAP
- matplotlib, seaborn

---

## 📊 Workflow

### 1. **Data Preprocessing**
- Loaded data, checked for nulls
- Scaled `Time` and `Amount` features
- Split into train/test

### 2. **Exploratory Data Analysis**
- Checked class imbalance
- Analyzed fraud vs. non-fraud transactions by `Amount` and `Time`

### 3. **Model Training**
- Used `XGBClassifier`
- Tuned `scale_pos_weight` for class imbalance
- Evaluated using metrics: confusion matrix, precision, recall, F1-score, ROC-AUC, PR-AUC

### 4. **Threshold Optimization**
- Swept threshold from 0 to 1
- Found optimal threshold ≈ 0.88 (maximizes F1 score)

### 5. **Model Interpretation**
- Used SHAP to explain which features contributed to fraud predictions
- Plotted global feature importance and local waterfall plots

---

## 📈 Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.977 |
| PR-AUC | 0.703 |
| Optimal Threshold | 0.88 |
| F1 Score (at 0.88) | High, with a balance between precision and recall |

---

## 🔑 Key Takeaways

- **Threshold tuning** is critical in imbalanced datasets — default (0.5) is usually suboptimal
- **Precision-Recall AUC** is more informative than ROC-AUC for rare event detection
- **SHAP** helps explain *why* the model flagged a transaction as fraud

---

## 🧠 Future Improvements

- Try alternative models: Logistic Regression, Isolation Forest, or Ensemble
- Handle imbalance with undersampling or SMOTE
- Build a Flask or FastAPI app for real-time fraud prediction

---

## 🗃️ How to Run

1. Clone the repo  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
