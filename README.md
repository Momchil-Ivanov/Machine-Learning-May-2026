# Clinical Trial Outcome ML

Machine Learning final project (May 2026).

Predicting whether a clinical trial will reach completion or be terminated/withdrawn,
using only information available at trial registration — protocol text and structured
metadata from ClinicalTrials.gov.

## Approach

- **Data:** ~2,000 trials downloaded via the ClinicalTrials.gov API v2
- **Features:** TF-IDF bigrams on protocol text + structured metadata (phase, study type, enrollment count)
- **Models:** Logistic Regression, Linear SVM, XGBoost
- **Target:** Binary — `Completed` (1) vs `Terminated / Withdrawn / Suspended` (0)
- **Key metric:** Macro F1 (accounts for class imbalance)

## Results

| Model | Macro F1 | PR-AUC |
|---|---|---|
| Logistic Regression | 0.60 | 0.93 |
| Linear SVM | 0.46 | 0.93 |
| **XGBoost (best)** | **0.77** | **0.95** |

Ablation study shows that structured metadata alone (Macro F1 = 0.72) outperforms
text alone (Macro F1 = 0.51). Combining both features achieves the best overall
performance.

## Repository Structure

```
clinical-trial-outcome/
├── notebooks/
│   └── main_analysis.ipynb   # Full analysis: EDA → models → ablation → conclusions
├── src/
│   ├── fetch_data.py          # Download trials from ClinicalTrials.gov API v2
│   └── preprocess.py          # Clean, label, and featurise raw data
├── data/                      # Ignored by git — download locally
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

## Reproducing the Results

```bash
pip install -r requirements.txt

# 1. Download data (~2000 trials)
python src/fetch_data.py --output data/raw/trials.csv

# 2. Preprocess (leakage removal enabled by default)
python src/preprocess.py --input data/raw/trials.csv --output data/processed/clean.csv

# 3. Open and run the notebook
jupyter notebook notebooks/main_analysis.ipynb
```
