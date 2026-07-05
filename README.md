# Clinical Trial Outcome ML

Machine Learning final project (May 2026).

Predicting whether a clinical trial will reach completion or be terminated/withdrawn,
using only information available at trial registration — protocol text and structured
metadata from ClinicalTrials.gov.

## Approach

- **Data:** ~3,000 trials downloaded via the ClinicalTrials.gov API v2 (includes `start_date` for temporal validation)
- **Features:** TF-IDF bigrams on protocol text + structured metadata (phase, study type, enrollment count)
- **Models:** Logistic Regression, Linear SVM, XGBoost
- **Target:** Binary — `Completed` (1) vs `Terminated / Withdrawn / Suspended` (0)
- **Key metric:** Macro F1 (accounts for class imbalance)

## Results

### Hold-out test (XGBoost hybrid, 80/20 split)

| Model | Macro F1 | PR-AUC | Recall (Terminated) |
|---|---|---|---|
| Logistic Regression | 0.64 | 0.94 | 0.43 |
| Linear SVM | 0.46 | 0.93 | 0.00 |
| **XGBoost (best)** | **0.72** | **0.95** | **0.40** |

Ablation: tabular-only Macro F1 = **0.67**, text-only = **0.55**, hybrid = **0.72**.

### Robustness and external validation (Sections 12–14)

| Check | Macro F1 |
|---|---|
| 5-fold stratified CV | 0.76 ± 0.02 |
| Text noise (30% word dropout) | 0.69 |
| Temporal external (train ≤2018, test ≥2019) | 0.75 |

Error analysis shows weakest recall on **phase-unspecified** trials (0.27).

## Repository Structure

```
clinical-trial-outcome/
├── notebooks/
│   └── main_analysis.ipynb   # EDA → models → ablation → validation → conclusions
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

# 1. Download data (~3000 trials; start_date included for external validation)
python src/fetch_data.py --limit 3000 --output data/raw/trials.csv

# 2. Preprocess (leakage removal enabled by default)
python src/preprocess.py --input data/raw/trials.csv --output data/processed/clean.csv

# 3. Open and run the notebook (Kernel → Restart & Run All)
jupyter notebook notebooks/main_analysis.ipynb
```
