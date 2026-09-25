# Clinical ML Models

This repository contains two independent supervised machine learning pipelines for clinical risk prediction: a **diabetes status classifier** and a **meningitis severity classifier**. Each was built and evaluated separately, but both follow the same workflow  compare Logistic Regression, Random Forest, and XGBoost via cross-validation, then select the best performer on F1 score.

| Model | Task | Best Model | Best CV F1 |
|---|---|---|---|
| [Diabetes Risk Prediction](#diabetes-risk-prediction-model) | Binary (negative / positive) | Random Forest | 0.9944 |
| [Meningitis Severity Prediction](#meningitis-severity-prediction-model) | Multi-class (Low / Moderate / High) | XGBoost | 0.8111 |

---

## Diabetes Risk Prediction Model

A model for predicting diabetes status from a combination of clinical laboratory values, physical measurements, and self-reported symptoms.

### Overview

Diabetes is often diagnosed too late, after symptoms have already progressed. This model explores how far a lightweight, interpretable feature set  a couple of standard lab markers plus commonly self-reported symptoms  can go toward flagging at-risk individuals early and accurately. The result leans almost entirely on two clinical markers, with everything else playing only a supporting role.

### Dataset & Features

| Category | Features |
|---|---|
| Lab markers | Fasting Plasma Glucose (FPG), HbA1c |
| Physical | BMI, Age, Gender |
| Symptoms | Polyuria, Polydipsia, Polyphagia, Nocturia, Weight loss, Vomiting, Nausea, Headache |

### Model Comparison

| Model | CV F1 Score |
|---|---|
| Logistic Regression | 0.9921 |
| **Random Forest** | **0.9950** |
| XGBoost | 0.9935 |

**Best model: Random Forest**  Best F1 score: **0.9944**

### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 222 | 4 |
| **Actual Positive** | 0 | 457 |

Out of 683 held-out cases, the model produced zero false negatives  no positive case was missed  and only 4 false positives. In a screening context, that asymmetry matters: the cost of a missed diabetes case is far higher than the cost of an unnecessary follow-up test, and the model's error profile reflects that priority well.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 1.000 | 0.982 | 0.991 | 226 |
| Positive | 0.991 | 1.000 | 0.996 | 457 |
| **Accuracy** | | | **0.994** | 683 |
| Macro avg | 0.996 | 0.991 | 0.993 | 683 |
| Weighted avg | 0.994 | 0.994 | 0.994 | 683 |

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | FPG | 0.1217 |
| 2 | HbA1c | 0.0760 |
| 3 | BMI | 0.0037 |
| 4 | Nocturia | 0.0007 |
| 5 | Polyuria | 0.0007 |
| 6 | Weight loss | 0.0007 |
|  | Age, Vomiting, Nausea, Polydipsia, Polyphagia, Headache, Gender | 0.0000 |

**What this says about the model's reasoning:**

- **FPG and HbA1c dominate.** Together they account for the overwhelming majority of the model's predictive signal  consistent with clinical practice, where both are frontline diagnostic markers for diabetes. FPG captures glucose at a single point in time, while HbA1c reflects average blood glucose over the preceding 2–3 months; the model effectively leans on one acute marker and one chronic marker in tandem.
- **BMI plays a minor but non-trivial role**, adding some refinement once glucose markers are accounted for.
- **Symptom features are nearly silent.** Classic diabetes symptoms  Polydipsia, Polyphagia, Vomiting, Nausea, Headache  and demographic features (Age, Gender) contribute essentially nothing.
- **Practical implication:** this is best understood as a lab-value classifier with a symptom checklist attached, not a symptom-based screening tool. A version without FPG/HbA1c would likely perform far worse.

### Limitations

- Performance is likely inflated by how strongly FPG and HbA1c correlate with the diagnostic labels themselves  read this as decision support, not a diagnostic replacement.
- Class imbalance (457 positive vs. 226 negative) should be kept in mind when interpreting precision/recall trade-offs on new data.

---

## Meningitis Severity Prediction Model

A multi-class model for predicting meningitis severity (**Low**, **Moderate**, **High**) from diagnosis type, infection markers, and patient vitals.

### Overview

Unlike the diabetes model's binary split, meningitis severity is a three-way call, and the classes are far from balanced  most cases in this dataset are Low or High severity, with Moderate cases comparatively rare. That imbalance shows up directly in where the model struggles.

### Model Comparison

| Model | CV F1 Score |
|---|---|
| Logistic Regression | 0.7769 |
| Random Forest | 0.8106 |
| **XGBoost** | **0.8111** |

**Best model: XGBoost**  Best F1 score: **0.8111**

All three models land within a few points of each other, with tree-based methods (Random Forest, XGBoost) modestly outperforming Logistic Regression  suggesting the feature interactions here (e.g. diagnosis type combined with lab markers) aren't fully linear.

### Confusion Matrix

|  | Predicted Low | Predicted Moderate | Predicted High |
|---|---|---|---|
| **Actual Low** | 106 | 3 | 13 |
| **Actual Moderate** | 8 | 9 | 8 |
| **Actual High** | 10 | 5 | 78 |

Low and High severity are predicted reliably (106/122 and 78/93 correct respectively), but **Moderate is the model's weak point**  only 9 of 25 actual Moderate cases were correctly classified, with the rest split roughly evenly toward Low and High. In practice, this means a genuinely moderate case is about as likely to be called Low or High as it is to be correctly flagged as Moderate.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 – Low | 0.855 | 0.869 | 0.862 | 122 |
| 1 – Moderate | 0.529 | 0.360 | 0.429 | 25 |
| 2 – High | 0.788 | 0.839 | 0.812 | 93 |
| **Accuracy** | | | **0.804** | 240 |
| Macro avg | 0.724 | 0.689 | 0.701 | 240 |
| Weighted avg | 0.795 | 0.804 | 0.798 | 240 |

The gap between macro avg (0.701) and weighted avg (0.798) F1 is itself informative: overall accuracy (80.4%) looks solid, but it's propped up by the two larger classes (Low, High). The Moderate class  the smallest and clinically most ambiguous  drags the macro-averaged score down substantially, since macro avg treats all three classes equally regardless of size.

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Diagnosis: Bacterial | 12.45 |
| 2 | CRP Level | 5.81 |
| 3 | Glucose Level | 5.78 |
| 4 | Diagnosis: Unknown | 3.27 |
| 5 | Diagnosis: Viral | 1.94 |
| 6 | Age | 1.19 |
| 7 | Pathogen Present: No | 1.17 |
| 8 | Protein Level | 1.11 |
| 9 | WBC Count | 1.02 |
| 10 | WBC Blood Count | 0.89 |
| 11 | Outcome: Deceased | 0.88 |
| 12 | Patient ID | 0.75 |
| 13 | Platelets | 0.72 |
| 14 | Hemoglobin | 0.68 |
| 15 | Gender: Female | 0.46 |

**What this says about the model's reasoning:**

- **Diagnosis type is the single strongest signal**, with `Diagnosis_Bacterial` roughly double the importance of any other feature. This tracks clinically: bacterial meningitis tends to progress more aggressively than viral meningitis, so knowing the causative agent is highly predictive of severity on its own.
- **CRP and Glucose levels are the next tier**, at roughly equal weight. Both are standard inflammatory/infection markers  elevated CRP signals systemic inflammation, and abnormal glucose (particularly low CSF glucose relative to blood glucose) is a classic marker of bacterial meningitis severity  so the model is picking up on genuinely meaningful biomarkers, not just the diagnosis label.
- **Protein Level, WBC Count, and WBC Blood Count**  all standard cerebrospinal fluid and blood infection markers  contribute smaller but consistent weight, reinforcing that the model draws on a coherent cluster of infection-severity indicators rather than one feature alone.
- **Age and Gender contribute comparatively little**, suggesting severity here is driven more by the infection's clinical profile than by patient demographics.
- **A caution worth flagging:** `Patient_ID` shows non-zero importance (0.75), which is a red flag rather than a real signal  an identifier should carry no predictive value, and its appearance here likely reflects incidental correlation or leakage from how the dataset was constructed/ordered. It's worth excluding `Patient_ID` from the feature set and re-running before trusting the importance ranking or deploying the model.
- **`Outcome_Deceased` appearing as a predictive feature is also worth scrutinizing**  if outcome data was recorded after severity was assessed, including it risks leaking future information into the prediction, which would inflate performance in a way that won't hold up on new patients.

### Limitations

- The Moderate class is small (25 of 240 test cases) and hardest to classify  treat Moderate predictions with more caution than Low/High.
- `Patient_ID` and `Outcome_Deceased` should be investigated for leakage before this model is trusted for deployment; re-run feature importance after removing/auditing them.
- Multi-class accuracy (80.4%) is respectable but macro-F1 (0.701) is the more honest number given the class imbalance.

---

## Usage

```bash
pip install -r requirements.txt

# Diabetes model
python train_diabetes.py
python predict_diabetes.py --input sample_input.csv

# Meningitis model
python train_meningitis.py
python predict_meningitis.py --input sample_input.csv
```

## License

MIT
