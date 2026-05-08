# NTI Training — Telecom Churn & HR Attrition Prediction

Two end-to-end machine learning projects completed as part of NTI (National Telecommunication Institute) training, focusing on binary classification for business-critical prediction tasks. The **Project Team** work addresses telecom customer churn, while the **Solo Project** tackles HR employee attrition — both using exploratory data analysis, feature engineering, class imbalance handling (SMOTE), and multi-model comparison to identify the best predictive model.

---

## Highlights

- **Two Complete ML Pipelines** — Both projects follow a rigorous end-to-end workflow: data loading → EDA → feature engineering → preprocessing → model training → evaluation → comparison, providing a comprehensive demonstration of applied machine learning on real-world business datasets.
- **SMOTE for Class Imbalance** — Both datasets are heavily imbalanced (churned customers and attritioned employees are minorities). The pipelines use SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples of the minority class, significantly improving recall and F1 scores for the positive class without simply inflating accuracy.
- **Multi-Model Benchmarking** — Rather than settling on a single algorithm, both projects train and compare 8–11 classification models side by side (Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes, XGBoost, LightGBM, CatBoost), reporting Accuracy, Precision, Recall, and F1-Score for each in a summary table.
- **Feature Engineering** — The telecom churn project creates derived features like `total_call_minutes`, `total_charges`, `active_plans_count`, `frequent_support_calls`, and `intl_usage_rate`. The HR project drops redundant columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`) and applies Label Encoding to categorical variables.
- **Interactive Dashboards** — The HR project includes Tableau and Power BI dashboards for visual exploration of attrition patterns, along with a Python script version for headless execution.

---

## Project Structure

```
nti-training/
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
│
├── project-team/                                # Telecom Customer Churn
│   ├── notebook/
│   │   └── Telecom_Churn.ipynb                  # Complete churn analysis & prediction
│   └── data/
│       ├── churn-bigml-80.csv                   # Training dataset (80% split)
│       └── churn-bigml-20.csv                   # Test dataset (20% split)
│
└── solo-project/                                # HR Employee Attrition
    ├── notebook/
    │   ├── final_HR_analytics.ipynb             # Full EDA & modeling pipeline
    │   ├── final_hr.ipynb                       # Earlier version
    │   └── final_hr_final.ipynb                 # Final version with all 11 models
    ├── src/
    │   └── final_hr.py                          # Standalone Python script version
    ├── data/
    │   └── HR-Employee-Attrition.csv            # IBM HR dataset (1,470 employees)
    └── dashboard/
        ├── page_1.png                           # Dashboard screenshot — overview
        ├── page_2.png                           # Dashboard screenshot — demographics
        └── page_3.png                           # Dashboard screenshot — compensation
```

---

## Project 1: Telecom Customer Churn (Project Team)

### Problem Statement

Telecom companies face significant revenue loss when customers switch to competitors. Identifying customers at high risk of churn before they leave enables proactive retention strategies — targeted offers, service improvements, and personalized outreach — that are far cheaper than acquiring new customers.

### Dataset

The **Churn BigML** dataset from Kaggle contains information about telecom customers including account details (state, account length, area code), plan subscriptions (international plan, voice mail plan), usage metrics (day/evening/night/international minutes, calls, charges), and customer service interaction data. The 80/20 split is provided as separate CSV files with 2,666 training and 667 test samples. The target variable `Churn` is boolean (True/False).

### Feature Engineering

Several powerful derived features are created from the raw data to improve model performance and interpretability:

- **`total_call_minutes`** — Sum of day, evening, night, and international call minutes, capturing the customer's overall usage intensity.
- **`total_calls`** — Total number of calls across all time periods, indicating engagement level.
- **`total_charges`** — Aggregate charges from all periods, directly related to cost-sensitivity which is a key churn driver.
- **`active_plans_count`** — Count of active subscription plans (international + voicemail), proxy for service bundling.
- **`account_tenure_category`** — Categorized account length (Short-term / Medium-term / Long-term / Loyal), capturing nonlinear tenure effects.
- **`frequent_support_calls`** — Binary flag for customers with more than 3 service calls, a strong churn signal.
- **`intl_usage_rate`** — Ratio of international minutes to total minutes, identifying international-heavy users.
- **`region`** — US geographic region mapped from state code, enabling regional churn analysis.

### Modeling Pipeline

1. **Preprocessing** — One-hot encoding of categorical features (`International plan`, `Voice mail plan`, `region`, `account_tenure_category`) with `drop_first=True` to avoid multicollinearity.
2. **SMOTE Oversampling** — Applied to the training set to balance the churn/non-churn ratio, generating synthetic minority class samples.
3. **Model Comparison** — Logistic Regression, KNN, and Random Forest trained and evaluated; scaled features used for distance-based models (LR, KNN) via `StandardScaler`.
4. **Evaluation** — Accuracy, Precision, Recall, and F1-Score reported for each model.

### Key Findings

- Customers with the **International Plan** churn at a significantly higher rate.
- High **Customer Service Calls** (especially >3) are one of the strongest churn predictors.
- **Random Forest** with SMOTE generally achieves the best balance of precision and recall.

---

## Project 2: HR Employee Attrition (Solo Project)

### Problem Statement

Employee attrition is costly — replacing an employee typically costs 50–200% of their annual salary in recruiting, onboarding, and lost productivity. Predicting which employees are likely to leave enables HR teams to intervene early with retention strategies, targeted career development, and compensation adjustments.

### Dataset

The **IBM HR Analytics Employee Attrition & Performance** dataset contains 1,470 employee records with 35 features covering demographics (age, gender, marital status), job details (department, role, level, years in role), compensation (monthly income, stock options, percent salary hike), work-life factors (overtime, distance from home, business travel), and satisfaction metrics (job satisfaction, environment satisfaction, work-life balance, relationship satisfaction). The target variable `Attrition` is binary (Yes/No), with approximately 16% positive rate — a significant class imbalance.

### Data Cleaning

- **Redundant columns removed**: `EmployeeCount` (constant 1), `StandardHours` (constant 80), `Over18` (constant Y), `EmployeeNumber` (arbitrary ID).
- **Duplicates checked** and none found.
- **Attrition converted** from "Yes"/"No" to 1/0 for binary classification.
- **Label Encoding** applied to all remaining categorical features (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime).
- **Feature Scaling** with `StandardScaler` on numerical columns after train/test split to prevent data leakage.

### Modeling Pipeline

1. **Stratified Train/Test Split** — 80/20 split with `stratify=y` to maintain the attrition rate across both sets.
2. **StandardScaler** applied to numerical features in the training set; same transformation applied to test set.
3. **11 Classification Models** trained and compared:

| # | Model | Type |
|---|-------|------|
| 1 | Linear Regression | Regression (thresholded at 0.5) |
| 2 | Logistic Regression | Linear classifier |
| 3 | Decision Trees | Tree-based |
| 4 | Random Forest | Ensemble (bagging) |
| 5 | SVM | Kernel-based |
| 6 | KNN | Distance-based |
| 7 | Naive Bayes | Probabilistic |
| 8 | XGBoost | Gradient boosting |
| 9 | LightGBM | Gradient boosting |
| 10 | CatBoost | Gradient boosting |

4. **Evaluation Metrics** — Accuracy, Precision, Recall, F1-Score computed for each model and summarized in a comparison table.
5. **Best Model Selection** — The model with the highest accuracy is automatically identified and reported.

### Dashboard

The project includes a Tableau/Power BI dashboard with three pages:
- **Page 1 — Overview**: Attrition rate, headcount, key KPIs at a glance
- **Page 2 — Demographics**: Attrition breakdown by age, gender, marital status, education
- **Page 3 — Compensation**: Salary distribution, stock options, and percent salary hike vs. attrition

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | scikit-learn (LabelEncoder, StandardScaler, train_test_split) |
| **Oversampling** | imbalanced-learn (SMOTE) |
| **Classical ML** | scikit-learn (LR, DT, RF, SVM, KNN, NB) |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost |
| **Dashboards** | Tableau, Power BI |
| **Environments** | Google Colab, Jupyter Notebook |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/aliisnetalive/nti-training-ml-projects.git
cd nti-training-ml-projects

# Install dependencies
pip install -r requirements.txt
```

### Running the Telecom Churn Project

Open `project-team/notebook/Telecom_Churn.ipynb` in Jupyter and run all cells. The data files are in `project-team/data/`.

### Running the HR Attrition Project

Open `solo-project/notebook/final_hr_final.ipynb` in Jupyter or Google Colab and run all cells. The data file is in `solo-project/data/`. Alternatively, run the standalone Python script:

```bash
cd solo-project
python src/final_hr.py
```

---

## Datasets

| Project | Dataset | Source | Samples | Features |
|---------|---------|--------|---------|----------|
| Telecom Churn | Churn BigML | Kaggle | 3,333 | 20+ |
| HR Attrition | IBM HR Analytics | Kaggle | 1,470 | 35 |

---

## References

- [Churn BigML Dataset — Kaggle](https://www.kaggle.com/datasets/milanvaddoriya/old-car-price-prediction)
- [IBM HR Analytics Employee Attrition — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) — Chawla et al., 2002
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## License

This project is open source and available for educational and personal use.
