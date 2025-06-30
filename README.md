# 🧠 Model Evaluation & Hyperparameter Tuning: Data Science Salaries (2025)

This project demonstrates how to evaluate and optimize multiple machine learning models to predict global data science salaries using real-world salary data. The dataset captures salary trends in Data Science, AI, and ML roles reported from 2020 to 2025 across various job titles, experience levels, and geographies.

---

## 📊 Objective

- Train and evaluate machine learning models to predict `salary_in_usd`
- Optimize models using `GridSearchCV` and `RandomizedSearchCV`
- Analyze metrics to select the best-performing model

---

## 📁 Dataset Details

- **Source**: [Kaggle Dataset: "Data Science Salaries 2025"](https://www.kaggle.com)
- **File**: `salaries.csv` (7.3 MB)
- **Rows**: 100K+
- **Columns**: 11

### Key Features:
| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `work_year`        | Year salary was reported (2020–2025)                                        |
| `experience_level` | Entry-level (EN), Mid (MI), Senior (SE), Executive (EX)                     |
| `employment_type`  | Full-time (FT), Part-time (PT), Contract (CT), Freelance (FL)               |
| `job_title`        | Role title (Data Scientist, ML Engineer, etc.)                              |
| `salary`           | Annual salary in original currency                                          |
| `salary_currency`  | Currency code (USD, GBP, EUR, etc.)                                         |
| `salary_in_usd`    | Salary converted to USD (standardized using 2025 rates)                     |
| `employee_residence` | Country of employee (ISO 2-letter code)                                   |
| `remote_ratio`     | 0 = On-site, 50 = Hybrid, 100 = Remote                                       |
| `company_location` | HQ country of employer                                                      |
| `company_size`     | S = Small, M = Medium, L = Large company                                    |

---

## 🔧 ML Workflow

1. **Data Preprocessing**:
   - Handled categorical columns using `pd.get_dummies()`
   - Split into train/test using `train_test_split()`

2. **Models Trained**:
   - ✅ Random Forest Regressor
   - ✅ Support Vector Regressor (SVR)
   - ✅ Linear Regression

3. **Evaluation Metrics**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score

4. **Hyperparameter Tuning**:
   - `GridSearchCV` → Optimized Random Forest
   - `RandomizedSearchCV` → Tuned SVR model

---

## 📂 Project Structure

model-evaluation-tuning/
├── salaries.csv
├── main.py
└── README.md


---

## ▶️ How to Run

Make sure `salaries.csv` is in the same folder as `main.py`. Then run:

```bash
python main.py
