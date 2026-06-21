# P0A-2 Customer Churn Predictor (ML Foundations)

Welcome to the **Customer Churn Predictor** project, a core part of the **DevJam AI Engineer Roadmap under Machine Learning Foundations (P0A-2)**.

This end-to-end supervised learning classification project is designed to demonstrate how classical Machine Learning models are used to solve critical business problems (retaining customers and optimizing marketing spend) before transitioning into Generative AI and Agentic systems.

---

## 📂 Project Structure

```
customer-churn-predictor/
  README.md
  requirements.txt
  data/
    churn_data.csv
  notebooks/
    churn_analysis.ipynb
  src/
    data_generation.py
    preprocessing.py
    train_model.py
    evaluate_model.py
    predict.py
  app/
    streamlit_app.py
  models/
    churn_model.pkl
    preprocessor.pkl
    feature_names.pkl
    model_comparison.csv
  reports/
    model_metrics.md
    confusion_matrix.png
    roc_curve.png
    feature_importance.png
```

---

## 🎯 Problem Statement & Churn Analytics

### Why Churn Prediction Matters
Customer Churn (also known as customer attrition) occurs when customers stop doing business with a company or cancel their service. In subscription-based industries (SaaS, Telecom, Banking):
1. **High Cost of Acquisition (CAC):** Acquiring a new customer is 5x to 25x more expensive than retaining an existing one.
2. **Compound Loss of LTV:** When a loyal customer churns, the company loses not only their immediate monthly recurring revenue (MRR) but their entire Lifetime Value (LTV).
3. **Proactive Retention:** If we can identify *high-risk* customers before they cancel, the customer success team can proactively offer tailored discounts, premium support trials, or contract transitions to keep them.

### What is Classification?
Classification is a type of **Supervised Machine Learning** where the target variable is a discrete category or label. In this project, the target is binary:
- **`Yes` (1):** Customer will churn.
- **`No` (0):** Customer will not churn.

---

## 📊 Dataset Description

The project uses a realistic, synthetic dataset of 3,500 customers (`data/churn_data.csv`) that simulates a subscription service. Features include:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `customer_id` | Text (Dropped) | Unique customer identifier |
| `gender` | Categorical | Male or Female |
| `age` | Numerical | Age of the customer (18 to 80) |
| `tenure_months` | Numerical | Number of months the customer has been with the company |
| `monthly_charges`| Numerical | Monthly billing rate ($) |
| `total_charges` | Numerical | Cumulative charges ($) (includes small random NaNs for imputation practice) |
| `contract_type` | Categorical | Month-to-month, One year, Two year |
| `payment_method` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| `internet_service`| Categorical | DSL, Fiber optic, No |
| `support_tickets`| Numerical | Number of support tickets filed in the last month (0 to 15) |
| `last_login_days`| Numerical | Days since the last user login (0 to 30) |
| `usage_frequency`| Numerical | Active days of service usage in the last month (0 to 30) |
| `plan_type` | Categorical | Basic, Standard, Premium |
| `has_premium_support`| Categorical| Yes or No |
| `churn` | Categorical (Target)| **Yes / No** (Mapped to 1 / 0 during preprocessing) |

---

## 💻 App Tabs & Dashboard Features

The interactive dashboard is organized into three tabs:

1. **🎯 Predict Customer Churn**: Enter individual customer details (demographics, billing, support interaction) to generate a real-time risk assessment. The tab displays:
   - A highly readable, color-coded **Prediction Outcome Card**.
   - An aligned **Churn Probability Gauge** representing the exact calibrated probability.
   - **Key Churn Drivers** indicating customer pain points.
   - **Actionable Business Recommendations** for customer success agents.
2. **📊 Exploratory Data Analysis (EDA)**: Fully interactive visualizations showcasing data trends and correlation patterns including:
   - Churn Distribution Pie Chart
   - Churn by Contract Type Bar Chart
   - Tenure Distribution Box Plot
   - Monthly Charges histogram
   - Support Tickets Bar Chart
   - Usage Frequency Box Plot
   - Days Since Last Login Box Plot
   - Correlation Matrix Heatmap for numerical features
3. **📈 Model Comparison & Insights**: Deep-dive evaluation section featuring:
   - Model Metrics Table (comparing Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting).
   - Interactive F1-Score comparison bar chart.
   - Confusion Matrix for the best performing model.
   - Feature Importance Bar Chart showing top churn indicators.

---

## 💡 Business Recommendations Logic

Retention workflows are dynamically mapped to specific customer risk factors:
- **Month-to-month Contract**: Offer a discount incentive to switch to a 1-year or 2-year contract.
- **High Support Tickets (>3)**: Assign a customer success manager or priority support follow-up.
- **Low Usage Frequency (<10 days)**: Send onboarding emails, product tips, or usage-based nudges.
- **High Monthly Charges (>$80)**: Offer plan optimization or loyalty discount.
- **Long Inactive Period (>15 days since last login)**: Trigger a re-engagement campaign.
- **Short Tenure (<12 months)**: Provide onboarding support and early-life retention offers.
- **Low Risk / Stable Customers**: Customer appears stable. Continue regular engagement and monitor usage.

---

## 🧠 Core Machine Learning Concepts Covered

1. **Missing Value Imputation:** Handled via `SimpleImputer` using column medians for numeric inputs and modes for categorical inputs.
2. **Categorical Encoding:** Converting categories into numbers using `OneHotEncoder`.
3. **Feature Scaling:** Centering numerical distributions to mean=0 and variance=1 using `StandardScaler` to ensure coefficients are comparable.
4. **Data Leakage Prevention:** Splitting the dataset *before* fitting standardizers or imputers. We fit transformers only on the training set and transform the test set.
5. **Model Comparisons:** Benchmarking Logistic Regression (linear baseline), Decision Trees (non-linear boundary), Random Forests (bagging ensemble), and Gradient Boosting (boosting ensemble).

---

## 📈 Model Performance Benchmark

Models were trained on 2,800 records and evaluated on 700 unseen records:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.71%** | **86.79%** | 87.98% | **87.38%** | **0.947** |
| **Random Forest** | 86.14% | 84.40% | **90.16%** | 87.19% | 0.938 |
| **Gradient Boosting** | 86.00% | 85.64% | 87.98% | 86.79% | 0.938 |
| **Decision Tree** | 84.86% | 83.68% | 88.25% | 85.90% | 0.912 |

---

## ⚖️ Evaluation Metrics: Business Explanations

- **Recall (Sensitivity):** Out of all customers who actually churned, how many did the model predict?
  * *Business Impact:* High Recall is the top priority for churn prediction. Missing a churned customer (False Negative) means losing their entire future LTV.
- **Precision:** Out of all customers flagged as "likely to churn", how many actually churned?
  * *Business Impact:* High Precision ensures we don't waste customer success budgets offering discounts to customers who were going to stay anyway.
- **F1 Score:** The harmonic mean of Precision and Recall. It is the best single metric to balance both risks.
- **ROC-AUC:** Measures how well the model separates churners and non-churners.

---

## 📸 Screenshots
*(Add application screenshots showing live prediction, EDA charts, and metrics dashboard)*

---

## ⚙️ How to Setup and Run Locally

### 1. Clone the repository and navigate to folder
```bash
git clone <repository_url>
cd customer-churn-predictor
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset
Create the synthetic customer records:
```bash
python src/data_generation.py
```

### 4. Train the Models
Train and compare classifiers, saving the best one to disk:
```bash
python src/train_model.py
```

### 5. Generate Evaluation Reports & Plots
Produce performance figures and metrics documentation:
```bash
python src/evaluate_model.py
```

### 6. Launch the Interactive Dashboard
Run the Streamlit application to predict churn and test recommendations:
```bash
python -m streamlit run app/streamlit_app.py
```

---

## 🎯 Learning Outcomes
- Developed a complete machine learning lifecycle project.
- Learned how to write production-grade, modular Python script structures instead of raw notebook cells.
- Deepened understanding of the trade-offs between precision and recall for business problems.
- Created interactive dashboards that display model outcomes and actionable retention recommendations.

---

## 🚀 Future Improvements
- **Real Datasets:** Substitute synthetic data with Kaggle's Telco Churn dataset.
- **Explainable AI (SHAP):** Integrate SHAP values to explain individual predictions.
- **Customer Segmentation:** Apply unsupervised clustering to group customers by value and behavior.
- **Retention Campaign Simulator:** Build a calculator simulating return-on-investment (ROI) of discount campaigns.
- **Experiment Tracking:** Set up MLflow to log training parameters, model versions, and graphs.
- **API Endpoint:** Deploy a FastAPI model server to expose `/predict` endpoints.
- **Deploy on Streamlit Cloud:** Publicly host the app for portfolios.
