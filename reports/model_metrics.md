# Model Performance Evaluation Report

This report presents a thorough evaluation of the machine learning models trained for the **Customer Churn Predictor** project.

## 1. Executive Summary & Model Comparison

Four supervised binary classification models were evaluated on an independent test dataset (700 customer records, stratified to maintain a ~52% churn distribution).

### Comparative Performance Table
| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |   0.867143 |    0.867925 | 0.879781 |   0.873813 |  0.946746 |
| Decision Tree       |   0.848571 |    0.836788 | 0.882514 |   0.859043 |  0.911542 |
| Random Forest       |   0.861429 |    0.84399  | 0.901639 |   0.871863 |  0.93841  |
| Gradient Boosting   |   0.86     |    0.856383 | 0.879781 |   0.867925 |  0.937608 |

* **Selected Model:** `Logistic Regression` (highest F1 Score).

---

## 2. Understanding Classification Metrics (AI Engineer's Guide)

For a real business problem like customer churn, understanding what each metric means is critical:

### Why Accuracy Alone is Not Enough
Accuracy measures the overall percentage of correct predictions: 
$$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$
In cases where churn is imbalanced (e.g. only 5% of users churn), a naive model that predicts "No Churn" for everyone would be 95% accurate, but completely useless because it catches 0% of the customers leaving.

### Why Recall Matters for Churn Prediction
Recall (or Sensitivity) measures the fraction of actual churners that the model correctly identified:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
In customer retention campaigns, **Recall is the most critical metric**. If a churned customer is not detected (a False Negative), they leave the company forever. A high recall ensures that we catch as many churners as possible, even if it means raising some false alarms.

### Business Implications of Errors
* **False Positives (Type I Error):** The model predicts a customer will churn, but they were actually going to stay.
  * *Business Cost:* We might spend marketing budget offering them unnecessary discounts or incentives.
* **False Negatives (Type II Error):** The model predicts a customer will stay, but they actually churn.
  * *Business Cost:* We do nothing, and the customer leaves. The cost of losing a customer (Customer Acquisition Cost + Lifetime Value) is usually much higher than the cost of a discount campaign.

---

## 3. Confusion Matrix Breakdown
Based on the best model (`Logistic Regression`), here is the confusion matrix for the 700 test cases:

- **True Negatives (TN):** 285 (correctly predicted to stay)
- **False Positives (FP):** 49 (predicted to churn, actually stayed)
- **False Negatives (FN):** 44 (predicted to stay, actually churned)
- **True Positives (TP):** 322 (correctly predicted to churn)

---

## 4. Top Drivers of Churn (Feature Importance Insights)

According to Random Forest Gini Importance and Logistic Regression Coefficients:

1. **Contract Type (Month-to-month):** Month-to-month contracts are the strongest positive predictor of churn. These customers have no long-term commitment and are highly sensitive to billing issues or competitive offers.
2. **Support Tickets:** Customers who file multiple support tickets have a significantly higher probability of churn. This indicates unresolved customer issues or dissatisfaction.
3. **Tenure Months:** Longer tenure is a strong negative predictor of churn. Customers who stay past the first few months are highly loyal.
4. **Usage Frequency:** Lower usage frequency correlates directly with high churn risk. Customers who stop logging in or using the service are disengaged and likely to cancel.

---

## 5. Next Steps & Recommendations
- **Engage Month-to-Month customers** with incentives to transition to 1-year or 2-year contracts.
- **Proactively reach out** to any customer with 3+ support tickets in a month.
- **Create re-engagement campaigns** (e.g. email reminders, product feature guides) for customers whose usage frequency drops.
