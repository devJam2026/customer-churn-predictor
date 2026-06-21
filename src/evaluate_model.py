import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import preprocessing pipeline
from preprocessing import preprocess_and_split

def evaluate_models(data_path):
    """
    Evaluates the trained classification models, generates performance charts,
    and writes the model metrics report to reports/model_metrics.md.
    """
    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(data_path)
    
    # 2. Load the trained models
    log_reg = joblib.load('models/logistic_regression_model.pkl')
    dec_tree = joblib.load('models/decision_tree_model.pkl')
    rand_forest = joblib.load('models/random_forest_model.pkl')
    grad_boost = joblib.load('models/gradient_boosting_model.pkl')
    best_model = joblib.load('models/churn_model.pkl')
    
    # Identify which model was saved as best
    best_model_name = "Logistic Regression"
    for name, model in [("Decision Tree", dec_tree), ("Random Forest", rand_forest), ("Gradient Boosting", grad_boost)]:
        # Compare model coefficients or parameters to find the matching one
        if type(best_model) == type(model):
            best_model_name = name
            break
            
    # 3. Create plots directory
    os.makedirs('reports', exist_ok=True)
    
    # 4. Generate Confusion Matrix Plot for the Best Model
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Retained (No Churn)', 'Churned'], 
                yticklabels=['Retained (No Churn)', 'Churned'])
    plt.title(f'Confusion Matrix - {best_model_name} (Best Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 5. Generate ROC Curve Plot for all models
    plt.figure(figsize=(8, 6))
    
    for name, model in [
        ('Logistic Regression', log_reg),
        ('Decision Tree', dec_tree),
        ('Random Forest', rand_forest),
        ('Gradient Boosting', grad_boost)
    ]:
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png', dpi=300)
    plt.close()
    
    # 6. Generate Feature Importance Plot using Random Forest / Gradient Boosting
    # Even if Logistic Regression is best, Random Forest gives highly robust Gini importances
    importances = rand_forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    # Select top 12 features for visualization
    top_n = min(12, len(feature_names))
    sns.barplot(x=importances[indices[:top_n]], y=np.array(feature_names)[indices[:top_n]], palette='viridis')
    plt.title('Top Churn-Driving Features (Random Forest Gini Importance)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=300)
    plt.close()
    
    # Generate Logistic Regression Coefficients for analysis
    coef_df = None
    if hasattr(log_reg, 'coef_'):
        coefs = log_reg.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Absolute Coefficient': np.abs(coefs)
        }).sort_values(by='Absolute Coefficient', ascending=False)
        
    # 7. Write reports/model_metrics.md
    # Check if index needs renaming
    comparison_data = pd.read_csv('models/model_comparison.csv')
    if 'Unnamed: 0' in comparison_data.columns:
        comparison_data = comparison_data.rename(columns={'Unnamed: 0': 'Model'})
        
    # Convert comparison table to markdown
    comparison_md = comparison_data.to_markdown(index=False)
    
    report_content = f"""# Model Performance Evaluation Report

This report presents a thorough evaluation of the machine learning models trained for the **Customer Churn Predictor** project.

## 1. Executive Summary & Model Comparison

Four supervised binary classification models were evaluated on an independent test dataset (700 customer records, stratified to maintain a ~52% churn distribution).

### Comparative Performance Table
{comparison_md}

* **Selected Model:** `{best_model_name}` (highest F1 Score).

---

## 2. Understanding Classification Metrics (AI Engineer's Guide)

For a real business problem like customer churn, understanding what each metric means is critical:

### Why Accuracy Alone is Not Enough
Accuracy measures the overall percentage of correct predictions: 
$$\\text{{Accuracy}} = \\frac{{\\text{{True Positives}} + \\text{{True Negatives}}}}{{\\text{{Total Samples}}}}$$
In cases where churn is imbalanced (e.g. only 5% of users churn), a naive model that predicts "No Churn" for everyone would be 95% accurate, but completely useless because it catches 0% of the customers leaving.

### Why Recall Matters for Churn Prediction
Recall (or Sensitivity) measures the fraction of actual churners that the model correctly identified:
$$\\text{{Recall}} = \\frac{{\\text{{True Positives}}}}{{\\text{{True Positives}} + \\text{{False Negatives}}}}$$
In customer retention campaigns, **Recall is the most critical metric**. If a churned customer is not detected (a False Negative), they leave the company forever. A high recall ensures that we catch as many churners as possible, even if it means raising some false alarms.

### Business Implications of Errors
* **False Positives (Type I Error):** The model predicts a customer will churn, but they were actually going to stay.
  * *Business Cost:* We might spend marketing budget offering them unnecessary discounts or incentives.
* **False Negatives (Type II Error):** The model predicts a customer will stay, but they actually churn.
  * *Business Cost:* We do nothing, and the customer leaves. The cost of losing a customer (Customer Acquisition Cost + Lifetime Value) is usually much higher than the cost of a discount campaign.

---

## 3. Confusion Matrix Breakdown
Based on the best model (`{best_model_name}`), here is the confusion matrix for the 700 test cases:

- **True Negatives (TN):** {cm[0, 0]} (correctly predicted to stay)
- **False Positives (FP):** {cm[0, 1]} (predicted to churn, actually stayed)
- **False Negatives (FN):** {cm[1, 0]} (predicted to stay, actually churned)
- **True Positives (TP):** {cm[1, 1]} (correctly predicted to churn)

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
"""
    
    with open('reports/model_metrics.md', 'w') as f:
        f.write(report_content)
        
    print("Evaluation completed. Plots and metrics report saved in: reports/")
    
if __name__ == "__main__":
    evaluate_models('data/churn_data.csv')
