import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import preprocessing pipeline
from preprocessing import preprocess_and_split

def train_and_compare_models(data_path):
    """
    Trains and compares four classical Machine Learning classification models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting
    """
    # 1. Load and preprocess the data
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(data_path)
    
    # 2. Define the models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=4)
    }
    
    results = {}
    trained_models = {}
    
    print("\n--- Model Training & Comparison ---")
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of Churn (class 1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"{name} metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f} (Crucial for churn!)")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}\n")
        
    # Convert results to DataFrame for comparison
    results_df = pd.DataFrame(results).T
    
    # 3. Select the best model based on F1 Score
    # F1 score is a harmonic mean of Precision and Recall, which is balanced for binary classification.
    # In business churn, Recall is very important, but we also want to avoid targeting everybody (low Precision).
    # Thus, F1 or Recall are common choice metrics. Let's use F1 Score as the selection metric.
    best_model_name = results_df['F1 Score'].idxmax()
    best_model = trained_models[best_model_name]
    
    print(f"Best model based on F1 Score: {best_model_name} ({results_df.loc[best_model_name, 'F1 Score']:.4f})")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model.pkl')
    print("Saved best model to models/churn_model.pkl")
    
    # Also save metadata for report generation
    results_df.to_csv('models/model_comparison.csv')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save all models temporarily so evaluate_model can compile comparison visualizations
    for name, model in trained_models.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(model, f'models/{clean_name}_model.pkl')
        
    return results_df

if __name__ == "__main__":
    train_and_compare_models('data/churn_data.csv')
