import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """Loads dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No data file found at: {filepath}. Please run data_generation.py first.")
    return pd.read_csv(filepath)

def build_preprocessing_pipeline(num_cols, cat_cols):
    """
    Creates a preprocessor pipeline using Scikit-Learn.
    
    Why Scaling is Needed:
    ---------------------
    Distance-based algorithms (like Logistic Regression, SVMs, KNNs) are sensitive to feature scales.
    For instance, 'total_charges' can run into thousands, whereas 'support_tickets' is between 0 and 15.
    StandardScaler centers features around 0 with unit variance, preventing larger scale features from
    dominating the model gradients.
    
    Why Encoding is Required:
    -------------------------
    Machine learning models are mathematical equations that require numerical inputs.
    Categorical values (like 'gender' or 'contract_type') cannot be multiplied directly.
    We use One-Hot Encoding to convert categories into binary flags (0 or 1).
    """
    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Fills missing values with the median of the column
        ('scaler', StandardScaler())                 # Scales features to mean=0 and variance=1
    ])
    
    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing categories with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Encodes to binary arrays
    ])
    
    # Combine both pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    return preprocessor

def preprocess_and_split(data_path, test_size=0.2, random_state=42):
    """
    Main preprocessing execution:
    - Loads data
    - Performs label mapping
    - Splits train and test sets
    - Fits and applies preprocessing pipeline
    """
    df = load_data(data_path)
    
    # Separate features and target
    # We drop customer_id because it is unique and does not contain generalizable patterns
    X = df.drop(columns=['customer_id', 'churn'])
    
    # Encode target churn variable: Yes -> 1, No -> 0
    # Binary classification models output probability scores between 0 and 1.
    # Mapping our target column to 1 (positive case, churned) and 0 (negative case, retained) is standard practice.
    y = df['churn'].map({'Yes': 1, 'No': 0})
    
    # Identify numerical and categorical features
    numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                          'support_tickets', 'last_login_days', 'usage_frequency']
    categorical_features = ['gender', 'contract_type', 'payment_method', 'internet_service', 
                            'plan_type', 'has_premium_support']
    
    # Train / Test Split
    # ------------------
    # The golden rule of ML: Never test your model on the same data it trained on!
    # A train/test split reserves a portion of the data (e.g. 20%) to evaluate how well the model generalizes
    # to completely unseen customer behaviors, avoiding overfitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    # IMPORTANT: Prevent Data Leakage
    # We fit the scaler/imputer ONLY on X_train, and then transform both X_train and X_test.
    # Fitting on test data would leak information from the evaluation set to our training process.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after One-Hot Encoding for downstream use/interpretability
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + encoded_cat_names
    
    # Convert processed arrays back to Pandas DataFrames for convenience
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the fitted preprocessor for Streamlit/inference pipeline
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))
    print("Fitted preprocessor pipeline saved to models/preprocessor.pkl")
    
    return X_train_df, X_test_df, y_train, y_test, feature_names

if __name__ == "__main__":
    # Test execution
    X_train, X_test, y_train, y_test, features = preprocess_and_split('data/churn_data.csv')
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print("Features list:\n", features[:10], "... and more.")
