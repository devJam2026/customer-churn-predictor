import os
import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self, models_dir='models'):
        self.preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        self.model_path = os.path.join(models_dir, 'churn_model.pkl')
        self.feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
        
        # Load assets
        if not os.path.exists(self.preprocessor_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError("Model assets not found. Please train models first using train_model.py.")
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.feature_names = joblib.load(self.feature_names_path)
        
    def predict_single(self, customer_data):
        """
        Predicts churn risk for a single customer.
        
        Parameters:
        -----------
        customer_data : dict
            A dictionary containing customer features.
            Example:
            {
                'gender': 'Male',
                'age': 35,
                'tenure_months': 12,
                'monthly_charges': 70.0,
                'total_charges': 840.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic',
                'support_tickets': 3,
                'last_login_days': 5,
                'usage_frequency': 15,
                'plan_type': 'Standard',
                'has_premium_support': 'No'
            }
            
        Returns:
        --------
        dict
            Prediction probability, label, risk level, contributing factors, and recommendations.
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Process the features using the saved ColumnTransformer
        processed_data = self.preprocessor.transform(df)
        processed_df = pd.DataFrame(processed_data, columns=self.feature_names)
        
        # Make predictions
        prob = self.model.predict_proba(processed_df)[0, 1]
        prediction_label = "Likely to Churn" if prob >= 0.50 else "Not Likely to Churn"
        
        # Define risk level
        # Convert to percentage-based check for consistency (probability * 100)
        prob_pct = prob * 100
        if prob_pct < 30.0:
            risk_level = "Low"
        elif prob_pct < 60.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        # Determine top factors contributing to churn
        contributing_factors = []
        recommendations = []
        
        # Check specific indicators
        if customer_data.get('contract_type') == 'Month-to-month':
            contributing_factors.append("Month-to-month contract type increases churn risk because the customer has low commitment.")
            recommendations.append("Offer a discount incentive to switch to a 1-year or 2-year contract.")
            
        if customer_data.get('support_tickets', 0) > 3:
            contributing_factors.append("High number of support tickets may indicate dissatisfaction or unresolved product issues.")
            recommendations.append("Assign a customer success manager or priority support follow-up.")
            
        if customer_data.get('usage_frequency', 30) < 10:
            contributing_factors.append("Low usage frequency suggests weak product engagement.")
            recommendations.append("Send onboarding emails, product tips, or usage-based nudges.")
            
        if customer_data.get('tenure_months', 0) < 12:
            contributing_factors.append("Short tenure customers are more likely to churn before becoming loyal.")
            recommendations.append("Provide onboarding support and early-life retention offers.")
            
        if customer_data.get('monthly_charges', 0) > 80:
            contributing_factors.append("High monthly charges may increase price sensitivity.")
            recommendations.append("Offer plan optimization or loyalty discount.")
            
        if customer_data.get('last_login_days', 0) > 15:
            contributing_factors.append("Long time since last login suggests declining engagement.")
            recommendations.append("Trigger a re-engagement campaign.")
            
        if customer_data.get('has_premium_support') == 'No':
            contributing_factors.append("Lack of premium support may increase churn risk for customers needing assistance.")
            recommendations.append("Upsell or offer a trial of Premium Support to reduce network issue frustration.")

        # Default recommendation if risk is low or recommendations are empty
        if not recommendations or risk_level == "Low":
            # If there are no recommendations, or if the risk is low, we ensure we have the stable message.
            if risk_level == "Low":
                recommendations = ["Customer appears stable. Continue regular engagement and monitor usage."]
            elif not recommendations:
                recommendations.append("Customer appears stable. Continue regular engagement and monitor usage.")
            
        return {
            'churn_probability': float(prob),
            'prediction': prediction_label,
            'risk_level': risk_level,
            'top_factors': contributing_factors,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    # Test prediction
    predictor = ChurnPredictor()
    sample_customer = {
        'gender': 'Female',
        'age': 45,
        'tenure_months': 3,
        'monthly_charges': 95.0,
        'total_charges': 285.0,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic',
        'support_tickets': 4,
        'last_login_days': 12,
        'usage_frequency': 8,
        'plan_type': 'Standard',
        'has_premium_support': 'No'
    }
    
    result = predictor.predict_single(sample_customer)
    print("\n--- Sample Prediction ---")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print("\nFactors:")
    for f in result['top_factors']:
        print(f" - {f}")
    print("\nRecommendations:")
    for r in result['recommendations']:
        print(f" - {r}")
