import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_churn_data(num_samples=3000):
    """
    Generates a realistic synthetic customer churn dataset.
    
    Why is churn a binary classification problem?
    ---------------------------------------------
    Classification in machine learning is the task of predicting a discrete label or category.
    When there are only two possible outcomes (e.g., Churn vs. Retained, Yes vs. No, 1 vs. 0),
    it is called 'binary classification'. 
    
    Here, the goal is to predict if a customer will churn ('Yes') or stay ('No') based on their behavior features.
    """
    print(f"Generating {num_samples} synthetic customer records...")
    
    # Generate Customer IDs
    customer_ids = [f"CUST-{i:04d}" for i in range(1, num_samples + 1)]
    
    # Demographics
    genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.5, 0.5])
    ages = np.random.randint(18, 80, size=num_samples)
    
    # Contract & Services
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        size=num_samples, 
        p=[0.55, 0.25, 0.20]
    )
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
        size=num_samples, 
        p=[0.40, 0.20, 0.20, 0.20]
    )
    internet_services = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        size=num_samples, 
        p=[0.30, 0.50, 0.20]
    )
    plan_types = np.random.choice(
        ['Basic', 'Standard', 'Premium'], 
        size=num_samples, 
        p=[0.50, 0.35, 0.15]
    )
    has_premium_support = np.random.choice(
        ['Yes', 'No'], 
        size=num_samples, 
        p=[0.30, 0.70]
    )
    
    # Usage & Activity (highly correlated with churn)
    # Tenure: month-to-month customers tend to have lower tenure
    tenures = []
    for contract in contract_types:
        if contract == 'Month-to-month':
            tenures.append(int(np.random.exponential(scale=8))) # skew towards low tenure
        elif contract == 'One year':
            tenures.append(int(np.random.normal(loc=24, scale=6)))
        else:
            tenures.append(int(np.random.normal(loc=48, scale=12)))
    
    # Bound tenure between 1 and 72 months
    tenures = np.clip(tenures, 1, 72).astype(int)
    
    # Monthly charges: Fiber optic and Premium plans cost more
    monthly_charges = []
    for i in range(num_samples):
        base_charge = 20.0
        if internet_services[i] == 'Fiber optic':
            base_charge += 45.0
        elif internet_services[i] == 'DSL':
            base_charge += 20.0
            
        if plan_types[i] == 'Standard':
            base_charge += 15.0
        elif plan_types[i] == 'Premium':
            base_charge += 30.0
            
        if has_premium_support[i] == 'Yes':
            base_charge += 10.0
            
        # Add random fluctuation
        charge = base_charge + np.random.normal(loc=0, scale=5)
        monthly_charges.append(round(max(charge, 19.99), 2))
    
    monthly_charges = np.array(monthly_charges)
    
    # Total charges = tenure * monthly_charges + some noise
    total_charges = []
    for i in range(num_samples):
        tot = (tenures[i] * monthly_charges[i]) + np.random.normal(loc=0, scale=20)
        total_charges.append(round(max(tot, monthly_charges[i]), 2))
    
    total_charges = np.array(total_charges)
    
    # Support tickets: customers with issues make tickets
    support_tickets = []
    for i in range(num_samples):
        # Month-to-month / Fiber optic customers might have more setup/billing issues
        mean_tickets = 1.2
        if contract_types[i] == 'Month-to-month':
            mean_tickets += 1.5
        if internet_services[i] == 'Fiber optic':
            mean_tickets += 1.0
        if has_premium_support[i] == 'No':
            mean_tickets += 0.8
            
        tickets = np.random.poisson(lam=mean_tickets)
        support_tickets.append(min(tickets, 15))
        
    support_tickets = np.array(support_tickets)
    
    # Last login days (0 to 30)
    last_logins = np.random.randint(0, 31, size=num_samples)
    
    # Usage frequency (active days per month, 0 to 30)
    usage_freqs = []
    for i in range(num_samples):
        # Higher support tickets or month-to-month could correlate with lower activity (disengagement)
        mean_freq = 18
        if support_tickets[i] > 3:
            mean_freq -= 4
        if last_logins[i] > 15:
            mean_freq -= 6
            
        freq = int(np.random.normal(loc=mean_freq, scale=4))
        usage_freqs.append(np.clip(freq, 0, 30))
        
    usage_freqs = np.array(usage_freqs)
    
    # CHURN LOGIT MODEL (Probability of Churn)
    # We construct a logit function to calculate the probability of churn
    # logit = b0 + b1*X1 + b2*X2 + ...
    logits = []
    for i in range(num_samples):
        val = -1.2 # Intercept (base log-odds of remaining)
        
        # Risk factors (Positive coefficients increase churn probability)
        if contract_types[i] == 'Month-to-month':
            val += 2.0
        elif contract_types[i] == 'Two year':
            val -= 1.8
            
        if internet_services[i] == 'Fiber optic':
            val += 0.8
            if has_premium_support[i] == 'No':
                val += 0.7 # Extra risk for fiber users with no support help
                
        val += 0.45 * support_tickets[i]
        val -= 0.07 * tenures[i] # Longer tenure reduces churn
        val += 0.008 * monthly_charges[i] # Higher cost slightly increases churn
        val -= 0.12 * usage_freqs[i] # Highly active users stay
        val += 0.05 * last_logins[i] # Not logging in increases churn risk
        
        logits.append(val)
        
    logits = np.array(logits)
    
    # Convert log-odds (logits) to probabilities using sigmoid function: P = 1 / (1 + e^-z)
    probs = 1 / (1 + np.exp(-logits))
    
    # Assign churn label based on probability
    churn_labels = np.random.binomial(1, probs)
    churn = ['Yes' if c == 1 else 'No' for c in churn_labels]
    
    # Construct DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'age': ages,
        'tenure_months': tenures,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_services,
        'support_tickets': support_tickets,
        'last_login_days': last_logins,
        'usage_frequency': usage_freqs,
        'plan_type': plan_types,
        'has_premium_support': has_premium_support,
        'churn': churn
    })
    
    # Introduce small amount of random NaNs (e.g. missing total_charges for new customers)
    # This is useful for teaching preprocessing and missing value imputation!
    df.loc[df['tenure_months'] == 0, 'total_charges'] = np.nan
    # Let's also randomly set 1.5% of total_charges to NaN to simulate typical real-world missing data
    missing_idx = df.sample(frac=0.015, random_state=42).index
    df.loc[missing_idx, 'total_charges'] = np.nan
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    df = generate_churn_data(3500)
    
    # Save to CSV
    output_path = os.path.join('data', 'churn_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully and saved to: {output_path}")
    print(f"Churn rate in generated dataset: {df['churn'].value_counts(normalize=True)['Yes']:.2%}")
