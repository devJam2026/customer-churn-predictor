import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add the src folder to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import ChurnPredictor

# Set page config
st.set_page_config(
    page_title="P0A-2 Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe probability formatting helper
def format_probability(probability):
    if probability <= 1:
        return probability * 100
    return probability

# Premium Custom CSS with enhanced contrast
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        color: #ffffff;
    }
    .metric-title {
        color: #9ca3af;
        margin: 0 0 10px 0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        margin: 0 0 15px 0;
        font-size: 26px;
        font-weight: 700;
        border-radius: 6px;
        padding: 8px 12px;
        display: inline-block;
    }
    .value-high {
        color: #ef4444;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .value-low {
        color: #10b981;
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .metric-details {
        margin-top: 15px;
        font-size: 15px;
        color: #d1d5db;
        line-height: 1.6;
    }
    .metric-details b {
        color: #ffffff;
    }
    .recommendation-box {
        background-color: #1e3a8a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-top: 10px;
        color: #ffffff;
        font-size: 15px;
    }
    .factor-box {
        background-color: #374151;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #ef4444;
        color: #ffffff;
        font-size: 15px;
    }
    .insight-card {
        background-color: #111827;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        margin-top: 10px;
        color: #e5e7eb;
        font-size: 14px;
        line-height: 1.5;
    }
    .sidebar-header {
        font-weight: 700;
        font-size: 16px;
        color: #f3f4f6;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .sidebar-value {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache preloader and predictor
@st.cache_resource
def load_predictor():
    return ChurnPredictor(models_dir=os.path.join(os.path.dirname(__file__), '..', 'models'))

@st.cache_data
def load_raw_dataset():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'churn_data.csv')
    return pd.read_csv(path)

# Initialize predictor
try:
    predictor = load_predictor()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading model resources. Have you run the training pipeline? Details: {e}")
    data_loaded = False

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/process.png", width=80)
    st.title("Project Summary")
    st.markdown("---")
    
    st.markdown('<div class="sidebar-header">📌 Project Name</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Customer Churn Predictor</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">🧭 Roadmap Module</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">P0A-2 ML Foundations</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">🧠 Problem Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Binary Classification</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">🎯 Target Variable</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Churn (Yes / No)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">🏆 Best Model</div>', unsafe_allow_html=True)
    # Check best model type dynamically if possible, else default to Random Forest/Logistic Regression
    best_model_name = "Logistic Regression"
    if data_loaded:
        best_model_name = predictor.model.__class__.__name__
        if "Classifier" in best_model_name:
            best_model_name = best_model_name.replace("Classifier", "")
        # Add spaces if CamelCase
        best_model_name = "".join([" " + c if c.isupper() and i > 0 else c for i, c in enumerate(best_model_name)]).strip()
    st.markdown(f'<div class="sidebar-value">{best_model_name}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">📈 Focus Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Precision, Recall, F1-Score, ROC-AUC</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header">📚 Learning Goal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Mastering classical ML classification workflow, pipeline building, evaluation metrics, and business recommendations.</div>', unsafe_allow_html=True)

# App Header
st.title("🔮 Customer Churn Analytics & Prediction")
st.markdown("### DevJam AI Engineer Roadmap - Machine Learning Foundations (P0A-2)")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict Customer Churn", "📊 Exploratory Data Analysis (EDA)", "📈 Model Comparison & Insights"])

if data_loaded:
    df_raw = load_raw_dataset()

    # ------------------ TAB 1: PREDICTION ------------------
    with tab1:
        st.markdown("#### Enter Customer Profile Details")
        st.write("Modify the attributes below to score real-time churn risk.")

        # Input layout using columns inside tab
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age (Years)", 18, 80, 38)
            tenure_months = st.slider("Tenure (Months)", 1, 72, 12)

        with col2:
            st.subheader("💼 Contract & Billing")
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])
            plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure_months * monthly_charges))

        with col3:
            st.subheader("⚙️ Support & Engagement")
            internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
            has_premium_support = st.selectbox("Has Premium Support?", ["Yes", "No"])
            support_tickets = st.slider("Support Tickets Filed (Month)", 0, 15, 2)
            last_login_days = st.slider("Days Since Last Login", 0, 30, 4)
            usage_frequency = st.slider("Usage Frequency (Active Days/Month)", 0, 30, 20)

        # Predict Button or Live prediction
        customer_dict = {
            'gender': gender,
            'age': age,
            'tenure_months': tenure_months,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'internet_service': internet_service,
            'support_tickets': support_tickets,
            'last_login_days': last_login_days,
            'usage_frequency': usage_frequency,
            'plan_type': plan_type,
            'has_premium_support': has_premium_support
        }

        st.markdown("---")
        if st.button("🚀 Calculate Churn Probability", use_container_width=True):
            res = predictor.predict_single(customer_dict)
            
            # Display results
            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                st.markdown("#### Prediction Output")
                
                # Check status
                prob = res['churn_probability']
                formatted_prob = format_probability(prob)
                pred_label = res['prediction']
                risk = res['risk_level']
                
                # Dynamic Styling based on outcome
                if pred_label == "Likely to Churn":
                    outcome_class = "value-high"
                else:
                    outcome_class = "value-low"

                if risk == "High":
                    risk_color = "#ef4444"
                elif risk == "Medium":
                    risk_color = "#f59e0b"
                else:
                    risk_color = "#10b981"

                # High contrast prediction outcome card
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Prediction Outcome</div>
                    <div class="metric-value {outcome_class}">{pred_label}</div>
                    <div class="metric-details">
                        <p style="margin: 6px 0;">Churn Probability: <b>{formatted_prob:.2f}%</b></p>
                        <p style="margin: 6px 0;">Risk Level Category: <b style="color: {risk_color};">{risk}</b></p>
                        <p style="margin: 6px 0; color: #9ca3af; font-size: 13px; font-style: italic;">Model Used: Random Forest Classifier</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge plot
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = formatted_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability %", 'font': {'size': 18, 'color': "white"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#3b82f6"},
                        'bgcolor': "#1f2937",
                        'borderwidth': 2,
                        'bordercolor': "#374151",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=240, margin=dict(t=30, b=0, l=10, r=10))
                
                # Put gauge card in side-by-side columns inside res_col1
                pred_col1, pred_col2 = st.columns([1, 1])
                with pred_col1:
                    # Rerender prediction card inside the column layout
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Prediction Outcome</div>
                        <div class="metric-value {outcome_class}">{pred_label}</div>
                        <div class="metric-details">
                            <p style="margin: 6px 0;">Churn Probability: <b>{formatted_prob:.2f}%</b></p>
                            <p style="margin: 6px 0;">Risk Level Category: <b style="color: {risk_color};">{risk}</b></p>
                            <p style="margin: 6px 0; color: #9ca3af; font-size: 13px; font-style: italic;">Model Used: Random Forest Classifier</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with pred_col2:
                    st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.markdown("#### Key Churn Drivers Detected")
                if res['top_factors']:
                    for factor in res['top_factors']:
                        st.markdown(f"<div class='factor-box'>⚠️ {factor}</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ No critical risk drivers detected. Customer exhibits healthy usage behaviors.")

                st.markdown("#### Actionable Business Recommendations")
                for rec in res['recommendations']:
                    st.markdown(f"""
                    <div class="recommendation-box">
                        💡 <b>Recommendation:</b> {rec}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("ℹ️ How this prediction works"):
            st.markdown("""
            ##### The Classification Process Explained:
            * **Binary Classification Task:** Churn prediction is a classical classification problem where we assign each customer into one of two groups: **Churn** (1 / Yes) or **Retained** (0 / No).
            * **Input Features & Encoding:** The details you entered are encoded into a numerical format, matching the shape expected by the model. Numerical fields are scaled (centered and standardized), while categorical attributes are converted to 0/1 indicator flags using **One-Hot Encoding**.
            * **Output Probability Score:** The core machine learning model (e.g., Random Forest) evaluates the features and outputs a raw probability score representing the likelihood of the customer churning (ranging from 0.0 to 1.0).
            * **Risk Level Calibration:**
              * **Low Risk:** Churn probability is less than **30%**.
              * **Medium Risk:** Churn probability is between **30% and 59.9%**.
              * **High Risk:** Churn probability is **60% or higher**.
            * **Decision Support Tool:** This output is designed to help Customer Success and Marketing teams identify customers who require proactive retention efforts. It is not an absolute certainty, but a statistical prediction based on training patterns.
            """)

    # ------------------ TAB 2: INTERACTIVE EDA ------------------
    with tab2:
        st.markdown("#### Exploratory Data Analysis Dashboard")
        st.write("Analyze structural churn patterns and statistics within our customer base.")

        # Row 1
        eda_row1_col1, eda_row1_col2 = st.columns(2)
        with eda_row1_col1:
            fig_churn = px.pie(
                df_raw, 
                names='churn', 
                title='Overall Churn Distribution',
                color='churn',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
                hole=0.4
            )
            fig_churn.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_churn, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                📊 <b>Business Insight:</b> The customer base has a stratified distribution. Identifying patterns here helps isolate characteristics of customers who remain loyal versus those who leave.
            </div>
            """, unsafe_allow_html=True)

        with eda_row1_col2:
            fig_contract = px.histogram(
                df_raw,
                x='contract_type',
                color='churn',
                barmode='group',
                title='Churn Count by Contract Type',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
            )
            fig_contract.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_contract, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                💳 <b>Business Insight:</b> Customers on Month-to-month contracts have the highest churn rates. Offering long-term incentives can transition these users to low-churn yearly options.
            </div>
            """, unsafe_allow_html=True)

        # Row 2
        eda_row2_col1, eda_row2_col2 = st.columns(2)
        with eda_row2_col1:
            fig_tenure = px.box(
                df_raw,
                x='churn',
                y='tenure_months',
                color='churn',
                title='Tenure Distribution by Churn Status',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
            )
            fig_tenure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_tenure, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                ⏳ <b>Business Insight:</b> Churned customers usually have much lower tenure, indicating that customer onboarding and first-year retention are critical lifecycle stages.
            </div>
            """, unsafe_allow_html=True)

        with eda_row2_col2:
            fig_charges = px.histogram(
                df_raw,
                x='monthly_charges',
                color='churn',
                barmode='overlay',
                title='Monthly Charges Distribution by Churn Status',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
                marginal='box'
            )
            fig_charges.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_charges, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                💰 <b>Business Insight:</b> High monthly charges correlate with increased churn. Pricier packages raise customer expectations, making them sensitive to price vs value perceptions.
            </div>
            """, unsafe_allow_html=True)

        # Row 3
        eda_row3_col1, eda_row3_col2 = st.columns(2)
        with eda_row3_col1:
            ticket_churn = df_raw.groupby(['support_tickets', 'churn']).size().reset_index(name='count')
            fig_tickets = px.bar(
                ticket_churn,
                x='support_tickets',
                y='count',
                color='churn',
                title='Support Tickets Filed vs Churn Count',
                barmode='group',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
            )
            fig_tickets.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_tickets, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                🎫 <b>Business Insight:</b> Filing 3 or more support tickets indicates high risk. Service quality follow-ups are highly recommended for customers reaching this threshold.
            </div>
            """, unsafe_allow_html=True)

        with eda_row3_col2:
            fig_usage = px.box(
                df_raw,
                x='churn',
                y='usage_frequency',
                color='churn',
                title='Usage Frequency (Active Days/Month) by Churn Status',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
            )
            fig_usage.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_usage, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                ⚡ <b>Business Insight:</b> Active usage frequency is a strong health metric. Low engagement (fewer than 10 active days per month) indicates declining customer interest.
            </div>
            """, unsafe_allow_html=True)

        # Row 4
        eda_row4_col1, eda_row4_col2 = st.columns(2)
        with eda_row4_col1:
            fig_login = px.box(
                df_raw,
                x='churn',
                y='last_login_days',
                color='churn',
                title='Days Since Last Login by Churn Status',
                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'}
            )
            fig_login.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_login, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                🚪 <b>Business Insight:</b> Customers who have not logged in for 15+ days are disengaged, suggesting that auto-churn risk increases rapidly with inactivity.
            </div>
            """, unsafe_allow_html=True)

        with eda_row4_col2:
            # Numerical Correlation Heatmap
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
            corr = df_raw[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                title="Correlation Heatmap (Numeric Features)",
                color_continuous_scale='RdBu_r'
            )
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                🔗 <b>Business Insight:</b> Correlation matrix reveals high collinear relationships between tenure and total charges, and negative correlation between usage frequency and last login.
            </div>
            """, unsafe_allow_html=True)

    # ------------------ TAB 3: MODEL INSIGHTS ------------------
    with tab3:
        st.markdown("#### Classifier Model Performance & Evaluation")
        
        # Load comparison CSV
        comp_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_comparison.csv')
        if os.path.exists(comp_path):
            comp_df = pd.read_csv(comp_path)
            if 'Unnamed: 0' in comp_df.columns:
                comp_df = comp_df.rename(columns={'Unnamed: 0': 'Model'})
            
            # Show Metrics table
            st.markdown("##### Test Set Metrics Comparison")
            st.dataframe(comp_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']))
            
            # Bar chart comparing F1 score
            fig_f1 = px.bar(
                comp_df,
                x='Model',
                y='F1 Score',
                color='F1 Score',
                text_auto='.3f',
                title='F1-Score Comparison Across Models',
                color_continuous_scale='Blues'
            )
            fig_f1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_f1, use_container_width=True)

            # Educational Box
            st.markdown("""
            <div class="insight-card">
                🧠 <b>Metric Interpretations for Classifications:</b>
                <ul>
                    <li><b>Accuracy:</b> The ratio of correctly predicted instances to total instances. While intuitive, it can be deceptive for imbalanced datasets.</li>
                    <li><b>Recall (Sensitivity):</b> Out of all actual churned customers, how many did we predict correctly? In churn prediction, **Recall is paramount** because failing to detect a customer who leaves (False Negative) has high business costs.</li>
                    <li><b>Precision:</b> Out of all customers flagged as churn risk, how many actually churn? High precision avoids wasting incentive budgets on loyal customers (minimizing False Positives).</li>
                    <li><b>F1-Score:</b> The harmonic mean of Precision and Recall. It balances both metrics and serves as a reliable score for final model selection.</li>
                    <li><b>ROC-AUC:</b> Represents the model's ability to distinguish between churners and non-churners. A score closer to 1.0 indicates superior classifier separation.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Display static evaluation plots
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            st.markdown("##### Confusion Matrix (Best Model)")
            cm_img_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'confusion_matrix.png')
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Ensure evaluation script was run.")

            st.markdown("##### ROC Curve (Model Comparison)")
            roc_img_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'roc_curve.png')
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, use_container_width=True)
            else:
                st.info("ROC Curve image not found.")

        with eval_col2:
            st.markdown("##### Feature Importance (Random Forest)")
            fi_img_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'feature_importance.png')
            if os.path.exists(fi_img_path):
                st.image(fi_img_path, use_container_width=True)
            else:
                st.info("Feature importance image not found.")
                
            st.markdown("""
            ##### Feature Importance Breakdown & Explanations:
            * **Contract Type (Month-to-month):** This feature indicates whether a customer is bound to a contract. Customers without long-term commitment (month-to-month) exhibit significantly higher churn rates as leaving incurs no contractual friction.
            * **Support Tickets:** A high number of customer support tickets is an explicit proxy for dissatisfaction or unresolved bugs/billing issues.
            * **Tenure:** Customers who stay longer build maturity, adapt to the product ecosystem, and are inherently less likely to cancel.
            * **Usage Frequency:** Low engagement or drop in active days indicates weak product utility perception, signaling high churn risk.
            """)
