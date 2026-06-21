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

# Premium Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        color: #ef4444;
        font-weight: bold;
        font-size: 24px;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: bold;
        font-size: 24px;
        background-color: rgba(245, 158, 11, 0.1);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .risk-low {
        color: #10b981;
        font-weight: bold;
        font-size: 24px;
        background-color: rgba(16, 185, 129, 0.1);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .recommendation-box {
        background-color: #1e3a8a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-top: 10px;
    }
    .factor-box {
        background-color: #374151;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Cache preloader and predictor
@st.cache_resource
def load_predictor():
    # Return predictor instance
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
        st.write("Modify the attributes in the left sidebar or the form below to score real-time churn risk.")

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
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("#### Prediction Output")
                
                # Check status
                prob = res['churn_probability']
                pred_label = res['prediction']
                risk = res['risk_level']
                
                if risk == "High":
                    risk_class = "risk-high"
                elif risk == "Medium":
                    risk_class = "risk-medium"
                else:
                    risk_class = "risk-low"

                st.markdown(f"""
                <div class="metric-card">
                    <h3>Prediction Outcome</h3>
                    <h2 class="{risk_class}">{pred_label}</h2>
                    <p style="margin-top:15px; font-size:16px;">Churn Probability: <b>{prob:.2%}</b></p>
                    <p style="font-size:16px;">Risk Level Category: <b>{risk}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge plot
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability %", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#3b82f6"},
                        'bgcolor': "#1f2937",
                        'borderwidth': 2,
                        'bordercolor': "#374151",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=280)
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

    # ------------------ TAB 2: INTERACTIVE EDA ------------------
    with tab2:
        st.markdown("#### Exploratory Data Analysis Dashboard")
        st.write("Analyze structural churn patterns and statistics within our customer base.")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            # 1. Churn distribution
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

            # 2. Tenure vs Churn
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

        with eda_col2:
            # 3. Monthly charges vs Churn
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

            # 4. Support Tickets vs Churn
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

        # Numerical Correlation Heatmap
        st.subheader("Numerical Features Correlation Matrix")
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
            
            # Explanations
            st.markdown("""
            > 💡 **AI Engineer Note:** 
            > * **Recall** measures how many of the actual churners were identified. High recall is critical to avoid missing churners (minimizing False Negatives).
            > * **Precision** measures the accuracy of our churn alerts. High precision prevents us from wasting retention budgets on loyal customers (minimizing False Positives).
            > * **F1 Score** balances both metrics, serving as a robust metric for model selection.
            """)

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
            ##### Key Takeaways:
            - **Contract Type (Month-to-month)** is structural risk.
            - **Support Tickets** reflect dissatisfaction.
            - **Tenure** shows customer maturity and loyalty.
            - **Usage Frequency** acts as a health indicator.
            """)
