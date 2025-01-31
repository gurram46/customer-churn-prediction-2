import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Load the trained model
model = joblib.load('best_churn_model1.pkl')

def predict(features):
    proba = model.predict_proba([features])[0]
    return proba[1]  # Probability of churn

st.title('Customer Churn Prediction Analysis')

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Customer Details")
    state_mapping = { 
        # ... (keep your existing state mapping here)
    }
    
    # Input fields (keep your existing input fields)
    # ...

# --- Main Analysis Section ---
if st.sidebar.button('Analyze Customer'):
    churn_prob = predict(features)
    churn_verdict = "Customer will LEAVE the company" if churn_prob >= 0.5 else "Customer will STAY with the company"
    
    # Clear verdict display
    st.markdown(f"<h2 style='text-align: center; color: {'#ff4b4b' if churn_prob >= 0.5 else '#2dc937'};'>{churn_verdict}</h2>", 
                unsafe_allow_html=True)
    
    # Probability gauge
    fig = px.bar(x=[churn_prob], y=["Risk Level"], 
                 orientation='h', 
                 color_discrete_sequence=['#ff6361' if churn_prob > 0.5 else '#58508d'],
                 labels={'x': 'Probability', 'y': ''},
                 height=200)
    fig.update_layout(showlegend=False, 
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    margin=dict(l=20, r=20, t=40, b=20))
    fig.add_vline(x=0.5, line_dash="dot", line_color="yellow")
    st.plotly_chart(fig, use_container_width=True)
    
    # Key factors visualization
    st.subheader("Key Churn Factors Analysis")
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Service Calls
        fig1 = px.bar(x=[customer_calls], y=['Service Calls'], 
                     orientation='h',
                     color_discrete_sequence=['#ff6361' if customer_calls > 3 else '#58508d'],
                     title=f"Service Calls ({customer_calls})")
        fig1.update_layout(showlegend=False, xaxis_range=[0,10])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # International Charges
        fig2 = px.bar(x=[intl_charge], y=['Intl Charges'], 
                     orientation='h',
                     color_discrete_sequence=['#ff6361' if intl_charge > 3 else '#58508d'],
                     title=f"Intl Charges (${intl_charge:.2f})")
        fig2.update_layout(showlegend=False, xaxis_range=[0,5])
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed report
    st.markdown("---")
    st.subheader("Churn Analysis Report")
    
    # Risk factors analysis
    risk_factors = []
    if customer_calls > 3:
        risk_factors.append(("High Service Calls", f"{customer_calls} calls (Industry avg: 2.8)"))
    if intl_charge > 3:
        risk_factors.append(("High Intl Charges", f"${intl_charge:.2f} (Avg: $2.75)"))
    if total_charge > 75:
        risk_factors.append(("High Total Charges", f"${total_charge:.2f} (Avg: $64.50)"))
    if voice_plan == 0:
        risk_factors.append(("No Voice Plan", "35% higher churn risk"))

    if risk_factors:
        st.write("### 🚨 Risk Factors Detected:")
        for factor, detail in risk_factors:
            st.markdown(f"🔴 **{factor}**: {detail}")
    else:
        st.success("### ✅ No Significant Risk Factors Detected")

    # Feature comparison table
    st.markdown("### 📊 Feature Comparison")
    st.table(pd.DataFrame({
        'Metric': ['Service Calls', 'Intl Charges', 'Total Charge', 'Total Minutes'],
        'Customer Value': [customer_calls, f"${intl_charge:.2f}", f"${total_charge:.2f}", f"{total_mins:.1f}"],
        'Industry Average': ['2.8', '$2.75', '$64.50', '535.2']
    }))

# --- Troubleshooting Note ---
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ If visualizations don't load:\n1. Check all required packages are installed\n2. Ensure model file exists\n3. Verify input values are valid")
