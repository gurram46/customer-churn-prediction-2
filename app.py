import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

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
        'Alabama': 0, 'Alaska': 1, 'Arizona': 2, 'Arkansas': 3, 'California': 4,
        'Colorado': 5, 'Connecticut': 6, 'Delaware': 7, 'Florida': 8, 'Georgia': 9,
        'Hawaii': 10, 'Idaho': 11, 'Illinois': 12, 'Indiana': 13, 'Iowa': 14,
        'Kansas': 15, 'Kentucky': 16, 'Louisiana': 17, 'Maine': 18, 'Maryland': 19,
        'Massachusetts': 20, 'Michigan': 21, 'Minnesota': 22, 'Mississippi': 23,
        'Missouri': 24, 'Montana': 25, 'Nebraska': 26, 'Nevada': 27, 'New Hampshire': 28,
        'New Jersey': 29, 'New Mexico': 30, 'New York': 31, 'North Carolina': 32,
        'North Dakota': 33, 'Ohio': 34, 'Oklahoma': 35, 'Oregon': 36, 'Pennsylvania': 37,
        'Rhode Island': 38, 'South Carolina': 39, 'South Dakota': 40, 'Tennessee': 41,
        'Texas': 42, 'Utah': 43, 'Vermont': 44, 'Virginia': 45, 'Washington': 46,
        'West Virginia': 47, 'Wisconsin': 48, 'Wyoming': 49
    }

    state_name = st.selectbox('State', list(state_mapping.keys()))
    state = state_mapping[state_name]
    area_code = st.number_input('Area Code', min_value=400, max_value=999, step=1)
    voice_plan = st.radio('Voice Plan', ['No', 'Yes'])
    intl_plan = st.radio('International Plan', ['No', 'Yes'])

    # Map Yes/No to 0/1
    voice_plan = 1 if voice_plan == 'Yes' else 0
    intl_plan = 1 if intl_plan == 'Yes' else 0

    no_voice_messages = st.slider('Voice Messages', min_value=0, max_value=100, step=1)
    intl_mins = st.slider('International Minutes', min_value=0.0, max_value=300.0, step=0.1)
    no_of_international_calls = st.slider('International Calls', min_value=0, max_value=100, step=1)
    intl_charge = st.slider('International Charge', min_value=0.0, max_value=50.0, step=0.1)
    customer_calls = st.slider('Customer Service Calls', min_value=0, max_value=10, step=1)
    total_mins = st.slider('Total Minutes', min_value=0.0, max_value=1000.0, step=1.0)
    total_calls = st.slider('Total Calls', min_value=0, max_value=500, step=1)
    total_charge = st.slider('Total Charge', min_value=0.0, max_value=500.0, step=1.0)

    # Collect the features into a list
    features = [state, area_code, voice_plan, no_voice_messages, intl_plan, intl_mins, no_of_international_calls, intl_charge, customer_calls, total_mins, total_calls, total_charge]

# --- Main Analysis Section ---
if st.sidebar.button('Analyze Customer'):
    churn_prob = predict(features)
    churn_verdict = "Customer will LEAVE the company" if churn_prob >= 0.5 else "Customer will STAY with the company"
    
    # Clear verdict display
    st.markdown(
        f"<h2 style='text-align: center; color: {'#ff4b4b' if churn_prob >= 0.5 else '#2dc937'};'>{churn_verdict}</h2>", 
        unsafe_allow_html=True
    )
    
    # Probability gauge
    fig = px.bar(
        x=[churn_prob], 
        y=["Risk Level"], 
        orientation='h', 
        color_discrete_sequence=['#ff6361' if churn_prob > 0.5 else '#58508d'],
        labels={'x': 'Probability', 'y': ''},
        height=200
    )
    fig.update_layout(
        showlegend=False, 
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="yellow")
    st.plotly_chart(fig, use_container_width=True)
    
    # Key factors visualization
    st.subheader("Key Churn Factors Analysis")
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Service Calls
        fig1 = px.bar(
            x=[customer_calls], 
            y=['Service Calls'], 
            orientation='h',
            color_discrete_sequence=['#ff6361' if customer_calls > 3 else '#58508d'],
            title=f"Service Calls ({customer_calls})"
        )
        fig1.update_layout(showlegend=False, xaxis_range=[0,10])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # International Charges
        fig2 = px.bar(
            x=[intl_charge], 
            y=['Intl Charges'], 
            orientation='h',
            color_discrete_sequence=['#ff6361' if intl_charge > 3 else '#58508d'],
            title=f"Intl Charges (${intl_charge:.2f})"
        )
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
st.sidebar.info(
    "ℹ️ If visualizations don't load:\n"
    "1. Check all required packages are installed\n"
    "2. Ensure model file exists\n"
    "3. Verify input values are valid"
)
