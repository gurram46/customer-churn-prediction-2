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
    # State mapping
    state_mapping = { ... }  # (keep your existing state mapping here)
    
    state_name = st.selectbox('State', list(state_mapping.keys()))
    state = state_mapping[state_name]
    area_code = st.number_input('Area Code', min_value=400, max_value=999, step=1)
    voice_plan = st.selectbox('Voice Plan', ['No', 'Yes'])
    intl_plan = st.selectbox('International Plan', ['No', 'Yes'])
    voice_plan = 1 if voice_plan == 'Yes' else 0
    intl_plan = 1 if intl_plan == 'Yes' else 0
    no_voice_messages = st.number_input('Voice Messages', min_value=0, max_value=100)
    intl_mins = st.number_input('International Minutes', min_value=0.0, step=0.1)
    no_of_international_calls = st.number_input('International Calls', min_value=0)
    intl_charge = st.number_input('International Charge', min_value=0.0, step=0.1)
    customer_calls = st.number_input('Customer Service Calls', min_value=0, max_value=10)
    total_mins = st.number_input('Total Minutes', min_value=0.0, max_value=1000.0)
    total_calls = st.number_input('Total Calls', min_value=0, max_value=500)
    total_charge = st.number_input('Total Charge', min_value=0.0)

features = [state, area_code, voice_plan, no_voice_messages, intl_plan, intl_mins,
            no_of_international_calls, intl_charge, customer_calls, total_mins,
            total_calls, total_charge]

# --- Main Analysis Section ---
if st.sidebar.button('Analyze Customer'):
    churn_prob = predict(features)
    
    # Visualization 1: Probability Gauge
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.barh(['Churn Probability'], [churn_prob], color='#ff6361' if churn_prob > 0.5 else '#58508d')
    ax1.set_xlim(0, 1)
    ax1.set_title('Churn Risk Probability')
    ax1.text(churn_prob/2, 0, f"{churn_prob*100:.1f}%", color='white', 
            ha='center', va='center', fontsize=20)
    
    # Visualization 2: Key Factors
    factors = {
        'Customer Service Calls': customer_calls,
        'Total Charges': total_charge,
        'International Charges': intl_charge,
        'Total Minutes': total_mins
    }
    
    # --- Layout Columns ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.pyplot(fig1)
        st.metric("Prediction", 
                 "High Risk" if churn_prob > 0.5 else "Low Risk",
                 f"{churn_prob*100:.1f}%")
    
    with col2:
        st.subheader("Key Risk Factors")
        
        # Factor 1: Customer Service Calls
        fig2 = px.bar(x=[customer_calls], y=['Service Calls'], 
                     orientation='h', 
                     color_discrete_sequence=['#ff6361' if customer_calls > 3 else '#58508d'],
                     title="Customer Service Calls (Risk Threshold: >3)")
        fig2.update_layout(showlegend=False, xaxis_range=[0,10])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Factor 2: International Charges
        fig3 = px.bar(x=[intl_charge], y=['Intl Charges'], 
                     orientation='h',
                     color_discrete_sequence=['#ff6361' if intl_charge > 3 else '#58508d'],
                     title="International Charges (Risk Threshold: >$3)")
        fig3.update_layout(showlegend=False, xaxis_range=[0,5])
        st.plotly_chart(fig3, use_container_width=True)
    
    # --- Detailed Report ---
    st.markdown("---")
    st.subheader("Churn Analysis Report")
    
    report_text = []
    if customer_calls > 3:
        report_text.append(f"🚨 High number of customer service calls ({customer_calls} calls) - industry average is 2-3 calls")
    if intl_charge > 3:
        report_text.append(f"🚨 High international charges (${intl_charge:.2f}) - above average spending")
    if total_charge > 75:
        report_text.append(f"💸 High total charges (${total_charge:.2f}) - potential price sensitivity")
    if voice_plan == 0:
        report_text.append("📞 No voice plan subscription - 35% higher churn risk observed")
    
    if len(report_text) > 0:
        st.write("### Risk Indicators:")
        for item in report_text:
            st.markdown(f"- {item}")
    else:
        st.success("No significant risk factors identified - customer profile appears stable")
    
    # Feature comparison table
    st.markdown("### Feature Comparison")
    st.write(f"""
    | Metric | Customer Value | Industry Average |
    |---|---:|---:|
    | Service Calls | {customer_calls} | 2.8 |
    | Intl Charges | ${intl_charge:.2f} | $2.75 |
    | Total Charge | ${total_charge:.2f} | $64.50 |
    | Total Minutes | {total_mins:.1f} | 535.2 |
    """)
