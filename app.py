import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = joblib.load('best_churn_model1.pkl')

def predict_proba(features):
    return model.predict_proba([features])[0]

st.title('Customer Churn Prediction')

# Layout: three columns
left_col, middle_col, right_col = st.columns([2, 3, 1])

with left_col:
    st.header("Input Features")
    
    # Create input fields for each feature with toggles
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

if st.button('Predict'):
    probabilities = predict_proba(features)
    result = "Churn customer will quit the company" if probabilities[1] > 0.5 else "No Churn customer will not quit the company"

    with middle_col:
        st.header("Prediction Results")
        st.write(f'Prediction: {result}')
        st.write(f'Probability of Churn: {probabilities[1]*100:.2f}%')
        st.write(f'Probability of No Churn: {probabilities[0]*100:.2f}%')

        # Display prediction result as a pie chart with some Yes and No in the result
        labels = ['Churn', 'No Churn']
        sizes = [probabilities[1], probabilities[0]]
        colors = ['red', 'green']
        explode = (0.1, 0)  # explode the churn slice

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)

        # Detailed Analysis: Most deciding factors
        st.header("Detailed Analysis")
        st.write(f'You selected: State - {state_name}, Area Code - {area_code}, Voice Plan - {"Yes" if voice_plan else "No"}, International Plan - {"Yes" if intl_plan else "No"}')

        # Factors affecting churn comparison
        st.subheader("Factors Affecting Churn")
        factors = ['International Plan', 'Voice Plan', 'Total Calls', 'Customer Service Calls', 'Total Charge']
        values = [intl_plan, voice_plan, total_calls, customer_calls, total_charge]
        mean_values = [0.6, 0.4, 300, 3, 200]  # Example mean values for the sake of comparison

        fig2, ax2 = plt.subplots()
        bar_width = 0.35
        index = np.arange(len(factors))

        bar1 = ax2.bar(index, values, bar_width, label='Customer')
        bar2 = ax2.bar(index + bar_width, mean_values, bar_width, label='Average')

        ax2.set_xlabel('Factors')
        ax2.set_ylabel('Values')
        ax2.set_title('Factors Comparison')
        ax2.set_xticks(index + bar_width / 2)
        ax2.set_xticklabels(factors)
        ax2.legend()

        st.pyplot(fig2)

        st.write("The above comparison shows that higher customer service calls, total charge, and international charges are key factors indicating a higher chance
