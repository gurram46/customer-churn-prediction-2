import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
data = pd.read_csv(r"data (1).csv")



model = joblib.load('best_churn_model1.pkl')  # Example filename


def predict(features):
    return model.predict([features])[0]

st.title('Customer Churn Prediction')

# Create feature selection on the right side
with st.sidebar:
    st.header('Select Features')
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
    voice_plan = st.selectbox('Voice Plan', ['No', 'Yes'])
    intl_plan = st.selectbox('International Plan', ['No', 'Yes'])

    # Map Yes/No to 0/1
    voice_plan = 1 if voice_plan == 'Yes' else 0
    intl_plan = 1 if intl_plan == 'Yes' else 0

    no_voice_messages = st.number_input('Voice Messages', min_value=0, max_value=100, step=1)
    intl_mins = st.number_input('International Minutes', min_value=0.0, step=0.1)
    no_of_international_calls = st.number_input('International Calls', min_value=0, max_value=100, step=1)
    intl_charge = st.number_input('International Charge', min_value=0.0, step=0.1)
    customer_calls = st.number_input('Customer Service Calls', min_value=0, max_value=10, step=1)
    total_mins = st.number_input('Total Minutes', min_value=0.0, max_value=1000.0, step=1.0)
    total_calls = st.number_input('Total Calls', min_value=0, max_value=500, step=1)
    total_charge = st.number_input('Total Charge', min_value=0.0, step=1.0)

# Collect the features into a list
features = [state, area_code, voice_plan, no_voice_messages, intl_plan, intl_mins, no_of_international_calls, intl_charge, customer_calls, total_mins, total_calls, total_charge]

# Print the features being passed to the model for debugging
st.write("Features being passed to the model:", features)

# Predict button
if st.button('Predict'):
    result = predict(features)
    st.write(f'Prediction: {"Churn customer will quit the company" if result == 1 else "No Churn customer will not quit the company"}')

    # Filter the dataset based on the selected state number
    state_data = data[data['state'] == state]

    # Show some visualizations
    st.subheader(f'Visualizations for {state_name}')
    
    # Plot 1: Histogram of Total Charges
    st.write("**Distribution of Total Charges**")
    st.write("This histogram shows the distribution of total charges for customers in the selected state. High charges might lead to higher churn rates.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(state_data['total_charge'], ax=ax, kde=True)
    ax.set_title('Distribution of Total Charges')
    st.pyplot(fig)

    # Plot 2: Countplot of Voice Plan
    st.write("**Count of Voice Plans**")
    st.write("This countplot shows the number of customers with and without a voice plan. Customers without a voice plan might have different churn rates.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x='voice.plan', data=state_data, ax=ax)
    ax.set_title('Count of Voice Plans')
    st.pyplot(fig)

    # Plot 3: Boxplot of Total Minutes by Churn
    st.write("**Total Minutes by Churn**")
    st.write("This boxplot compares the total minutes used by customers who churned versus those who did not. Differences in usage patterns might indicate churn risk.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x='churn', y='total_mins', data=state_data, ax=ax)
    ax.set_title('Total Minutes by Churn')
    st.pyplot(fig)

    # Plot 4: Countplot of International Plan
    st.write("**Count of International Plans**")
    st.write("This countplot shows the number of customers with and without an international plan. Lack of an international plan might be a factor in churn for customers who make international calls.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x='intl.plan', data=state_data, ax=ax)
    ax.set_title('Count of International Plans')
    st.pyplot(fig)

    # Plot 5: Pie Chart of Churn vs. Non-Churn
    st.write("**Churn vs. Non-Churn Customers**")
    st.write("This pie chart shows the proportion of customers who churned versus those who didn't. It helps to visualize the churn rate among the customers in the selected state.")
    fig, ax = plt.subplots(figsize=(9, 6))
    churn_counts = state_data['churn'].value_counts()
    ax.pie(churn_counts, labels=['Non-Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    ax.axis('equal')
    ax.set_title('Churn vs. Non-Churn Customers')
    st.pyplot(fig)

    # Plot 6: Bar Plot of Customer Service Calls
    st.write("**Customer Service Calls vs. Total Charge**")
    st.write("This bar plot shows the relationship between the number of customer service calls and total charges. Frequent customer service calls might indicate dissatisfaction leading to churn.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x='customer.calls', y='total_charge', data=state_data, ax=ax)
    ax.set_title('Customer Service Calls vs. Total Charge')
    st.pyplot(fig)

    # Detailed report with visualizations
    st.subheader('Detailed Report')
    churn_reasons = []

    if customer_calls > 4:
        churn_reasons.append("High number of customer service calls")
    if total_charge > state_data['total_charge'].mean():
        churn_reasons.append("Higher than average total charges")
    if not voice_plan:
        churn_reasons.append("No voice plan")
    if not intl_plan:
        churn_reasons.append("No international plan")

    if churn_reasons:
        st.write("The customer is likely to leave the company due to the following reasons:")
        for reason in churn_reasons:
            st.write(f"- {reason}")

        # Recommendations with supporting visualizations
        st.subheader('Recommendations to Avoid Churn')

        if "High number of customer service calls" in churn_reasons:
            st.write("- Improve the quality of customer support to address issues promptly and effectively.")
            st.write("**Visualization: Customer Satisfaction with Effective Support**")
            st.write("This bar plot shows the average total charges for customers with different numbers of customer service calls. Addressing customer issues promptly can lead to lower total charges and higher satisfaction.")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(x='customer.calls', y='total_charge', data=state_data, ax=ax)
            ax.set_title('Customer Service Calls vs. Total Charge')
            st.pyplot(fig)

        if "Higher than average total charges" in churn_reasons:
            st.write("- Review and optimize pricing plans to ensure they are competitive and align with customer expectations.")
            
        if "Higher than average total charges" in churn_reasons:
            st.write("- Review and optimize pricing plans to ensure they are competitive and align with customer expectations.")
            st.write("**Visualization: Impact of Total Charges on Churn**")
            st.write("This histogram shows that higher total charges are associated with increased churn rates. By offering competitive pricing, customer churn can be reduced.")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.histplot(state_data[state_data['churn'] == 1]['total_charge'], ax=ax, kde=True, color='red', label='Churned')
            sns.histplot(state_data[state_data['churn'] == 0]['total_charge'], ax=ax, kde=True, color='blue', label='Not Churned')
            ax.set_title('Impact of Total Charges on Churn')
            ax.legend()
            st.pyplot(fig)

        if "No voice plan" in churn_reasons:
            st.write("- Consider offering a voice plan that matches the customer's needs.")
            st.write("**Visualization: Churn Rate for Customers Without a Voice Plan**")
            st.write("This countplot shows that customers without a voice plan have a higher churn rate. Offering suitable voice plans can help reduce churn.")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.countplot(x='churn', data=state_data[state_data['voice.plan'] == 0], ax=ax)
            ax.set_title('Churn Rate for Customers Without a Voice Plan')
            st.pyplot(fig)

        if "No international plan" in churn_reasons:
            st.write("- Consider offering an international plan to customers who frequently make international calls.")
            st.write("**Visualization: Churn Rate for Customers Without an International Plan**")
            st.write("This countplot shows that customers without an international plan have a higher churn rate. Providing international plans can help retain these customers.")
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.countplot(x='churn', data=state_data[state_data['intl.plan'] == 0], ax=ax)
            ax.set_title('Churn Rate for Customers Without an International Plan')
            st.pyplot(fig)

    else:
        st.write("The customer is unlikely to leave the company based on the selected features.")
    
