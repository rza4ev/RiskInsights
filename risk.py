import streamlit as st
import pandas as pd
import pickle
import os

# Encoding mappings
process_name_map = {
    'Customer Support': 0, 'Data Management': 1, 'Financial Reporting': 2, 'IT Support': 3,
    'Inventory Control': 4, 'Marketing Campaign': 5, 'Order Processing': 6, 'Payroll': 7,
    'Procurement': 8, 'Product Development': 9
}

process_description_map = {
    'Designing and developing new products': 0, 'Handling company data': 1, 'Managing IT services and support': 2,
    'Managing and processing customer orders': 3, 'Planning and executing marketing campaigns': 4,
    'Preparing financial statements': 5, 'Processing employee payroll': 6, 'Providing support to customers': 7,
    'Purchasing goods and services': 8, 'Tracking and managing inventory levels': 9
}

potential_risk_map = {
    'Customer complaints': 0, 'Data breaches': 1, 'Financial misstatements': 2, 'Misleading advertisements': 3,
    'Order errors': 4, 'Payroll calculation errors': 5, 'Project delays': 6, 'Stockouts': 7,
    'Supplier non-compliance': 8, 'System downtimes': 9
}

control_measure_map = {
    'Accurate information vetting': 0, 'Automated payroll system': 1, 'Comprehensive training': 2,
    'Double-checking orders': 3, 'External audits': 4, 'Regular inventory audits': 5, 'Regular system maintenance': 6,
    'Strict project management': 7, 'Strong encryption practices': 8, 'Supplier audits': 9
}

department_map = {
    'Administration': 0, 'Customer Service': 1, 'Finance': 2, 'HR': 3, 'IT': 4, 'Marketing': 5,
    'Operations': 6, 'Procurement': 7, 'R&D': 8, 'Sales': 9
}

process_frequency_map = {
    'Daily': 0, 'Monthly': 1, 'Quarterly': 2, 'Weekly': 3, 'Yearly': 4
}

model_file = 'extra_trees_model.pkl'

# Check if the model file exists
if os.path.isfile(model_file):
    # Load the model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    # Streamlit app
    st.title('Risk Likelihood Prediction')

    # User inputs
    process_name = st.selectbox('Select Process Name', list(process_name_map.keys()))
    process_description = st.selectbox('Select Process Description', list(process_description_map.keys()))
    potential_risk = st.selectbox('Select Potential Risk', list(potential_risk_map.keys()))
    risk_impact = st.number_input('Enter Risk Impact', min_value=1, max_value=3, step=1)
    control_measure = st.selectbox('Select Control Measure', list(control_measure_map.keys()))
    control_effectiveness = st.number_input('Enter Control Effectiveness', min_value=1, max_value=3, step=1)
    department = st.selectbox('Select Department', list(department_map.keys()))
    process_frequency = st.selectbox('Select Process Frequency', list(process_frequency_map.keys()))
    process_criticality = st.number_input('Enter Process Criticality', min_value=1, max_value=3, step=1)
    risk_score = st.number_input('Enter Risk Score', min_value=1, max_value=9, step=1)
    control_score = st.number_input('Enter Control Score', min_value=1, max_value=9, step=1)

    # Encoding the inputs
    encoded_data = [
        process_name_map[process_name],
        process_description_map[process_description],
        potential_risk_map[potential_risk],
        risk_impact,
        control_measure_map[control_measure],
        control_effectiveness,
        department_map[department],
        process_frequency_map[process_frequency],
        process_criticality,
        risk_score,
        control_score
    ]

    # Convert to DataFrame
    features = pd.DataFrame([encoded_data], columns=[
        'ProcessName', 'ProcessDescription', 'PotentialRisk', 'RiskImpact',
        'ControlMeasure', 'ControlEffectiveness', 'Department', 'ProcessFrequency',
        'ProcessCriticality', 'RiskScore', 'ControlScore'
    ])

    # Predict button
    if st.button('Predict Risk Likelihood'):
        prediction = model.predict(features)
        st.write('Predicted Risk Likelihood:', prediction[0])
else:
    st.error(f"Model dosyası '{model_file}' bulunamadı. Lütfen dosyanın doğru dizinde olduğundan emin olun.")

