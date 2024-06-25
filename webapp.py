import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Load the model and scaler
model_path = r"C:\Users\jusep\OneDrive\Desktop\Anesthesia Prediction Model\logistic_regression_model.joblib"
scaler_path = r"C:\Users\jusep\OneDrive\Desktop\Anesthesia Prediction Model\standard_scaler.joblib"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
# Manually define the list of feature names based on the model's training dataset
feature_names = [
    'Age', 'Sex', 'Weight', 'ASA Score', 'Mental Health Diagnosis',
    'Chronic Pain', 'Respiratory Disease', 'Medications', 'Narcotic Exposure',
    'Cannabis Exposure', 'Cannabis Frequency', 'Alcohol', 'Smoking',
    'Procedure Code', 'Duration', 'Propofol/min', 'Propofol/kg/min'
]

# Define the list of resource URLs for Anesthesia Resources
asa_high_risk_url = "https://www.asahq.org/madeforthismoment/preparing-for-surgery/risks/"
asa_procedure_url = "https://www.asahq.org/madeforthismoment/preparing-for-surgery/procedures/upper-endoscopy/"

# Function to display links based on the risk prediction
def show_resources(prediction):
      # Link for all predictions explaining the endoscopy procedure
    st.markdown(f"[Learn More About Upper Endoscopy]({asa_procedure_url})")
    if prediction == 1:  # If the prediction is high risk
        st.error('The predicted anesthesia-related risk is High.')
        # Link explaining risks associated with anesthesia
        st.markdown(f"[Understand Anesthesia Risks]({asa_high_risk_url})")
    else:
        st.success('The predicted anesthesia-related risk is Low.')

# Define a function to calculate feature contributions
def get_feature_contributions(input_data_scaled):
    # Assuming logistic regression model, retrieve the coefficients
    coef = model.coef_[0]
    # Calculate contributions for each feature
    contributions = coef * input_data_scaled.flatten()
    return contributions
# Function to plot the feature contributions
def plot_feature_contributions(contributions):
    contrib_df = pd.DataFrame({'Feature': feature_names, 'Contribution': contributions})
    contrib_df = contrib_df.sort_values(by='Contribution', key=abs, ascending=False)
    fig = px.bar(contrib_df, x='Contribution', y='Feature', orientation='h')
    return fig

# Define a function to make predictions
def predict_risk(input_data):
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Predict the risk
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit app
def main():
    st.title('Anesthesia-Related Risk Prediction for Endoscopic Procedures')
    # Input fields for patient data
    age = st.number_input('Age', value=30.0)
    sex_code = st.selectbox('Sex (0 for Male, 1 for Female)', options=[0, 1])
    weight = st.number_input('Weight (kg)', value=70.0)
    asa_score = st.selectbox('ASA Score', options=[1, 2, 3, 4, 5])
    mh_dx = st.number_input('Mental Health Diagnosis (0=No, 1=Yes)', value=0.0)
    chronic_pain = st.number_input('Chronic Pain (0=No, 1=Yes)', value=0.0)
    resp_ds = st.selectbox('Respiratory Disease (0=No, 1=Yes)', options=[0, 1])
    meds = st.number_input('Number of Anxiolytic/Antidepressant Medications', value=0)
    narcotic = st.selectbox('Narcotic Exposure (0=No, 1=Yes)', options=[0, 1])
    cannabis = st.selectbox('Cannabis Exposure (0=No, 1=Yes)', options=[0, 1])
    cannabis_frequency = st.number_input('Frequency of Recreational Cannabis Use (0=None, 1=Daily, 2=Monthly, 3=Weekly, 4=Other, 5=Never)', value=0)
    alcohol = st.number_input('Alcohol (drinks/week)', value=0.0)
    smoking = st.selectbox('Smoking (0=No, 1=Yes)', options=[0, 1])
    procedure_code = st.selectbox('Procedure Code (0=EGD, 1=Colon, 2=Double)', options=[0, 1, 2])
    duration = st.number_input('Duration (minutes)', value=15)
    propofol_min = st.number_input('Propofol/min', value=10.0)
    propofol_kg_min = st.number_input('Propofol/kg/min', value=0.2)

    # Button to predict risk
    # When the Predict button is clicked
    if st.button('Predict Anesthesia-Related Risk'):
        input_data = np.array([[
            age, sex_code, weight, asa_score, mh_dx, chronic_pain, resp_ds, meds, 
            narcotic, cannabis, cannabis_frequency, alcohol, smoking, 
            procedure_code, duration, propofol_min, propofol_kg_min
        ]])
        prediction = predict_risk(input_data)
        show_resources(prediction[0])
        
        # Call function to get feature contributions
        input_data_scaled = scaler.transform(input_data)
        contributions = get_feature_contributions(input_data_scaled)

        # Call function to plot the feature contributions without passing X.columns
        fig = plot_feature_contributions(contributions)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()



