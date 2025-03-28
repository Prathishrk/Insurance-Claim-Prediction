import streamlit as st
import joblib
import numpy as np

# Load models
model_0 = joblib.load("model_xgb_0.pkl")
model_1 = joblib.load("model_xgb_1.pkl")


# Streamlit UI
st.title("Insurance Claim Amount Prediction")
st.sidebar.header("Input Features")

# Input fields
vehicle_claim = st.sidebar.number_input("Vehicle Claim", min_value=0.0, value=1000.0)
property_claim = st.sidebar.number_input("Property Claim", min_value=0.0, value=500.0)
injury_claim = st.sidebar.number_input("Injury Claim", min_value=0.0, value=300.0)
incident_hour_of_the_day = st.sidebar.slider("Incident Hour", 0, 23, 12)
number_of_vehicles_involved = st.sidebar.slider("Number of Vehicles", 1, 5, 1)

# Cluster selection
cluster = st.sidebar.radio("Select Cluster", [0, 1])

# Make prediction
input_data = np.array([vehicle_claim, property_claim, injury_claim, incident_hour_of_the_day, number_of_vehicles_involved,cluster]).reshape(1, -1)

if st.sidebar.button("Predict"):
    if cluster == 0:
        prediction = model_0.predict(input_data)[0]
    else:
        prediction = model_1.predict(input_data)[0]

    st.success(f"Predicted Claim Amount: ${prediction:.2f}")
