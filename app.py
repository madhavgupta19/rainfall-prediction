import streamlit as st
import numpy as np
import pickle

st.title("🌧️ Rainfall Prediction App")

# Load trained model
with open("rainfall_prediction_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["feature_names"]  # Load feature names

st.header("Enter the weather features:")

# Create input fields for all features
input_values = {}
for feature in feature_names:
    if feature == 'pressure':
        input_values[feature] = st.number_input("Pressure", value=1015.0)
    elif feature == 'dewpoint':
        input_values[feature] = st.number_input("Dewpoint", value=20.0)
    elif feature == 'humidity':
        input_values[feature] = st.slider("Humidity", 0, 100, 50)
    elif feature == 'cloud':
        # Add explanation for cloud input
        st.markdown("Cloud cover is typically measured in percentage (0-100). "
                    "Please enter a numerical value representing percentage of cloud cover.")
        input_values[feature] = st.number_input("Cloud Cover", value=5.0)
    elif feature == 'sunshine':
        sunshine_options = ["Low", "Medium", "High"]
        selected_sunshine = st.selectbox("Sunshine", options=sunshine_options)
        sunshine_mapping = {
            "Low": 2.0,
            "Medium": 6.0,
            "High": 10.0
        }
        input_values[feature] = sunshine_mapping[selected_sunshine]
    elif feature == 'winddirection':
        direction_options = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
        selected_direction = st.selectbox("Wind Direction", options=direction_options)
        direction_mapping = {
            "North": 0.0,
            "North-East": 45.0,
            "East": 90.0,
            "South-East": 135.0,
            "South": 180.0,
            "South-West": 225.0,
            "West": 270.0,
            "North-West": 315.0
        }
        input_values[feature] = direction_mapping[selected_direction]
    elif feature == 'windspeed':
        st.markdown("Please give the windspeed in km/h")
        input_values[feature] = st.number_input("Windspeed", value=15.0)

# Ensure input data is in the correct order
input_data = np.array([[input_values[feature] for feature in feature_names]])

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "🌧️ Rain expected" if prediction[0] == 1 else "☀️ No rain expected"
    st.success(f"Prediction: {result}")