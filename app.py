import streamlit as st
import numpy as np
import pickle
import time  # Import the time module

# --- Page Configuration ---
st.set_page_config(
    page_title="🌧️ Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown(
    """
    <style>
    .title {
        color: #1E88E5;
        text-align: center;
        padding-bottom: 1.5rem;
    }
    .header {
        color: #42A5F5;
        padding-top: 1rem;
    }
    .subheader {
        color: #64B5F6;
    }
    .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #1976D2;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #388E3C;
    }
    .stSuccess {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .signature {
        color: gray;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.markdown("<h1 class='title'>🌧️ Rainfall Prediction App</h1>", unsafe_allow_html=True)

# --- Sidebar for Instructions/Information ---
with st.sidebar:
    st.subheader("About This App")
    st.markdown(
        """
        This application predicts whether it will rain based on several weather features. 
        Please enter the values for each feature in the main panel.
        """
    )
    st.subheader("Feature Information")
    st.markdown(
        """
        - **Pressure:** Atmospheric pressure (typically in hPa).
        - **Dewpoint:** The temperature to which air must be cooled to become saturated with water vapor (°C).
        - **Humidity:** The amount of water vapor in the air, expressed as a percentage (%).
        - **Cloud Cover:** The extent to which the sky is covered by clouds, as a percentage (0-100).
        - **Sunshine:** A qualitative measure of sunshine (Low, Medium, High).
        - **Wind Direction:** The direction from which the wind is blowing (e.g., North, East).
        - **Windspeed:** The speed of the wind in kilometers per hour (km/h).
        """
    )
    st.markdown("---")
    st.markdown("Made with ❤️ by Madhav Gupta")

# --- Main Panel ---
st.header("Enter the Weather Features:")

# Load trained model with a spinner
with st.spinner("Loading the rainfall prediction model..."):
    try:
        with open("rainfall_prediction_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        feature_names = model_data["feature_names"]
        time.sleep(2)  # Simulate loading time (remove in production)
    except FileNotFoundError:
        st.error(
            "Error: The rainfall prediction model file (rainfall_prediction_model.pkl) was not found. "
            "Please make sure it's in the same directory as this app."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Create input fields for all features
input_values = {}
for feature in feature_names:
    if feature == 'pressure':
        input_values[feature] = st.number_input("Pressure (hPa)", value=1015.0)
    elif feature == 'dewpoint':
        input_values[feature] = st.number_input("Dewpoint (°C)", value=20.0)
    elif feature == 'humidity':
        input_values[feature] = st.slider("Humidity (%)", 0, 100, 50)
    elif feature == 'cloud':
        st.markdown("<p class='subheader'>Cloud Cover (%)</p>", unsafe_allow_html=True)
        input_values[feature] = st.number_input("Enter percentage (0-100)", min_value=0, max_value=100, value=5)
    elif feature == 'sunshine':
        st.markdown("<p class='subheader'>Sunshine</p>", unsafe_allow_html=True)
        sunshine_options = ["Low", "Medium", "High"]
        selected_sunshine = st.selectbox("Select level of sunshine", options=sunshine_options)
        sunshine_mapping = {
            "Low": 2.0,
            "Medium": 6.0,
            "High": 10.0
        }
        input_values[feature] = sunshine_mapping[selected_sunshine]
    elif feature == 'winddirection':
        st.markdown("<p class='subheader'>Wind Direction</p>", unsafe_allow_html=True)
        direction_options = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
        selected_direction = st.selectbox("Select wind direction", options=direction_options)
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
        st.markdown("<p class='subheader'>Windspeed (km/h)</p>", unsafe_allow_html=True)
        input_values[feature] = st.number_input("Enter windspeed in km/h", value=15.0)

# Ensure input data is in the correct order
input_data = np.array([[input_values[feature] for feature in feature_names]])

# Predict and display result
if st.button("Predict Rainfall"):
    prediction = model.predict(input_data)
    result_text = "🌧️ Rain expected" if prediction[0] == 1 else "☀️ No rain expected"
    st.success(f"Prediction: {result_text}")

# --- Signature ---
st.markdown("<p class='signature'>Made by Madhav Gupta</p>", unsafe_allow_html=True)
