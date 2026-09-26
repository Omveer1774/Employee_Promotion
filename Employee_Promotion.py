import streamlit as st
import pandas as pd
import numpy as np
# Agar aap TensorFlow use kar rahe hain toh ise uncomment karein:
# import tensorflow as tf
# Agar aap PyTorch use kar rahe hain toh ise uncomment karein:
# import torch
# import torch.nn as nn

# --- 1. PAGE SETUP (Visual Improvement) ---
st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="🏆",
    layout="centered"
)

# --- 2. LOAD MODEL (Place your model loading logic here) ---
@st.cache_resource
def load_my_model():
    """
    Function to load your trained model.
    Replace the dummy logic with your actual model loading code.
    """
    # EXAMPLE FOR TENSORFLOW:
    # model = tf.keras.models.load_model('your_model.h5')
    # return model

    # DUMMY MODEL (Replace this with actual loading)
    # This is just a placeholder because I don't have your .h5/.pth file.
    class DummyModel:
        def predict(self, data):
            # Returns a random prediction (Yes/No) and a dummy probability
            prob = np.random.rand()
            return prob

    st.warning("🔄 Loading Dummy Model. Replace with your actual model loading logic in the code.")
    return DummyModel()

# Load the model
my_model = load_my_model()

# --- 3. CUSTOM CSS FOR BETTER LOOKS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A; /* Dark Blue */
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #172554; /* Darker Blue on hover */
        border-color: #172554;
    }
    .prediction-container {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .promoted {
        background-color: #D1FAE5; /* Light Green */
        color: #065F46; /* Dark Green */
    }
    .not-promoted {
        background-color: #FEE2E2; /* Light Red */
        color: #991B1B; /* Dark Red */
    }
</style>
""", unsafe_allow_html=True)

# --- 4. APP TITLE & HEADER ---
st.markdown('<h1 class="main-title">Employee Promotion Prediction 🏆</h1>', unsafe_allow_html=True)
st.write("Enter the employee's details below to predict their likelihood of promotion.")

# --- 5. THE USER INTERFACE (INPUT FORM) ---
# We use st.form to group the inputs and submit button together
with st.form("employee_details_form"):
    st.subheader("📋 Employee Input Details")
    
    # Arrange inputs in columns for cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.selectbox(
            "Highest Education Level",
            ("Bachelors", "Masters", "Below Secondary")
        )
        gender = st.radio("Gender", ("Male", "Female"), horizontal=True)
        no_of_trainings = st.number_input(
            "Number of Trainings Completed",
            min_value=1, max_value=10, value=1, step=1
        )
        age = st.number_input(
            "Employee Age",
            min_value=18, max_value=60, value=30, step=1
        )
        
    with col2:
        length_of_service = st.number_input(
            "Length of Service (in years)",
            min_value=1, max_value=40, value=5, step=1
        )
        kpis_met = st.selectbox(
            "KPIs Met (>80%)?",
            ("Yes", "No")
        )
        awards_won = st.selectbox(
            "Awards Won in Last Year?",
            ("Yes", "No")
        )
        avg_training_score = st.slider(
            "Average Training Score (0-100)",
            min_value=0.0, max_value=100.0, value=65.0, step=0.1
        )

    # Submit button for the form
    submit_button = st.form_submit_button(label="Analyze & Predict")

# --- 6. PREDICTION LOGIC ---
if submit_button:
    # --- 6a. PREPROCESS INPUTS ---
    # Convert inputs to the format your model expects (e.g., one-hot encoding, normalization)
    
    # Step 1: Create a dictionary from user inputs
    input_data = {
        'no_of_trainings': no_of_trainings,
        'age': age,
        'previous_year_rating': 3.0, # Dummy value if not input
        'length_of_service': length_of_service,
        'kpis_met': 1 if kpis_met == "Yes" else 0,
        'awards_won': 1 if awards_won == "Yes" else 0,
        'avg_training_score': avg_training_score,
        'education': education,
        'gender': gender
    }
    
    # Step 2: Convert to DataFrame for easier preprocessing
    input_df = pd.DataFrame([input_data])
    
    # Step 3: APPLY ACTUAL PREPROCESSING (This depends on how you trained your model)
    # Examples:
    # 1. One-hot encoding for categorical variables: 'education', 'gender'
    #    (You might need to make sure the columns match your training data exactly)
    # 2. Scaling numerical variables: 'age', 'avg_training_score', etc.

    # DUMMY PREPROCESSING (Replace with actual)
    # For now, we just pass raw numbers for the 7 numerical inputs to match the Dummy model
    preprocessed_data = np.array([[
        no_of_trainings, age, 3.0, length_of_service, 
        1 if kpis_met == "Yes" else 0, 
        1 if awards_won == "Yes" else 0, 
        avg_training_score
    ]])

    # Show a progress spinner while the model predicts
    with st.spinner("Analyzing data and generating prediction..."):
        # --- 6b. MAKE PREDICTION ---
        try:
            # The structure must match how you preprocessed and fed data during training
            prediction_probability = my_model.predict(preprocessed_data)
            
            # --- 6c. DISPLAY RESULTS ---
            st.divider()
            st.subheader("🎯 Prediction Result")

            # Determine promotion status based on probability threshold (e.g., 0.5)
            # Adjust the threshold as per your model performance
            threshold = 0.5
            is_promoted = prediction_probability >= threshold
            
            # Formatted output
            if is_promoted:
                st.balloons()
                st.markdown(f"""
                <div class="prediction-container promoted">
                    <strong>Prediction: PROMOTED (YES)</strong><br>
                    Probability of Promotion: {prediction_probability:.2f}
                </div>
                """, unsafe_allow_html=True)
                st.success("Analysis suggests this employee has a high chance of being recommended for promotion.")
            else:
                st.markdown(f"""
                <div class="prediction-container not-promoted">
                    <strong>Prediction: NOT PROMOTED (NO)</strong><br>
                    Probability of Promotion: {prediction_probability:.2f}
                </div>
                """, unsafe_allow_html=True)
                st.warning("Analysis suggests this employee has a lower chance of being recommended for promotion at this time.")
                
        except Exception as e:
            st.error(f"Error during prediction. Please check model input compatibility. Error: {e}")

# --- 7. FOOTER ---
st.write("---")
st.caption("Developed by Omveer | Performance Metrics App")