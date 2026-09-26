import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Employee Promotion AI Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CUSTOM CSS (PRO DESIGN) ---
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Container */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        padding: 2.5rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Cards Styling */
    .css-card {
        background: rgba(30, 41, 59, 0.6);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }

    /* Input Section Header */
    .section-title {
        color: #38bdf8;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Banners */
    .result-banner-yes {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
        border: 1px solid #10b981;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
    }
    .result-banner-no {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
        border: 1px solid #ef4444;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
    }

    /* Custom Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0284c7, #6366f1);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #4f46e5);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DUMMY MODEL LOAD (Replace with your model) ---
@st.cache_resource
def load_ann_model():
    # Replace this with: return tf.keras.models.load_model('model.h5')
    return None

model = load_ann_model()

# --- 4. HEADER ---
st.markdown("""
<div class="main-header">
    <h1>Employee Promotion Intelligence</h1>
    <p>Artificial Neural Network (ANN) Powered Evaluation Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- 5. SIDEBAR - INPUT FORM ---
st.sidebar.markdown("## 🎛️ Candidate Parameters")
st.sidebar.markdown("Fill in the employee details to run the ANN assessment.")

with st.sidebar.form("input_form"):
    st.markdown("### 👤 Demographic & Education")
    education = st.selectbox("Education Level", ["Bachelors", "Masters & above", "Below Secondary"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age (Years)", 20, 60, 30)
    
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    department = st.selectbox("Department", ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "HR"])
    no_of_trainings = st.number_input("Trainings Completed", 1, 10, 1)
    avg_training_score = st.slider("Avg Training Score (0-100)", 30, 100, 65)
    previous_year_rating = st.selectbox("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    
    st.markdown("---")
    st.markdown("### 🏆 Achievements")
    length_of_service = st.slider("Length of Service (Years)", 1, 30, 5)
    kpis_met = st.checkbox("KPIs Met > 80%?", value=True)
    awards_won = st.checkbox("Awards Won in Last Year?", value=False)

    predict_btn = st.form_submit_button("🚀 Run Evaluation")

# --- 6. MAIN CONTENT AREA ---
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("""
    <div class="css-card">
        <div class="section-title">📌 Candidate Snapshot</div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Department", department)
    m2.metric("Experience", f"{length_of_service} Yrs")
    m3.metric("Last Rating", f"{previous_year_rating} ⭐")

    m4, m5, m6 = st.columns(3)
    m4.metric("Trainings", no_of_trainings)
    m5.metric("Avg Score", f"{avg_training_score}/100")
    m6.metric("Awards", "Yes" if awards_won else "No")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="css-card">
        <div class="section-title">📊 Analytics Overview</div>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            The model analyzes key factors such as <b>KPI completion</b>, <b>training scores</b>, and <b>years of service</b> using deep neural layers.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. PREDICTION RESULT LOGIC ---
if predict_btn:
    with st.spinner("Processing features through Neural Network..."):
        time.sleep(1) # Visual effect
        
        # PREPROCESSING LOGIC PLACEHOLDER
        # Format your input features as required by scaler/encoder
        # e.g., features = np.array([[...]])
        
        # DUMMY PREDICTION (Replace with your actual model logic)
        # score = model.predict(features)[0][0]
        score = np.random.uniform(0.3, 0.95) # Dummy dynamic probability for demo
        is_promoted = score >= 0.5

    st.markdown("---")
    
    if is_promoted:
        st.balloons()
        st.markdown(f"""
        <div class="result-banner-yes">
            <h2 style="color: #10b981; margin: 0;">🎉 High Likelihood of Promotion</h2>
            <h1 style="color: #ffffff; font-size: 3.5rem; margin: 0.5rem 0;">{score*100:.1f}%</h1>
            <p style="color: #a7f3d0; margin: 0;">This employee satisfies key threshold metrics for recommendation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner-no">
            <h2 style="color: #ef4444; margin: 0;">⚠️ Promotion Not Recommended</h2>
            <h1 style="color: #ffffff; font-size: 3.5rem; margin: 0.5rem 0;">{score*100:.1f}%</h1>
            <p style="color: #fca5a5; margin: 0;">Employee needs improvement in performance rating or training score metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar visualization
    st.write("")
    st.write("### Model Confidence Gauge")
    st.progress(float(score))

st.markdown("<br><hr><center style='color:#64748b;'>ANN Employee Promotion Dashboard • Omveer</center>", unsafe_allow_html=True)