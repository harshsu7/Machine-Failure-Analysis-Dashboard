import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. SET PAGE CONFIG (Makes it look like a real app)
st.set_page_config(page_title="Machine Health Monitor", layout="wide")

# 2. LOAD ARTIFACTS
# Make sure these filenames match exactly what you saved in the notebook
try:
    model = joblib.load('machine_failure_model.pkl')
    scaler = joblib.load('app_scaler.pkl')
except:
    st.error("Model or Scaler files not found! Please run the Jupyter Notebook first.")

# 3. UI HEADER
st.title(" Predictive Maintenance Dashboard")
st.markdown("Enter the machine's real-time sensor data to predict potential failure.")
st.divider()

# 4. INPUT SECTION (Using columns for a cleaner look)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Temperature Data")
    air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0)
    proc_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0)

with col2:
    st.subheader(" Mechanical Data")
    rpm = st.number_input("Rotational Speed [rpm]", min_value=1200, max_value=3000, value=1500)
    torque = st.number_input("Torque [Nm]", min_value=3.0, max_value=80.0, value=40.0)

with col3:
    st.subheader(" Tool Metrics")
    tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=250, value=0)
    # Mapping 'Type' if you included it in training (L=0, M=1, H=2)
    m_type = st.selectbox("Machine Quality Type", options=["Low (L)", "Medium (M)", "High (H)"])
    type_map = {"Low (L)": 0, "Medium (M)": 1, "High (H)": 2}

st.divider()

# 5. PREDICTION LOGIC
if st.button("Analyze Machine Status", type="primary"):
    
    # Create input array (Ensure order matches your training columns!)
    # Assuming order: [Type, Air temp, Process temp, Speed, Torque, Tool wear]
# If you didn't use 'Type' in training, remove it from the array here!
    features = np.array([[air_temp, proc_temp, rpm, torque, tool_wear]])

# Now it has 5 features.
    scaled_features = scaler.transform(features)
    
    # IMPORTANT: Use .transform(), NOT .fit_transform()
    
    # Get Prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features) # Show confidence
    
    # 6. DISPLAY RESULTS
    if prediction[0] == 1:
        st.error("###  WARNING: Machine Failure Predicted!")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success("###  System Normal: No Failure Detected.")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

# 7. SIDEBAR INFO
st.sidebar.info("This app uses a Random Forest model trained on the AI4I 2020 Predictive Maintenance Dataset.")