import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Customer Insight Pro", layout="wide", page_icon="🎯")

# 2. Load Model & Sample Data
try:
    model = joblib.load("C:\\Users\\NewUser\\Downloads\\spending_model.pkl")
    # We load the CSV just to show the distribution charts
    df = pd.read_csv("C:\\Users\\NewUser\\Downloads\\Mall_Customers.csv") 
except:
    st.error("Model or Dataset not found. Please ensure files are in the same folder.")

# SIDEBAR: INPUTS
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
    st.header("Customer Profile")
    st.info("Adjust the parameters below to predict the Spending Score.")
    
    Genre = st.radio("Select Genre", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income (k$)", 15, 150, 50)
    
    # Process inputs
    Genre_value = 1 if Genre == "Male" else 0
    input_df = pd.DataFrame({'Genre': [Genre_value], 'Age': [age], 'Annual Income (k$)': [income]})

# --- MAIN AREA: ROWS & COLUMNS ---
st.title(" Mall Customer Predictive Analytics")
st.markdown("---")

# ROW 1: Metrics & Prediction
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("Input Summary")
    st.write(f"**Genre:** {Genre}")
    st.write(f"**Age:** {age}")
    st.write(f"**Income:** ${income}k")

with col2:
    st.subheader("Prediction")
    if st.button("Calculate Score"):
        prediction = model.predict(input_df)[0]
        st.metric(label="Predicted Spending Score", value=f"{prediction:.1f}/100")
        
        if prediction > 70:
            st.success("Target: High-Value")
        elif prediction > 40:
            st.info("Target: Average")
        else:
            st.warning("Target: Low-Value")
    else:
        st.write("Click button to run model")

with col3:
    st.subheader("Strategic Insight")
    st.markdown("""
    This prediction uses a **Random Forest Regressor** to estimate spending potential. 
    By analyzing Age and Income, we can determine the **Variance** in consumer behavior.
    """)

st.markdown("---")

# ROW 2: Visualization
st.subheader("📊 Customer Distribution Analysis")
c1, c2 = st.columns(2)

with c1:
    # Show where THIS customer sits in terms of Income vs Age
    fig = px.scatter(df, x="Age", y="Annual Income (k$)", color="Gender", 
                     title="Market Overview (Income vs Age)")
    # Add a gold star for the current input
    fig.add_scatter(x=[age], y=[income], mode='markers', 
                    marker=dict(size=15, color='Gold', symbol='star'),
                    name="Current Input")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Distribution of the target variable
    fig2 = px.histogram(df, x="Spending Score (1-100)", nbins=20, 
                        title="Global Spending Score Distribution", color_discrete_sequence=['white'])
    st.plotly_chart(fig2, use_container_width=True)