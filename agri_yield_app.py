# agri_yield_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Setup Streamlit
st.set_page_config(page_title="Agricultural Yield Dashboard", page_icon="🌾", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_yield_model.pkl')

model = load_model()

# Load the encoders
@st.cache_resource
def load_encoders():
    country_encoder = joblib.load('country_label_encoder.pkl')
    crop_encoder = joblib.load('crop_label_encoder.pkl')
    return country_encoder, crop_encoder

country_mapping, crop_mapping = load_encoders()

# Tabs (Home, About)
tab1, tab3 = st.tabs(["🌾 Prediction Tool", "ℹ️ About"])

# ===================== TAB 1: Prediction Tool =======================
with tab1:
    st.title("🌾 Agricultural Yield Prediction")
    st.subheader("Make smarter farming decisions using machine learning")

    # Sidebar Inputs
    country = st.sidebar.selectbox('Country/Area', country_mapping.classes_)
    crop = st.sidebar.selectbox('Crop Type', crop_mapping.classes_)
    year = st.sidebar.number_input('Year', min_value=2000, max_value=2030, step=1)
    rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0, step=1.0)
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=-10.0, step=0.1)
    pesticide_use = st.sidebar.number_input('Pesticide Use (tonnes)', min_value=0.0, step=0.1)

    # Build DataFrame (keeping original country and crop names for display)
    input_features = pd.DataFrame({
        'Country': [country],
        'Temperature': [temperature],
        'Rainfall': [rainfall],
        'Year': [year],
        'Pesticides': [pesticide_use],
        'Crop': [crop]
    })

    # Save a copy for display
    display_features = input_features.copy()

    # Encode for model input (convert to numerical labels)
    input_features['Country'] = country_mapping.transform([country])
    input_features['Crop'] = crop_mapping.transform([crop])

    # Display
    st.write("### 📋 Input Summary")
    st.dataframe(display_features)

    # Prediction
    if st.button('Predict Yield'):
        with st.spinner('Predicting...'):
            prediction = model.predict(input_features)
        
        # Reverse the encoding to show the original names
        original_country = country_mapping.inverse_transform(input_features['Country'])
        original_crop = crop_mapping.inverse_transform(input_features['Crop'])
        
        st.success(f"🌱 Predicted Crop Yield: **{prediction[0]:.2f} tons per hectare**")
        st.write(f"Predicted for: {original_country[0]} - {original_crop[0]}")




# ===================== TAB 3: About =======================
with tab3:
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Agricultural Yield Prediction System** is a machine learning-powered dashboard  
    designed to forecast crop yields based on area/country, weather, pesticide use, and historical trends.

    **Key Features**:
    - Predicts crop yields accurately
    - Helps farmers optimize resource use

    **Technologies Used**:
    - Python, XGBoost, Streamlit, Random Forest
    - Pandas, NumPy, Scikit-learn

    **Developer**: Allan Matthew Mohono  
    **Institution**: Jomo Kenyatta University of Agriculture and Technology (JKUAT)

    > Built with LOVE for a food-secure future.
    """)
    st.caption("© 2025 Agricultural Yields Prediction System | All Rights Reserved.")