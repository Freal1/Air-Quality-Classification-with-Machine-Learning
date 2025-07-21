import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')  

st.title("Prediksi Kualitas Udara")

with st.form("pollution_form"):
    pm10 = st.number_input("PM10", min_value=0.0, step=0.1)
    so2 = st.number_input("SO2", min_value=0.0, step=0.1)
    co = st.number_input("CO", min_value=0.0, step=0.1)
    o3 = st.number_input("O3", min_value=0.0, step=0.1)
    no2 = st.number_input("NO2", min_value=0.0, step=0.1)
    max_val = max(pm10, so2, co, o3, no2)
    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    input_data = np.array([[pm10, so2, co, o3, no2, max_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    label = le.inverse_transform(prediction)  
    st.success(f"Prediksi Kategori Udara: {label[0]}")
