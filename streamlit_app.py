import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature column names
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🧠 Obesity Level Predictor")

# UI Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
height = st.slider("Height (in meters)", 1.0, 2.5, 1.70)
weight = st.slider("Weight (in kg)", 30.0, 200.0, 65.0)
family_history = st.selectbox("Family History of Overweight", ["Yes", "No"])
favc = st.selectbox("High Calorie Food (FAVC)", ["Yes", "No"])
fcvc = st.slider("Veggie Intake Frequency (FCVC)", 1.0, 3.0, 2.0)
ncp = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0)
caec = st.selectbox("Snacking (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you Smoke?", ["Yes", "No"])
ch2o = st.slider("Water Intake (CH2O)", 1.0, 3.0, 2.0)
scc = st.selectbox("Do you Monitor Calorie Intake? (SCC)", ["Yes", "No"])
faf = st.slider("Physical Activity (FAF)", 0.0, 3.0, 1.0)
tue = st.slider("Tech Use per Day (TUE)", 0.0, 2.0, 1.0)
calc = st.selectbox("Alcohol Consumption (CALC)", [0, 1, 2, 3])
mtrans = st.selectbox("Transportation Method", ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])

# Manual encodings
gender = 1 if gender == "Male" else 0
family_history = 1 if family_history == "Yes" else 0
favc = 1 if favc == "Yes" else 0
smoke = 1 if smoke == "Yes" else 0
scc = 1 if scc == "Yes" else 0
caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
caec = caec_map[caec]

# MTRANS One-hot Encoding (drop_first=True in training)
mtrans_map = {
    "Automobile": [0, 0, 0, 0],
    "Motorbike": [1, 0, 0, 0],
    "Bike": [0, 1, 0, 0],
    "Public Transportation": [0, 0, 1, 0],
    "Walking": [0, 0, 0, 1]
}
mtrans_encoded = mtrans_map[mtrans]

# Combine features
features = np.array([[
    gender, age, height, weight, family_history, favc, fcvc, ncp, caec,
    smoke, ch2o, scc, faf, tue, calc
] + mtrans_encoded])

# Create DataFrame with correct column names
input_df = pd.DataFrame(features, columns=feature_columns)

# Prediction
features_scaled = scaler.transform(input_df)

# Mapping from label to description
category_map = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    readable_label = category_map.get(prediction, "Unknown")
    st.success(f"🏷️ Predicted Obesity Category: **{readable_label}** (class {prediction})")

