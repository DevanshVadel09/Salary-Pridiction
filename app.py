
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

st.set_page_config(page_title="Salary Prediction", page_icon="💼")
st.title("💼 Employee Salary Prediction")

# Load saved files
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

try:
    model = tf.keras.models.load_model(
        MODEL_DIR / "salary_ann.keras", compile=False
    )
    preprocessor = joblib.load(
        MODEL_DIR / "salary_preprocessor.joblib"
    )
    scaler = joblib.load(
        MODEL_DIR / "salary_scaler.joblib"
    )
    config = joblib.load(MODEL_DIR / "salary_config.joblib")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Get feature names
features = config.get("feature_names")
df = pd.read_csv(BASE / "job_salary_prediction_dataset.csv") \
    if (BASE / "job_salary_prediction_dataset.csv").exists() else None

if not features and df is not None:
    target = config.get("target_name", "salary")
    features = [c for c in df.columns if c != target]

if not features:
    st.error("Feature names not found in configuration or dataset.")
    st.stop()

# Employee input
inputs = {}

with st.form("salary_form"):
    for feature in features:
        if df is not None and feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(df[feature].median())
                )
            else:
                options = df[feature].dropna().astype(str).unique().tolist()
                inputs[feature] = st.selectbox(
                    feature.replace("_", " ").title(), options
                )
        else:
            inputs[feature] = st.text_input(feature)

    submit = st.form_submit_button("Predict Salary")

# Predict
if submit:
    try:
        X = pd.DataFrame([inputs], columns=features)
        X = preprocessor.transform(X)

        if hasattr(X, "toarray"):
            X = X.toarray()

        prediction = model.predict(np.asarray(X), verbose=0)
        salary = scaler.inverse_transform(
            np.asarray(prediction).reshape(-1, 1)
        )[0, 0]

        st.success(f"Predicted Salary: {salary:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
