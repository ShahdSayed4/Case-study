import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime

# ----------------------------
# 1. App Configuration
# ----------------------------
st.set_page_config(page_title="Cairo Real Estate Pricing", layout="centered")
st.title("🏠 PropMatch Cairo — Price Prediction Tool")
st.write("Estimate the fair market price of 2–3 bedroom apartments in **New Cairo**.")

# ----------------------------
# 2. Load Trained RandomForest Model
# ----------------------------
model_path = r"C:\Users\shahd\Depi Data Science\Case Study\chat\models\RandomForest_model.joblib"

try:
    rf_model = joblib.load(model_path)
    st.success("✅ RandomForest model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load the RandomForest model. Predictions are disabled. ({e})")
    st.stop()

# Get expected feature names from the trained model
expected_features = rf_model.feature_names_in_

# ----------------------------
# 3. Collect User Input
# ----------------------------
st.subheader("🏘️ Apartment Details")

district = st.selectbox("District", ["Fifth Settlement", "Rehab", "Madinaty", "Katameya"])
area = st.number_input("Area (sqm)", 80, 400, 150)
bedrooms = st.selectbox("Bedrooms", [2, 3])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
floor = st.slider("Floor Number", 1, 20, 3)
building_age = st.slider("Building Age (years)", 0, 30, 5)
finishing = st.selectbox("Finishing Type", ["Super Lux", "Lux", "Semi-finished", "Unfinished"])
view = st.selectbox("View Type", ["Street", "Garden", "Nile", "Compound"])
distance_auc = st.slider("Distance to AUC (km)", 0.5, 20.0, 5.0)
distance_mall = st.slider("Distance to Nearest Mall (km)", 0.5, 15.0, 3.0)
distance_metro = st.slider("Distance to Metro (km)", 0.5, 15.0, 10.0)

input_df = pd.DataFrame([{
    "district": district,
    "finishing_type": finishing,
    "view_type": view,
    "area_sqm": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floor_number": floor,
    "building_age_years": building_age,
    "distance_to_auc_km": distance_auc,
    "distance_to_mall_km": distance_mall,
    "distance_to_metro_km": distance_metro
}])

# ----------------------------
# 4. Manual One-Hot Encoding
# ----------------------------
# Create an empty DataFrame with the model's expected features
X_pred = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)

# Fill numeric features
numeric_features = ["area_sqm", "bedrooms", "bathrooms", "floor_number",
                    "building_age_years", "distance_to_auc_km",
                    "distance_to_mall_km", "distance_to_metro_km"]

for col in numeric_features:
    if col in expected_features:
        X_pred[col] = input_df[col].values[0]

# Fill one-hot categorical features
for feature in expected_features:
    if feature.startswith("district_") and input_df["district"].values[0] in feature:
        X_pred[feature] = 1
    if feature.startswith("finishing_type_") and input_df["finishing_type"].values[0] in feature:
        X_pred[feature] = 1
    if feature.startswith("view_type_") and input_df["view_type"].values[0] in feature:
        X_pred[feature] = 1

# ----------------------------
# 5. Predict and Display
# ----------------------------
if st.button("💰 Predict Price"):
    try:
        prediction = rf_model.predict(X_pred)[0]

        # Estimate confidence ±10%
        lower = prediction * 0.9
        upper = prediction * 1.1

        st.success(f"🏡 **Estimated Price:** {prediction:,.0f} EGP")
        st.caption(f"Confidence range: {lower:,.0f} – {upper:,.0f} EGP")

        # Log input and prediction to MLflow
        with mlflow.start_run(run_name="streamlit_prediction"):
            mlflow.log_params(input_df.iloc[0].to_dict())
            mlflow.log_metric("predicted_price", float(prediction))
            mlflow.log_metric("timestamp", datetime.now().timestamp())
            st.info("Prediction logged to MLflow ✅")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----------------------------
# 6. Optional MLflow Dashboard
# ----------------------------
st.markdown("---")
st.subheader("📈 MLflow Dashboard")
if st.button("📊 View MLflow Dashboard"):
    st.info("Open your browser to [MLflow Dashboard](http://127.0.0.1:5001) to view all logged predictions and model runs.")
