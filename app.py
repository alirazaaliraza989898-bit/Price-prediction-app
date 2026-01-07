import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load Artifacts ---
try:
    rf_model = joblib.load('rf_model_compressed.pkl')
    model_features = joblib.load('model_features.pkl')
    feature_defaults = joblib.load('feature_defaults.pkl')
    categorical_options = joblib.load('categorical_options.pkl')
    
    top_10_features = [
        'accommodates', 'distance_to_center_km', 'minimum_nights',
        'availability_365', 'amenities_count', 'host_experience_days',
        'availability_60', 'bedrooms', 'availability_90', 'availability_30'
    ]

except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- Helper Functions ---
def assign_price_score(ratio):
    if ratio < 0.7: return 1
    elif ratio < 0.9: return 2
    elif ratio <= 1.1: return 3
    elif ratio <= 1.3: return 4
    else: return 5

def create_model_input_df(user_inputs):
    df = pd.DataFrame({col: [feature_defaults[col]] for col in model_features})
    for k,v in user_inputs.items():
        if k in df.columns:
            df[k] = v
    return df

def predict_and_score(user_price, user_inputs):
    df = create_model_input_df(user_inputs)
    predicted = rf_model.predict(df)[0]
    ratio = user_price / predicted
    score = assign_price_score(ratio)
    labels = ['Very Cheap', 'Cheap', 'Fair', 'Expensive', 'Very Expensive']
    return predicted, labels[score-1], score, ratio

# --- Page Config ---
st.set_page_config(page_title="Munich Airbnb Price Predictor", layout="wide", page_icon="🏠")
st.title("🏠 Munich Airbnb Price Predictor")
st.markdown("Enter your listing details to get a price prediction and fairness rating!")

# --- Input Section ---
with st.expander("Enter Your Listing Details 🔧", expanded=True):
    imagined_price = st.number_input("Your Imagined Price ($)", min_value=1.0, value=100.0, step=5.0)
    
    user_inputs = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(top_10_features):
        default = feature_defaults.get(feature, 0)
        # Set some smarter defaults
        if 'availability_30' in feature: default = 30
        if 'availability_60' in feature: default = 60
        if 'availability_90' in feature: default = 90
        if 'availability_365' in feature: default = 200
        if 'distance_to_center_km' in feature: default = 5.0
        
        col = col1 if i % 2 == 0 else col2
        user_inputs[feature] = col.number_input(feature.replace('_',' ').title(), min_value=0.0, value=float(default), step=1.0)

# --- Prediction ---
if st.button("Predict Price & Fairness"):
    predicted_price, rating_label, rating_score, price_ratio = predict_and_score(imagined_price, user_inputs)
    
    # --- Output Section ---
    st.header("💰 Prediction Results")
    out_col1, out_col2, out_col3 = st.columns(3)
    
    out_col1.metric(label="Your Price", value=f"${imagined_price:.2f}", delta="")
    out_col2.metric(label="Predicted Market Price", value=f"${predicted_price:.2f}", delta=f"${predicted_price - imagined_price:+.2f}")
    out_col3.metric(label="Price Ratio", value=f"{price_ratio:.2f}", delta="")
    
    # Color-coded fairness
    if rating_score <= 2:
        st.success(f"✅ Fairness Rating: {rating_label} (Score: {rating_score})")
        st.info("Your price is cheaper than market — great deal!")
    elif rating_score == 3:
        st.info(f"⚖️ Fairness Rating: {rating_label} (Score: {rating_score})")
        st.info("Your price aligns well with the market price.")
    else:
        st.warning(f"⚠️ Fairness Rating: {rating_label} (Score: {rating_score})")
        st.warning("Your price is higher than the market — consider adjusting it.")

    # Quick tip
    st.markdown("---")
    st.markdown("*Tip: Adjust `accommodates`, `bedrooms`, `amenities count` or `distance to center` to see how market price changes in real time!*")
