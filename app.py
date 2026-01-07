import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Saved Artifacts ---
# Ensure these files are in the same directory as your streamlit_app.py
# or provide the full path.
try:
    rf_model = joblib.load('rf_model_compressed.pkl')
    model_features = joblib.load('model_features.pkl')
    feature_defaults = joblib.load('feature_defaults.pkl')
    categorical_options = joblib.load('categorical_options.pkl')
    
    # Assuming top_10_features was saved or can be re-derived
    # For this outline, let's hardcode it based on previous output for simplicity
    top_10_features = [
        'accommodates', 'distance_to_center_km', 'minimum_nights',
        'availability_365', 'amenities_count', 'host_experience_days',
        'availability_60', 'bedrooms', 'availability_90', 'availability_30'
    ]

    st.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Make sure all .pkl files are present.")
    st.stop() # Stop the app if essential files are missing

# --- Helper Functions (Copied from previous steps) ---
def assign_price_score(ratio):
    if ratio < 0.7: return 1
    elif ratio < 0.9: return 2
    elif ratio <= 1.1: return 3
    elif ratio <= 1.3: return 4
    else: return 5

def create_model_input_df_streamlit(user_feature_inputs, feature_defaults, categorical_options, model_feature_columns):
    input_data = {col: [feature_defaults[col]] for col in model_feature_columns}
    input_df = pd.DataFrame(input_data)

    for feature, value in user_feature_inputs.items():
        is_original_categorical = False
        for cat_name in categorical_options.keys():
            if feature == cat_name:
                is_original_categorical = True
                break

        if not is_original_categorical and feature in input_df.columns:
            input_df[feature] = value

    for original_cat_feature, chosen_value in user_feature_inputs.items():
        if original_cat_feature in categorical_options:
            for ohe_col_prefix in categorical_options[original_cat_feature]:
                col_name = f"{original_cat_feature}_{ohe_col_prefix}"
                if col_name in input_df.columns:
                    input_df[col_name] = 0
            col_to_set_one = f"{original_cat_feature}_{chosen_value}"
            if col_to_set_one in input_df.columns:
                input_df[col_to_set_one] = 1
            if str(chosen_value) in ['0', '1'] and original_cat_feature in ['host_is_superhost', 'instant_bookable']:
                 if original_cat_feature in input_df.columns:
                     input_df[original_cat_feature] = int(chosen_value)

    bool_like_features = ['host_is_superhost', 'instant_bookable', 'is_oktoberfest_snapshot', 'is_holiday_snapshot']
    for b_feature in bool_like_features:
        if b_feature in input_df.columns:
            input_df[b_feature] = input_df[b_feature].astype(int)

    input_df = input_df[model_feature_columns]
    return input_df

def predict_and_score_listing(user_imagined_price, user_feature_inputs):
    model_input_df = create_model_input_df_streamlit(user_feature_inputs, feature_defaults, categorical_options, model_features)
    predicted_price = rf_model.predict(model_input_df)[0]
    price_ratio = user_imagined_price / predicted_price
    fairness_score = assign_price_score(price_ratio)
    score_labels = ['Very Cheap', 'Cheap', 'Fair', 'Expensive', 'Very Expensive']
    fairness_rating_label = score_labels[fairness_score - 1]
    return predicted_price, fairness_rating_label, fairness_score


# --- Streamlit UI ---
st.set_page_config(page_title="Munich Airbnb Price Predictor", layout="centered")
st.title("🏠 Munich Airbnb Price Predictor")
st.markdown("Enter your listing details below to get a price prediction and fairness score!")

# User Input Section
st.header("1. Your Listing Details")

user_inputs = {}

# Imagined Price Input
imagined_price = st.number_input("Enter your imagined price for the listing ($)", min_value=1.0, value=100.0, step=5.0)

# Input for Top 10 Features
for feature in top_10_features:
    is_categorical = False
    for cat_name in categorical_options.keys():
        if feature == cat_name:
            is_categorical = True
            original_cat_feature = cat_name
            break

    if is_categorical:
        # Ensure the feature is in the keys of categorical_options
        if original_cat_feature in categorical_options:
            options = categorical_options[original_cat_feature]
            user_inputs[original_cat_feature] = st.selectbox(f"Select {original_cat_feature.replace('_', ' ').title()}", options, index=0)
        else:
            # Fallback if a top feature is somehow identified as categorical but not in categorical_options
            user_inputs[feature] = st.text_input(f"Enter {feature.replace('_', ' ').title()}")
    else:
        # Numerical input, using median as default value for better user experience
        default_value = float(feature_defaults.get(feature, 0.0))
        if default_value == 0.0 and 'availability' in feature: # Handle availability to show full availability as default
            default_value = 30 if '30' in feature else (60 if '60' in feature else (90 if '90' in feature else 365))
        elif 'distance_to_center_km' in feature: # Give a reasonable default distance
            default_value = 5.0
        
        # Clamp default_value to be at least 0 if it somehow ends up negative
        default_value = max(0.0, default_value)

        user_inputs[feature] = st.number_input(f"Enter value for {feature.replace('_', ' ').title()}",
                                                min_value=0.0,
                                                value=float(default_value),
                                                step=1.0)


# Prediction Button
if st.button("Predict Price & Score"):    
    # Run prediction and scoring
    predicted_price, fairness_rating_label, fairness_score = predict_and_score_listing(
        imagined_price, user_inputs
    )

    st.header("2. Prediction Results")
    st.metric(label="Your Imagined Price", value=f"${imagined_price:,.2f}")
    st.metric(label="Predicted Market Price", value=f"${predicted_price:,.2f}")
    
    price_ratio = imagined_price / predicted_price
    st.metric(label="Price Ratio (Imagined / Predicted)", value=f"{price_ratio:.2f}")

    if fairness_score == 1:
        st.success(f"**Fairness Rating: {fairness_rating_label} (Score: {fairness_score})** 🚀\nThis is a great deal! Your price is significantly lower than the market prediction.")
    elif fairness_score == 2:
        st.info(f"**Fairness Rating: {fairness_rating_label} (Score: {fairness_score})** ✨\nYour price is slightly lower than the market prediction.")
    elif fairness_score == 3:
        st.success(f"**Fairness Rating: {fairness_rating_label} (Score: {fairness_score})** 👍\nYour price is fair and aligns with the market prediction.")
    elif fairness_score == 4:
        st.warning(f"**Fairness Rating: {fairness_rating_label} (Score: {fairness_score})** 💸\nYour price is slightly higher than the market prediction. Consider adjusting it.")
    else:
        st.error(f"**Fairness Rating: {fairness_rating_label} (Score: {fairness_score})** 🚨\nYour price is significantly higher than the market prediction. It might be challenging to attract bookings.")

    st.markdown(f"*Based on your inputs, the model suggests your imagined price is **{fairness_rating_label.lower()}**.*")