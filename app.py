import streamlit as st
import pandas as pd
import pickle

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("model/num_imputer.pkl", "rb") as f:
    num_imputer = pickle.load(f)

with open("model/cat_imputer.pkl", "rb") as f:
    cat_imputer = pickle.load(f)

st.title("House Price Prediction")

area = st.number_input("Area (sq ft)", min_value=300.0)
bedrooms = st.number_input("Bedrooms", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
stories = st.number_input("Stories", min_value=1)
parking = st.number_input("Parking", min_value=0)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

input_df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = input_df.select_dtypes(include=["object"]).columns

input_df[num_cols] = num_imputer.transform(input_df[num_cols])
input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

input_df = pd.get_dummies(input_df, drop_first=True)

for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

input_scaled = scaler.transform(input_df)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted House Price: ₹ {int(prediction[0]):,}")
