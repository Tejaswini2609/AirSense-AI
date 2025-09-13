import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load model and feature names
model = joblib.load("flight_fare_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # ensures correct 18 columns

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")

st.title("✈️ Flight Fare Prediction System")
st.write("Enter the flight details below to predict the fare.")

# ------------------------------
# Input fields
# ------------------------------

journey_date = st.date_input("Date of Journey", min_value=datetime.today())
journey_day = journey_date.day
journey_month = journey_date.month

dep_time = st.time_input("Departure Time")
arrival_time = st.time_input("Arrival Time")

dep_datetime = datetime.combine(journey_date, dep_time)
arrival_datetime = datetime.combine(journey_date, arrival_time)
if arrival_datetime < dep_datetime:
    arrival_datetime = arrival_datetime + pd.Timedelta(days=1)
duration = (arrival_datetime - dep_datetime).seconds / 3600

airline = st.selectbox(
    "Airline",
    ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "GoAir", "Vistara"]
)
source = st.selectbox("Source", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
destination = st.selectbox("Destination", ["Cochin", "Kolkata", "Delhi", "Hyderabad", "Bangalore"])
total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

# ------------------------------
# Convert to DataFrame
# ------------------------------
input_dict = {
    "Total_Stops": total_stops,
    "Duration": duration,
    "Journey_day": journey_day,
    "Journey_month": journey_month,
    "Airline": airline,
    "Source": source,
    "Destination": destination
}

input_df = pd.DataFrame([input_dict])

# One-hot encode exactly like training
input_encoded = pd.get_dummies(input_df)

# Reindex to match training columns (fill missing with 0)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Fare"):
    prediction = model.predict(input_encoded)
    st.success(f"Predicted Flight Fare: ₹{prediction[0]:,.2f}")
