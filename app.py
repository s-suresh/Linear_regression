import streamlit as st
import pickle
import numpy as np
import sklearn
# Load trained Linear Regression model
with open("linear_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🏡 House Price Predictor")
st.write("Enter the median income to predict the price.")

# User input
MedInc = st.number_input("Enter Median Income in $:", min_value=1.064848, max_value=14.996047)

if st.button("Predict Price"):
    prediction = model.predict(np.array([[MedInc]]))[0]
    st.success(f"💰 Predicted Price: ${float(prediction)}")


