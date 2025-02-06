import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib

# Load trained Linear Regression model
with open("linear_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🏡 House Price Predictor")
st.write("Enter the median income to predict the price.")

# User input for median income
MedInc = st.number_input("Enter Median Income in $:", min_value=1.064848, max_value=14.996047)

if st.button("Predict Price"):
    prediction = model.predict(np.array([[MedInc]]))[0]
    st.success(f"💰 Predicted Price: ${float(prediction):,.2f}")
prediction=model.predict(np.array([[MedInc]]))[0]
# Generate a range of median income values for line plot (increase points for smooth curve)
medinc_values = np.linspace(1.064848, 14.996047, 100).reshape(-1, 1)

# Predict prices for the range of median income values
predicted_prices = model.predict(medinc_values)

# Create a DataFrame for the data (flattening arrays to 1D)
data = pd.DataFrame({
    'Median Income ($)': medinc_values.flatten(),  # Flattened to 1D array
    'Predicted Price ($)': predicted_prices.flatten()  # Flattened to 1D array
})

# Display the line plot with highlighted points
st.write("### Predicted House Prices for Different Median Incomes")

# Create the figure and axes for plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the line
ax.plot(data['Median Income ($)'], data['Predicted Price ($)'], label='Predicted Price', color='blue')

# Highlight points on the line with red markers
ax.scatter(MedInc, prediction, color='red', zorder=5, label="Data Points")

# Add labels and title
ax.set_xlabel('Median Income ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('House Price Prediction Based on Median Income')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)




