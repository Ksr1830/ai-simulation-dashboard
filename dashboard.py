import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Market Strategy AI Agent")

# Simulated data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

model = LinearRegression().fit(X, y)

user_input = st.slider("Choose a market size", 0, 10)
prediction = model.predict(np.array([[user_input]]))

st.write(f"Predicted Revenue: ${prediction[0]:.2f}M")
sentiment = st.selectbox("Market Sentiment", ["Optimistic", "Neutral", "Pessimistic"])
adjustment = {"Optimistic": 1.2, "Neutral": 1.0, "Pessimistic": 0.8}[sentiment]
adjusted_prediction = prediction[0] * adjustment
st.write(f"Adjusted Revenue based on Sentiment: ${adjusted_prediction:.2f}M")
import matplotlib.pyplot as plt

# Data for trend visualization
sizes = np.arange(1, 11)
revenues = model.predict(sizes.reshape(-1, 1))
adjusted_revenues = revenues * adjustment

# Create plot
fig, ax = plt.subplots()
ax.plot(sizes, adjusted_revenues, marker='o', color='teal')
ax.set_xlabel("Market Size")
ax.set_ylabel("Adjusted Revenue (Millions)")
ax.set_title("Market Size vs Revenue Forecast")

# Show plot
st.pyplot(fig)
st.header("📊 Forecast Model Comparison")

model_type = st.selectbox("Select Forecast Type", ["Linear", "Polynomial"])
market_size = st.slider("Market Size", 0, 10, 5)

# Define training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

# Create models
linear_model = LinearRegression().fit(X, y)
poly_features = np.polyfit(X.flatten(), y, deg=2)
poly_model = np.poly1d(poly_features)

# Generate predictions
if model_type == "Linear":
    prediction = linear_model.predict(np.array([[market_size]]))[0]
else:
    prediction = poly_model(market_size)

st.success(f"{model_type} Forecast: ${prediction:.2f}M")
