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