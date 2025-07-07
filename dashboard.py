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
sizes = np.arange(1, 11)
linear_preds = linear_model.predict(sizes.reshape(-1, 1))
poly_preds = poly_model(sizes)

fig, ax = plt.subplots()
ax.plot(sizes, linear_preds, label="Linear Model", linestyle="--", marker="o")
ax.plot(sizes, poly_preds, label="Polynomial Model", linestyle=":", marker="s")
ax.set_xlabel("Market Size")
ax.set_ylabel("Revenue Forecast")
ax.set_title("Forecast Model Comparison")
ax.legend()
st.pyplot(fig)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 🎯 App Header
st.markdown("<h1 style='text-align: center;'>📈 AI Strategy Forecast Simulator</h1>", unsafe_allow_html=True)

# 📑 Tab Layout
tab1, tab2 = st.tabs(["📊 Forecasting Hub", "📦 Scenario Packs"])

with tab1:
    st.subheader("Forecast Model Comparison")

    # Model selection
    model_type = st.selectbox("Select Forecast Type", ["Linear", "Polynomial"])
    market_size = st.slider("Market Size", 0, 10, 5)
    sentiment = st.selectbox("Market Sentiment", ["Optimistic", "Neutral", "Pessimistic"])
    adjustment = {"Optimistic": 1.2, "Neutral": 1.0, "Pessimistic": 0.8}[sentiment]

    # Model logic
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5, 7, 9, 11, 13])
    linear_model = LinearRegression().fit(X, y)
    poly_model = np.poly1d(np.polyfit(X.flatten(), y, deg=2))

    # Prediction based on model type
    if model_type == "Linear":
        prediction = linear_model.predict(np.array([[market_size]]))[0]
    else:
        prediction = poly_model(market_size)

    adjusted_prediction = prediction * adjustment
    confidence = np.random.uniform(85, 99)

    # Outputs
    st.success(f"{model_type} Forecast: ${prediction:.2f}M")
    st.info(f"Adjusted for Sentiment ({sentiment}): ${adjusted_prediction:.2f}M")
    st.write(f"🔐 Model Confidence: {confidence:.1f}%")

    # 📈 Plot comparison
    sizes = np.arange(1, 11)
    linear_preds = linear_model.predict(sizes.reshape(-1, 1))
    poly_preds = poly_model(sizes)

    fig, ax = plt.subplots()
    ax.plot(sizes, linear_preds, "--o", label="Linear Model")
    ax.plot(sizes, poly_preds, ":s", label="Polynomial Model")
    ax.set_xlabel("Market Size")
    ax.set_ylabel("Revenue Forecast")
    ax.set_title("Forecast Model Trends")
    ax.legend()
    st.pyplot(fig)

    # 📥 Downloadable CSV report
    report = pd.DataFrame({
        "Market Size": [market_size],
        "Forecast Model": [model_type],
        "Sentiment": [sentiment],
        "Adjusted Revenue": [adjusted_prediction],
        "Confidence (%)": [confidence]
    })
    csv = report.to_csv(index=False)
    st.download_button("📥 Download Full Forecast Report", data=csv, file_name="forecast_report.csv", mime="text/csv")

    # 🧠 Strategy recommendation logic
    st.markdown("### 🧠 Strategy Recommendation")
    if sentiment == "Optimistic" and adjusted_prediction > 15:
        st.success("✅ Consider aggressive expansion—market conditions are favorable.")
    elif sentiment == "Pessimistic":
        st.warning("⚠️ Focus on retention and efficiency—revenue outlook is constrained.")
    else:
        st.info("📈 Maintain steady growth and monitor sentiment trends.")

with tab2:
    st.subheader("🏪 Retail Scenario")
    base_cost = st.number_input("Cost per Unit ($)", min_value=0.0)
    units = st.slider("Units Sold", 0, 5000, 1000)
    retail_margin = 1.5
    retail_revenue = units * base_cost * retail_margin
    st.write(f"Estimated Retail Revenue: ${retail_revenue:.2f}")

    st.subheader("🌐 SaaS Scenario")
    subs = st.slider("Monthly Subscribers", 0, 10000, 5000)
    price = st.number_input("Subscription Price ($)", value=49.0)
    churn = st.slider("Churn Rate (%)", 0, 100, 5)
    active_subs = subs * (1 - churn / 100)
    saas_revenue = active_subs * price
    st.write(f"Estimated SaaS Revenue: ${saas_revenue:.2f}")
