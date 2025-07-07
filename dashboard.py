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

    # Model selection with unique keys
    model_type = st.selectbox("Select Forecast Type", ["Linear", "Polynomial"], key="forecast_model")
    market_size = st.slider("Market Size", 0, 10, 5, key="market_slider")
    sentiment = st.selectbox("Market Sentiment", ["Optimistic", "Neutral", "Pessimistic"], key="sentiment_selector")

    adjustment = {"Optimistic": 1.2, "Neutral": 1.0, "Pessimistic": 0.8}[sentiment]

    # Model setup
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5, 7, 9, 11, 13])
    linear_model = LinearRegression().fit(X, y)
    poly_model = np.poly1d(np.polyfit(X.flatten(), y, deg=2))

    if model_type == "Linear":
        prediction = linear_model.predict(np.array([[market_size]]))[0]
    else:
        prediction = poly_model(market_size)

    adjusted_prediction = prediction * adjustment
    confidence = np.random.uniform(85, 99)

    # Display results
    st.success(f"{model_type} Forecast: ${prediction:.2f}M")
    st.info(f"Adjusted for Sentiment ({sentiment}): ${adjusted_prediction:.2f}M")
    st.write(f"🔐 Model Confidence: {confidence:.1f}%")

    # Plot comparison
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

    # CSV Download
    report = pd.DataFrame({
        "Market Size": [market_size],
        "Forecast Model": [model_type],
        "Sentiment": [sentiment],
        "Adjusted Revenue": [adjusted_prediction],
        "Confidence (%)": [confidence]
    })
    csv = report.to_csv(index=False)
    st.download_button("📥 Download Full Forecast Report", data=csv, file_name="forecast_report.csv", mime="text/csv", key="download_button")

    # Recommendations
    st.markdown("### 🧠 Strategy Recommendation")
    if sentiment == "Optimistic" and adjusted_prediction > 15:
        st.success("✅ Consider aggressive expansion—market conditions are favorable.")
    elif sentiment == "Pessimistic":
        st.warning("⚠️ Focus on retention and efficiency—revenue outlook is constrained.")
    else:
        st.info("📈 Maintain steady growth and monitor sentiment trends.")

with tab2:
    st.subheader("🏪 Retail Scenario")
    base_cost = st.number_input("Cost per Unit ($)", min_value=0.0, key="retail_cost")
    units = st.slider("Units Sold", 0, 5000, 1000, key="retail_units")
    retail_margin = 1.5
    retail_revenue = units * base_cost * retail_margin
    st.write(f"Estimated Retail Revenue: ${retail_revenue:.2f}")

    st.subheader("🌐 SaaS Scenario")
    subs = st.slider("Monthly Subscribers", 0, 10000, 5000, key="saas_subs")
    price = st.number_input("Subscription Price ($)", value=49.0, key="saas_price")
    churn = st.slider("Churn Rate (%)", 0, 100, 5, key="saas_churn")
    active_subs = subs * (1 - churn / 100)
    saas_revenue = active_subs * price
    st.write(f"Estimated SaaS Revenue: ${saas_revenue:.2f}")
