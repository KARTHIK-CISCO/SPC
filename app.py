import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 Stock Price Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Feature engineering
    features = ['Open', 'High', 'Low', 'Volume']
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)

    X_ml = df[features + [f'lag_{i}' for i in range(1,6)]]
    y_ml = df['Close']

    # Train last 80% of data, predict last 20%
    split = int(len(df)*0.8)
    X_train, X_test = X_ml[:split], X_ml[split:]
    y_train, y_test = y_ml[:split], y_ml[split:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # Prediction
    df['Predicted'] = rf.predict(X_ml)

    # ---------------------------
    # Select date for prediction
    # ---------------------------
    st.subheader("📅 Select Date for Prediction")
    selected_date = st.date_input("Select a date", df['Date'].max())
    predicted_value = df[df['Date'] == pd.to_datetime(selected_date)]['Predicted'].values
    if predicted_value.size > 0:
        st.write(f"Predicted Close Price for {selected_date}: ₹{predicted_value[0]:.2f}")
    else:
        st.write("Date not in dataset. Please select a valid date.")

    # ---------------------------
    # Plot historical + predicted
    # ---------------------------
    st.subheader("📉 Historical Close vs Predicted Close")
    plt.figure(figsize=(12,5))
    plt.plot(df['Date'], df['Close'], label='Historical Close', color='blue')
    plt.plot(df['Date'], df['Predicted'], label='Predicted Close', color='green')
    plt.scatter(selected_date, predicted_value, color='red', s=100, label='Selected Day Prediction')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Stock Price: Historical vs Predicted")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
