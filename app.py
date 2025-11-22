#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Safe imports for sklearn
# ---------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
except ModuleNotFoundError:
    st.error("⚠️ scikit-learn is not installed. Add it to requirements.txt")
    st.stop()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 Stock Price Prediction using Random Forest")
st.write("Upload your stock dataset (Date, Open, High, Low, Close, Volume).")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Date Parsing
    # ---------------------------
    date_col = [col for col in df.columns if 'date' in col.lower()]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
        df.dropna(subset=[date_col[0]], inplace=True)
    else:
        st.warning("No 'Date' column found in the uploaded CSV.")

    # ---------------------------
    # Feature Engineering (Lag Features)
    # ---------------------------
    st.subheader("🔧 Feature Engineering (Lag Creation)")
    features = ['Open', 'High', 'Low', 'Volume']

    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)

    features_ml = features + [f'lag_{i}' for i in range(1, 6)]
    X_ml = df[features_ml]
    y_ml = df['Close']

    # ---------------------------
    # Random Forest Hyperparameter Tuning
    # ---------------------------
    st.subheader("🎯 Hyperparameter Tuning with GridSearchCV")
    tscv = TimeSeriesSplit(n_splits=5)
    rf_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}

    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_squared_error')
    rf_grid.fit(X_ml, y_ml)
    best_rf = rf_grid.best_estimator_
    st.write("Best Parameters:", rf_grid.best_params_)

    # ---------------------------
    # In-sample Performance
    # ---------------------------
    y_pred_in_sample = best_rf.predict(X_ml)
    rmse = np.sqrt(mean_squared_error(y_ml, y_pred_in_sample))
    mape = np.mean(np.abs((y_ml - y_pred_in_sample)/y_ml))*100
    r2 = r2_score(y_ml, y_pred_in_sample)

    st.subheader("📊 Model Performance (In-Sample)")
    st.write(f"**RMSE:** {rmse:.6f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.4f}")

    # ---------------------------
    # Forecast Next 30 Days
    # ---------------------------
    st.subheader("📅 Forecast for Next 30 Days")
    forecast_days = 30
    last_row = X_ml.iloc[-1].copy()
    forecast = []

    for i in range(forecast_days):
        pred = best_rf.predict([last_row.values])[0]
        forecast.append(pred)
        # Shift lag features
        for lag in range(5, 1, -1):
            last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        last_row['lag_1'] = pred

    st.write(forecast)

    # ---------------------------
    # Plot Historical + Forecast
    # ---------------------------
    plt.figure(figsize=(12,5))
    plt.plot(df['Date'], df['Close'], label='Historical Close', color='blue')
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    plt.plot(future_dates, forecast, label='Forecast 30 Days', marker='o', color='green')
    plt.title("Stock Price: Historical + 30 Days Forecast")
    plt.x
