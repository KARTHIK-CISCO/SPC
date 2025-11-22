#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np

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
# Import Plotly
# ---------------------------
import plotly.graph_objects as go

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
        df.rename(columns={date_col[0]: 'Date'}, inplace=True)
    else:
        st.warning("No 'Date' column found in the uploaded CSV. Ensure your CSV has a 'Date' column.")
        st.stop()

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
    # Forecast Days Slider
    # ---------------------------
    st.subheader("📅 Select Forecast Days")
    forecast_days = st.slider("Number of days to forecast:", min_value=1, max_value=90, value=30)

    # ---------------------------
    # Forecast Next N Days
    # ---------------------------
    last_row = X_ml.iloc[-1].copy()
    forecast = []

    for i in range(forecast_days):
        pred = best_rf.predict([last_row.values])[0]
        forecast.append(pred)
        # Shift lag features
        for lag in range(5, 1, -1):
            last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        last_row['lag_1'] = pred

    # ---------------------------
    # Prepare Future Dates
    # ---------------------------
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": forecast})

    # ---------------------------
    # Select Specific Forecast Day
    # ---------------------------
    st.subheader("🔹 Select Specific Forecast Day to View Prediction")
    specific_day = st.slider("Select forecast day:", min_value=1, max_value=forecast_days, value=1)

    specific_forecast_date = future_dates[specific_day-1]
    specific_forecast_value = forecast[specific_day-1]

    st.write(f"Predicted Close Price for Day {specific_day} ({specific_forecast_date.date()}): **{specific_forecast_value:.2f}**")

    # ---------------------------
    # Plot Historical + Forecast with Plotly
    # ---------------------------
    st.subheader("📈 Interactive Forecast Plot")

    fig = go.Figure()

    # Historical Close
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='blue')
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green', dash='dash'),
        marker=dict(size=6)
    ))

    # Highlight specific forecast day
    fig.add_trace(go.Scatter(
        x=[specific_forecast_date],
        y=[specific_forecast_value],
        mode='markers',
        name=f'Day {specific_day} Forecast',
        marker=dict(color='red', size=12, symbol='circle')
    ))

    fig.update_layout(
        title="Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template='plotly_white',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Optional: Download Forecast CSV
    # ---------------------------
    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f'forecast_{forecast_days}_days.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to proceed.")
