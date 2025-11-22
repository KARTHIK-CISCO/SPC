import streamlit as st
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
except ModuleNotFoundError:
    st.error("⚠️ scikit-learn is not installed. Add it to requirements.txt")
    st.stop()

import plotly.graph_objects as go

st.title("📈 Stock Price Prediction (Smooth Forecast)")
st.write("Upload your stock data (Date, Open, High, Low, Close, Volume).")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    date_col = [col for col in df.columns if 'date' in col.lower()]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
        df.dropna(subset=[date_col[0]], inplace=True)
        df.rename(columns={date_col[0]: 'Date'}, inplace=True)
    else:
        st.warning("No 'Date' column found in the uploaded CSV. Ensure your CSV has a 'Date' column.")
        st.stop()

    features = ['Open', 'High', 'Low', 'Volume']
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)

    features_ml = features + [f'lag_{i}' for i in range(1, 6)]
    X_ml = df[features_ml]
    y_ml = df['Close']
    rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7]
        }

    # Set parameters directly as you requested
    rf = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42)
    rf.fit(X_ml, y_ml)

    y_pred_in_sample = rf.predict(X_ml)
    rmse = np.sqrt(mean_squared_error(y_ml, y_pred_in_sample))
    r2 = r2_score(y_ml, y_pred_in_sample)
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    st.subheader("📅 Select Forecast Days")
    forecast_days = st.slider("Forecast Days", 1, 90, 30)

    last_row = X_ml.iloc[-1].copy()
    forecast = []
    prev_close = df['Close'].iloc[-1]
    for i in range(forecast_days):
        pred = rf.predict([last_row.values])[0]
        smoothed_pred = 0.7 * prev_close + 0.3 * pred
        forecast.append(smoothed_pred)
        prev_close = smoothed_pred
        for lag in range(5, 1, -1):
            last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        last_row['lag_1'] = smoothed_pred

    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": forecast})

    st.subheader("🔹 Pick Forecast Day")
    specific_day = st.slider("Forecast day:", 1, forecast_days, 1)
    st.write(f"Predicted Close Price Day {specific_day} ({future_dates[specific_day-1].date()}): **{forecast[specific_day-1]:.2f}**")
    
    df = df.sort_values('Date')
    

    st.subheader("📈 Interactive Smoothed Forecast Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines+markers',
        name='Forecast (Smoothed)',
        line=dict(color='green', dash='dot'),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=[future_dates[specific_day-1]],
        y=[forecast[specific_day-1]],
        mode='markers',
        name=f'Day {specific_day} Forecast',
        marker=dict(color='red', size=11, symbol='circle')
    ))
    fig.update_layout(
        title="Stock Price Forecast (Smoothed)",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f'forecast_{forecast_days}_days.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to proceed.")
