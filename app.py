import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
#from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt

# ---------------------------
#  Streamlit UI
# ---------------------------
st.title("📈 Stock Price Prediction using Random Forest")
st.write("Upload your stock dataset (Date, Open, High, Low, Close, Volume).")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv("/content/AAPL.csv")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    #  Feature Engineering
    # ---------------------------
    st.subheader("🔧 Feature Engineering (Lag Creation)")

    df['Date'] = pd.to_datetime(df['Date'])

    features = ['Open', 'High', 'Low', 'Volume']

    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    df.dropna(inplace=True)

    features_ml = features + [f'lag_{i}' for i in range(1, 6)]
    X_ml = df[features_ml]
    y_ml = df['Close']

    # ---------------------------
    # Train-Test Split (Last 30 days)
    # ---------------------------
    split_ix = -30
    X_train, X_test = X_ml.iloc[:split_ix], X_ml.iloc[split_ix:]
    y_train, y_test = y_ml.iloc[:split_ix], y_ml.iloc[split_ix:]

    # ---------------------------
    # Random Forest Tuning
    # ---------------------------
    st.subheader("🎯 Hyperparameter Tuning with GridSearchCV")

    tscv = TimeSeriesSplit(n_splits=5)

    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }

    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_squared_error')
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_

    st.write("Best Parameters:", rf_grid.best_params_)

    # ---------------------------
    # Predictions
    # ---------------------------
    rf_pred = best_rf.predict(X_test)

    # ---------------------------
    # Metrics
    # ---------------------------
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    rf_mape = mape(y_test, rf_pred)
    rf_accuracy = (1 - rf_rmse) * 100

    st.subheader("📊 Model Performance")
    st.write(f"**RMSE:** {rf_rmse:.6f}")
    st.write(f"**MAPE:** {rf_mape:.2f}%")
    st.write(f"**Accuracy:** {rf_accuracy:.2f}%")

    # ---------------------------
    # Plot Actual vs Predicted
    # ---------------------------
    st.subheader("📉 Actual vs Predicted Close Price")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values, label='Actual Close', color='black')
    ax.plot(rf_pred, label='RF Tuned Pred', color='blue')
    ax.set_title("Random Forest: Actual vs Predicted")
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Close Price")

    plt.legend()
    st.pyplot(fig)



else:
    st.info("Please upload a CSV file to proceed.")

