import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import yfinance as yf

def predict_stock_price(stock_name, ahead, d):
    stock = yf.Ticker(stock_name)
    hist = stock.history(period=f"{d}y")
    df = hist

    n = int(hist.shape[0] * 0.8)
    training_set = df.iloc[:n, 1:2].values
    test_set = df.iloc[n:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set_scaled = sc.transform(test_set)

    X_train = []
    y_train = []
    for i in range(60, n - ahead):  # Assuming 60 trading days per month
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i + ahead, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = load_model("keras_model.keras")

    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    df['Date'] = df.index

    # Visualize training data
    fig_train, ax_train = plt.subplots(figsize=(16, 6))
    ax_train.plot(df.iloc[:n].index, dataset_train.values, label='Training Data')
    ax_train.set_title('Training Data', fontsize=16, fontweight='bold')
    ax_train.set_xlabel('Time', fontsize=14)
    ax_train.set_ylabel('Price', fontsize=14)
    ax_train.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)

    # Visualize testing data
    fig_test, ax_test = plt.subplots(figsize=(16, 6))
    ax_test.plot(df.iloc[n:].index, dataset_test.values, label='Testing Data')
    ax_test.set_title('Testing Data', fontsize=16, fontweight='bold')
    ax_test.set_xlabel('Time', fontsize=14)
    ax_test.set_ylabel('Price', fontsize=14)
    ax_test.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)

    # Visualize predictions
    fig_pred, ax_pred = plt.subplots(figsize=(16, 6))
    ax_pred.plot(df.iloc[n:].index, dataset_test.values, color='red', label='Actual Price')
    ax_pred.plot(df.iloc[n:].index, predicted_stock_price, color='blue', label='Predicted Price')
    ax_pred.set_title('Stock Price Prediction', fontsize=16, fontweight='bold')
    ax_pred.set_xlabel('Time', fontsize=14)
    ax_pred.set_ylabel('Price', fontsize=14)
    ax_pred.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)

    # Print next month's predicted stock price
    next_month_price = predicted_stock_price[-1][0]
    st.write(f"The predicted stock price for the next day is: ${next_month_price:.2f}")

    return fig_train, fig_test, fig_pred

def main():
    st.set_page_config(page_title="Stock Price Prediction", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title(":chart_with_upwards_trend: Stock Price Prediction")

    with st.container():
            stock_name = st.text_input("Enter stock name (e.g., AAPL, GOOG, TSLA)", key="stock_name")
    
            ahead = st.number_input("Enter number of months ahead to predict", min_value=1, value=1, step=1, key="ahead")

        
            d = st.number_input("Enter number of years of data to consider", min_value=1, value=5, step=1, key="data_years")
       

            predict_button = st.button("Predict the stock price of next day", key="predict_button")

    if predict_button:
        if stock_name:
            with st.spinner("Predicting stock price..."):
                fig_train, fig_test, fig_pred = predict_stock_price(stock_name, ahead, d)

            st.subheader("Training Data")
            st.pyplot(fig_train)

            st.subheader("Testing Data")
            st.pyplot(fig_test)

            st.subheader("Stock Price Prediction")
            st.pyplot(fig_pred)
        else:
            st.warning("Please enter a stock name.")

if __name__ == "__main__":
    main()
