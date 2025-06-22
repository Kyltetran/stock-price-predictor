# [LSTM-Based Nasdaq Stock Price Predictor]([url](https://nasdaq-10-stock-price-predictor.streamlit.app))
<img width="1395" alt="image" src="https://github.com/user-attachments/assets/dc479780-a9eb-440e-b9a5-34ea2fe74355" />

This is a web application that predicts future stock prices of technology companies listed on the Nasdaq using a Long Short-Term Memory (LSTM) neural network.

## What It Does

Given the ticker symbol of a technology company (e.g., `AAPL`, `MSFT`, `GOOGL`), the app:

- Retrieves historical stock data using **Yahoo Finance**
- Uses a trained **LSTM model** to forecast future prices
- Displays a plot comparing actual vs predicted prices

## Why I Built This

This project was built as part of an academic exploration into time-series forecasting using deep learning. I chose the **LSTM model** for its strength in learning long-term dependencies in sequential financial data.

## How It Works

- **Data Source**: Yahoo Finance via the `yfinance` Python package
- **Features Used**: `Open`, `High`, `Low`, `Close`, and `Volume`
- **Preprocessing**:
  - Custom time-series splitting to preserve date order
  - Normalization and denormalization for accurate prediction and visualization
- **Model**: LSTM neural network trained on historical data from multiple technology companies

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, TensorFlow (LSTM)
- **Deployment**: Render (free hosting)

## Try It Yourself

1. Open the website.
2. Choose a ticker.
3. Choose how many days ahead you want to predict.
4. View the plot showing the predicted vs actual price.

## Notes

- This is a **learning project**, not financial advice.
- The model focuses on overall trends, not short-term trading accuracy.

## Feedback

If you have suggestions or want to collaborate, feel free to reach out!
