from tensorflow.keras.losses import MeanSquaredError
from pyod.models.auto_encoder import AutoEncoder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import glob
import json
import warnings
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        color: #155724;
    }
    .stAlert > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions


@st.cache_data
def load_nasdaq_data():
    """Load the NASDAQ dataset"""
    possible_paths = [
        'processed_combined_nasdaq.csv',
        './stock-price-predictor/processed_combined_nasdaq.csv',
        './processed_combined_nasdaq.csv',
        '../processed_combined_nasdaq.csv'
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                nasdaq = pd.read_csv(path)
                return nasdaq
        except Exception:
            continue
    return None


@st.cache_data
def get_available_companies():
    """Get list of companies with saved models"""
    possible_model_dirs = [
        "saved_models",
        "./saved_models",
        "../saved_models",
    ]

    possible_metadata_dirs = [
        "model_metadata",
        "./model_metadata",
        "../model_metadata",
    ]

    model_dir = next(
        (d for d in possible_model_dirs if os.path.exists(d)), None)
    metadata_dir = next(
        (d for d in possible_metadata_dirs if os.path.exists(d)), None)

    if not model_dir or not metadata_dir:
        return []

    model_files = glob.glob(os.path.join(model_dir, "*.h5"))
    companies = []

    for model_file in model_files:
        try:
            model_filename = os.path.basename(model_file)
            company_name = model_filename.replace(
                '_model.h5', '').replace('_', ' ')
            metadata_filename = model_filename.replace(
                '_model.h5', '_metadata.json')
            metadata_path = os.path.join(metadata_dir, metadata_filename)

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    companies.append({
                        'name': metadata['company'],
                        'clean_name': company_name,
                        'mse': metadata['mse_score'],
                        'training_date': metadata['training_date']
                    })
        except Exception:
            continue

    return companies


@st.cache_resource
def load_saved_model(company):
    """Load a previously saved model"""
    possible_model_dirs = ["saved_models", "./saved_models", "../saved_models"]
    possible_metadata_dirs = ["model_metadata",
                              "./model_metadata", "../model_metadata"]

    clean_company_name = "".join(
        c for c in company if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_company_name = clean_company_name.replace(' ', '_')

    model_dir = next(
        (d for d in possible_model_dirs if os.path.exists(d)), None)
    metadata_dir = next(
        (d for d in possible_metadata_dirs if os.path.exists(d)), None)

    if not model_dir or not metadata_dir:
        return None, None

    model_path = os.path.join(model_dir, f"{clean_company_name}_model.h5")
    metadata_path = os.path.join(
        metadata_dir, f"{clean_company_name}_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None

    try:
        from tensorflow.keras.metrics import MeanSquaredError
        from tensorflow.keras.losses import MeanSquaredError as mse_loss

        custom_objects = {
            'mse': MeanSquaredError,
            'MeanSquaredError': MeanSquaredError,
            'mse_loss': mse_loss
        }

        try:
            model = load_model(model_path, custom_objects=custom_objects)
        except Exception:
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            except Exception:
                return None, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return model, metadata
    except Exception:
        return None, None


def create_demo_mode():
    """Create a demo mode with sample data when no models are available"""
    st.info("🎯 **Demo Mode** - No trained models found")

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    # Create sample predictions
    future_dates = pd.date_range(
        start=dates[-1] + timedelta(days=1), periods=7, freq='D')
    future_prices = prices[-1] + np.cumsum(np.random.randn(7) * 0.3)

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates[-30:],
        y=prices[-30:],
        mode='lines',
        name='Historical Prices',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines+markers',
        name='Predictions',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Demo: Sample Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        **To use the full application:**
        1. Train your models using the main notebook
        2. Ensure model files are saved in `saved_models/` directory
        3. Ensure metadata files are saved in `model_metadata/` directory
        4. Place `processed_combined_nasdaq.csv` in the root directory
        """)


def features_prediction(company_data, window_size):
    """Prepare features for prediction"""
    feature_columns = ['Open', 'High', 'Low', 'Volume']
    X_predict_original = company_data[feature_columns].iloc[-window_size:]
    X_predict = np.array(X_predict_original)

    close_index = feature_columns.index('High')
    X_predict = X_predict[:, close_index:close_index+1]
    X_predict = X_predict.reshape(1, window_size, 1)

    scaler = MinMaxScaler()
    X_predict_norm = scaler.fit_transform(
        X_predict.reshape(-1, X_predict.shape[2]))
    X_predict_norm = X_predict_norm.reshape(
        X_predict.shape[0], X_predict.shape[1], X_predict.shape[2])

    return X_predict, X_predict_norm, scaler


def predict_future_days(model, X_predict_norm, k):
    """Predict k days into the future"""
    predictions = []

    for i in range(k):
        y_pred_norm = model.predict(X_predict_norm, verbose=0)
        predictions.append(y_pred_norm[0])

        y_pred_reshaped = np.zeros((1, 1, X_predict_norm.shape[2]))
        y_pred_reshaped[0, 0, :1] = y_pred_norm
        X_predict_norm = np.concatenate(
            (X_predict_norm[:, 1:, :], y_pred_reshaped), axis=1)

    return np.array(predictions)

# Main App


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load data and companies
    nasdaq = load_nasdaq_data()
    companies = get_available_companies()

    if not companies or nasdaq is None:
        st.error("❌ Required data or models not found.")
        create_demo_mode()
        return

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration")

        # Company selection
        company_names = [comp['name'] for comp in companies]
        selected_company = st.selectbox("📊 Select Company", company_names)

        # Days to predict
        days_to_predict = st.slider(
            "📅 Days to Predict", min_value=1, max_value=30, value=7)

        # Show model info
        selected_company_info = next(
            comp for comp in companies if comp['name'] == selected_company)

        st.markdown("### 📋 Model Information")
        st.markdown(f"""
        <div class="model-info">
            <strong>Company:</strong> {selected_company}<br>
            <strong>MSE Score:</strong> {selected_company_info['mse']:.6f}<br>
            <strong>Training Date:</strong> {selected_company_info['training_date'][:10]}
        </div>
        """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 📈 Prediction Results")

        # Load model and make predictions
        with st.spinner("Loading model and generating predictions..."):
            model, metadata = load_saved_model(selected_company)

            if model is None:
                st.error("❌ Could not load the selected model.")
                st.stop()

            # Get company data
            company_data = nasdaq[nasdaq.company_name ==
                                  selected_company].copy()

            if company_data.empty:
                st.error(f"❌ No data found for company: {selected_company}")
                st.stop()

            # Handle missing values
            for column in company_data.columns:
                if np.issubdtype(company_data[column].dtype, np.number):
                    company_data[column] = company_data[column].fillna(
                        company_data[column].mean())
                else:
                    company_data[column] = company_data[column].fillna(
                        method='ffill')

            # Make predictions
            window_size = metadata['window_size']
            X_predict, X_predict_norm, scaler = features_prediction(
                company_data, window_size)
            predictions = predict_future_days(
                model, X_predict_norm, days_to_predict)

            # Create prediction dataframe
            last_date = pd.to_datetime(company_data['Date'].iloc[-1])
            future_dates = [
                last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]

            # Get recent actual prices for comparison
            recent_data = company_data.tail(30)
            recent_dates = pd.to_datetime(recent_data['Date'])
            recent_prices = recent_data['Close'].values

        # Plot predictions
        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=recent_dates,
            y=recent_prices,
            mode='lines',
            name='Historical Prices',
            line=dict(color='#1f77b4', width=2)
        ))

        # Predicted prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions.flatten(),
            mode='lines+markers',
            name=f'Predicted Prices ({days_to_predict} days)',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8)
        ))

        # Add vertical line to separate historical and predicted
        fig.add_shape(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dot"),
        )

        # Add annotation for the vertical line
        fig.add_annotation(
            x=last_date,
            y=max(recent_prices),
            text="Prediction Start",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            bgcolor="white",
            bordercolor="gray"
        )

        fig.update_layout(
            title=f'Stock Price Prediction for {selected_company}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("## 📊 Key Metrics")

        # Current price info
        current_price = company_data['Close'].iloc[-1]
        predicted_price = predictions[-1][0]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        # Prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <h3>💰 Price Prediction</h3>
            <h2>${predicted_price:.2f}</h2>
            <p>in {days_to_predict} days</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Expected Change",
                  f"${price_change:.2f}", f"{price_change_pct:.1f}%")
        st.metric("Model Accuracy (MSE)", f"{metadata['mse_score']:.6f}")

        # Prediction table
        st.markdown("### 📅 Daily Predictions")
        pred_df = pd.DataFrame({
            'Day': range(1, days_to_predict + 1),
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Predicted Price': [f"${p[0]:.2f}" for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True)

        # Download predictions
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f'{selected_company}_predictions.csv',
            mime='text/csv'
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>📈 Stock Price Predictor | Built with Streamlit & TensorFlow</p>
        <p><small>⚠️ This is for educational purposes only. Not financial advice.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
