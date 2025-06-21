# from tensorflow.keras.losses import MeanSquaredError
# from pyod.models.auto_encoder import AutoEncoder
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime, timedelta
# import os
# import glob
# import json
# import warnings
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import yfinance as yf

# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Stock Price Predictor",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .prediction-box {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 1rem;
#         text-align: center;
#         margin: 1rem 0;
#     }
#     .debug-info {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #007bff;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Helper Functions


# def check_directory_structure():
#     """Check and display current directory structure"""
#     current_dir = os.getcwd()
#     st.markdown("### 🔍 Directory Structure Debug")

#     with st.expander("Current Directory Structure", expanded=False):
#         st.code(f"Current working directory: {current_dir}")

#         # Check for required directories
#         required_dirs = ['saved_models', 'model_metadata']
#         for dir_name in required_dirs:
#             if os.path.exists(dir_name):
#                 files = os.listdir(dir_name)
#                 st.success(f"✅ {dir_name}/ exists with {len(files)} files")
#                 if files:
#                     st.write("Files:")
#                     for file in files[:10]:  # Show first 10 files
#                         st.write(f"  - {file}")
#             else:
#                 st.error(f"❌ {dir_name}/ directory not found")

#         # Check for CSV file
#         csv_files = ['processed_combined_nasdaq.csv',
#                      './stock-price-predictor/processed_combined_nasdaq.csv']
#         csv_found = False
#         for csv_file in csv_files:
#             if os.path.exists(csv_file):
#                 st.success(f"✅ Found CSV: {csv_file}")
#                 csv_found = True
#                 break
#         if not csv_found:
#             st.error("❌ processed_combined_nasdaq.csv not found")


# @st.cache_data
# def load_nasdaq_data():
#     """Load the NASDAQ dataset"""
#     # Try multiple possible locations
#     possible_paths = [
#         'processed_combined_nasdaq.csv',
#         './stock-price-predictor/processed_combined_nasdaq.csv',
#         './processed_combined_nasdaq.csv',
#         '../processed_combined_nasdaq.csv'
#     ]

#     for path in possible_paths:
#         try:
#             if os.path.exists(path):
#                 nasdaq = pd.read_csv(path)
#                 st.success(f"✅ Successfully loaded data from: {path}")
#                 return nasdaq
#         except Exception as e:
#             st.warning(f"⚠️ Could not load from {path}: {e}")

#     st.error("❌ NASDAQ dataset not found in any expected location.")
#     st.info(
#         "Please ensure 'processed_combined_nasdaq.csv' is in one of these locations:")
#     for path in possible_paths:
#         st.write(f"  - {path}")
#     return None


# @st.cache_data
# def get_available_companies():
#     """Get list of companies with saved models"""
#     # Check if directories exist
#     if not os.path.exists("saved_models"):
#         st.error("❌ 'saved_models' directory not found")
#         return []

#     if not os.path.exists("model_metadata"):
#         st.error("❌ 'model_metadata' directory not found")
#         return []

#     model_files = glob.glob("saved_models/*.h5")
#     st.info(f"🔍 Found {len(model_files)} model files")

#     companies = []
#     for model_file in model_files:
#         try:
#             company_name = os.path.basename(model_file).replace(
#                 '_model.h5', '').replace('_', ' ')
#             metadata_file = f"model_metadata/{os.path.basename(model_file).replace('_model.h5', '_metadata.json')}"

#             if os.path.exists(metadata_file):
#                 with open(metadata_file, 'r') as f:
#                     metadata = json.load(f)
#                     companies.append({
#                         'name': metadata['company'],
#                         'clean_name': company_name,
#                         'mse': metadata['mse_score'],
#                         'training_date': metadata['training_date']
#                     })
#                     st.success(f"✅ Loaded model for: {metadata['company']}")
#             else:
#                 st.warning(f"⚠️ Metadata file not found for: {company_name}")
#         except Exception as e:
#             st.error(f"❌ Error processing {model_file}: {e}")

#     return companies


# @st.cache_resource
# def load_saved_model(company):
#     """Load a previously saved model"""
#     clean_company_name = "".join(
#         c for c in company if c.isalnum() or c in (' ', '-', '_')).rstrip()
#     clean_company_name = clean_company_name.replace(' ', '_')

#     model_filename = f"saved_models/{clean_company_name}_model.h5"
#     metadata_filename = f"model_metadata/{clean_company_name}_metadata.json"

#     if not os.path.exists(model_filename) or not os.path.exists(metadata_filename):
#         st.error(f"❌ Model files not found for {company}")
#         st.write(f"Looking for: {model_filename}")
#         st.write(f"Looking for: {metadata_filename}")
#         return None, None

#     try:
#         # Define custom objects to handle compatibility issues
#         from tensorflow.keras.metrics import MeanSquaredError
#         from tensorflow.keras.losses import MeanSquaredError as mse_loss

#         custom_objects = {
#             'mse': MeanSquaredError,
#             'MeanSquaredError': MeanSquaredError,
#             'mse_loss': mse_loss
#         }

#         # Try loading with custom objects first
#         try:
#             model = load_model(model_filename, custom_objects=custom_objects)
#             st.success(f"✅ Successfully loaded model with custom objects")
#         except Exception as e1:
#             st.warning(f"⚠️ Failed with custom objects: {e1}")
#             # Try loading without custom objects
#             try:
#                 model = load_model(model_filename, compile=False)
#                 # Recompile the model with current TensorFlow version
#                 model.compile(
#                     optimizer='adam',
#                     loss='mse',
#                     metrics=['mse']
#                 )
#                 st.success(f"✅ Successfully loaded and recompiled model")
#             except Exception as e2:
#                 st.error(
#                     f"❌ Failed to load model even without compilation: {e2}")
#                 return None, None

#         with open(metadata_filename, 'r') as f:
#             metadata = json.load(f)
#         return model, metadata
#     except Exception as e:
#         st.error(f"❌ Error loading model: {e}")
#         return None, None


# def create_demo_mode():
#     """Create a demo mode with sample data when no models are available"""
#     st.warning("🔄 Running in Demo Mode - No trained models found")

#     # Generate sample data
#     dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
#     np.random.seed(42)
#     prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

#     # Create sample predictions
#     future_dates = pd.date_range(
#         start=dates[-1] + timedelta(days=1), periods=7, freq='D')
#     future_prices = prices[-1] + np.cumsum(np.random.randn(7) * 0.3)

#     # Plot
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=dates[-30:],
#         y=prices[-30:],
#         mode='lines',
#         name='Sample Historical Prices',
#         line=dict(color='#1f77b4', width=2)
#     ))

#     fig.add_trace(go.Scatter(
#         x=future_dates,
#         y=future_prices,
#         mode='lines+markers',
#         name='Sample Predictions',
#         line=dict(color='#ff7f0e', width=3, dash='dash'),
#         marker=dict(size=8)
#     ))

#     fig.update_layout(
#         title='Demo: Sample Stock Price Prediction',
#         xaxis_title='Date',
#         yaxis_title='Price ($)',
#         hovermode='x unified',
#         template='plotly_white',
#         height=500
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.info("""
#     📋 **To use the full application:**
#     1. Train your models using the main notebook
#     2. Ensure model files are saved in 'saved_models/' directory
#     3. Ensure metadata files are saved in 'model_metadata/' directory
#     4. Place 'processed_combined_nasdaq.csv' in the root directory
#     """)


# def features_prediction(company_data, window_size):
#     """Prepare features for prediction"""
#     feature_columns = ['Open', 'High', 'Low', 'Volume']

#     # Extract the last 'window_size' rows for the selected features
#     X_predict_original = company_data[feature_columns].iloc[-window_size:]

#     # Convert to numpy array
#     X_predict = np.array(X_predict_original)

#     # Use 'high' as the prediction feature (you can modify this)
#     close_index = feature_columns.index('High')
#     X_predict = X_predict[:, close_index:close_index+1]
#     X_predict = X_predict.reshape(1, window_size, 1)

#     # Normalize
#     scaler = MinMaxScaler()
#     X_predict_norm = scaler.fit_transform(
#         X_predict.reshape(-1, X_predict.shape[2]))
#     X_predict_norm = X_predict_norm.reshape(
#         X_predict.shape[0], X_predict.shape[1], X_predict.shape[2])

#     return X_predict, X_predict_norm, scaler


# def predict_future_days(model, X_predict_norm, k):
#     """Predict k days into the future"""
#     predictions = []

#     for i in range(k):
#         y_pred_norm = model.predict(X_predict_norm, verbose=0)
#         predictions.append(y_pred_norm[0])

#         # Prepare next input
#         y_pred_reshaped = np.zeros((1, 1, X_predict_norm.shape[2]))
#         y_pred_reshaped[0, 0, :1] = y_pred_norm
#         X_predict_norm = np.concatenate(
#             (X_predict_norm[:, 1:, :], y_pred_reshaped), axis=1)

#     return np.array(predictions)


# def get_recent_stock_data(symbol, days=30):
#     """Fetch recent stock data from Yahoo Finance"""
#     try:
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=days)
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(start=start_date, end=end_date)
#         return data
#     except:
#         return None

# # Main App


# def main():
#     # Header
#     st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>',
#                 unsafe_allow_html=True)
#     st.markdown("---")

#     # Debug information
#     with st.expander("🔧 Debug Information", expanded=False):
#         check_directory_structure()

#     # Load data
#     nasdaq = load_nasdaq_data()

#     # Get available companies
#     companies = get_available_companies()

#     if not companies:
#         st.error("❌ No trained models found.")

#         # Show available options
#         st.markdown("### 🔧 Setup Instructions")
#         st.markdown("""
#         To fix this issue:

#         1. **Create required directories:**
#         ```bash
#         mkdir saved_models
#         mkdir model_metadata
#         ```

#         2. **Train your models** using your main notebook

#         3. **Ensure your model files** are saved as `saved_models/CompanyName_model.h5`

#         4. **Ensure your metadata files** are saved as `model_metadata/CompanyName_metadata.json`

#         5. **Place your CSV file** as `processed_combined_nasdaq.csv` in the root directory
#         """)

#         # Show demo mode
#         st.markdown("---")
#         st.markdown("## 🎯 Demo Mode")
#         create_demo_mode()
#         return

#     # Continue with normal app flow if models are available
#     with st.sidebar:
#         st.image("https://via.placeholder.com/200x100/1f77b4/white?text=Stock+Predictor",
#                  use_column_width=True)
#         st.markdown("## 🔧 Configuration")

#         # Company selection
#         company_names = [comp['name'] for comp in companies]
#         selected_company = st.selectbox("📊 Select Company", company_names)

#         # Days to predict
#         days_to_predict = st.slider(
#             "📅 Days to Predict", min_value=1, max_value=30, value=7)

#         # Show model info
#         selected_company_info = next(
#             comp for comp in companies if comp['name'] == selected_company)
#         st.markdown("### 📋 Model Information")
#         st.info(f"""
#         **Company:** {selected_company}
#         **MSE Score:** {selected_company_info['mse']:.6f}
#         **Training Date:** {selected_company_info['training_date'][:10]}
#         """)

#     # Main content
#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.markdown("## 📈 Prediction Results")

#         # Load model and make predictions
#         with st.spinner("Loading model and making predictions..."):
#             model, metadata = load_saved_model(selected_company)

#             if model is None:
#                 st.error("❌ Could not load the selected model.")
#                 st.stop()

#             # Get company data
#             company_data = nasdaq[nasdaq.company_name ==
#                                   selected_company].copy()

#             if company_data.empty:
#                 st.error(f"❌ No data found for company: {selected_company}")
#                 st.stop()

#             # Handle missing values
#             for column in company_data.columns:
#                 if np.issubdtype(company_data[column].dtype, np.number):
#                     company_data[column] = company_data[column].fillna(
#                         company_data[column].mean())
#                 else:
#                     company_data[column] = company_data[column].fillna(
#                         method='ffill')

#             # Make predictions
#             window_size = metadata['window_size']
#             X_predict, X_predict_norm, scaler = features_prediction(
#                 company_data, window_size)
#             predictions = predict_future_days(
#                 model, X_predict_norm, days_to_predict)

#             # Create prediction dataframe
#             last_date = pd.to_datetime(company_data['Date'].iloc[-1])
#             future_dates = [
#                 last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]

#             # Get recent actual prices for comparison
#             recent_data = company_data.tail(30)
#             recent_dates = pd.to_datetime(recent_data['Date'])
#             recent_prices = recent_data['Close'].values

#         # Plot predictions
#         fig = go.Figure()

#         # Historical prices
#         fig.add_trace(go.Scatter(
#             x=recent_dates,
#             y=recent_prices,
#             mode='lines',
#             name='Historical Prices',
#             line=dict(color='#1f77b4', width=2)
#         ))

#         # Predicted prices
#         fig.add_trace(go.Scatter(
#             x=future_dates,
#             y=predictions.flatten(),
#             mode='lines+markers',
#             name=f'Predicted Prices ({days_to_predict} days)',
#             line=dict(color='#ff7f0e', width=3, dash='dash'),
#             marker=dict(size=8)
#         ))

#         # FIX: Use timestamp instead of string for vertical line
#         # Add vertical line to separate historical and predicted
#         fig.add_shape(
#             type="line",
#             x0=last_date,
#             x1=last_date,
#             y0=0,
#             y1=1,
#             yref="paper",
#             line=dict(
#                 color="gray",
#                 width=2,
#                 dash="dot",
#             ),
#         )

#         # Add annotation for the vertical line
#         fig.add_annotation(
#             x=last_date,
#             y=max(recent_prices),
#             text="Prediction Start",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="gray",
#             bgcolor="white",
#             bordercolor="gray"
#         )

#         fig.update_layout(
#             title=f'Stock Price Prediction for {selected_company}',
#             xaxis_title='Date',
#             yaxis_title='Price ($)',
#             hovermode='x unified',
#             template='plotly_white',
#             height=500
#         )

#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         st.markdown("## 📊 Key Metrics")

#         # Current price info
#         current_price = company_data['Close'].iloc[-1]
#         predicted_price = predictions[-1][0]
#         price_change = predicted_price - current_price
#         price_change_pct = (price_change / current_price) * 100

#         # Prediction box
#         st.markdown(f"""
#         <div class="prediction-box">
#             <h3>💰 Price Prediction</h3>
#             <h2>${predicted_price:.2f}</h2>
#             <p>in {days_to_predict} days</p>
#         </div>
#         """, unsafe_allow_html=True)

#         # Metrics
#         st.metric("Current Price", f"${current_price:.2f}")
#         st.metric("Expected Change",
#                   f"${price_change:.2f}", f"{price_change_pct:.1f}%")
#         st.metric("Model MSE", f"{metadata['mse_score']:.6f}")

#         # Prediction table
#         st.markdown("### 📅 Daily Predictions")
#         pred_df = pd.DataFrame({
#             'Day': range(1, days_to_predict + 1),
#             'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
#             'Predicted Price': [f"${p[0]:.2f}" for p in predictions]
#         })
#         st.dataframe(pred_df)

#         # Download predictions
#         csv = pred_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Predictions",
#             data=csv,
#             file_name=f'{selected_company}_predictions.csv',
#             mime='text/csv'
#         )

#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #666; margin-top: 2rem;'>
#         <p>📈 Stock Price Predictor | Built with Streamlit & TensorFlow</p>
#         <p><small>⚠️ This is for educational purposes only. Not financial advice.</small></p>
#     </div>
#     """, unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()


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
    if not os.path.exists("saved_models") or not os.path.exists("model_metadata"):
        return []

    model_files = glob.glob("saved_models/*.h5")
    companies = []

    for model_file in model_files:
        try:
            company_name = os.path.basename(model_file).replace(
                '_model.h5', '').replace('_', ' ')
            metadata_file = f"model_metadata/{os.path.basename(model_file).replace('_model.h5', '_metadata.json')}"

            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
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
    clean_company_name = "".join(
        c for c in company if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_company_name = clean_company_name.replace(' ', '_')

    model_filename = f"saved_models/{clean_company_name}_model.h5"
    metadata_filename = f"model_metadata/{clean_company_name}_metadata.json"

    if not os.path.exists(model_filename) or not os.path.exists(metadata_filename):
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
            model = load_model(model_filename, custom_objects=custom_objects)
        except Exception:
            try:
                model = load_model(model_filename, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            except Exception:
                return None, None

        with open(metadata_filename, 'r') as f:
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
