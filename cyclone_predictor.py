import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import math
import warnings
import logging

# Set Matplotlib backend for Windows compatibility
plt.switch_backend('TkAgg')

# Set up logging for detailed debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Protobuf and TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=Warning, module='tensorflow')

# Define features globally
FEATURES = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'SPEED', 'BEARING']

# Set Cartopy cache directory
CARTOPY_CACHE_DIR = os.path.expanduser('~/.cartopy_cache')
os.makedirs(CARTOPY_CACHE_DIR, exist_ok=True)
os.environ['CARTOPY_DATA_DIR'] = CARTOPY_CACHE_DIR

# Function to calculate Haversine distance (in km) between two lat/lon points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Function to calculate bearing (direction) between two lat/lon points
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

# Load and preprocess IBTrACS data
def load_and_preprocess_data(file_path, basin='NI'):
    logger.info("Checking if file exists...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The file '{file_path}' was not found. Please ensure:\n"
            f"1. The file 'ibtracs.last3years.list.v04r01.csv' is in '{os.path.dirname(file_path)}'.\n"
            f"2. Download it from https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ if missing.\n"
            f"3. Verify the file name matches exactly (case-sensitive).\n"
            f"4. Update the 'file_path' variable in the script if the file is in a different directory."
        )
    logger.info(f"File '{file_path}' found.")
    
    logger.info("Starting to load IBTrACS data...")
    try:
        chunks = pd.read_csv(file_path, low_memory=False, chunksize=10000)
        df_list = []
        for chunk in chunks:
            if basin:
                chunk = chunk[chunk['BASIN'] == 'NI']
            chunk['YEAR'] = pd.to_datetime(chunk['ISO_TIME'], errors='coerce').dt.year
            chunk = chunk[chunk['YEAR'] >= 2015]
            df_list.append(chunk)
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully loaded IBTrACS data from '{file_path}' with {len(df)} records.")
    except Exception as e:
        logger.error(f"Error loading file '{file_path}': {e}")
        raise
    
    if df.empty:
        logger.warning(f"No data found for basin '{basin}'. Try removing the basin filter or checking the dataset.")
    
    columns = ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
    df = df[columns].dropna(subset=['LAT', 'LON'])
    logger.info(f"Selected columns and dropped NaN LAT/LON: {len(df)} records.")
    
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['WMO_WIND'] = pd.to_numeric(df['WMO_WIND'], errors='coerce').fillna(0)
    df['WMO_PRES'] = pd.to_numeric(df['WMO_PRES'], errors='coerce').fillna(1010)
    
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df.sort_values(['SID', 'ISO_TIME'])
    logger.info("Data preprocessing completed.")
    
    return df

# Create sequences for LSTM training
def create_sequences(df, seq_length=3):
    logger.info("Creating sequences for LSTM training...")
    sequences = []
    targets = []
    
    for sid in df['SID'].unique():
        cyclone_data = df[df['SID'] == sid].sort_values('ISO_TIME')
        if len(cyclone_data) < seq_length + 1:
            continue
        
        cyclone_data = cyclone_data.copy()
        cyclone_data['SPEED'] = 0.0
        cyclone_data['BEARING'] = 0.0
        for i in range(1, len(cyclone_data)):
            lat1, lon1 = cyclone_data.iloc[i-1][['LAT', 'LON']]
            lat2, lon2 = cyclone_data.iloc[i][['LAT', 'LON']]
            time_diff = (cyclone_data.iloc[i]['ISO_TIME'] - cyclone_data.iloc[i-1]['ISO_TIME']).total_seconds() / 3600
            if time_diff > 0:
                cyclone_data.iloc[i, cyclone_data.columns.get_loc('SPEED')] = haversine_distance(lat1, lon1, lat2, lon2) / time_diff
                cyclone_data.iloc[i, cyclone_data.columns.get_loc('BEARING')] = calculate_bearing(lat1, lon1, lat2, lon2)
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cyclone_data[FEATURES])
        
        for i in range(len(scaled_data) - seq_length):
            seq = scaled_data[i:i+seq_length]
            target = scaled_data[i+seq_length, [0, 1]]  # Predict LAT and LON
            sequences.append(seq)
            targets.append(target)
    
    logger.info(f"Created {len(sequences)} sequences.")
    return np.array(sequences), np.array(targets), scaler

# Build LSTM model
def build_lstm_model(seq_length, n_features):
    logger.info("Building LSTM model...")
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(32, return_sequences=True),
        LSTM(16),
        Dense(8, activation='relu'),
        Dense(2)  # Output: latitude and longitude
    ])
    model.compile(optimizer='adamw', loss='mse')
    return model

# Fetch real-time cyclone data using Indian API
def fetch_realtime_data(df, seq_length=3):
    logger.info("Fetching real-time cyclone data from Indian API...")
    api_key = 'sk-live-BixWVrhAPqbFE3l0ryfVrI2Lr6SdwQSPYuSO46XT'  # Replace with your Indian API key
    url = 'https://indianapi.in/api/v1/cyclone'  # Adjust if endpoint differs
    params = {'lat': 15.0, 'lng': 85.0, 'limit': seq_length}  # Sample coords (Bay of Bengal)
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and data.get('data'):
                cyclone_data = []
                for event in data['data'][:seq_length]:
                    cyclone_data.append({
                        'LAT': event.get('latitude', 0.0),
                        'LON': event.get('longitude', 0.0),
                        'WMO_WIND': event.get('wind_speed', 0.0),  # Adjust if field differs
                        'WMO_PRES': event.get('pressure', 1010.0),  # Adjust if field differs
                        'ISO_TIME': pd.to_datetime(event.get('timestamp', datetime.now()))
                    })
                if len(cyclone_data) >= seq_length:
                    logger.info("Successfully fetched real-time cyclone data from Indian API.")
                    return pd.DataFrame(cyclone_data)
                else:
                    logger.warning("Insufficient cyclone data from Indian API.")
            else:
                logger.warning("No cyclone data found in Indian API response.")
        else:
            logger.error(f"Indian API request failed with status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching Indian API data: {e}")
    
    # Fallback: Use the last cyclone from the dataset
    logger.info("Using last cyclone from dataset as fallback.")
    try:
        last_cyclone = df[df['SID'] == df['SID'].iloc[-1]].tail(seq_length)
        if len(last_cyclone) >= seq_length:
            cyclone_data = {
                'LAT': last_cyclone['LAT'].values,
                'LON': last_cyclone['LON'].values,
                'WMO_WIND': last_cyclone['WMO_WIND'].values,
                'WMO_PRES': last_cyclone['WMO_PRES'].values,
                'ISO_TIME': last_cyclone['ISO_TIME'].values
            }
            return pd.DataFrame(cyclone_data)
        else:
            logger.warning("Insufficient data in last cyclone for prediction.")
            return None
    except Exception as e:
        logger.error(f"Error fetching fallback data: {e}")
        return None

# Plot predicted vs actual tracks
def plot_tracks(actual_lats, actual_lons, pred_lats, pred_lons, title="Cyclone Track Prediction"):
    logger.info("Generating track plot...")
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.plot(actual_lons, actual_lats, 'b-', label='Actual Track', marker='o')
    ax.plot(pred_lons, pred_lats, 'r--', label='Predicted Track', marker='x')
    ax.set_extent([60, 100, 0, 30])  # Focus on North Indian Ocean
    plt.title(title)
    plt.legend()
    plt.show(block=True)
    logger.info("Plot displayed.")

# Main function
def main():
    file_path = 'C:\\Users\\bhavi\\ibtracs.last3years.list.v04r01.csv'
    
    logger.info("Starting main execution...")
    df = load_and_preprocess_data(file_path, basin='NI')
    
    if df.empty:
        logger.error("No valid data after preprocessing. Exiting.")
        return
    
    seq_length = 3
    X, y, scaler = create_sequences(df, seq_length)
    
    if len(X) == 0:
        logger.error("No valid sequences created. Check if data contains enough valid records for North Indian Ocean cyclones.")
        return
    
    train_size = int(0.9 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    model = build_lstm_model(seq_length, X.shape[2])
    logger.info("Training LSTM model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)
    
    logger.info("Saving model...")
    model.save('cyclone_predictor.keras')
    logger.info("Model saved in native Keras format to avoid HDF5 warning.")
    
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    logger.info(f"Test MSE: {mse:.4f}")
    
    realtime_df = fetch_realtime_data(df, seq_length)
    if realtime_df is not None:
        realtime_df['SPEED'] = 0.0
        realtime_df['BEARING'] = 0.0
        for i in range(1, len(realtime_df)):
            lat1, lon1 = realtime_df.iloc[i-1][['LAT', 'LON']]
            lat2, lon2 = realtime_df.iloc[i][['LAT', 'LON']]
            time_diff = (realtime_df.iloc[i]['ISO_TIME'] - realtime_df.iloc[i-1]['ISO_TIME']).total_seconds() / 3600
            if time_diff > 0:
                realtime_df.iloc[i, realtime_df.columns.get_loc('SPEED')] = haversine_distance(lat1, lon1, lat2, lon2) / time_diff
                realtime_df.iloc[i, realtime_df.columns.get_loc('BEARING')] = calculate_bearing(lat1, lon1, lat2, lon2)
        
        scaled_realtime = scaler.transform(realtime_df[FEATURES])
        
        if len(scaled_realtime) >= seq_length:
            seq = scaled_realtime[-seq_length:].reshape(1, seq_length, len(FEATURES))
            pred = model.predict(seq)
            pred_lat_lon = scaler.inverse_transform(np.concatenate([pred, np.zeros((pred.shape[0], len(FEATURES)-2))], axis=1))[:, :2]
            logger.info(f"Predicted next position: Latitude {pred_lat_lon[0,0]:.2f}°N, Longitude {pred_lat_lon[0,1]:.2f}°E")
            
            plot_tracks(realtime_df['LAT'].values, realtime_df['LON'].values, [pred_lat_lon[0,0]], [pred_lat_lon[0,1]], title="Real-Time Cyclone Track Prediction")
        else:
            logger.warning("Insufficient real-time data for prediction")
    else:
        logger.info("Using test data for fallback prediction...")
        test_seq = X_test[0].reshape(1, seq_length, -1)
        test_pred = model.predict(test_seq)
        test_pred_lat_lon = scaler.inverse_transform(np.concatenate([test_pred, np.zeros((test_pred.shape[0], len(FEATURES)-2))], axis=1))[:, :2]
        test_actual = scaler.inverse_transform(np.concatenate([y_test[0].reshape(1, -1), np.zeros((1, len(FEATURES)-2))], axis=1))[:, :2]
        plot_tracks([test_actual[0,0]], [test_actual[0,1]], [test_pred_lat_lon[0,0]], [test_pred_lat_lon[0,1]], title="Test Cyclone Track Prediction")

if __name__ == "__main__":
    main()