
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Verify dataset files exist
if not os.path.exists('flood_risk_dataset_india.csv'):
    raise FileNotFoundError("Missing flood_risk_dataset_india.csv in C:\\Users\\bhavi")
if not os.path.exists('Natural_Disasters_in_india .csv'):
    raise FileNotFoundError("Missing Natural_Disasters_in_india .csv in C:\\Users\\bhavi")

# Load datasets
flood_risk_data = pd.read_csv('flood_risk_dataset_india.csv')
historical_data = pd.read_csv('Natural_Disasters_in_india .csv')

# Encode categorical variables
le_land_cover = LabelEncoder()
le_soil_type = LabelEncoder()
flood_risk_data['Land Cover'] = le_land_cover.fit_transform(flood_risk_data['Land Cover'])
flood_risk_data['Soil Type'] = le_soil_type.fit_transform(flood_risk_data['Soil Type'])

# Update Historical Floods for Uttarakhand based on Natural_Disasters_in_India.csv
flood_risk_data.loc[flood_risk_data['Latitude'].between(29.0, 31.5) & 
                    flood_risk_data['Longitude'].between(77.5, 81.0), 'Historical Floods'] = 1

# Prepare features and target
X = flood_risk_data.drop(columns=['Flood Occurred'])
y = flood_risk_data['Flood Occurred']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to fetch real-time weather data
def get_weather_data(lat, lon, api_key):
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{today}?key={api_key}&include=days&elements=precip,temp,humidity"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'Rainfall (mm)': data['days'][0]['precip'] * 25.4,  # Convert inches to mm
            'Temperature (°C)': (data['days'][0]['temp'] - 32) * 5/9,  # Convert °F to °C
            'Humidity (%)': data['days'][0]['humidity']
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {'Rainfall (mm)': 1.704, 'Temperature (°C)': 22.0, 'Humidity (%)': 87.9}  # Fallback

# Function to predict flood with real-time and static inputs
def predict_flood_real_time(lat, lon, api_key, river_discharge, water_level, elevation, land_cover, soil_type, population_density, infrastructure, historical_floods):
    weather = get_weather_data(lat, lon, api_key)
    try:
        land_cover_encoded = int(le_land_cover.transform([land_cover])[0])  # Convert to Python int
        soil_type_encoded = int(le_soil_type.transform([soil_type])[0])  # Convert to Python int
    except ValueError as e:
        print(f"Error: Invalid Land Cover or Soil Type. Valid options: {le_land_cover.classes_}, {le_soil_type.classes_}")
        return None
    input_data = {
        'Latitude': float(lat),
        'Longitude': float(lon),
        'Rainfall (mm)': float(weather['Rainfall (mm)']),
        'Temperature (°C)': float(weather['Temperature (°C)']),
        'Humidity (%)': float(weather['Humidity (%)']),
        'River Discharge (m³/s)': float(river_discharge),
        'Water Level (m)': float(water_level),
        'Elevation (m)': float(elevation),
        'Land Cover': land_cover_encoded,
        'Soil Type': soil_type_encoded,
        'Population Density': float(population_density),
        'Infrastructure': int(infrastructure),
        'Historical Floods': int(historical_floods)
    }
    input_df = pd.DataFrame([input_data])
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1] * 100
    return {
        'Prediction': 'Flood Likely' if prediction == 1 else 'Flood Unlikely',
        'Probability': probability,
        'Input Data': input_data
    }

# Function to validate coordinates
def validate_coordinates(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
        if not (8 <= lat <= 37 and 68 <= lon <= 97):
            raise ValueError("Coordinates out of India's range (Lat: 8–37°N, Lon: 68–97°E)")
        return lat, lon
    except ValueError as e:
        print(f"Error: Invalid coordinates. {str(e)}")
        return None, None

# Main execution with user input
if __name__ == "__main__":
    print("Enter coordinates for flood prediction (India: Lat 8–37°N, Lon 68–97°E)")
    lat = input("Enter latitude (e.g., 30.0668 for Almora): ")
    lon = input("Enter longitude (e.g., 79.0193 for Almora): ")
    
    lat, lon = validate_coordinates(lat, lon)
    if lat is None or lon is None:
        print("Exiting due to invalid coordinates.")
        exit()

    api_key = "W9YBBNGHFGYQSNR98KUQWPHM6"  # Replace with actual Visual Crossing API key
    # Default inputs for Almora; adjust for other locations as needed
    result = predict_flood_real_time(
        lat=lat, lon=lon, api_key=api_key,
        river_discharge=3000, water_level=5, elevation=1600,
        land_cover='Forest', soil_type='Loam', population_density=200,
        infrastructure=0, historical_floods=1
    )
    if result:
        print("\nFlood Prediction Result:")
        print(f"Prediction: {result['Prediction']}")
        print(f"Probability: {result['Probability']:.2f}%")
        print(f"Location: Latitude {lat}, Longitude {lon}")
        print(f"Input Data: {json.dumps(result['Input Data'], indent=2)}")
