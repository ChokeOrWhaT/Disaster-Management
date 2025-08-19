import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import curve_fit
import argparse
from obspy.clients.fdsn import Client as FDSNClient
from obspy import UTCDateTime

class EarthquakePredictor:
    def __init__(self, csv_file=None, min_magnitude=3.0, lat_center=0.0, lon_center=0.0, radius_km=500, time_window_days=30):
        self.min_magnitude = min_magnitude
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.radius_km = radius_km
        self.time_window_days = time_window_days
        self.csv_file = csv_file
        self.fdsn_client = FDSNClient("IRIS")
        self.load_and_fit_data()

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def load_and_fit_data(self):
        # Initialize empty DataFrame with explicit dtypes
        df = pd.DataFrame(columns=['time', 'mag', 'latitude', 'longitude', 'datetime'])

        # Load static data from CSV if provided
        if self.csv_file:
            try:
                df_csv = pd.read_csv(self.csv_file, usecols=['time', 'mag', 'latitude', 'longitude'])
                # Convert to timezone-naive datetime
                df_csv['datetime'] = pd.to_datetime(df_csv['time'], utc=True).dt.tz_localize(None)
                df_csv = df_csv[df_csv['mag'] >= self.min_magnitude].dropna()
                if not df_csv.empty:
                    df = pd.concat([df, df_csv], ignore_index=True)
                    print(f"Loaded {len(df_csv)} events from CSV.")
                else:
                    print("No valid events in CSV after filtering.")
            except FileNotFoundError:
                print(f"Error: File '{self.csv_file}' not found.")
            except KeyError:
                print("Error: CSV must contain 'time', 'mag', 'latitude', 'longitude' columns.")

        # Fetch real-time data from IRIS FDSN
        try:
            endtime = UTCDateTime.now()
            starttime = endtime - (self.time_window_days * 86400)
            catalog = self.fdsn_client.get_events(
                starttime=starttime,
                endtime=endtime,
                minmagnitude=self.min_magnitude,
                latitude=self.lat_center,
                longitude=self.lon_center,
                maxradius=self.radius_km / 111.32
            )
            data = [
                {"time": event.origins[0].time.datetime.replace(tzinfo=None),  # Remove timezone
                 "mag": event.magnitudes[0].mag,
                 "latitude": event.origins[0].latitude,
                 "longitude": event.origins[0].longitude}
                for event in catalog if event.magnitudes and event.origins
            ]
            df_rt = pd.DataFrame(data)
            # Ensure timezone-naive datetime
            df_rt['datetime'] = pd.to_datetime(df_rt['time'], utc=True).dt.tz_localize(None)
            if not df_rt.empty:
                df = pd.concat([df, df_rt], ignore_index=True)
                print(f"Loaded {len(df_rt)} events from IRIS real-time data.")
            else:
                print("No events from IRIS within the specified criteria.")
        except Exception as e:
            print(f"Error fetching IRIS data: {e}")

        # Remove duplicates based on time, lat, lon, and mag
        df = df.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])

        # Filter by radius
        if not df.empty:
            df['distance'] = self.haversine(self.lat_center, self.lon_center, df['latitude'], df['longitude'])
            df = df[df['distance'] <= self.radius_km]

        # Check if data is available
        if df.empty:
            print("No events found within the specified region and magnitude threshold.")
            self.a_value, self.b_value, self.event_rate = 0, 0, 0
            return

        # Calculate time span in years
        time_span_years = (df['datetime'].max() - df['datetime'].min()).days / 365.25
        if time_span_years == 0:
            time_span_years = self.time_window_days / 365.25
        magnitudes = df['mag'].values

        # Fit Gutenberg-Richter law
        mag_bins = np.arange(self.min_magnitude, magnitudes.max() + 0.1, 0.1)
        hist, bin_edges = np.histogram(magnitudes, bins=mag_bins, density=False)
        cumulative_counts = np.cumsum(hist[::-1])[::-1] / time_span_years
        valid = cumulative_counts > 0
        mag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mag_centers = mag_centers[valid]
        log_counts = np.log10(cumulative_counts[valid])

        def gr_law(m, a, b):
            return a - b * m

        try:
            popt, _ = curve_fit(gr_law, mag_centers, log_counts, p0=[3.0, 1.0])
            self.a_value, self.b_value = popt
            self.event_rate = 10 ** (self.a_value - self.b_value * self.min_magnitude)
        except RuntimeError:
            print("Curve fitting failed. Using default a=3.0, b=1.0.")
            self.a_value, self.b_value = 3.0, 1.0
            self.event_rate = 0
        print(f"Fitted: a={self.a_value:.2f}, b={self.b_value:.2f}")
        print(f"Total events processed: {len(magnitudes)}")

    def update_data(self):
        print("Updating model with new data...")
        self.load_and_fit_data()

    def predict_probability(self, magnitude, time_window_days=365):
        if self.event_rate == 0:
            return 0.0
        rate = 10 ** (self.a_value - self.b_value * magnitude)
        annual_prob = 1 - np.exp(-rate)
        if annual_prob >= 1:
            annual_prob = 0.9999
        daily_rate = -np.log(1 - annual_prob) / 365
        return 1 - np.exp(-daily_rate * time_window_days)

def get_user_input():
    try:
        lat = float(input("Enter latitude of the location (e.g., 23.3 for Kutch, Gujarat): "))
        lon = float(input("Enter longitude of the location (e.g., 70.4 for Kutch, Gujarat): "))
        radius = float(input("Enter radius in kilometers (default 500): ") or 500)
        return lat, lon, radius
    except ValueError:
        print("Invalid input. Using default values: lat=23.3, lon=70.4, radius=500.")
        return 23.3, 70.4, 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earthquake probability predictor using real-time and static data.")
    parser.add_argument("--lat", type=float, default=None, help="Latitude of the location")
    parser.add_argument("--lon", type=float, default=None, help="Longitude of the location")
    parser.add_argument("--radius", type=float, default=500, help="Radius in kilometers")
    parser.add_argument("--time-window", type=int, default=30, help="Time window in days for real-time data")
    parser.add_argument("--file", type=str, default="earthquakes_2023_global.csv",
                        help="Path to USGS earthquake CSV file (optional)")

    args = parser.parse_args()

    if args.lat is None or args.lon is None:
        lat, lon, radius = get_user_input()
    else:
        lat, lon, radius = args.lat, args.lon, args.radius

    predictor = EarthquakePredictor(
        csv_file=args.file,
        min_magnitude=3.0,
        lat_center=lat,
        lon_center=lon,
        radius_km=radius,
        time_window_days=args.time_window
    )
    prob = predictor.predict_probability(5.5, 30) * 100
    print(f"Probability of M≥5.5 in 30 days near ({lat}, {lon}): {prob:.2f}%")