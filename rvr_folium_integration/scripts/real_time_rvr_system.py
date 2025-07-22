import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the live predictor
from live_rvr_predictor import LiveRVRPredictor

class RealTimeRVRSystem:
    """
    Real-time RVR prediction system that:
    1. Reads latest RVR logs and weather data
    2. Generates predictions using trained models
    3. Stores results in CSV format for map visualization
    4. Updates continuously in real-time
    """
    
    def __init__(self, 
                 rvr_logs_dir="data/raw/rvr_logs",
                 weather_dir="data/raw/weather",
                 output_dir="data/real_time_predictions",
                 update_interval=60):  # Update every 60 seconds
        
        self.rvr_logs_dir = Path(rvr_logs_dir)
        self.weather_dir = Path(weather_dir)
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the live predictor
        print("🚀 Initializing Live RVR Predictor...")
        self.predictor = LiveRVRPredictor()
        
        # Initialize data storage
        self.latest_rvr_data = None
        self.latest_weather_data = {}
        self.prediction_history = []
        
        # Threading for continuous updates
        self.running = False
        self.update_thread = None
        
        print(f"✅ Real-time RVR system initialized")
        print(f"   RVR logs directory: {self.rvr_logs_dir}")
        print(f"   Weather directory: {self.weather_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Update interval: {self.update_interval} seconds")
    
    def load_latest_rvr_data(self):
        """Load the most recent RVR data from CSV files"""
        print(f"\n📊 Loading latest RVR data...")
        
        rvr_files = list(self.rvr_logs_dir.glob("RVR_2024.csv"))
        
        if not rvr_files:
            print(f"   ❌ No RVR 2024 data found in {self.rvr_logs_dir}")
            return None
        
        rvr_file = rvr_files[0]
        print(f"   📁 Loading: {rvr_file.name}")
        
        try:
            # Load RVR data
            rvr_df = pd.read_csv(rvr_file)
            
            # Parse datetime in yyyy-mm-dd HH:MM format
            rvr_df['Datetime'] = pd.to_datetime(rvr_df['Datetime'], format='%Y-%m-%d %H:%M', errors='coerce')
            
            # If parsing fails, try d/m/Y H:M as fallback
            if rvr_df['Datetime'].isnull().all():
                rvr_df['Datetime'] = pd.to_datetime(rvr_df['Datetime'], format='%d/%m/%Y %H:%M', errors='coerce')
            
            rvr_df = rvr_df.dropna(subset=['Datetime'])
            
            print(f"   📅 Date range: {rvr_df['Datetime'].min()} to {rvr_df['Datetime'].max()}")
            
            # Get the latest data point
            latest_rvr = rvr_df.iloc[-1]
            print(f"   🕐 Latest timestamp: {latest_rvr['Datetime']}")
            
            self.latest_rvr_data = latest_rvr
            return latest_rvr
            
        except Exception as e:
            print(f"   ❌ Error loading RVR data: {e}")
            return None
    
    def load_latest_weather_data(self):
        """Load the most recent weather data from Excel files"""
        print(f"\n🌤️ Loading latest weather data...")
        
        # Find all 2024 weather files
        weather_files = list(self.weather_dir.glob("*_2024.xlsx"))
        print(f"   📁 Found {len(weather_files)} weather files for 2024")
        
        weather_data = {}
        
        for weather_file in weather_files:
            try:
                # Extract runway from filename
                runway = weather_file.stem.split('_')[0]  # e.g., 'RUNWAY11'
                print(f"   📊 Loading weather for {runway}...")
                
                # Load Excel file
                weather_df = pd.read_excel(weather_file)
                print(f"   ✅ Loaded {runway}: {weather_df.shape}")
                
                # Get the latest data point
                if len(weather_df) > 0:
                    latest_weather = weather_df.iloc[-1]
                    weather_data[runway] = latest_weather
                    print(f"   🕐 Latest {runway} timestamp: {latest_weather.get('Datetime', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Error loading {weather_file.name}: {e}")
        
        self.latest_weather_data = weather_data
        print(f"   📊 Loaded weather data for {len(weather_data)} runways")
        return weather_data
    
    def prepare_sensor_data_for_prediction(self):
        """Prepare sensor data in the format expected by the live predictor"""
        print(f"\n🔧 Preparing sensor data for prediction...")
        
        if self.latest_rvr_data is None:
            print(f"   ❌ No RVR data available")
            return None
        
        # Map RVR columns to runway zones
        rvr_column_mapping = {
            'RWY 09 (BEG)': 'RWY_09_BEG',
            'RWY 09 (TDZ)': 'RWY_09_TDZ',
            'RWY 10 (TDZ)': 'RWY_10_TDZ',
            'RWY 11 (BEG)': 'RWY_11_BEG',
            'RWY 11 (TDZ)': 'RWY_11_TDZ',
            'RWY 27 (MID)': 'RWY_27_MID',
            'RWY 28 (BEG)': 'RWY_28_BEG',
            'RWY 28 (MID)': 'RWY_28_MID',
            'RWY 28 (TDZ)': 'RWY_28_TDZ',
            'RWY 29 (BEG)': 'RWY_29_BEG',
            'RWY 29 (MID)': 'RWY_29_MID',
        }
        
        sensor_data = {}
        timestamp = self.latest_rvr_data['Datetime']
        
        print(f"   🕐 Using timestamp: {timestamp}")
        print(f"   📊 Available RVR columns: {list(self.latest_rvr_data.index)}")
        
        for rvr_col, runway_zone in rvr_column_mapping.items():
            if rvr_col in self.latest_rvr_data.index:
                value = self.latest_rvr_data[rvr_col]
                
                # Handle different types of missing/invalid data
                if pd.notna(value) and value != 3333.0 and value != '' and str(value).strip() != '':
                    try:
                        float_value = float(value)
                        if float_value > 0 and float_value < 3333.0:  # Valid RVR range
                            sensor_data[runway_zone] = float_value
                            print(f"   📍 {runway_zone}: {float_value:.1f}m")
                        else:
                            print(f"   ⚠️ {runway_zone}: Invalid value {float_value} (using default)")
                    except (ValueError, TypeError):
                        print(f"   ⚠️ {runway_zone}: Non-numeric value '{value}' (using default)")
                else:
                    print(f"   ❌ {runway_zone}: Missing/invalid data '{value}' (using default)")
        
        # If no valid sensor data found, try to get some recent valid data from the file
        if not sensor_data:
            print(f"   🔍 No valid sensor data in latest row, searching for recent valid data...")
            try:
                # Load the full RVR file and find recent valid data
                rvr_file = list(self.rvr_logs_dir.glob("RVR_2024.csv"))[0]
                rvr_df = pd.read_csv(rvr_file)
                rvr_df['Datetime'] = pd.to_datetime(rvr_df['Datetime'], format='%d/%m/%Y %H:%M')
                
                # Look for the last 100 rows for valid data
                for idx in range(len(rvr_df) - 1, max(0, len(rvr_df) - 100), -1):
                    row = rvr_df.iloc[idx]
                    for rvr_col, runway_zone in rvr_column_mapping.items():
                        if rvr_col in row.index and runway_zone not in sensor_data:
                            value = row[rvr_col]
                            if pd.notna(value) and value != 3333.0 and value != '' and str(value).strip() != '':
                                try:
                                    float_value = float(value)
                                    if float_value > 0 and float_value < 3333.0:
                                        sensor_data[runway_zone] = float_value
                                        print(f"   📍 {runway_zone}: {float_value:.1f}m (from row {idx})")
                                        break  # Found valid data for this zone
                                except (ValueError, TypeError):
                                    continue
                
                # Update timestamp to the row where we found data
                if sensor_data:
                    timestamp = row['Datetime']
                    print(f"   🕐 Updated timestamp to: {timestamp}")
                    
            except Exception as e:
                print(f"   ❌ Error searching for valid data: {e}")
        
        print(f"   ✅ Prepared sensor data for {len(sensor_data)} zones")
        return sensor_data, timestamp
    
    def generate_predictions(self):
        """Generate predictions for all runway zones"""
        print(f"\n🔮 Generating RVR predictions...")
        # Prepare sensor data
        sensor_data_result = self.prepare_sensor_data_for_prediction()
        if sensor_data_result is None:
            print(f"   ❌ Could not prepare sensor data")
            # Return default predictions and a dummy timestamp
            predictions = {zone: 1000.0 for zone in self.predictor.runway_zones}
            timestamp = pd.Timestamp('2024-01-01')
            return predictions, timestamp
        sensor_data, timestamp = sensor_data_result
        
        # Update predictor with latest sensor data
        print(f"   📡 Updating predictor with sensor data...")
        self.predictor.update_sensor_data(sensor_data, timestamp)
        
        # Generate predictions for all zones
        predictions = {}
        
        for runway_zone in self.predictor.runway_zones:
            print(f"\n   🎯 Predicting for {runway_zone}...")
            prediction = self.predictor.predict_rvr(runway_zone)
            
            if prediction is not None:
                predictions[runway_zone] = prediction
                print(f"   ✅ {runway_zone}: {prediction:.1f}m")
            else:
                # Use current sensor value if available, otherwise default
                current_value = sensor_data.get(runway_zone, 1000.0)
                predictions[runway_zone] = current_value
                print(f"   ⚠️ {runway_zone}: Using current value {current_value:.1f}m")
        
        print(f"\n   📊 Generated predictions for {len(predictions)} zones")
        return predictions, timestamp
    
    def create_prediction_record(self, predictions, timestamp):
        """Create a prediction record with all necessary data"""
        print(f"\n📝 Creating prediction record...")
        
        # Base record with timestamp
        record = {
            'Datetime': timestamp,
        }
        
        # Add predictions with proper column names
        prediction_columns = {
            'RWY_09_BEG': 'RWY_09_BEG_predicted',
            'RWY_09_TDZ': 'RWY_09_TDZ_predicted',
            'RWY_10_TDZ': 'RWY_10_TDZ_predicted',
            'RWY_11_BEG': 'RWY_11_BEG_predicted',
            'RWY_11_TDZ': 'RWY_11_TDZ_predicted',
            'RWY_27_MID': 'RWY_27_MID_predicted',
            'RWY_28_BEG': 'RWY_28_BEG_predicted',
            'RWY_28_MID': 'RWY_28_MID_predicted',
            'RWY_28_TDZ': 'RWY_28_TDZ_predicted',
            'RWY_29_BEG': 'RWY_29_BEG_predicted',
            'RWY_29_MID': 'RWY_29_MID_predicted',
        }
        
        for zone, column in prediction_columns.items():
            record[column] = predictions.get(zone, 1000.0)
        
        # Add current sensor values if available
        if self.latest_rvr_data is not None:
            rvr_column_mapping = {
                'RWY 09 (BEG)': 'RWY_09_BEG_current',
                'RWY 09 (TDZ)': 'RWY_09_TDZ_current',
                'RWY 10 (TDZ)': 'RWY_10_TDZ_current',
                'RWY 11 (BEG)': 'RWY_11_BEG_current',
                'RWY 11 (TDZ)': 'RWY_11_TDZ_current',
                'RWY 27 (MID)': 'RWY_27_MID_current',
                'RWY 28 (BEG)': 'RWY_28_BEG_current',
                'RWY 28 (MID)': 'RWY_28_MID_current',
                'RWY 28 (TDZ)': 'RWY_28_TDZ_current',
                'RWY 29 (BEG)': 'RWY_29_BEG_current',
                'RWY 29 (MID)': 'RWY_29_MID_current',
            }
            
            for rvr_col, current_col in rvr_column_mapping.items():
                if rvr_col in self.latest_rvr_data.index:
                    value = self.latest_rvr_data[rvr_col]
                    if pd.notna(value) and value != 3333.0:
                        record[current_col] = float(value)
                    else:
                        record[current_col] = 1000.0
                else:
                    record[current_col] = 1000.0
        
        print(f"   ✅ Created prediction record with {len(record)} columns")
        return record
    
    def save_predictions_to_csv(self, prediction_record):
        """Save predictions to CSV file for map visualization"""
        print(f"\n💾 Saving predictions to CSV...")
        
        # Create filename with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"real_time_predictions_{current_date}.csv"
        csv_path = self.output_dir / csv_filename
        
        # Check if file exists
        if csv_path.exists():
            # Append to existing file
            print(f"   📁 Appending to existing file: {csv_filename}")
            existing_df = pd.read_csv(csv_path)
            new_df = pd.DataFrame([prediction_record])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # Create new file
            print(f"   📁 Creating new file: {csv_filename}")
            combined_df = pd.DataFrame([prediction_record])
        
        # Save to CSV
        combined_df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved {len(combined_df)} records to {csv_path}")
        
        # Also create a latest predictions file for immediate access
        latest_csv_path = self.output_dir / "latest_predictions.csv"
        combined_df.to_csv(latest_csv_path, index=False)
        print(f"   ✅ Updated latest predictions file")
        
        return csv_path
    
    def update_system(self):
        """Perform one complete update cycle"""
        print(f"\n🔄 Starting update cycle at {datetime.now()}")
        
        try:
            # Step 1: Load latest data
            self.load_latest_rvr_data()
            self.load_latest_weather_data()
            
            # Step 2: Generate predictions
            prediction_result = self.generate_predictions()
            if prediction_result is None:
                print(f"   ❌ Failed to generate predictions")
                return
            
            predictions, timestamp = prediction_result
            
            # Step 3: Create prediction record
            prediction_record = self.create_prediction_record(predictions, timestamp)
            
            # Step 4: Save to CSV
            csv_path = self.save_predictions_to_csv(prediction_record)
            
            # Step 5: Store in history
            self.prediction_history.append({
                'timestamp': timestamp,
                'predictions': predictions,
                'csv_path': csv_path
            })
            
            # Keep only last 100 predictions in memory
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            print(f"   ✅ Update cycle completed successfully")
            
        except Exception as e:
            print(f"   ❌ Error in update cycle: {e}")
            import traceback
            traceback.print_exc()
    
    def start_real_time_updates(self):
        """Start continuous real-time updates"""
        print(f"\n🚀 Starting real-time RVR prediction system...")
        print(f"   Update interval: {self.update_interval} seconds")
        print(f"   Press Ctrl+C to stop")
        
        self.running = True
        
        # Perform initial update
        self.update_system()
        
        # Start continuous updates
        while self.running:
            try:
                time.sleep(self.update_interval)
                if self.running:
                    self.update_system()
            except KeyboardInterrupt:
                print(f"\n⏹️ Stopping real-time updates...")
                self.running = False
                break
            except Exception as e:
                print(f"   ❌ Error in update loop: {e}")
                time.sleep(10)  # Wait before retrying
        
        print(f"   ✅ Real-time updates stopped")
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            'running': self.running,
            'last_update': None,
            'prediction_count': len(self.prediction_history),
            'available_models': len(self.predictor.models),
            'runway_zones': self.predictor.runway_zones,
            'output_directory': str(self.output_dir),
            'latest_rvr_data': self.latest_rvr_data is not None,
            'weather_data_count': len(self.latest_weather_data)
        }
        
        if self.prediction_history:
            status['last_update'] = self.prediction_history[-1]['timestamp']
        
        return status
    
    def batch_predict_for_time_range(self, start_time, end_time, freq='10min'):
        """
        Generate predictions for a range of timestamps and save to a single CSV.
        Args:
            start_time: Start datetime (inclusive)
            end_time: End datetime (inclusive)
            freq: Frequency string for time steps (default '10min')
        """
        print(f"\n🚀 Batch prediction from {start_time} to {end_time} every {freq}...")
        # Load full RVR data
        rvr_file = list(self.rvr_logs_dir.glob("RVR_2024.csv"))[0]
        rvr_df = pd.read_csv(rvr_file)
        # Parse datetime in yyyy-mm-dd HH:MM format
        rvr_df['Datetime'] = pd.to_datetime(rvr_df['Datetime'], format='%Y-%m-%d %H:%M', errors='coerce')
        # If parsing fails, try d/m/Y H:M as fallback
        if rvr_df['Datetime'].isnull().all():
            rvr_df['Datetime'] = pd.to_datetime(rvr_df['Datetime'], format='%d/%m/%Y %H:%M', errors='coerce')
        # Drop rows with invalid dates
        rvr_df = rvr_df.dropna(subset=['Datetime'])
        # Filter to desired range
        mask = (rvr_df['Datetime'] >= start_time) & (rvr_df['Datetime'] <= end_time)
        rvr_df = rvr_df.loc[mask].reset_index(drop=True)
        print(f"   Filtered to {len(rvr_df)} rows in range.")
        all_records = []
        for idx, row in rvr_df.iterrows():
            self.latest_rvr_data = row
            predictions, timestamp = self.generate_predictions()
            record = self.create_prediction_record(predictions, timestamp)
            # Ensure Datetime is in yyyy-mm-dd HH:MM format
            record['Datetime'] = pd.to_datetime(record['Datetime']).strftime('%Y-%m-%d %H:%M')
            all_records.append(record)
            if (idx+1) % 100 == 0:
                print(f"   Processed {idx+1} rows...")
        batch_csv = self.output_dir / f"batch_predictions_{start_time:%Y%m%d}_{end_time:%Y%m%d}.csv"
        pd.DataFrame(all_records).to_csv(batch_csv, index=False)
        print(f"   ✅ Batch predictions saved to {batch_csv}")
        return batch_csv

def main():
    """Main function to run the real-time RVR system"""
    print("=" * 60)
    print("🌐 REAL-TIME RVR PREDICTION SYSTEM")
    print("=" * 60)
    print("This system will:")
    print("1. Read latest RVR logs and weather data")
    print("2. Generate predictions using trained models")
    print("3. Save results to CSV for map visualization")
    print("4. Update continuously in real-time")
    print("=" * 60)
    
    # Initialize the system
    system = RealTimeRVRSystem(update_interval=60)  # Update every minute
    
    # Show initial status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Start real-time updates
    try:
        system.start_real_time_updates()
    except KeyboardInterrupt:
        print(f"\n👋 System stopped by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # Example: python scripts/real_time_rvr_system.py batch 2024-01-01 2024-01-02
        if len(sys.argv) >= 4:
            start = pd.to_datetime(sys.argv[2])
            end = pd.to_datetime(sys.argv[3])
        else:
            # Default: one week in 2024
            start = pd.to_datetime('2024-01-01')
            end = pd.to_datetime('2024-01-07')
        system = RealTimeRVRSystem()
        system.batch_predict_for_time_range(start, end)
        return

if __name__ == "__main__":
    main() 