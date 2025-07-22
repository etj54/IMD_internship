import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load as joblib_load
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Tuple

class LiveRVRPredictor:
    """
    Real-time RVR prediction system for Delhi Airport
    Takes live sensor data and predicts RVR values for different runway zones
    """
    
    def __init__(self, model_dir: str = "saved_models"):
        """
        Initialize the live RVR predictor
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.runway_zones = []
        
        # Load all trained models
        self._load_models()
        
        # Initialize historical data storage for lag features
        self.historical_data = {}
        self.max_lag = 3  # Maximum lag period
        
        print(f"Loaded {len(self.models)} models for live prediction")
    
    def _load_models(self):
        """Load all trained models from the model directory"""
        print(f"🔍 Searching for models in: {self.model_dir}")
        print(f"   Directory exists: {self.model_dir.exists()}")
        
        model_files = list(self.model_dir.glob("rvr_model_RWY_*.pkl"))
        print(f"   Found {len(model_files)} model files")
        
        for model_file in model_files:
            try:
                print(f"\n   📁 Processing: {model_file.name}")
                print(f"   File size: {model_file.stat().st_size / 1024:.1f} KB")
                
                # Parse runway and zone from filename
                parts = model_file.stem.split("_")
                print(f"   Filename parts: {parts}")
                
                if len(parts) >= 5:
                    runway = parts[3]  # e.g., '09'
                    zone = parts[4]    # e.g., 'BEG'
                    runway_id = f"RWY_{runway}_{zone}"
                    print(f"   Parsed runway: {runway}, zone: {zone}")
                    print(f"   Runway ID: {runway_id}")
                    
                    # Load model data
                    print(f"   Loading model data...")
                    model_data = joblib_load(model_file)
                    
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.models[runway_id] = model_data['model']
                        self.scalers[runway_id] = model_data.get('scaler')
                        self.feature_columns[runway_id] = model_data.get('feature_columns', [])
                        self.runway_zones.append(runway_id)
                        
                        print(f"   ✅ Successfully loaded model for {runway_id}")
                        print(f"   Model type: {type(model_data['model']).__name__}")
                        print(f"   Has scaler: {self.scalers[runway_id] is not None}")
                        print(f"   Feature columns: {len(self.feature_columns[runway_id])} columns")
                        if self.feature_columns[runway_id]:
                            print(f"   Sample features: {self.feature_columns[runway_id][:3]}")
                    else:
                        print(f"   ❌ Invalid model format for {model_file.name}")
                        print(f"   Model data type: {type(model_data)}")
                        if isinstance(model_data, dict):
                            print(f"   Available keys: {list(model_data.keys())}")
                        
                else:
                    print(f"   ❌ Could not parse filename: {model_file.name}")
                    print(f"   Expected format: rvr_model_RWY_XX_YY.pkl")
                        
            except Exception as e:
                print(f"   ❌ Error loading model {model_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 Model Loading Summary:")
        print(f"   Total models loaded: {len(self.models)}")
        print(f"   Runway zones: {self.runway_zones}")
        print(f"   Models with scalers: {sum(1 for scaler in self.scalers.values() if scaler is not None)}")
    
    def update_sensor_data(self, sensor_data: Dict[str, float], timestamp: Optional[datetime] = None):
        """
        Update historical data with new sensor readings
        
        Args:
            sensor_data: Dictionary of sensor readings for each runway zone
                        Format: {'RWY_09_BEG': 850.0, 'RWY_09_TDZ': 900.0, ...}
            timestamp: Timestamp for the sensor data (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store the new data
        for runway_id, value in sensor_data.items():
            if runway_id not in self.historical_data:
                self.historical_data[runway_id] = []
            
            self.historical_data[runway_id].append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Keep only the last max_lag + 1 entries
            if len(self.historical_data[runway_id]) > self.max_lag + 1:
                self.historical_data[runway_id] = self.historical_data[runway_id][-self.max_lag-1:]
    
    def create_lag_features(self, runway_id: str) -> Optional[np.ndarray]:
        """
        Create lag features for a specific runway zone
        
        Args:
            runway_id: Runway zone identifier (e.g., 'RWY_09_BEG')
            
        Returns:
            Array of lag features or None if insufficient data
        """
        if runway_id not in self.historical_data:
            print(f"   ❌ No historical data for {runway_id}")
            return None
        
        data = self.historical_data[runway_id]
        print(f"   📊 Historical data points for {runway_id}: {len(data)}")
        
        # Need at least max_lag + 1 data points
        if len(data) < self.max_lag + 1:
            print(f"   ❌ Insufficient data for {runway_id}: {len(data)} < {self.max_lag + 1}")
            return None
        
        # Get the most recent values
        recent_values = [entry['value'] for entry in data[-self.max_lag-1:]]
        print(f"   📈 Recent values for {runway_id}: {recent_values}")
        
        # Create lag features (lag1, lag2, lag3)
        lag_features = []
        for lag in range(1, self.max_lag + 1):
            lag_value = recent_values[-lag-1]  # -1 because we want lagged values
            lag_features.append(lag_value)
            print(f"   Lag{lag}: {lag_value:.1f}m")
        
        features_array = np.array(lag_features).reshape(1, -1)
        print(f"   ✅ Created features for {runway_id}: {features_array.shape}")
        return features_array
    
    def predict_rvr(self, runway_id: str) -> Optional[float]:
        """
        Predict RVR for a specific runway zone
        
        Args:
            runway_id: Runway zone identifier (e.g., 'RWY_09_BEG')
            
        Returns:
            Predicted RVR value or None if prediction not possible
        """
        print(f"\n🔮 Predicting RVR for {runway_id}")
        
        if runway_id not in self.models:
            print(f"   ❌ No model available for {runway_id}")
            print(f"   Available models: {list(self.models.keys())}")
            return None
        
        print(f"   ✅ Model found for {runway_id}")
        
        # Create lag features
        print(f"   Creating lag features...")
        features = self.create_lag_features(runway_id)
        if features is None:
            print(f"   ❌ Insufficient historical data for {runway_id}")
            return None
        
        print(f"   ✅ Features created: {features.shape}")
        print(f"   Feature values: {features.flatten()}")
        
        try:
            # Scale features if scaler is available
            if runway_id in self.scalers and self.scalers[runway_id] is not None:
                print(f"   🔧 Scaling features...")
                features = self.scalers[runway_id].transform(features)
                print(f"   ✅ Features scaled: {features.shape}")
                print(f"   Scaled values: {features.flatten()}")
            else:
                print(f"   ⚠️ No scaler available for {runway_id}, using raw features")
            
            # Make prediction
            print(f"   🎯 Making prediction...")
            prediction = self.models[runway_id].predict(features)[0]
            print(f"   ✅ Prediction for {runway_id}: {prediction:.1f}m")
            return prediction
            
        except Exception as e:
            print(f"   ❌ Error predicting RVR for {runway_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_all_zones(self) -> Dict[str, float]:
        """
        Predict RVR for all runway zones
        
        Returns:
            Dictionary of predictions for each runway zone
        """
        predictions = {}
        
        for runway_id in self.runway_zones:
            prediction = self.predict_rvr(runway_id)
            if prediction is not None:
                predictions[runway_id] = prediction
        
        return predictions
    
    def get_prediction_status(self) -> Dict[str, Dict]:
        """
        Get status of predictions for all zones
        
        Returns:
            Dictionary with status information for each zone
        """
        status = {}
        
        for runway_id in self.runway_zones:
            data_points = len(self.historical_data.get(runway_id, []))
            can_predict = data_points >= self.max_lag + 1
            
            status[runway_id] = {
                'data_points': data_points,
                'required_points': self.max_lag + 1,
                'can_predict': can_predict,
                'latest_value': self.historical_data.get(runway_id, [{}])[-1].get('value', None) if self.historical_data.get(runway_id) else None,
                'latest_timestamp': self.historical_data.get(runway_id, [{}])[-1].get('timestamp', None) if self.historical_data.get(runway_id) else None
            }
        
        return status
    
    def simulate_sensor_data(self, duration_minutes: int = 60, interval_seconds: int = 10):
        """
        Simulate sensor data for testing purposes
        
        Args:
            duration_minutes: Duration of simulation in minutes
            interval_seconds: Interval between sensor readings in seconds
        """
        print(f"Simulating sensor data for {duration_minutes} minutes...")
        
        # Generate realistic RVR values with some variation
        base_values = {
            'RWY_09_BEG': 800,
            'RWY_09_TDZ': 750,
            'RWY_10_TDZ': 900,
            'RWY_11_BEG': 850,
            'RWY_11_TDZ': 700,
            'RWY_27_MID': 950,
            'RWY_28_BEG': 800,
            'RWY_28_MID': 850,
            'RWY_28_TDZ': 750,
            'RWY_29_BEG': 900,
            'RWY_29_MID': 850
        }
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            # Generate sensor data with some random variation
            sensor_data = {}
            for runway_id, base_value in base_values.items():
                # Add random variation (±20% of base value)
                variation = np.random.normal(0, base_value * 0.1)
                sensor_data[runway_id] = max(50, base_value + variation)  # Minimum 50m
            
            # Update predictor with new data
            self.update_sensor_data(sensor_data)
            
            # Make predictions
            predictions = self.predict_all_zones()
            
            # Print results
            print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
            print("Sensor Data:")
            for runway_id, value in sensor_data.items():
                print(f"  {runway_id}: {value:.1f}m")
            
            print("\nPredictions:")
            for runway_id, pred in predictions.items():
                print(f"  {runway_id}: {pred:.1f}m")
            
            # Wait for next interval
            time.sleep(interval_seconds)

# ─── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
def main():
    """Example usage of the LiveRVRPredictor"""
    
    print("🚀 Starting Live RVR Predictor Test")
    print("=" * 50)
    
    # Initialize predictor
    print("\n📦 Initializing LiveRVRPredictor...")
    predictor = LiveRVRPredictor()
    
    # Example 1: Single prediction
    print("\n" + "=" * 50)
    print("🧪 Example 1: Single Prediction")
    print("=" * 50)
    
    # Simulate some historical data
    print("\n📊 Setting up historical data...")
    historical_data = {
        'RWY_09_BEG': [850, 900, 950],  # lag3, lag2, lag1
        'RWY_09_TDZ': [800, 850, 900],
        'RWY_10_TDZ': [900, 950, 1000],
        'RWY_11_BEG': [850, 900, 950],
        'RWY_11_TDZ': [700, 750, 800],
        'RWY_27_MID': [950, 1000, 1050],
        'RWY_28_BEG': [800, 850, 900],
        'RWY_28_MID': [850, 900, 950],
        'RWY_28_TDZ': [750, 800, 850],
        'RWY_29_BEG': [900, 950, 1000],
        'RWY_29_MID': [850, 900, 950]
    }
    
    # Add historical data
    print("   Adding historical data points...")
    for runway_id, values in historical_data.items():
        print(f"   {runway_id}: {values}")
        for i, value in enumerate(values):
            timestamp = datetime.now() - timedelta(minutes=(len(values)-i)*10)
            predictor.update_sensor_data({runway_id: value}, timestamp)
    
    print(f"\n✅ Historical data added for {len(historical_data)} runway zones")
    
    # Make predictions
    print("\n🔮 Making predictions for all zones...")
    predictions = predictor.predict_all_zones()
    
    print("\n📈 RVR Predictions:")
    print("-" * 40)
    for runway_id, pred in predictions.items():
        print(f"  {runway_id}: {pred:.1f}m")
    
    print(f"\n✅ Successfully predicted for {len(predictions)} zones")
    
    # Example 2: Real-time simulation
    print("\n" + "=" * 50)
    print("🔄 Example 2: Real-time Simulation")
    print("=" * 50)
    print("Starting 2-minute simulation with 5-second intervals...")
    print("This will simulate live sensor data and show predictions in real-time")
    print("Press Ctrl+C to stop early")
    
    try:
        # Run simulation for 2 minutes
        predictor.simulate_sensor_data(duration_minutes=2, interval_seconds=5)
    except KeyboardInterrupt:
        print("\n⏹️ Simulation stopped by user")
    
    print("\n✅ Live RVR Predictor test completed!")

if __name__ == "__main__":
    main() 