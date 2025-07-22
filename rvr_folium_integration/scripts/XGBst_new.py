#%%
import os
import glob
import pickle
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# seaborn is used only for its palette settings
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#%%
class RVRPredictor:
    """
    A class to load, preprocess, train, and evaluate runway visual range (RVR) prediction models
    using XGBoost for multiple runways at an airport.
    """
    def __init__(self,
                 base_path='C:/Users/alwyn/OneDrive/Desktop/IMD_Internship',
                 target_runways=None):
        # Base directories for RVR and weather data
        self.base_path = base_path
        self.rvr_path = os.path.join(base_path, 'Processed_RVR_Logs_Combined')
        self.weather_path = os.path.join(base_path, 'Processed_Weather_AllMonths')

        # List of runways to process (default if none provided)
        self.target_runways = target_runways or ["RWY 09 (BEG)", "RWY 09 (TDZ)", "RWY 09 (MID)",
        "RWY 10 (BEG)", "RWY 10 (TDZ)", "RWY 10 (MID)",
        "RWY 11 (BEG)", "RWY 11 (TDZ)", "RWY 11 (MID)",
        "RWY 27 (BEG)", "RWY 27 (TDZ)", "RWY 27 (MID)",
        "RWY 28 (BEG)", "RWY 28 (TDZ)", "RWY 28 (MID)",
        "RWY 29 (BEG)", "RWY 29 (TDZ)", "RWY 29 (MID)",]

        # Placeholders for data and models
        self.rvr_data = None
        self.weather_data = None
        self.merged_data = None
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}

        # Attributes updated per runway during pipeline
        self.target_runway = None
        self.feature_columns = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_train_pred = self.y_test_pred = None

    def load_rvr_data(self):
        """Load and concatenate all RVR CSV files into a single DataFrame."""
        print("Loading RVR data...")
        rvr_files = glob.glob(os.path.join(self.rvr_path, 'RVR_*.csv'))
        if not rvr_files:
            print("No RVR files found!")
            return

        dfs = []
        for file in rvr_files:
            try:
                df = pd.read_csv(file)
                # Find and standardize datetime column
                dt_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                if dt_col:
                    df['datetime'] = pd.to_datetime(df[dt_col], errors='coerce', dayfirst=True)
                    df.drop(columns=[dt_col], inplace=True)
                    df.dropna(subset=['datetime'], inplace=True)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if dfs:
            self.rvr_data = pd.concat(dfs, ignore_index=True).sort_values('datetime')
            print(f"Loaded {len(self.rvr_data)} RVR records")
        else:
            print("No valid RVR data loaded!")

    def load_weather_data(self):
        """Load and concatenate all weather Excel files into a single DataFrame."""
        print("Loading weather data...")
        weather_files = glob.glob(os.path.join(self.weather_path, 'RUNWAY*.xlsx'))
        if not weather_files:
            print("No weather files found!")
            return

        dfs = []
        for file in weather_files:
            try:
                df = pd.read_excel(file)
                # Handle separate date/time columns or a single datetime column
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['datetime'] = pd.to_datetime(
                        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                        errors='coerce', dayfirst=True
                    )
                    df.drop(columns=['Date', 'Time'], inplace=True)
                else:
                    dt_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                    if dt_col:
                        df['datetime'] = pd.to_datetime(df[dt_col], errors='coerce', dayfirst=True)
                        df.drop(columns=[dt_col], inplace=True)
                df.dropna(subset=['datetime'], inplace=True)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if dfs:
            self.weather_data = pd.concat(dfs, ignore_index=True).sort_values('datetime')
            print(f"Loaded {len(self.weather_data)} weather records")
        else:
            print("No valid weather data loaded!")

    def merge_data(self):
        """
        Merge RVR and weather data on nearest timestamps within a 30-minute tolerance.
        """
        print("Merging data...")
        if self.rvr_data is None or self.weather_data is None:
            print("Skipping merge - missing data!")
            return

        rvr_df = self.rvr_data.copy().sort_values('datetime')
        weather_df = self.weather_data.copy().sort_values('datetime')

        # Identify RVR columns (runway readings)
        rwy_cols = [c for c in rvr_df.columns if 'RWY' in c or 'RW' in c]
        if not rwy_cols:
            print("No runway columns found in RVR data!")
            return

        # Ensure numeric types for merge
        for col in rwy_cols:
            rvr_df[col] = pd.to_numeric(rvr_df[col], errors='coerce')

        # Ensure no NaT in datetime columns before merging
        rvr_df = rvr_df.dropna(subset=['datetime'])
        weather_df = weather_df.dropna(subset=['datetime'])
        # As-of merge to align nearest times
        self.merged_data = pd.merge_asof(
            rvr_df, weather_df,
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('30min')
        )
        print(f"Merged data shape: {self.merged_data.shape}")

    def create_features(self):
        """
        Generate time-based, lag, rolling, and weather-derived features
        for the current target runway.
        """
        if self.merged_data is None or self.target_runway not in self.merged_data.columns:
            print(f"Skipping feature creation for {self.target_runway} - no valid data!")
            return

        print(f"Creating features for {self.target_runway}...")
        df = self.merged_data.copy()

        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_year'] = df['datetime'].dt.dayofyear

        # Cyclical encoding for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Lag features to capture recent history
        for i in range(1, 4):
            df[f'{self.target_runway}_lag{i}'] = df[self.target_runway].shift(i)

        # Rolling mean features
        df[f'{self.target_runway}_rolling_3'] = df[self.target_runway].rolling(window=3, min_periods=1).mean()
        df[f'{self.target_runway}_rolling_6'] = df[self.target_runway].rolling(window=6, min_periods=1).mean()

        # Rename key weather columns for consistency
        weather_map = {
            'Temperature1MinAvg (DEG C)': 'temperature',
            'Humidity1MinAvg (%)': 'humidity',
            'Wind Speed': 'wind_speed',
            'DewPoint1MinAvg (DEG C)': 'dew_point'
        }
        for old, new in weather_map.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

        # Derived weather features
        if 'temperature' in df and 'humidity' in df:
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        if 'temperature' in df and 'dew_point' in df:
            df['temp_dewpoint_diff'] = df['temperature'] - df['dew_point']

        # Drop rows with missing lag values
        df.dropna(subset=[f'{self.target_runway}_lag1',
                          f'{self.target_runway}_lag2',
                          f'{self.target_runway}_lag3'], inplace=True)

        self.merged_data = df
        print(f"Feature creation complete for {self.target_runway}. Data shape: {self.merged_data.shape}")

    def preprocess_data(self):
        """
        Impute missing feature values and define the set of predictors
        for model training.
        """
        if self.merged_data is None or self.target_runway not in self.merged_data.columns:
            print(f"Skipping preprocessing for {self.target_runway} - no valid data!")
            return

        print(f"Preprocessing data for {self.target_runway}...")
        df = self.merged_data.copy()

        # Remove rows where the target itself is missing
        df.dropna(subset=[self.target_runway], inplace=True)

        # Select numeric features (exclude the target)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.feature_columns = [c for c in numeric_cols if c != self.target_runway]

        # Median imputation for any remaining missing values
        if self.feature_columns:
            imputer = SimpleImputer(strategy='median')
            df[self.feature_columns] = imputer.fit_transform(df[self.feature_columns])
        else:
            print(f"No numeric features found for {self.target_runway}!")

        self.merged_data = df
        print(f"Preprocessed data shape for {self.target_runway}: {self.merged_data.shape}")

    def train_model(self, test_size=0.2, random_state=42):
        """
        Split data, scale features, and train an XGBRegressor
        for the current runway.
        """
        if self.merged_data is None or not self.feature_columns:
            print(f"Skipping training for {self.target_runway} - no valid data or features!")
            return

        print(f"Training model for {self.target_runway}...")
        df = self.merged_data

        # Time-based train/test split to respect temporal order
        split_idx = int(len(df) * (1 - test_size))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

        X_train = train_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        y_train = train_df[self.target_runway].values
        y_test = test_df[self.target_runway].values

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and fit the XGBoost model
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Store for later evaluation and prediction
        self.models[self.target_runway] = model
        self.scalers[self.target_runway] = scaler
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.y_train_pred = model.predict(X_train_scaled)
        self.y_test_pred = model.predict(X_test_scaled)

        print(f"Model training complete for {self.target_runway}!")

    def evaluate_model(self):
        """
        Compute and print performance metrics, then plot feature importances
        and actual vs. predicted values for the test set.
        """
        if self.target_runway not in self.models:
            print(f"Skipping evaluation for {self.target_runway} - no trained model!")
            return

        print(f"Evaluating model for {self.target_runway}...")

        # Regression metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)

        # Store results
        self.evaluation_results[self.target_runway] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        # Print results
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        # Plot top 10 feature importances
        feature_imp = pd.Series(
            self.models[self.target_runway].feature_importances_,
            index=self.feature_columns
        )
        top_features = feature_imp.sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        top_features.sort_values().plot(kind='barh')
        plt.title(f'Top 10 Feature Importances for {self.target_runway}')
        plt.tight_layout()
        plt.show()

        # Plot actual vs. predicted
        if self.y_test is not None and self.y_test_pred is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, self.y_test_pred, alpha=0.3)
            plt.plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()],
                     'r--', lw=2)
            plt.xlabel('Actual RVR')
            plt.ylabel('Predicted RVR')
            plt.title(f'Actual vs Predicted RVR Values for {self.target_runway}')
            plt.grid(True)
            plt.show()
        else:
            print("No test predictions available for plotting.")

    def save_model(self, runway, save_dir='./models'):
        """
        Save the model, scaler, and feature information for a specific runway.
        """
        if runway not in self.models:
            print(f"No model trained for runway {runway} - skipping save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare the data to save
        model_data = {
            'model': self.models[runway],
            'scaler': self.scalers[runway],
            'feature_columns': self.feature_columns,
            'target_runway': runway
        }
        
        # Create safe filename by replacing special characters
        safe_runway = runway.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        filename = os.path.join(save_dir, f'rvr_model_{safe_runway}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved model for {runway} to {filename}")

    def save_all_models(self, save_dir='saved_models'):
        """
        Save all trained models in the pipeline.
        """
        print("\nSaving all models...")
        for runway in self.models.keys():
            self.save_model(runway, save_dir)
        print("All models saved successfully!")

    def load_model(self, runway, model_path):
        """
        Load a saved model for a specific runway.
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[runway] = model_data['model']
        self.scalers[runway] = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Loaded model for {runway} from {model_path}")

    def _reset_runway_specific_attributes(self):
        """
        Reset per-runway attributes so the next runway starts fresh.
        """
        for attr in ['X_train', 'X_test', 'y_train', 'y_test',
                     'y_train_pred', 'y_test_pred', 'feature_columns']:
            setattr(self, attr, None)

    def run_complete_pipeline(self):
        """
        Execute the full pipeline: load data once, then for each runway:
        merge, create features, preprocess, train, evaluate, and summarize.
        """
        print("\n" + "="*50)
        print("Starting Multi-Runway RVR Prediction Pipeline")
        print("="*50)

        # Load and merge data
        self.load_rvr_data()
        self.load_weather_data()
        self.merge_data()

        if self.merged_data is None:
            print("Pipeline aborted - no valid merged data!")
            return

        # Loop through each runway
        for runway in self.target_runways:
            print(f"\n>>> Processing runway: {runway}")
            self.target_runway = runway

            if runway not in self.merged_data.columns:
                print(f"Runway {runway} not found in data - skipping")
                continue

            self.create_features()
            self.preprocess_data()
            self.train_model()
            self.evaluate_model()
            self._reset_runway_specific_attributes()

        print("\nPipeline completed for all runways!")
        self.show_summary()
        self.save_all_models()  # Save all models at the end

    def predict(self, runway, features):
        """
        Predict RVR for a single runway given a feature vector.
        """
        if runway not in self.models:
            raise ValueError(f"No model trained for runway {runway}")
        scaled = self.scalers[runway].transform([features])
        return self.models[runway].predict(scaled)[0]
    
    def generate_unified_predictions(self, future_hours=24, freq='15T'):
        """
        Generate a unified table of predictions for all runways at future time points.
        
        Parameters:
        -----------
        future_hours : int
            Number of hours to predict into the future
        freq : str
            Frequency of predictions (e.g., '15T' for 15-minute intervals)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: Datetime, RWY_11_BEG, RWY_11_TDZ, etc.
        """
        if not self.models:
            raise ValueError("No models trained - run pipeline first")
        
        if self.merged_data is None:
            raise ValueError("No merged data available - run pipeline first")

        # Create future time points
        last_datetime = self.merged_data['datetime'].max()
        future_datetimes = pd.date_range(
            start=last_datetime,
            periods=int(future_hours * (60 / int(freq[:-1]))) + 1,
            freq=freq
        )
        
        # Initialize prediction table
        prediction_table = pd.DataFrame({'Datetime': future_datetimes})
        
        # Generate predictions for each runway
        for runway in self.target_runways:
            if runway not in self.models:
                print(f"Skipping {runway} - no trained model")
                continue
                
            try:
                # Create safe column name
                col_name = (runway.replace(" ", "_")
                                .replace("(", "")
                                .replace(")", ""))
                
                # Get the feature columns for this runway
                if hasattr(self, 'feature_columns') and self.feature_columns:
                    runway_features = self.feature_columns
                else:
                    # Try to load from saved model
                    model_file = os.path.join('saved_models', f'rvr_model_{col_name}.pkl')
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            model_data = pickle.load(f)
                        runway_features = model_data['feature_columns']
                    else:
                        print(f"Warning: No feature columns available for {runway}")
                        continue
                
                # Prepare feature template using last available data point
                try:
                    last_features = self.merged_data.iloc[-1][runway_features].values.reshape(1, -1)
                except KeyError as e:
                    print(f"Missing features for {runway}: {str(e)}")
                    continue
                
                # Initialize predictions array
                predictions = []
                current_features = last_features.copy()
                
                # Generate multi-step predictions
                for _ in range(len(future_datetimes)):
                    try:
                        # Scale features
                        scaled_features = self.scalers[runway].transform(current_features)
                        
                        # Predict
                        pred = self.models[runway].predict(scaled_features)[0]
                        predictions.append(pred)
                        
                        # Update features with prediction (for autoregressive forecasting)
                        for i in [1, 2, 3]:  # Update all lag features
                            lag_col = f'{runway}_lag{i}'
                            if lag_col in runway_features:
                                idx = runway_features.index(lag_col)
                                if i == 1:
                                    current_features[0][idx] = pred
                                else:
                                    prev_lag = f'{runway}_lag{i-1}'
                                    if prev_lag in runway_features:
                                        prev_idx = runway_features.index(prev_lag)
                                        current_features[0][idx] = current_features[0][prev_idx]
                    
                    except Exception as e:
                        print(f"Prediction error for {runway}: {str(e)}")
                        predictions.append(np.nan)  # Append NaN if prediction fails
                        continue
                
                # Add to prediction table
                prediction_table[col_name] = predictions
                
            except Exception as e:
                print(f"Error processing {runway}: {str(e)}")
                continue
        
        # Format output
        prediction_table = prediction_table.set_index('Datetime')
        prediction_table = prediction_table.round(1)  # Round to 1 decimal place
        
        return prediction_table

    def show_summary(self):
        """
        Print and plot a summary of train & test R² scores across all runways.
        """
        if not self.evaluation_results:
            print("No evaluation results available!")
            return

        print("\n" + "="*50)
        print("Model Performance Summary Across All Runways")
        print("="*50)

        summary_df = pd.DataFrame.from_dict(self.evaluation_results, orient='index')
        print(summary_df)

        # Plot R² scores
        plt.figure(figsize=(12, 6))
        ax = summary_df[['train_r2', 'test_r2']].plot(kind='bar')
        plt.title('R² Scores Across Runways')
        plt.ylabel('R² Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Compute overall average test R²
        overall_test_r2 = summary_df['test_r2'].mean()
        print(f"Overall Test R² Score: {overall_test_r2:.4f}")

#%%
if __name__ == "__main__":
    # Define all runways of interest
    all_runways = [
        "RWY 09 (BEG)", "RWY 09 (TDZ)", "RWY 09 (MID)",
        "RWY 10 (BEG)", "RWY 10 (TDZ)", "RWY 10 (MID)",
        "RWY 11 (BEG)", "RWY 11 (TDZ)", "RWY 11 (MID)",
        "RWY 27 (BEG)", "RWY 27 (TDZ)", "RWY 27 (MID)",
        "RWY 28 (BEG)", "RWY 28 (TDZ)", "RWY 28 (MID)",
        "RWY 29 (BEG)", "RWY 29 (TDZ)", "RWY 29 (MID)",
    ]

    # Initialize and execute the pipeline
    predictor = RVRPredictor(target_runways=all_runways)
    predictor.run_complete_pipeline()

    # # Generate and save unified predictions
    # try:
    #     unified_predictions = predictor.generate_unified_predictions(future_hours=24, freq='15T')
    #     unified_predictions.to_csv('runway_predictions.csv')
    #     print("Successfully generated unified predictions:")
    #     print(unified_predictions.head())
    # except Exception as e:
    #     print(f"Failed to generate unified predictions: {str(e)}")