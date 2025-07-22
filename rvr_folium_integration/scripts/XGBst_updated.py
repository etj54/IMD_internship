import os
import glob
import pickle
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class RVRPredictorUpdated:
    """
    Updated RVR predictor that works with the current data structure
    """
    def __init__(self, base_path='.', target_runways=None):
        # Updated base path to work with current structure
        self.base_path = base_path
        self.rvr_path = os.path.join(base_path, 'data', 'raw', 'rvr_logs')
        self.weather_path = os.path.join(base_path, 'data', 'raw', 'weather')
        
        # Available runways based on your data
        self.target_runways = target_runways or [
            "RWY 09 (BEG)", "RWY 09 (TDZ)",
            "RWY 10 (TDZ)",
            "RWY 11 (BEG)", "RWY 11 (TDZ)",
            "RWY 27 (MID)",
            "RWY 28 (BEG)", "RWY 28 (MID)", "RWY 28 (TDZ)",
            "RWY 29 (BEG)", "RWY 29 (MID)"
        ]
        
        # Data storage
        self.rvr_data = None
        self.weather_data = None
        self.merged_data = None
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.training_logs = []  # Store training logs for CSV
        
        # Training configuration
        self.epochs_list = [10, 25, 20]  # Different epoch configurations
        self.use_hyperparameter_tuning = True  # Enable hyperparameter tuning
        
        # Hyperparameter search space
        self.param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Current runway being processed
        self.target_runway = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.feature_columns = None

    def load_rvr_data(self):
        """Load and combine RVR data from CSV files"""
        print("Loading RVR data...")
        rvr_files = glob.glob(os.path.join(self.rvr_path, "RVR_*.csv"))
        
        if not rvr_files:
            print(f"No RVR files found in {self.rvr_path}")
            return None
            
        print(f"Found {len(rvr_files)} RVR files")
        
        all_rvr_data = []
        for file in rvr_files:
            print(f"  Loading {os.path.basename(file)}")
            try:
                df = pd.read_csv(file)
                # Convert datetime - try multiple formats
                try:
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')
                except:
                    try:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
                    except:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
                all_rvr_data.append(df)
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                
        if all_rvr_data:
            self.rvr_data = pd.concat(all_rvr_data, ignore_index=True)
            self.rvr_data = self.rvr_data.sort_values('Datetime').reset_index(drop=True)
            print(f"Loaded {len(self.rvr_data)} RVR records")
            print(f"Date range: {self.rvr_data['Datetime'].min()} to {self.rvr_data['Datetime'].max()}")
            return self.rvr_data
        else:
            print("No valid RVR data loaded")
            return None

    def load_weather_data(self):
        """Load and combine weather data from Excel files"""
        print("Loading weather data...")
        weather_files = glob.glob(os.path.join(self.weather_path, "RUNWAY*.xlsx"))
        
        if not weather_files:
            print(f"No weather files found in {self.weather_path}")
            return None
            
        print(f"Found {len(weather_files)} weather files")
        
        all_weather_data = []
        for file in weather_files:
            print(f"  Loading {os.path.basename(file)}")
            try:
                df = pd.read_excel(file)
                # Add runway identifier
                runway_id = os.path.basename(file).replace('.xlsx', '')
                df['runway'] = runway_id
                all_weather_data.append(df)
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                
        if all_weather_data:
            self.weather_data = pd.concat(all_weather_data, ignore_index=True)
            print(f"Loaded {len(self.weather_data)} weather records")
            return self.weather_data
        else:
            print("No valid weather data loaded")
            return None

    def create_features(self, df, target_runway):
        """Create features for modeling"""
        features_df = df.copy()
        
        # Temporal features
        features_df['hour'] = features_df['Datetime'].dt.hour
        features_df['day_of_week'] = features_df['Datetime'].dt.dayofweek
        features_df['month'] = features_df['Datetime'].dt.month
        features_df['day_of_year'] = features_df['Datetime'].dt.dayofyear
        
        # Lag features for target runway
        if target_runway in features_df.columns:
            for lag in [1, 2, 3]:
                features_df[f'{target_runway}_lag_{lag}'] = features_df[target_runway].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12]:
                features_df[f'{target_runway}_rolling_mean_{window}'] = features_df[target_runway].rolling(window).mean()
                features_df[f'{target_runway}_rolling_std_{window}'] = features_df[target_runway].rolling(window).std()
        
        # Add other runway values as features (cross-runway dependencies)
        runway_cols = [col for col in features_df.columns if 'RWY' in col and col != target_runway]
        
        return features_df, runway_cols

    def prepare_data_for_modeling(self, target_runway):
        """Prepare data for a specific runway"""
        print(f"\nPreparing data for {target_runway}...")
        
        if self.rvr_data is None:
            print("No RVR data available")
            return None, None, None, None
            
        # Create features
        features_df, runway_cols = self.create_features(self.rvr_data, target_runway)
        
        # Select feature columns
        feature_cols = ['hour', 'day_of_week', 'month', 'day_of_year']
        
        # Add lag features
        lag_cols = [col for col in features_df.columns if f'{target_runway}_lag_' in col]
        feature_cols.extend(lag_cols)
        
        # Add rolling features
        rolling_cols = [col for col in features_df.columns if f'{target_runway}_rolling_' in col]
        feature_cols.extend(rolling_cols)
        
        # Add other runway features (limited to avoid overfitting)
        feature_cols.extend(runway_cols[:5])  # Use up to 5 other runways as features
        
        # Prepare target and features
        y = features_df[target_runway]
        X = features_df[feature_cols]
        
        # Remove rows with missing target values
        valid_mask = ~y.isna()
        y = y[valid_mask]
        X = X[valid_mask]
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Remove remaining invalid values
        final_mask = ~(X_imputed.isna().any(axis=1) | y.isna() | np.isinf(y))
        X_final = X_imputed[final_mask]
        y_final = y[final_mask]
        
        print(f"Final dataset shape: {X_final.shape}, Target shape: {y_final.shape}")
        
        if len(X_final) < 100:
            print(f"Warning: Only {len(X_final)} valid samples for {target_runway}")
            return None, None, None, None
            
        return X_final, y_final, feature_cols, imputer

    def perform_hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        print("    Performing hyperparameter tuning...")
        
        # Base model
        base_model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Use RandomizedSearchCV for efficiency (faster than GridSearchCV)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_grid,
            n_iter=50,  # Number of parameter combinations to try
            scoring='r2',  # Use R² as scoring metric
            cv=3,  # 3-fold cross-validation
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the random search
        random_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        best_cv_score = random_search.best_score_
        
        # Make predictions with best model
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"    Best CV R²: {best_cv_score:.4f}")
        print(f"    Best Test R²: {test_r2:.4f}")
        print(f"    Best parameters: {best_params}")
        
        return {
            'model': best_model,
            'params': best_params,
            'cv_score': best_cv_score,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
        }

    def train_model_with_fixed_params(self, X_train, y_train, X_test, y_test, epoch):
        """Train model with fixed parameters (original approach)"""
        model = XGBRegressor(
            n_estimators=epoch,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        return {
            'model': model,
            'params': {'n_estimators': epoch, 'method': 'fixed'},
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
        }

    def train_model(self, target_runway):
        """Train XGBoost model for a specific runway with hyperparameter tuning"""
        print(f"\nTraining model for {target_runway}...")
        
        X, y, feature_cols, imputer = self.prepare_data_for_modeling(target_runway)
        
        if X is None:
            print(f"Skipping {target_runway} - insufficient data")
            return False
            
        # Temporal split (older data for training, newer for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        best_model = None
        best_scaler = None
        best_score = -float('inf')
        best_method = None
        best_predictions = None
        all_results = []
        
        # Method 1: Hyperparameter Tuning
        if self.use_hyperparameter_tuning:
            print(f"\n  Method 1: Hyperparameter Tuning")
            try:
                tuning_result = self.perform_hyperparameter_tuning(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                test_r2 = tuning_result['metrics']['test_r2']
                method_name = f"tuned_{tuning_result['params']['n_estimators']}"
                
                # Log results
                log_entry = {
                    'runway': target_runway,
                    'method': 'hyperparameter_tuning',
                    'epoch': tuning_result['params'].get('n_estimators', 'tuned'),
                    'train_rmse': tuning_result['metrics']['train_rmse'],
                    'test_rmse': tuning_result['metrics']['test_rmse'],
                    'train_r2': tuning_result['metrics']['train_r2'],
                    'test_r2': tuning_result['metrics']['test_r2'],
                    'train_mae': tuning_result['metrics']['train_mae'],
                    'test_mae': tuning_result['metrics']['test_mae'],
                    'best_params': str(tuning_result['params']),
                    'cv_score': tuning_result.get('cv_score', 'N/A'),
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.training_logs.append(log_entry)
                all_results.append(('Hyperparameter Tuning', tuning_result))
                
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = tuning_result['model']
                    best_scaler = scaler
                    best_method = method_name
                    best_predictions = tuning_result
                    
                print(f"    Hyperparameter Tuning - Test R²: {test_r2:.4f}")
                
            except Exception as e:
                print(f"    Hyperparameter tuning failed: {e}")
        
        # Method 2: Fixed Epochs (Original approach)
        print(f"\n  Method 2: Fixed Epochs {self.epochs_list}")
        for epoch in self.epochs_list:
            print(f"\n    Training with {epoch} estimators...")
            
            try:
                fixed_result = self.train_model_with_fixed_params(
                    X_train_scaled, y_train, X_test_scaled, y_test, epoch
                )
                
                test_r2 = fixed_result['metrics']['test_r2']
                
                # Log results
                log_entry = {
                    'runway': target_runway,
                    'method': 'fixed_params',
                    'epoch': epoch,
                    'train_rmse': fixed_result['metrics']['train_rmse'],
                    'test_rmse': fixed_result['metrics']['test_rmse'],
                    'train_r2': fixed_result['metrics']['train_r2'],
                    'test_r2': fixed_result['metrics']['test_r2'],
                    'train_mae': fixed_result['metrics']['train_mae'],
                    'test_mae': fixed_result['metrics']['test_mae'],
                    'best_params': str(fixed_result['params']),
                    'cv_score': 'N/A',
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.training_logs.append(log_entry)
                all_results.append((f'Fixed Epoch {epoch}', fixed_result))
                
                print(f"      Epoch {epoch} - Test R²: {test_r2:.4f}, Test RMSE: {fixed_result['metrics']['test_rmse']:.2f}")
                
                # Check if this is the best model
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = fixed_result['model']
                    best_scaler = scaler
                    best_method = f'epoch_{epoch}'
                    best_predictions = fixed_result
                    
            except Exception as e:
                print(f"      Error training with epoch {epoch}: {e}")
        
        if best_model is None:
            print(f"  Failed to train any model for {target_runway}")
            return False
        
        # Store the best model
        self.models[target_runway] = best_model
        self.scalers[target_runway] = best_scaler
        self.feature_columns = feature_cols
        
        # Store for evaluation
        self.target_runway = target_runway
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = best_predictions['train_pred']
        self.y_test_pred = best_predictions['test_pred']
        
        # Store best model results
        self.evaluation_results[target_runway] = {
            'best_method': best_method,
            'best_params': best_predictions['params'],
            'train_rmse': best_predictions['metrics']['train_rmse'],
            'test_rmse': best_predictions['metrics']['test_rmse'],
            'train_r2': best_predictions['metrics']['train_r2'],
            'test_r2': best_predictions['metrics']['test_r2'],
            'train_mae': best_predictions['metrics']['train_mae'],
            'test_mae': best_predictions['metrics']['test_mae']
        }
        
        print(f"\n  🏆 Best model for {target_runway}: {best_method} with Test R²: {best_score:.4f}")
        print(f"  📊 All results comparison:")
        for method_name, result in all_results:
            r2 = result['metrics']['test_r2']
            rmse = result['metrics']['test_rmse']
            print(f"    - {method_name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
        
        print(f"Model trained successfully for {target_runway}")
        return True

    def evaluate_model(self):
        """Evaluate the best trained model"""
        if self.target_runway not in self.models:
            print(f"No model found for {self.target_runway}")
            return
            
        print(f"\nEvaluating best model for {self.target_runway}...")
        
        # Get stored results from best model
        results = self.evaluation_results[self.target_runway]
        
        # Print results
        print(f"Best Method: {results['best_method']}")
        print(f"Best Parameters: {results['best_params']}")
        print(f"Train RMSE: {results['train_rmse']:.2f}")
        print(f"Test RMSE: {results['test_rmse']:.2f}")
        print(f"Train R²: {results['train_r2']:.4f}")
        print(f"Test R²: {results['test_r2']:.4f}")
        print(f"Train MAE: {results['train_mae']:.2f}")
        print(f"Test MAE: {results['test_mae']:.2f}")

    def save_training_logs(self, log_file="training_logs.csv"):
        """Save training logs to CSV file"""
        if not self.training_logs:
            print("No training logs to save")
            return
            
        logs_df = pd.DataFrame(self.training_logs)
        logs_df.to_csv(log_file, index=False)
        print(f"Training logs saved to {log_file}")
        
        # Print summary of logs
        print(f"Total training runs logged: {len(self.training_logs)}")
        print(f"Runways trained: {logs_df['runway'].nunique()}")
        print(f"Epochs tested: {sorted(logs_df['epoch'].unique())}")

    def save_model(self, target_runway, save_dir="../saved_models"):
        """Save trained model"""
        if target_runway not in self.models:
            print(f"No model to save for {target_runway}")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Clean filename
        safe_name = target_runway.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"rvr_model_{safe_name}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        # Save model and scaler together
        model_data = {
            'model': self.models[target_runway],
            'scaler': self.scalers[target_runway],
            'feature_columns': self.feature_columns,
            'runway': target_runway
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("=" * 60)
        print("Starting RVR Prediction Pipeline")
        print("=" * 60)
        
        # Load data
        if self.load_rvr_data() is None:
            print("Failed to load RVR data")
            return
            
        print(f"\nAvailable runways in data: {[col for col in self.rvr_data.columns if 'RWY' in col]}")
        
        # Train models for each runway
        successful_models = []
        
        for runway in self.target_runways:
            if runway not in self.rvr_data.columns:
                print(f"Skipping {runway} - not found in data")
                continue
                
            try:
                if self.train_model(runway):
                    self.evaluate_model()
                    self.save_model(runway)
                    successful_models.append(runway)
                    
            except Exception as e:
                print(f"Error processing {runway}: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        # Save training logs to CSV
        self.save_training_logs()
        
        if self.evaluation_results:
            summary_df = pd.DataFrame.from_dict(self.evaluation_results, orient='index')
            print("\nBest Model Results:")
            print(summary_df)
            
            # Calculate averages
            avg_test_r2 = summary_df['test_r2'].mean()
            avg_test_rmse = summary_df['test_rmse'].mean()
            avg_test_mae = summary_df['test_mae'].mean()
            
            print(f"\nOverall Performance (Best Models):")
            print(f"Average Test R²: {avg_test_r2:.4f}")
            print(f"Average Test RMSE: {avg_test_rmse:.2f}")
            print(f"Average Test MAE: {avg_test_mae:.2f}")
            
            print(f"\nSuccessfully trained {len(successful_models)} models:")
            for model in successful_models:
                best_method = self.evaluation_results[model]['best_method']
                best_r2 = self.evaluation_results[model]['test_r2']
                improvement = "🚀" if best_r2 > 0.9923 else "✅" if best_r2 > 0.99 else "📈"
                print(f"  {improvement} {model}: {best_method} (R² = {best_r2:.4f})")
                
            # Check if we beat the 99.23% target
            models_above_target = sum(1 for model in successful_models 
                                    if self.evaluation_results[model]['test_r2'] > 0.9923)
            
            print(f"\n🎯 TARGET ANALYSIS (vs 99.23% baseline):")
            print(f"Models exceeding 99.23%: {models_above_target}/{len(successful_models)}")
            if avg_test_r2 > 0.9923:
                improvement = ((avg_test_r2 - 0.9923) / 0.9923) * 100
                print(f"🏆 ACHIEVEMENT UNLOCKED! Average R² improved by {improvement:.2f}%")
            else:
                gap = ((0.9923 - avg_test_r2) / 0.9923) * 100
                print(f"📊 Gap to target: {gap:.2f}% below 99.23%")
                
        else:
            print("No models were successfully trained")

if __name__ == "__main__":
    # Initialize predictor with current directory structure
    predictor = RVRPredictorUpdated(base_path='..')
    
    # Run the complete pipeline
    predictor.run_complete_pipeline()
