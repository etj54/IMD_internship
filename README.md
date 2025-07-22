# Real-Time RVR Prediction and Visualization System

## Project Overview
This project provides a real-time Runway Visual Range (RVR) prediction and visualization system for airport runways, with a focus on Delhi Airport. It integrates live sensor data, weather data, and advanced machine learning models with hyperparameter optimization to predict RVR values for different runway zones. The system achieves exceptional prediction accuracy with an average R² score of **99.37%** through comprehensive hyperparameter tuning. The results are visualized on an interactive map with a time slider, enabling users to see predicted visibility conditions over time.

## Features
- **Real-Time RVR Prediction:** Continuously reads the latest RVR logs and weather data, generating predictions for multiple runway zones using trained machine learning models with advanced hyperparameter optimization.
- **Advanced ML Pipeline:** Employs XGBoost models with RandomizedSearchCV hyperparameter tuning, achieving 99.37% average R² accuracy across 11 runway zones.
- **Interactive Map Visualization:** Generates an HTML map with Folium, displaying predicted RVR values for each runway zone. Includes a time slider to animate predictions over time.
- **Historical and Real-Time Data Support:** Handles both historical predictions and real-time updates, saving results as CSV files for further analysis or visualization.
- **Comprehensive Logging:** Detailed training logs with performance metrics for model comparison and optimization tracking.
- **Extensible and Testable:** Modular code structure with test scripts and simulation capabilities for development and validation.

## Directory Structure
```
├── data/
│   ├── raw/
│   │   ├── rvr_logs/           # Raw RVR log CSV files (by year)
│   │   └── weather/            # Weather data Excel files (by runway/year)
│   ├── predicted_rvr/          # Historical predicted RVR CSVs
│   └── real_time_predictions/  # Real-time prediction output CSVs
├── saved_models/               # Trained ML models for each runway zone
├── scripts/                    # All main Python scripts
│   ├── generate_rvr_map.py         # Script to generate the interactive RVR map
│   ├── live_rvr_predictor.py       # Core real-time RVR prediction logic
│   ├── real_time_rvr_system.py     # Main real-time system orchestrator
│   ├── test_real_time_system.py    # Test script for the real-time system
│   ├── XGBst_updated.py            # Advanced XGBoost training with hyperparameter optimization
│   ├── all_rvr_cleaned.py          # Cleans and aggregates raw RVR log text files
│   └── all_data_cleaned.py         # Cleans and processes weather data Excel files
├── training_logs.csv           # Detailed training experiment logs and performance metrics
├── rvr_map_with_slider.html    # Output HTML map with time slider
├── README.md
```

## Setup & Installation
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   
   The main dependencies are:
   - `pandas`
   - `numpy`
   - `folium`
   - `geopy`
   - `joblib`
   - `openpyxl` (for reading Excel files)
   - `matplotlib`, `seaborn`, `xgboost`, `scikit-learn` (for model training scripts)

   You can install them with:
   ```bash
   pip install pandas numpy folium geopy joblib openpyxl matplotlib seaborn xgboost scikit-learn
   ```

3. **Prepare data:**
   - Place RVR log CSVs in `data/raw/rvr_logs/` (e.g., `RVR_2024.csv`).
   - Place weather Excel files in `data/raw/weather/` (e.g., `RUNWAY11_2024.xlsx`).
   - Ensure trained model files are in `saved_models/`.

## Usage
### 1. Real-Time Prediction System
Run the real-time prediction system to continuously generate and update RVR predictions:
```bash
python scripts/real_time_rvr_system.py
```
- This will read the latest RVR and weather data, generate predictions, and save results to `data/real_time_predictions/`.
- To test a single update cycle, run:
```bash
python scripts/test_real_time_system.py
```

### 2. Generate Interactive Map
After predictions are available, generate the interactive map:
```bash
python scripts/generate_rvr_map.py
```
- This will create `rvr_map_with_slider.html` in the project root.
- Open this HTML file in your browser to view the map with a time slider and RVR status color coding.

### 3. Simulate Live Predictions (Development)
You can run the predictor in simulation mode for development/testing:
```bash
python scripts/live_rvr_predictor.py
```
- This will simulate sensor data and print predictions in real-time.

### 4. Data Cleaning and Model Training with Hyperparameter Optimization
- To clean and aggregate raw RVR logs:
  ```bash
  python scripts/all_rvr_cleaned.py
  ```
- To clean and process weather data:
  ```bash
  python scripts/all_data_cleaned.py
  ```
- To train and evaluate XGBoost models with advanced hyperparameter tuning:
  ```bash
  python scripts/XGBst_updated.py
  ```
  This will:
  - Perform comprehensive hyperparameter optimization using RandomizedSearchCV
  - Compare multiple training approaches (tuned vs fixed parameters)
  - Generate detailed training logs in `training_logs.csv`
  - Save optimized models achieving 99.37% average R² accuracy

## Output
- **CSV Files:** Real-time and historical predictions are saved in `data/real_time_predictions/` and `data/predicted_rvr/`.
- **HTML Map:** The interactive map is saved as `rvr_map_with_slider.html`.
- **Training Logs:** Comprehensive training experiments and performance metrics are logged in `training_logs.csv`.
- **Optimized Models:** Hyperparameter-tuned models achieving 99.37% average R² accuracy are saved in `saved_models/`.

## Performance Achievements
🏆 **Breakthrough Results:** The system achieves exceptional performance through advanced hyperparameter optimization:
- **Average R² Score:** 99.37% (exceeding 99.23% baseline)
- **Models Above Target:** 8 out of 11 runway models exceed 99.23% accuracy
- **Best Performer:** RWY 09 TDZ with 99.75% R² score
- **Optimization Method:** RandomizedSearchCV with 50 parameter combinations per model
- **Training Dataset:** 217,330 records spanning 2020-2024

## Credits
- Developed for real-time RVR prediction and visualization at Delhi Airport.
- Uses open-source libraries: [Folium](https://python-visualization.github.io/folium/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Geopy](https://geopy.readthedocs.io/), [Joblib](https://joblib.readthedocs.io/), [OpenPyXL](https://openpyxl.readthedocs.io/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/).

---
For questions or contributions, please open an issue or pull request. 