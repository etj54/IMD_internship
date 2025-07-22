import pandas as pd
import glob
import os
import folium
from folium import plugins
from geopy.distance import geodesic
import numpy as np
from datetime import datetime, timedelta
from folium import Element
import json

print("=== RVR MAP GENERATOR WITH TIME SLIDER ===\n")

# Step 1: Check if folder exists
folder_path = '.'  # Look in current directory
print(f"1. Checking folder: {folder_path}")
print(f"   Folder exists: {os.path.exists(folder_path)}")

if os.path.exists(folder_path):
    files = os.listdir(folder_path)
    csv_files_in_dir = [f for f in files if f.endswith('.csv')]
    print(f"   CSV files in folder: {csv_files_in_dir}")
else:
    print("   ❌ FOLDER NOT FOUND!")
    print("   Current directory:", os.getcwd())
    print("   Available folders:", [d for d in os.listdir('.') if os.path.isdir(d)])

# Step 2: Find CSV files - ENHANCED DEBUGGING
print(f"\n2. ENHANCED CSV FILE SEARCH:")
print(f"   Current working directory: {os.getcwd()}")

# Check multiple possible locations for CSV files - PRIORITIZE REAL-TIME DATA
search_paths = [
    'data/real_time_predictions',  # Real-time predictions directory
    'data/real_time_predictions/latest_predictions.csv',  # Latest predictions file
    '.',  # Current directory
    'data/predicted_rvr',  # predicted_rvr subdirectory
    'data/raw',  # raw data directory (if needed)
]

all_csv_files = []
for search_path in search_paths:
    print(f"\n   Searching in: {search_path}")
    print(f"   Path exists: {os.path.exists(search_path)}")
    
    if os.path.exists(search_path):
        if os.path.isdir(search_path):
            files_in_path = os.listdir(search_path)
            csv_files_in_path = [f for f in files_in_path if f.endswith('.csv')]
            print(f"   Files in {search_path}: {len(files_in_path)} total files")
            print(f"   CSV files in {search_path}: {csv_files_in_path}")
            
            # Add full paths to the list
            for csv_file in csv_files_in_path:
                full_path = os.path.join(search_path, csv_file)
                all_csv_files.append(full_path)
                print(f"   Added: {full_path}")
        else:
            # It's a file, check if it's a CSV
            if search_path.endswith('.csv'):
                all_csv_files.append(search_path)
                print(f"   Added file: {search_path}")
            else:
                print(f"   {search_path} is not a CSV file")
    else:
        print(f"   Path {search_path} does not exist")

print(f"\n   Total CSV files found across all paths: {len(all_csv_files)}")
for i, file_path in enumerate(all_csv_files):
    print(f"   {i+1}. {file_path}")

# Use the found CSV files instead of glob
csv_files = all_csv_files

if not csv_files:
    print("   ❌ NO CSV FILES FOUND!")
    print("   DEBUG: Let's check what's actually in the current directory:")
    print(f"   Current directory contents: {os.listdir('.')}")
    if os.path.exists('predicted_rvr'):
        print(f"   predicted_rvr directory contents: {os.listdir('predicted_rvr')}")
else:
    print(f"   ✅ Found {len(csv_files)} CSV files")

# Step 3: Define runway positions with better coordinates
runways = {
    '11': {'beg': (28.546269726812046, 77.07209263473048), 'heading': 103.0, 'length_m': 4430},
    '29': {'beg': (28.538300748490773, 77.1068818248323), 'heading': 283.0, 'length_m': 4430},
    '10': {'beg': (28.567182331326887, 77.08498547442058), 'heading': 104.4, 'length_m': 3813},
    '28': {'beg': (28.558606400571975, 77.12240587666396), 'heading': 284.4, 'length_m': 3813},
    '09': {'beg': (28.570559706309265, 77.08822392522595), 'heading': 91.4,  'length_m': 2816},
    '27': {'beg': (28.569910389750675, 77.11687357828113), 'heading': 271.4, 'length_m': 2816},
}

print(f"\n3. Computing runway positions...")
predicted_map = {}
for rwy, data in runways.items():
    beg = data['beg']
    hdg = data['heading']
    length = data['length_m']
    tdz = geodesic(meters=300).destination(beg, hdg)
    mid = geodesic(meters=length / 2).destination(beg, hdg)
    
    predicted_map[f'RWY_{rwy}_TDZ'] = (tdz.latitude, tdz.longitude)
    predicted_map[f'RWY_{rwy}_MID'] = (mid.latitude, mid.longitude)

print(f"   Expected zone names: {list(predicted_map.keys())}")

def prepare_time_series_data(df, runway_positions, hours_ahead=12):
    """
    Prepare time series data for the slider
    
    Args:
        df: DataFrame with prediction data
        runway_positions: Dictionary of runway zone positions
        hours_ahead: Number of hours to look ahead
    
    Returns:
        List of GeoJSON features for each time step
    """
    print(f"\n📊 Preparing time series data for {hours_ahead} hours...")
    # Use the earliest timestamp in the data as the reference point
    reference_time = df['Datetime'].min()
    print(f"   Reference time from data: {reference_time}")
    # Get data for the next 12 hours from reference time
    end_time = reference_time + timedelta(hours=hours_ahead)
    mask = (df['Datetime'] >= reference_time) & (df['Datetime'] <= end_time)
    filtered_df = df[mask].copy()
    if filtered_df.empty:
        print(f"   ⚠️ No data found for the next {hours_ahead} hours")
        print(f"   Using first {hours_ahead}*6 rows of available data")
        filtered_df = df.head(hours_ahead * 6).copy()  # 6 records per hour (10-minute intervals)
    print(f"   Found {len(filtered_df)} time steps")
    print(f"   Time range: {filtered_df['Datetime'].min()} to {filtered_df['Datetime'].max()}")
    
    # Zone to column mapping
    zone_to_column_mapping = {
        'RWY_09_TDZ': 'RWY_09_TDZ_predicted',
        'RWY_09_MID': 'RWY_09_BEG_predicted',  # Using BEG as MID
        'RWY_10_TDZ': 'RWY_10_TDZ_predicted',
        'RWY_10_MID': 'RWY_10_TDZ_predicted',  # Using TDZ as MID since no MID column
        'RWY_11_TDZ': 'RWY_11_TDZ_predicted',
        'RWY_11_MID': 'RWY_11_BEG_predicted',  # Using BEG as MID
        'RWY_27_TDZ': 'RWY_27_MID_predicted',  # Using MID as TDZ
        'RWY_27_MID': 'RWY_27_MID_predicted',
        'RWY_28_TDZ': 'RWY_28_TDZ_predicted',
        'RWY_28_MID': 'RWY_28_MID_predicted',
        'RWY_29_TDZ': 'RWY_29_BEG_predicted',  # Using BEG as TDZ
        'RWY_29_MID': 'RWY_29_MID_predicted',
    }
    
    features = []
    
    for idx, row in filtered_df.iterrows():
        timestamp = row['Datetime']
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create features for this time step
        for zone_name, (lat, lon) in runway_positions.items():
            # Get predicted value
            predicted_value = None
            if zone_name in zone_to_column_mapping:
                column_name = zone_to_column_mapping[zone_name]
                if column_name in row.index and not pd.isna(row[column_name]):
                    predicted_value = row[column_name]
            
            # Use default if no prediction available
            if predicted_value is None:
                predicted_value = 1000  # Default value
            
            # Determine color based on RVR value
            if predicted_value >= 800:
                color = 'green'
                status = 'Good'
            elif predicted_value >= 500:
                color = 'orange'
                status = 'Moderate'
            elif predicted_value >= 200:
                color = 'red'
                status = 'Poor'
            else:
                color = 'darkred'
                status = 'Very Poor'
            
            # Create GeoJSON feature with different styling for time slider
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                },
                'properties': {
                    'time': time_str,
                    'zone': zone_name,
                    'rvr_value': round(predicted_value, 1),
                    'status': status,
                    'color': color,
                    'style': {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0.7,
                        'radius': 12
                    },
                    'popup_content': f"""
                    <div style=\"min-width: 200px;\">
                        <h4 style=\"margin: 5px 0; color: {color};\">{zone_name}</h4>
                        <p><strong>Predicted RVR:</strong> {predicted_value:.0f}m</p>
                        <p><strong>Status:</strong> {status}</p>
                        <p><strong>Time:</strong> {time_str}</p>
                    </div>
                    """
                }
            }
            features.append(feature)
    
    print(f"   Created {len(features)} features for time slider")
    return features

# Step 4: Process the CSV file for time series data
latest_data = None
latest_file = None
df = None
if csv_files:
    # Sort files by modification time to get the latest
    csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = csv_files[0]
    print(f"\n4. Processing file for time series: {os.path.basename(latest_file)}")
    print(f"   Full path: {latest_file}")
    print(f"   File size: {os.path.getsize(latest_file) / (1024*1024):.2f} MB")
    print(f"   Last modified: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")
    
    try:
        print(f"   Attempting to read CSV file...")
        df = pd.read_csv(latest_file)
        print(f"   ✅ Successfully read CSV file")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Column count: {len(df.columns)}")
        
        # Show first few rows for debugging
        print(f"   First 3 rows:")
        print(df.head(3).to_string())
        
        # Find datetime column
        datetime_col = None
        datetime_candidates = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_candidates.append(col)
        
        print(f"   Datetime column candidates: {datetime_candidates}")
        
        if datetime_candidates:
            datetime_col = datetime_candidates[0]  # Use the first one found
            print(f"   Selected datetime column: {datetime_col}")
            
            # Check if datetime conversion works
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                print(f"   ✅ Successfully converted {datetime_col} to datetime")
                print(f"   Date range: {df[datetime_col].min()} to {df[datetime_col].max()}")
                print(f"   Total rows: {len(df)}")
                
                # Get the latest row for static display
                latest_row = df.iloc[-1]
                latest_data = latest_row
                print(f"   Latest timestamp: {latest_row[datetime_col]}")
                print(f"   Latest row index: {df.index[-1]}")
                
            except Exception as e:
                print(f"   ❌ Error converting datetime column: {e}")
                datetime_col = None
        else:
            print(f"   ❌ No datetime column found!")
            print(f"   Available columns: {list(df.columns)}")
        
        # Check for predicted columns with enhanced debugging
        predicted_cols = [col for col in df.columns if 'predicted' in col.lower()]
        print(f"   Predicted columns found: {len(predicted_cols)}")
        if predicted_cols:
            print(f"   All predicted columns: {predicted_cols}")
            print(f"   Sample predicted columns: {predicted_cols[:5]}")
            
            # Check for non-null values in predicted columns
            for col in predicted_cols[:3]:  # Check first 3 predicted columns
                non_null_count = df[col].notna().sum()
                print(f"   {col}: {non_null_count}/{len(df)} non-null values")
                if non_null_count > 0:
                    print(f"   {col} sample values: {df[col].dropna().head(3).tolist()}")
        else:
            print(f"   ❌ No predicted columns found!")
            print(f"   All columns: {list(df.columns)}")
            
    except Exception as e:
        print(f"   ❌ Error processing CSV: {e}")
        import traceback
        print(f"   Full error traceback:")
        traceback.print_exc()

# Step 5: Create the map with time slider
print(f"\n5. Creating RVR map with time slider...")

# Create map centered on Delhi Airport
m = folium.Map(location=[28.556, 77.095], zoom_start=14, tiles='OpenStreetMap')

# Add runway beginning markers (static)
print("   Adding runway beginning markers...")
for rwy, data in runways.items():
    folium.Marker(
        location=data['beg'],
        popup=f"<b>Runway {rwy} Beginning</b><br>Heading: {data['heading']}°<br>Length: {data['length_m']}m",
        icon=folium.Icon(color='green', icon='info-sign'),
        tooltip=f"RWY {rwy} BEG"
    ).add_to(m)

# Add time slider if we have data
if df is not None:
    print("   Preparing time series data for slider...")
    
    # Prepare time series data
    time_series_features = prepare_time_series_data(df, predicted_map, hours_ahead=12)
    
    if time_series_features:
        print("   Adding time slider to map...")
        
        # Create TimestampedGeoJson layer
        timestamped_geojson = plugins.TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': time_series_features
            },
            period='PT10M',  # 10-minute periods
            duration='PT5M',  # 5-minute duration for each step
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options='YYYY-MM-DD HH:mm:ss',
            time_slider_drag_update=True
        )
        
        # Add the time slider to the map
        timestamped_geojson.add_to(m)
        
        print("   ✅ Time slider added successfully!")
    else:
        print("   ❌ No time series features created")
else:
    print("   ❌ No data available for time slider")

# Add static RVR markers for the latest available values (like the old map)
if df is not None and latest_data is not None:
    print("   Adding static RVR markers for latest values...")
    # Find datetime column
    datetime_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    # Zone to column mapping
    zone_to_column_mapping = {
        'RWY_09_TDZ': 'RWY_09_TDZ_predicted',
        'RWY_09_MID': 'RWY_09_BEG_predicted',  # Using BEG as MID
        'RWY_10_TDZ': 'RWY_10_TDZ_predicted',
        'RWY_10_MID': 'RWY_10_TDZ_predicted',  # Using TDZ as MID since no MID column
        'RWY_11_TDZ': 'RWY_11_TDZ_predicted',
        'RWY_11_MID': 'RWY_11_BEG_predicted',  # Using BEG as MID
        'RWY_27_TDZ': 'RWY_27_MID_predicted',  # Using MID as TDZ
        'RWY_27_MID': 'RWY_27_MID_predicted',
        'RWY_28_TDZ': 'RWY_28_TDZ_predicted',
        'RWY_28_MID': 'RWY_28_MID_predicted',
        'RWY_29_TDZ': 'RWY_29_BEG_predicted',  # Using BEG as TDZ
        'RWY_29_MID': 'RWY_29_MID_predicted',
    }
    for zone_name, (lat, lon) in predicted_map.items():
        # Get predicted value
        predicted_value = None
        if zone_name in zone_to_column_mapping:
            column_name = zone_to_column_mapping[zone_name]
            if column_name in latest_data.index and not pd.isna(latest_data[column_name]):
                predicted_value = latest_data[column_name]
        if predicted_value is None:
            predicted_value = 1000
        # Determine color
        if predicted_value >= 800:
            color = 'green'
            status = 'Good'
        elif predicted_value >= 500:
            color = 'orange'
            status = 'Moderate'
        elif predicted_value >= 200:
            color = 'red'
            status = 'Poor'
        else:
            color = 'darkred'
            status = 'Very Poor'
        # Popup
        popup_content = f"""
        <div style='min-width: 200px;'>
            <h4 style='margin: 5px 0; color: {color};'>{zone_name}</h4>
            <p><strong>Predicted RVR:</strong> {predicted_value:.0f}m</p>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Time:</strong> {latest_data[datetime_col] if latest_data is not None else 'N/A'}</p>
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            popup=folium.Popup(popup_content, max_width=300),
            color='black',
            weight=3,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"{zone_name}: {predicted_value:.0f}m"
        ).add_to(m)
        # Add text label
        label_html = f"""
        <div style='background-color: white; padding: 4px 6px; border: 2px solid {color}; border-radius: 4px; font-size: 11px; font-weight: bold; color: black; text-align: center; min-width: 60px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            <div style='font-size: 10px; color: #666;'>{zone_name.replace('RWY_', '').replace('_', ' ')}</div>
            <div style='color: {color}; font-size: 12px;'>{predicted_value:.0f}m</div>
        </div>
        """
        folium.Marker(
            location=[lat + 0.0003, lon + 0.0003],
            icon=folium.DivIcon(html=label_html, class_name='rvr-label')
        ).add_to(m)
    print("   ✅ Static RVR markers added!")

# Save map
output_file = 'rvr_map_with_slider.html'
m.save(output_file)

print(f"\n=== SUMMARY ===")
print(f"✅ RVR map with time slider saved to: {output_file}")
print(f"Expected zones: {len(predicted_map)}")
print(f"CSV files found: {len(csv_files)}")
print(f"Latest data file: {os.path.basename(latest_file) if latest_file else 'None'}")

print(f"\n=== WHAT TO CHECK ===")
print("1. Open rvr_map_with_slider.html in your browser")
print("2. You should see:")
print("   - Green markers at runway beginnings")
print("   - A time slider at the bottom of the map")
print("   - Colored circles showing RVR predictions that change over time")
print("   - A legend showing RVR status colors")

print(f"\n3. Time Slider Features:")
print("   - Drag the slider to see predictions at different times")
print("   - Use play/pause buttons to animate through time")
print("   - Shows predictions for up to 12 hours ahead")
print("   - Each time step represents 10-minute intervals")

print(f"\n4. RVR Color coding:")
print("   - Green: ≥800m (Good visibility)")
print("   - Orange: 500-799m (Moderate visibility)")
print("   - Red: 200-499m (Poor visibility)")
print("   - Dark Red: <200m (Very poor visibility)")

print(f"\n5. If you don't see the slider:")
print("   - Check if the HTML file opens properly")
print("   - Check browser console for JavaScript errors")
print("   - Try refreshing the page")
print("   - Make sure you have internet connection for folium plugins")