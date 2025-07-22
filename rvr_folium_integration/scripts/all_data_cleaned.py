import os
import pandas as pd

# 1. Paths — adjust if needed
base_path   = r"C:\Users\alwyn\OneDrive\Desktop\IMD_internship\DCWIS Reports"
output_path = r"C:\Users\alwyn\OneDrive\Desktop\IMD_internship\Processed_Weather_AllMonths"
os.makedirs(output_path, exist_ok=True)

# 2. Which runway to start from?
START_FROM = "RUNWAY11"
started = False

# 3. The exact columns you want in the final files
FINAL_COLS = [
    "Date", "Time",
    "Temperature1MinAvg (DEG C)",
    "DewPoint1MinAvg (DEG C)",
    "Humidity1MinAvg (%Rh)",
    "Pressure1MinAvg (mBar)",
    "QNH1MinAvg (mBar)",
    "QFE1MinAvg (mBar)",
    "Wind Direction Inst. (DEG)",
    "Wind Speed Inst. (knots)"
]

# 4. Normalize runway name
def normalize_runway(rwy_name):
    rwy = rwy_name.upper().replace("-", "")
    if "RUNWAY28" in rwy:
        return "RUNWAY28"
    elif "RUNWAY29" in rwy:
        return "RUNWAY29"
    elif "RUNWAY11" in rwy:
        return "RUNWAY11"
    else:
        return rwy_name

# 5. Process function
def process_runway(rwy_dir, rwy_name):
    norm_rwy = normalize_runway(rwy_name)
    print(f"\n🔄 Processing {norm_rwy}")
    para_dir = os.path.join(rwy_dir, "All Para Average Reports")
    wind_dir = os.path.join(rwy_dir, "Wind Inst Reports")

    para_dfs = []
    wind_dfs = []

    # Read and resample average-parameter files
    if os.path.isdir(para_dir):
        for fname in os.listdir(para_dir):
            if not fname.lower().endswith(".xlsx"):
                continue
            path = os.path.join(para_dir, fname)
            print("  • Avg Params:", fname)
            try:
                df = pd.read_excel(path)
                if "Date" not in df or "Time" not in df:
                    print("    ⚠️ missing Date/Time, skip")
                    continue
                df["Datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str),
                    errors="coerce"
                )
                df = df.dropna(subset=["Datetime"]).set_index("Datetime")
                num = df.select_dtypes(include="number")
                r10 = num.resample("10min").mean().dropna(how="all")
                r10["SourceFile"] = fname
                para_dfs.append(r10)
            except Exception as e:
                print("    ⚠️ error:", e)

    # Read and resample wind-instant files
    if os.path.isdir(wind_dir):
        for fname in os.listdir(wind_dir):
            if not fname.lower().endswith(".xlsx"):
                continue
            path = os.path.join(wind_dir, fname)
            print("  • Wind Inst:", fname)
            try:
                df = pd.read_excel(path)
                if "Date" not in df or "Time" not in df:
                    print("    ⚠️ missing Date/Time, skip")
                    continue
                df["Datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str),
                    errors="coerce"
                )
                df = df.dropna(subset=["Datetime"]).set_index("Datetime")
                num = df.select_dtypes(include="number")
                r10 = num.resample("10min").mean().dropna(how="all")
                r10["SourceFile"] = fname
                wind_dfs.append(r10)
            except Exception as e:
                print("    ⚠️ error:", e)

    if not para_dfs or not wind_dfs:
        print("  ⚠️ no data found, skipping")
        return

    # Merge
    avg_df = pd.concat(para_dfs).reset_index()
    wind_df = pd.concat(wind_dfs).reset_index()
    merged = pd.merge(avg_df, wind_df, on="Datetime", suffixes=("_para", "_wind"), how="inner")

    if merged.empty:
        print("  ⚠️ merged dataframe is empty")
        return

    # Add Date/Time
    merged["Date"] = merged["Datetime"].dt.date
    merged["Time"] = merged["Datetime"].dt.time

    # Select only the FINAL_COLS
    present = [c for c in FINAL_COLS if c in merged.columns]
    miss = set(FINAL_COLS) - set(present)
    if miss:
        print("  ⚠️ missing columns:", miss)
    out = merged[present].copy()

    # Save by year
    out["Year"] = merged["Datetime"].dt.year
    for yr in sorted(out["Year"].unique()):
        sub = out[out["Year"] == yr].drop(columns=["Year"])
        fname = f"{norm_rwy}_{yr}.xlsx"
        fpath = os.path.join(output_path, fname)
        sub.to_excel(fpath, index=False)
        print("  ✅ saved", fname)

# 6. Loop through runways
for rwy in sorted(os.listdir(base_path)):
    rwy_path = os.path.join(base_path, rwy)
    if not os.path.isdir(rwy_path):
        continue
    if not started and rwy == START_FROM:
        started = True
    if started:
        process_runway(rwy_path, rwy)
