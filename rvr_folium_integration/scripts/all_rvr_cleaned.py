import re
import pandas as pd
from pathlib import Path

# 1. Adjust these two paths to your actual folders:
root_dir   = Path(r"C:\Users\alwyn\OneDrive\Desktop\IMD_internship\RVR2")
output_dir = Path(r"C:\Users\alwyn\OneDrive\Desktop\IMD_internship\Processed_RVR_Logs_New")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Regex to parse each line of a log.txt:
log_re = re.compile(r"^(\d{2}-\d{2}-\d{4})\t(\d{2}:\d{2}:\d{2})\t(.+)$")

def normalize_runway(point: str) -> str:
    """
    Standardize runway names but KEEP any L/R suffix.
    E.g. "RW28-L" stays "RW28-L", "Runway 28R" -> "RWY 28R".
    """
    p = point.upper().replace("RUNWAY", "RWY").strip()
    # no stripping of -L or -R here
    return p

for year_dir in sorted(root_dir.iterdir()):
    if not year_dir.is_dir() or not year_dir.name.isdigit():
        continue
    year = int(year_dir.name)
    print(f"\n▶ Processing year {year}")

    records = []  # will hold (Datetime, Runway, RVR)

    # Recursively find every .txt under this year
    for txt_file in year_dir.rglob("*.txt"):
        print(f"  • Reading {txt_file.relative_to(root_dir)}")
        with open(txt_file, "r") as f:
            for line in f:
                m = log_re.match(line.strip())
                if not m:
                    continue
                date_s, time_s, rest = m.groups()
                dt = pd.to_datetime(f"{date_s} {time_s}",
                                    dayfirst=True, errors="coerce")
                if pd.isna(dt):
                    continue

                # split the trailing runway/visibility parts
                for part in rest.split("\t"):
                    p = part.strip()
                    if not p:
                        continue
                    # runway name + RVR value (or '-')
                    *rp, rvr_str = p.rsplit(" ", 1)
                    runway_point = normalize_runway(" ".join(rp))
                    try:
                        rvr_val = int(rvr_str) if rvr_str != "-" else None
                    except:
                        rvr_val = None
                    records.append((dt, runway_point, rvr_val))

    if not records:
        print(f"  ⚠️ No records for {year}, skipping.")
        continue

    # Build DataFrame
    df = pd.DataFrame(records, columns=["Datetime", "Runway", "RVR"])

    # Floor to 10-minute bins
    df["Datetime"] = df["Datetime"].dt.floor("10min")

    # Aggregate by runway & bin (mean of RVR)
    df_agg = (
        df
        .dropna(subset=["RVR"])
        .groupby(["Datetime", "Runway"])["RVR"]
        .mean()
        .unstack(fill_value=None)   # each Runway (including suffix) becomes a column
        .reset_index()
    )

    # Sort and save
    df_agg.sort_values("Datetime", inplace=True)
    out_file = output_dir / f"RVR_{year}.csv"
    df_agg.to_csv(out_file, index=False)
    print(f"  ✅ Saved {out_file.name}")

print("\nAll years done.")
