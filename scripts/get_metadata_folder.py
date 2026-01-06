import os
import csv
import sys
import argparse
from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--mask_folder", type=str, default="patches_UTM_1_99")

mask_folder = argparser.parse_args().__dict__["mask_folder"]
folder_path = os.path.join(DATA_FOLDER, mask_folder)

rows = []

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".tif"):
        continue

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    try:
        date = parts[1]
        fsc = int(parts[2].replace("FSC", ""))
        lat = float(parts[3])
        lon = float(parts[4])
    except (IndexError, ValueError):
        print(f"Skipping invalid filename: {filename}")
        continue

    rows.append([date, fsc, lat, lon, filename])

output_path = os.path.join(folder_path, f"{mask_folder}.csv")

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "FSC", "latitude", "longitude", "filename"])
    writer.writerows(rows)

print(f"Saved: {output_path}")
print(f"Parsed {len(rows)} files.")