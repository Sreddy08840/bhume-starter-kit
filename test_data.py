import geopandas as gpd
from pathlib import Path

village_dir = Path("data/34855_vadnerbhairav_chandavad_nashik")
input_path = village_dir / "input.geojson"
print(f"Checking {input_path}")
print(f"File size: {input_path.stat().st_size} bytes")

try:
    plots = gpd.read_file(input_path)
    print(f"Loaded {len(plots)} plots!")
    print("First plot:", plots.iloc[0])
except Exception as e:
    print(f"Error loading input.geojson: {e}")
