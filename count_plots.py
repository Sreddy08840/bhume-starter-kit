
import json
from pathlib import Path

path = Path("data/34855_vadnerbhairav_chandavad_nashik/input.geojson")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

plots = data.get("features", [])
plot_numbers = [f["properties"]["plot_number"] for f in plots]

print(f"Total plots in input.geojson: {len(plots)}")
print(f"First 5 plot numbers: {plot_numbers[:5]}")
print(f"Are example truth plots present?")
truth_plots = ["1145", "1403", "1476", "1710", "2647"]
for tp in truth_plots:
    present = tp in plot_numbers
    print(f"  {tp}: {'YES' if present else 'NO'}")
