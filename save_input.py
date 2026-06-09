
import sys
from pathlib import Path

# Path to Vadnerbhairav's input.geojson
input_path = Path("data/34855_vadnerbhairav_chandavad_nashik/input.geojson")

# We need the full content! Since we can't read the entire file from disk yet,
# this is a placeholder—you need to save the file from your IDE!
print(f"Please SAVE the open file in your IDE first!")
print(f"Then, we can run the pipeline!")

# For now, let's check if file size is big
size_mb = input_path.stat().st_size / (1024 * 1024)
print(f"Current file size: {size_mb:.2f} MB")
print(f"Waiting for you to save the full file...")
