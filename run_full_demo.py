#!/usr/bin/env python3
"""
Full project demo: run baseline and check score on both villages
"""
from pathlib import Path
from bhume import load, score, write_predictions
from bhume.baseline import global_median_shift
import sys

print("="*80)
print("BhuMe Cadastral Boundary Correction - Full Demo")
print("="*80)

for village_dir in [
    "data/34855_vadnerbhairav_chandavad_nashik",
    "data/Malatavari"
]:
    print(f"\n{'='*80}")
    print(f"Processing Village: {Path(village_dir).name}")
    print(f"{'='*80}")
    print("  Step 1: Loading village data...")
    village = load(village_dir)
    print("  Step 2: Applying baseline (global median shift)...")
    preds = global_median_shift(village)
    print("  Step 3: Writing predictions to predictions.geojson...")
    write_predictions(Path(village_dir)/"predictions.geojson", preds)
    print("  Step 4: Self-scoring...")
    if village.example_truths is not None:
        result = score(preds, village)
        print("  Score complete!")
        for line in str(result).split('\n'):
            print(f"  {line}")


print("\n"+"="*80)
print("Demo complete! Both villages have predictions.geojson files!")
print("="*80)
